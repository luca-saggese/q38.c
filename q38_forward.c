#include "q38_forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_forward_matrix_from_tensor(const q38_gguf *model,
                                    const q38_tensor *tensor, size_t rows,
                                    size_t cols, q38_forward_matrix *out,
                                    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!model || !tensor || !out || !rows || !cols ||
        tensor->ndim != 2 || tensor->dim[0] != rows ||
        tensor->dim[1] != cols)
        return fail(error, error_len, "invalid file-backed forward matrix");
    if (tensor->type != 30 && tensor->type != 0)
        return fail(error, error_len, "unsupported file-backed matrix type");
    const void *data = q38_gguf_tensor_data(model, tensor);
    if (!data) return fail(error, error_len, "tensor payload is outside mmap");
    *out = (q38_forward_matrix){data, rows, cols,
                                tensor->type == 30 ? Q38_FORWARD_BF16
                                                   : Q38_FORWARD_F32};
    return true;
}

static float bf16(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float matrix_at(const q38_forward_matrix *matrix, size_t row,
                       size_t col) {
    if (matrix->dtype == Q38_FORWARD_BF16)
        return bf16(((const uint16_t *)matrix->data)[row * matrix->cols + col]);
    return ((const float *)matrix->data)[row * matrix->cols + col];
}

static bool matrix_ok(const q38_forward_matrix *matrix, size_t rows,
                      size_t cols) {
    return matrix && matrix->data && matrix->rows == rows &&
           matrix->cols == cols &&
           (matrix->dtype == Q38_FORWARD_F32 ||
            matrix->dtype == Q38_FORWARD_BF16);
}

static void rms(float *x, const float *weight, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(sum / n) + 1e-6f);
    for (size_t i = 0; i < n; ++i) x[i] *= scale * (weight ? weight[i] : 1.0f);
}

static void rope(float *x, size_t n, size_t rotary, size_t position,
                 float theta) {
    (void)n;
    if (!rotary) return;
    const float base = powf(theta, -2.0f / (float)rotary);
    float frequency = 1.0f;
    for (size_t i = 0; i < rotary / 2; ++i) {
        const float angle = (float)position * frequency;
        const float c = cosf(angle), s = sinf(angle);
        const float a = x[i], b = x[i + rotary / 2];
        x[i] = a * c - b * s;
        x[i + rotary / 2] = a * s + b * c;
        frequency *= base;
    }
}

static bool project(const q38_forward_matrix *matrix, const float *input,
                    size_t tokens, float *output) {
    if (!matrix_ok(matrix, matrix->rows, matrix->cols)) return false;
    for (size_t t = 0; t < tokens; ++t)
        for (size_t r = 0; r < matrix->rows; ++r) {
            float sum = 0.0f;
            for (size_t c = 0; c < matrix->cols; ++c)
                sum += matrix_at(matrix, r, c) * input[t * matrix->cols + c];
            output[t * matrix->rows + r] = sum;
        }
    return true;
}

bool q38_forward_qsa_state_init(q38_qsa_state *state,
                                const q38_forward_qsa_weights *weights,
                                char *error, size_t error_len) {
    if (!weights || !weights->kv_heads || !weights->head_dim ||
        !weights->index_dim)
        return fail(error, error_len, "invalid forward QSA dimensions");
    return q38_qsa_state_init(
        state, weights->kv_heads * weights->head_dim * sizeof(float),
        weights->kv_heads * weights->head_dim * sizeof(float),
        weights->index_dim * sizeof(float), error, error_len);
}

static size_t select_prefix(const q38_forward_qsa_weights *w,
                            const float *index_keys, size_t visible,
                            const float *query, size_t query_position,
                            uint32_t *selected, size_t stride) {
    const size_t complete = visible / w->ratio;
    const size_t group_budget = w->budget / w->ratio;
    const size_t groups = complete < group_budget ? complete : group_budget;
    const size_t tail = visible - complete * w->ratio;
    if (groups > SIZE_MAX / w->ratio ||
        groups * w->ratio > stride ||
        tail > stride - groups * w->ratio)
        return 0;
    float *scores = (float *)calloc(complete, sizeof(float));
    size_t *order = (size_t *)malloc(complete * sizeof(size_t));
    if ((complete && (!scores || !order))) {
        free(scores); free(order); return 0;
    }
    for (size_t g = 0; g < complete; ++g) {
        float total = 0.0f;
        const size_t begin = g * w->ratio;
        for (size_t h = 0; h < w->index_heads; ++h) {
            float qnorm = 0.0f, knorm = 0.0f, dot = 0.0f;
            float *key = (float *)calloc(w->index_dim, sizeof(float));
            if (!key) { free(scores); free(order); return 0; }
            for (size_t d = 0; d < w->index_dim; ++d) {
                for (size_t j = 0; j < w->ratio; ++j)
                    key[d] += index_keys[(begin + j) * w->index_dim + d];
                key[d] /= (float)w->ratio;
            }
            rms(key, w->index_k_norm, w->index_dim);
            rope(key, w->index_dim, w->rotary_dims, begin, w->rope_theta);
            const float *q = query + h * w->index_dim;
            for (size_t d = 0; d < w->index_dim; ++d) {
                dot += q[d] * key[d];
                qnorm += q[d] * q[d];
                knorm += key[d] * key[d];
            }
            free(key);
            dot /= sqrtf(knorm / (float)w->index_dim + 1e-6f);
            dot /= sqrtf(qnorm / (float)w->index_dim + 1e-6f);
            total += dot > 0.0f ? dot : 0.0f;
        }
        scores[g] = total / sqrtf((float)w->index_dim);
        order[g] = g;
    }
    for (size_t candidate = 1; candidate < complete; ++candidate) {
        size_t at = candidate;
        while (at > 0) {
            const size_t previous = order[at - 1];
            if (scores[previous] > scores[candidate] ||
                (scores[previous] == scores[candidate] &&
                 previous < candidate))
                break;
            --at;
        }
        for (size_t j = candidate; j > at; --j) order[j] = order[j - 1];
        order[at] = candidate;
    }
    size_t out = 0;
    for (size_t rank = 0; rank < groups; ++rank)
        for (size_t j = 0; j < w->ratio; ++j)
            selected[out++] = (uint32_t)(order[rank] * w->ratio + j);
    for (size_t i = complete * w->ratio; i < visible; ++i)
        selected[out++] = (uint32_t)i;
    free(scores); free(order);
    (void)query_position;
    return out;
}

bool q38_forward_qsa_ref(const q38_forward_qsa_weights *w,
                         q38_qsa_state *state, const float *hidden,
                         size_t token_count, float *output,
                         uint32_t *selected, size_t selected_stride,
                         size_t *selected_counts, char *error,
                         size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!w || !state || !hidden || !token_count || !output || !selected ||
        !selected_counts || !selected_stride || !w->hidden || !w->query_heads ||
        !w->kv_heads || !w->head_dim || !w->index_heads || !w->index_dim ||
        !w->ratio || !w->budget || w->query_heads % w->kv_heads ||
        !matrix_ok(&w->q_proj, w->query_heads * w->head_dim * 2, w->hidden) ||
        !matrix_ok(&w->k_proj, w->kv_heads * w->head_dim, w->hidden) ||
        !matrix_ok(&w->v_proj, w->kv_heads * w->head_dim, w->hidden) ||
        !matrix_ok(&w->o_proj, w->hidden, w->query_heads * w->head_dim) ||
        !matrix_ok(&w->index_qk_proj,
                   (w->index_heads + 1) * w->index_dim, w->hidden) ||
        !w->q_norm || !w->k_norm || !w->index_q_norm || !w->index_k_norm ||
        w->rotary_dims > w->head_dim || w->rotary_dims % 2)
        return fail(error, error_len, "invalid forward QSA arguments");
    memset(output, 0, token_count * w->hidden * sizeof(*output));
    const size_t q_rows = w->query_heads * w->head_dim * 2;
    const size_t k_rows = w->kv_heads * w->head_dim;
    const size_t index_rows = (w->index_heads + 1) * w->index_dim;
    const size_t attention_width = w->query_heads * w->head_dim;
    float *qfull = calloc(token_count * q_rows, sizeof(float));
    float *keys = calloc(token_count * k_rows, sizeof(float));
    float *values = calloc(token_count * k_rows, sizeof(float));
    float *index = calloc(token_count * index_rows, sizeof(float));
    float *raw_index = calloc(token_count * w->index_dim, sizeof(float));
    float *queries = calloc(token_count * w->query_heads * w->head_dim, sizeof(float));
    float *indexq = calloc(token_count * w->index_heads * w->index_dim, sizeof(float));
    float *attention = calloc(token_count * attention_width, sizeof(float));
    if (!qfull || !keys || !values || !index || !raw_index || !queries ||
        !indexq || !attention) {
        free(qfull); free(keys); free(values); free(index); free(raw_index); free(queries);
        free(indexq);
        free(attention);
        return fail(error, error_len, "forward activation allocation failed");
    }
    if (!project(&w->q_proj, hidden, token_count, qfull) ||
        !project(&w->k_proj, hidden, token_count, keys) ||
        !project(&w->v_proj, hidden, token_count, values) ||
        !project(&w->index_qk_proj, hidden, token_count, index)) {
        free(qfull); free(keys); free(values); free(index); free(raw_index); free(queries);
        free(indexq);
        free(attention);
        return fail(error, error_len, "forward projection failed");
    }
    for (size_t t = 0; t < token_count; ++t) {
        for (size_t h = 0; h < w->query_heads; ++h) {
            float *q = queries + (t * w->query_heads + h) * w->head_dim;
            memcpy(q, qfull + t * q_rows + h * w->head_dim * 2,
                   w->head_dim * sizeof(float));
            rms(q, w->q_norm, w->head_dim);
            rope(q, w->head_dim, w->rotary_dims, state->position + t,
                 w->rope_theta);
        }
        for (size_t h = 0; h < w->kv_heads; ++h) {
            float *k = keys + (t * w->kv_heads + h) * w->head_dim;
            rms(k, w->k_norm, w->head_dim);
            rope(k, w->head_dim, w->rotary_dims, state->position + t,
                 w->rope_theta);
        }
        for (size_t h = 0; h < w->index_heads; ++h) {
            float *q = indexq + (t * w->index_heads + h) * w->index_dim;
            memcpy(q, index + t * index_rows + h * w->index_dim,
                   w->index_dim * sizeof(float));
            rms(q, w->index_q_norm, w->index_dim);
            rope(q, w->index_dim, w->rotary_dims, state->position + t,
                 w->rope_theta);
        }
        memcpy(raw_index + t * w->index_dim,
               index + t * index_rows + w->index_heads * w->index_dim,
               w->index_dim * sizeof(float));
    }
    if (!q38_qsa_state_append(state, keys, values, raw_index, token_count,
                              error, error_len)) {
        free(qfull); free(keys); free(values); free(index); free(raw_index); free(queries);
        free(indexq);
        return false;
    }
    for (size_t t = 0; t < token_count; ++t) {
        const size_t visible = state->position - token_count + t + 1;
        const size_t count = select_prefix(w, (const float *)state->index_k.data,
                                           visible, indexq + t * w->index_heads *
                                           w->index_dim, state->position,
                                           selected + t * selected_stride,
                                           selected_stride);
        if (!count || count > selected_stride)
            return fail(error, error_len, "forward QSA selection failed");
        selected_counts[t] = count;
        for (size_t h = 0; h < w->query_heads; ++h) {
            const size_t kvh = h / (w->query_heads / w->kv_heads);
            float max_score = -INFINITY;
            float *numerator = calloc(w->head_dim, sizeof(float));
            float denom = 0.0f;
            if (!numerator)
                return fail(error, error_len, "forward attention allocation failed");
            for (size_t j = 0; j < count; ++j) {
                const size_t row = selected[t * selected_stride + j];
                float score = 0.0f;
                const float *q = queries + (t * w->query_heads + h) * w->head_dim;
                const float *k = (const float *)state->main_k.data +
                    (row * w->kv_heads + kvh) * w->head_dim;
                for (size_t d = 0; d < w->head_dim; ++d) score += q[d] * k[d];
                score /= sqrtf((float)w->head_dim);
                if (score > max_score) max_score = score;
            }
            for (size_t j = 0; j < count; ++j) {
                const size_t row = selected[t * selected_stride + j];
                const float *q = queries + (t * w->query_heads + h) * w->head_dim;
                const float *k = (const float *)state->main_k.data +
                    (row * w->kv_heads + kvh) * w->head_dim;
                const float *v = (const float *)state->main_v.data +
                    (row * w->kv_heads + kvh) * w->head_dim;
                float score = 0.0f;
                for (size_t d = 0; d < w->head_dim; ++d) score += q[d] * k[d];
                const float weight = expf(score / sqrtf((float)w->head_dim) -
                                          max_score);
                denom += weight;
                for (size_t d = 0; d < w->head_dim; ++d)
                    numerator[d] += weight * v[d];
            }
            for (size_t d = 0; d < w->head_dim; ++d)
                {
                const float gate = 1.0f /
                    (1.0f + expf(-(qfull[t * q_rows + h * w->head_dim * 2 +
                                    w->head_dim + d])));
                attention[t * attention_width + h * w->head_dim + d] =
                    numerator[d] / denom * gate;
                }
            free(numerator);
        }
        float *projected = output + t * w->hidden;
        float *tmp = calloc(w->query_heads * w->head_dim, sizeof(float));
        if (!tmp) return fail(error, error_len, "forward output allocation failed");
        memcpy(tmp, attention + t * attention_width,
               attention_width * sizeof(float));
        for (size_t r = 0; r < w->o_proj.rows; ++r) {
            projected[r] = 0.0f;
            for (size_t c = 0; c < w->o_proj.cols; ++c)
                projected[r] += matrix_at(&w->o_proj, r, c) * tmp[c];
        }
        free(tmp);
    }
    free(qfull); free(keys); free(values); free(index); free(raw_index);
    free(queries); free(indexq); free(attention);
    return true;
}
