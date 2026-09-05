#include "q38_forward.h"
#include "q38_gdn_ref.h"
#include "q38_gr_ref.h"
#include "q38_moe.h"
#include "q38_moe_ref.h"
#include "q38_ple_ref.h"
#include "q38_quant.h"

#define Q38_FULL_QSA_SELECTED_STRIDE 2051u

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static void rms(float *x, const float *weight, size_t n, bool one_plus) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(sum / n) + 1e-6f);
    for (size_t i = 0; i < n; ++i) {
        const float multiplier = weight ? (one_plus ? 1.0f + weight[i]
                                                    : weight[i])
                                        : 1.0f;
        x[i] *= scale * multiplier;
    }
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
            float dot = 0.0f;
            float *key = (float *)calloc(w->index_dim, sizeof(float));
            if (!key) { free(scores); free(order); return 0; }
            for (size_t d = 0; d < w->index_dim; ++d) {
                for (size_t j = 0; j < w->ratio; ++j)
                    key[d] += index_keys[(begin + j) * w->index_dim + d];
                key[d] /= (float)w->ratio;
            }
            rms(key, w->index_k_norm, w->index_dim, true);
            rope(key, w->index_dim, w->rotary_dims, begin, w->rope_theta);
            const float *q = query + h * w->index_dim;
            for (size_t d = 0; d < w->index_dim; ++d)
                dot += q[d] * key[d];
            free(key);
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
    const size_t base_position = state->position;
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
        const size_t position = base_position + t;
        for (size_t h = 0; h < w->query_heads; ++h) {
            float *q = queries + (t * w->query_heads + h) * w->head_dim;
            memcpy(q, qfull + t * q_rows + h * w->head_dim * 2,
                   w->head_dim * sizeof(float));
            rms(q, w->q_norm, w->head_dim, true);
            rope(q, w->head_dim, w->rotary_dims, position,
                 w->rope_theta);
        }
        for (size_t h = 0; h < w->kv_heads; ++h) {
            float *k = keys + (t * w->kv_heads + h) * w->head_dim;
            rms(k, w->k_norm, w->head_dim, true);
            rope(k, w->head_dim, w->rotary_dims, position,
                 w->rope_theta);
        }
        for (size_t h = 0; h < w->index_heads; ++h) {
            float *q = indexq + (t * w->index_heads + h) * w->index_dim;
            memcpy(q, index + t * index_rows + h * w->index_dim,
                   w->index_dim * sizeof(float));
            rms(q, w->index_q_norm, w->index_dim, true);
            rope(q, w->index_dim, w->rotary_dims, position,
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

/* The complete graph below intentionally keeps tensor payloads in the GGUF
 * mapping.  Only the current layer's activations and one decoded row are
 * materialized; this is the reference path used to classify CUDA drift. */

static bool full_fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static q38_forward_matvec_backend full_backend;
static q38_forward_matrix_backend full_matrix_backend;
static q38_forward_expert_backend full_expert_backend;
static q38_forward_moe_layer_backend full_moe_layer_backend;
static void *full_backend_user;
static bool full_backend_strict;
static uint64_t full_backend_rows;
static uint64_t full_scalar_rows;
static uint64_t full_backend_declines;
static bool full_perf_strict;
static q38_forward_diagnostics *full_diagnostics;
static uint32_t full_current_layer;

static void full_backend_context(const q38_tensor *tensor, size_t rows,
                                 size_t cols, const char *stage) {
    if (full_diagnostics && full_diagnostics->backend_context)
        full_diagnostics->backend_context(full_current_layer, stage, tensor,
                                          rows, cols,
                                          full_diagnostics->trace_user);
}

static double full_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static bool full_emit_stage(q38_forward_diagnostics *diagnostics,
                            const char *name, uint64_t backend_rows,
                            uint64_t scalar_rows, uint64_t declines,
                            double elapsed_ms, char *error,
                            size_t error_len) {
    if (!diagnostics || !diagnostics->stage_trace) return true;
    const q38_forward_stage_usage usage = {
        .name = name,
        .matrix_calls = 1,
        .backend_rows = backend_rows,
        .scalar_rows = scalar_rows,
        .backend_declines = declines,
        .elapsed_ms = elapsed_ms,
        .layer = full_current_layer,
        .logical_stage = name,
    };
    return diagnostics->stage_trace(&usage, diagnostics->trace_user, error,
                                    error_len);
}

static bool full_mul(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

static bool full_tensor_rows(const q38_tensor *tensor, size_t *rows,
                             size_t *cols) {
    if (!tensor || tensor->ndim < 1 || tensor->ndim > 3 ||
        tensor->dim[tensor->ndim - 1] > SIZE_MAX)
        return false;
    size_t r = 1;
    for (uint32_t i = 0; i + 1 < tensor->ndim; ++i) {
        if (tensor->dim[i] > SIZE_MAX || !full_mul(r, (size_t)tensor->dim[i], &r))
            return false;
    }
    *rows = r;
    *cols = (size_t)tensor->dim[tensor->ndim - 1];
    return *rows && *cols;
}

static bool full_tensor_data(const q38_gguf *model, const q38_tensor *tensor,
                             const void **data, size_t *rows, size_t *cols,
                             size_t *row_bytes) {
    if (!model || !tensor || !data || !rows || !cols || !row_bytes ||
        !full_tensor_rows(tensor, rows, cols) ||
        tensor->bytes % *rows != 0 ||
        tensor->bytes / *rows > SIZE_MAX)
        return false;
    *data = q38_gguf_tensor_data(model, tensor);
    *row_bytes = (size_t)(tensor->bytes / *rows);
    return *data != NULL && *row_bytes != 0;
}

static float full_bf16_to_float(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static float full_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* Match the round-to-nearest-even conversion used by torch.bfloat16. */
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    return full_bf16_to_float((uint16_t)(bits >> 16));
}

static float full_cast_forward_dtype(float value, q38_forward_dtype dtype) {
    return dtype == Q38_FORWARD_BF16 ? full_float_to_bf16(value) : value;
}

static float full_tensor_scalar(const q38_gguf *model, const q38_tensor *tensor,
                                size_t row, size_t column, float *scratch,
                                size_t scratch_count) {
    const void *data;
    size_t rows, cols, row_bytes;
    if (!full_tensor_data(model, tensor, &data, &rows, &cols, &row_bytes) ||
        row >= rows || column >= cols)
        return NAN;
    if (tensor->type == 30) {
        uint16_t bits;
        memcpy(&bits, (const unsigned char *)data + row * row_bytes +
                   column * sizeof(bits), sizeof(bits));
        return full_bf16_to_float(bits);
    }
    if (tensor->type == 0) {
        float value;
        memcpy(&value, (const unsigned char *)data + row * row_bytes +
                   column * sizeof(value), sizeof(value));
        return value;
    }
    if (tensor->type == 8 && cols % 32u == 0) {
        const unsigned char *block =
            (const unsigned char *)data + row * row_bytes +
            (column / 32u) * 34u;
        uint16_t bits;
        memcpy(&bits, block, sizeof(bits));
        return q38_half_to_float(bits) *
            (float)((const int8_t *)(block + 2))[column % 32u];
    }
    if ((tensor->type == Q38_QUANT_Q2_K ||
         tensor->type == Q38_QUANT_Q4_K) &&
        scratch && scratch_count >= cols &&
        q38_quant_dequantize_row(tensor->type,
                                 (const unsigned char *)data + row * row_bytes,
                                 cols / Q38_QUANT_QK_K, scratch, cols, NULL, 0))
        return scratch[column];
    return NAN;
}

static float full_vector_scalar(const q38_gguf *model, const q38_tensor *tensor,
                                size_t index, float *scratch,
                                size_t scratch_count) {
    return full_tensor_scalar(model, tensor, 0, index, scratch, scratch_count);
}

static bool full_tensor_row(const q38_gguf *model, const q38_tensor *tensor,
                            size_t row, float *out, size_t count,
                            char *error, size_t error_len) {
    const void *data;
    size_t rows, cols, row_bytes;
    if (!full_tensor_data(model, tensor, &data, &rows, &cols, &row_bytes) ||
        row >= rows || count != cols)
        return full_fail(error, error_len, "invalid file-backed tensor row");
    if (tensor->type == 30 || tensor->type == 0 || tensor->type == 8) {
        for (size_t i = 0; i < cols; ++i)
            out[i] = full_tensor_scalar(model, tensor, row, i, NULL, 0);
        return true;
    }
    if (tensor->type == Q38_QUANT_Q2_K || tensor->type == Q38_QUANT_Q4_K)
        return q38_quant_dequantize_row(
            tensor->type, (const unsigned char *)data + row * row_bytes,
            cols / Q38_QUANT_QK_K, out, cols, error, error_len);
    return full_fail(error, error_len, "unsupported file-backed tensor type");
}

static bool full_row_dot(const q38_gguf *model, const q38_tensor *tensor,
                         size_t row, const float *input, size_t count,
                         float *scratch, float *out, char *error,
                         size_t error_len) {
    if (full_perf_strict) {
        return full_fail(error, error_len,
                         "Q38_PERF_STRICT forbids row-oriented backend path");
    }
    if (full_backend) {
        full_backend_context(tensor, 1, count, "row_matvec");
        if (full_backend(model, tensor, row, input, count, out,
                         full_backend_user, error, error_len)) {
            ++full_backend_rows;
            return true;
        }
        ++full_backend_declines;
        /*
         * A backend may decline very wide diagnostic-only matrices by
         * returning false without an error.  A populated error remains a
         * hard execution failure.
         */
        if (error && error_len && error[0] != '\0') return false;
        if (full_backend_strict) {
            if (error && error_len)
                snprintf(error, error_len,
                         "CUDA backend declined row %zu of type %u; scalar fallback disabled",
                         row, tensor ? tensor->type : UINT32_MAX);
            return false;
        }
    }
    ++full_scalar_rows;
    const void *data;
    size_t rows, cols, row_bytes;
    if (!input || !out ||
        !full_tensor_data(model, tensor, &data, &rows, &cols, &row_bytes) ||
        row >= rows || count != cols)
        return full_fail(error, error_len, "invalid tensor matvec geometry");
    if (tensor->type == 30 || tensor->type == 0) {
        float sum = 0.0f;
        const unsigned char *row_data =
            (const unsigned char *)data + row * row_bytes;
        if (tensor->type == 30) {
            const uint16_t *values = (const uint16_t *)row_data;
            for (size_t c = 0; c < cols; ++c)
                sum += full_bf16_to_float(values[c]) * input[c];
        } else {
            const float *values = (const float *)row_data;
            for (size_t c = 0; c < cols; ++c) sum += values[c] * input[c];
        }
        *out = sum;
        return true;
    }
    if (tensor->type == 8 && cols % 32u == 0) {
        float sum = 0.0f;
        const unsigned char *row_data =
            (const unsigned char *)data + row * row_bytes;
        for (size_t c = 0; c < cols; ++c) {
            const unsigned char *block = row_data + (c / 32u) * 34u;
            uint16_t bits;
            memcpy(&bits, block, sizeof(bits));
            sum += q38_half_to_float(bits) *
                   (float)((const int8_t *)(block + 2))[c % 32u] * input[c];
        }
        *out = sum;
        return true;
    }
    if (tensor->type != Q38_QUANT_Q2_K &&
        tensor->type != Q38_QUANT_Q4_K)
        return full_fail(error, error_len, "unsupported tensor matvec type");
    if (!scratch)
        return full_fail(error, error_len, "missing tensor row scratch");
    if (scratch == input)
        return full_fail(error, error_len,
                         "quantized tensor row scratch aliases input");
    if (!q38_quant_dequantize_row(
            tensor->type, (const unsigned char *)data + row * row_bytes,
            cols / Q38_QUANT_QK_K, scratch, cols, error, error_len))
        return false;
    float sum = 0.0f;
    for (size_t c = 0; c < cols; ++c) sum += scratch[c] * input[c];
    *out = sum;
    return true;
}

static bool full_matvec(const q38_gguf *model, const q38_tensor *tensor,
                        const float *input, size_t rows, size_t cols,
                        float *output, float *scratch, char *error,
                        size_t error_len, const char *stage) {
    size_t actual_rows, actual_cols;
    const void *data;
    size_t row_bytes;
    if (!full_tensor_data(model, tensor, &data, &actual_rows, &actual_cols,
                          &row_bytes) ||
        actual_rows != rows || actual_cols != cols)
        return full_fail(error, error_len, "tensor matrix shape mismatch");
    if (full_matrix_backend) {
        full_backend_context(tensor, rows, cols, stage ? stage : "matvec");
        const double started = full_now_ms();
        if (full_matrix_backend(model, tensor, input, rows, cols, output,
                                full_backend_user, error, error_len)) {
            full_backend_rows += rows;
            return full_emit_stage(full_diagnostics, stage ? stage : "matvec",
                                   rows, 0, 0, full_now_ms() - started,
                                   error, error_len);
        }
        ++full_backend_declines;
        if (error && error_len && error[0] != '\0') return false;
        if (full_backend_strict)
            return full_fail(error, error_len,
                             "CUDA matrix backend declined; scalar fallback disabled");
    }
    const uint64_t backend_before = full_backend_rows;
    const uint64_t scalar_before = full_scalar_rows;
    const uint64_t declines_before = full_backend_declines;
    const double started = full_now_ms();
    for (size_t row = 0; row < rows; ++row)
        if (!full_row_dot(model, tensor, row, input, cols, scratch,
                          &output[row], error, error_len))
            return false;
    return full_emit_stage(
        full_diagnostics, stage ? stage : "matvec",
        full_backend_rows - backend_before, full_scalar_rows - scalar_before,
        full_backend_declines - declines_before, full_now_ms() - started,
        error, error_len);
}

static void full_rms(float *x, const float *weight, size_t n, bool one_plus) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + 1e-6f);
    for (size_t i = 0; i < n; ++i) {
        const float multiplier = weight ? (one_plus ? 1.0f + weight[i]
                                                    : weight[i])
                                        : 1.0f;
        x[i] *= scale * multiplier;
    }
}

static void full_grouped_rms(float *x, const float *weight, size_t tokens,
                             size_t streams, size_t hidden, bool one_plus) {
    const size_t width = streams * hidden;
    for (size_t t = 0; t < tokens; ++t)
        for (size_t stream = 0; stream < streams; ++stream)
            full_rms(x + t * width + stream * hidden,
                     weight + stream * hidden, hidden, one_plus);
}

static bool full_decode_vector(const q38_gguf *model, const q38_tensor *tensor,
                               float *out, size_t n, char *error,
                               size_t error_len) {
    return full_tensor_row(model, tensor, 0, out, n, error, error_len);
}

static q38_tensor *full_named_global(const q38_weights *weights,
                                     const char *part) {
    for (uint32_t i = 0; i < weights->global_tensor_count; ++i) {
        q38_tensor *tensor = weights->global_tensor[i];
        const size_t n = strlen(part);
        bool found = false;
        for (uint64_t p = 0; p + n <= tensor->name.len; ++p)
            if (memcmp(tensor->name.ptr + p, part, n) == 0) found = true;
        if (found)
            return tensor;
    }
    return NULL;
}

static q38_tensor *full_named_ple(const q38_layer_weights *layer,
                                  const char *part) {
    for (uint32_t i = 0; i < layer->ple_tensor_count; ++i) {
        q38_tensor *tensor = layer->ple_tensor[i];
        const size_t n = strlen(part);
        bool found = false;
        for (uint64_t p = 0; p + n <= tensor->name.len; ++p)
            if (memcmp(tensor->name.ptr + p, part, n) == 0) found = true;
        if (found)
            return tensor;
    }
    return NULL;
}

static bool full_boundary_trace(uint32_t layer, const char *boundary,
                                const float *values, size_t token_count,
                                size_t width,
                                q38_forward_diagnostics *diagnostics,
                                char *error, size_t error_len) {
    if (!diagnostics || !diagnostics->boundary_trace ||
        (layer != 1 && layer != 9 && layer != UINT32_MAX))
        return true;
    return diagnostics->boundary_trace(
        layer, boundary, values, token_count, width,
        diagnostics->trace_user, error, error_len);
}

static bool full_gr_read(const q38_gguf *model, const q38_gr_weights *weights,
                         const float *residual, size_t tokens, float *input,
                         float *normed, float *down, float *up, float *scratch,
                         char *error, size_t error_len) {
    const size_t width = 4u * Q38_GR_HIDDEN;
    for (size_t t = 0; t < tokens; ++t) {
        memset(input + t * Q38_GR_HIDDEN, 0,
               Q38_GR_HIDDEN * sizeof(float));
        memcpy(normed + t * width, residual + t * width,
               width * sizeof(float));
        for (size_t s = 0; s < 4; ++s) {
            float gamma[Q38_GR_HIDDEN];
            for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
                gamma[d] = full_vector_scalar(model, weights->hc_norm,
                                              s * Q38_GR_HIDDEN + d, scratch,
                                              Q38_GR_HIDDEN);
            full_rms(normed + t * width + s * Q38_GR_HIDDEN, gamma,
                     Q38_GR_HIDDEN, true);
        }
        if (!full_matvec(model, weights->input_mix_weight_down,
                         normed + t * width, 320, width, down, scratch,
                         error, error_len, "gr_read_down"))
            return false;
        for (size_t r = 0; r < 320; ++r)
            down[r] = down[r] / 4.0f /
                      (1.0f + expf(-down[r] / 4.0f));
        if (!full_matvec(model, weights->input_mix_weight_up, down, width,
                         320, up, scratch, error, error_len, "gr_read_up"))
            return false;
        for (size_t s = 0; s < 4; ++s)
            for (size_t d = 0; d < Q38_GR_HIDDEN; ++d) {
                float gate = 1.0f /
                    (1.0f + expf(-up[s * Q38_GR_HIDDEN + d]));
                input[t * Q38_GR_HIDDEN + d] +=
                    gate * normed[t * width + s * Q38_GR_HIDDEN + d] / 4.0f;
            }
    }
    return true;
}

static bool full_gr_write(const q38_gguf *model, const q38_gr_weights *weights,
                          const float *residual, const float *block,
                          size_t tokens, float *updated, float *normed,
                          float *inject, float *scratch, char *error,
                          size_t error_len) {
    const size_t width = 4u * Q38_GR_HIDDEN;
    for (size_t t = 0; t < tokens; ++t) {
        memcpy(normed + t * width, residual + t * width,
               width * sizeof(float));
        for (size_t s = 0; s < 4; ++s) {
            float gamma[Q38_GR_HIDDEN];
            for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
                gamma[d] = full_vector_scalar(model, weights->hc_norm,
                                              s * Q38_GR_HIDDEN + d, scratch,
                                              Q38_GR_HIDDEN);
            full_rms(normed + t * width + s * Q38_GR_HIDDEN, gamma,
                     Q38_GR_HIDDEN, true);
        }
        if (!full_matvec(model, weights->block_inject_weight,
                         normed + t * width, 4, width, inject, scratch,
                         error, error_len, "gr_write_inject"))
            return false;
        for (size_t s = 0; s < 4; ++s) {
            const float scale = 2.0f /
                (1.0f + expf(-inject[s] / 4.0f));
            for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
                updated[t * width + s * Q38_GR_HIDDEN + d] =
                    residual[t * width + s * Q38_GR_HIDDEN + d] +
                    scale * block[t * Q38_GR_HIDDEN + d];
        }
    }
    return true;
}

static bool full_gdn(const q38_gguf *model, const q38_layer_weights *layer,
                     q38_forward_state *state, const float *input,
                     size_t tokens, uint32_t layer_number, float *output,
                     float *scratch, char *error, size_t error_len) {
    const size_t qkv_n = 10240, z_n = 6144, heads = 48, dim = 128;
    float *qkv = calloc(tokens * qkv_n, sizeof(float));
    float *z = calloc(tokens * z_n, sizeof(float));
    float *a = calloc(tokens * heads, sizeof(float));
    float *b = calloc(tokens * heads, sizeof(float));
    float *conv = calloc(tokens * qkv_n, sizeof(float));
    float *q = calloc(tokens * heads * dim, sizeof(float));
    float *k = calloc(tokens * heads * dim, sizeof(float));
    float *v = calloc(tokens * heads * dim, sizeof(float));
    float *decay = calloc(tokens * heads, sizeof(float));
    float *beta = calloc(tokens * heads, sizeof(float));
    float *gdn_out = calloc(tokens * heads * dim, sizeof(float));
    if (!qkv || !z || !a || !b || !conv || !q || !k || !v || !decay ||
        !beta || !gdn_out) {
        free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k);
        free(v); free(decay); free(beta); free(gdn_out);
        return full_fail(error, error_len, "GDN activation allocation failed");
    }
    const q38_tensor *proj[] = {layer->gdn.in_proj_qkv, layer->gdn.in_proj_z,
                                layer->gdn.in_proj_a, layer->gdn.in_proj_b};
    const size_t sizes[] = {qkv_n, z_n, heads, heads};
    float *outs[] = {qkv, z, a, b};
    for (size_t t = 0; t < tokens; ++t)
        for (size_t p = 0; p < 4; ++p)
            if (!full_matvec(model, proj[p],
                             input + t * Q38_GR_HIDDEN,
                             sizes[p], Q38_GR_HIDDEN,
                             outs[p] + t * sizes[p], scratch, error, error_len,
                             p == 0 ? "gdn_qkv_projection" :
                             p == 1 ? "gdn_z_projection" :
                             p == 2 ? "gdn_a_projection" : "gdn_b_projection"))
                goto fail;
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t c = 0; c < qkv_n; ++c) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < 4; ++tap) {
                const size_t current = 3 + t;
                const size_t source = current - (3 - tap);
                const float sample = source < 3
                    ? state->storage.conv_history[
                        (size_t)q38_gdn_slot_for_layer(
                            &state->storage.layout, layer_number) *
                        3u * qkv_n + source * qkv_n + c]
                    : qkv[(source - 3) * qkv_n + c];
                sum += full_tensor_scalar(model, layer->gdn.conv1d, c,
                                          tap, scratch, qkv_n) * sample;
            }
            conv[t * qkv_n + c] = sum / (1.0f + expf(-sum));
        }
    }
    {
        const int slot = q38_gdn_slot_for_layer(&state->storage.layout,
                                                layer_number);
        float *history = state->storage.conv_history +
            (size_t)slot * 3u * qkv_n;
        for (size_t tail = 0; tail < 3; ++tail) {
            const size_t source = tokens + tail;
            if (source < 3) memmove(history + tail * qkv_n,
                                    history + source * qkv_n,
                                    qkv_n * sizeof(float));
            else memcpy(history + tail * qkv_n,
                        qkv + (source - 3) * qkv_n,
                        qkv_n * sizeof(float));
        }
    }
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < 16; ++h)
            for (size_t d = 0; d < dim; ++d) {
                float qv = conv[t * qkv_n + h * dim + d];
                float kv = conv[t * qkv_n + 16 * dim + h * dim + d];
                for (size_t r = 0; r < 3; ++r) {
                    q[t * heads * dim + (h * 3 + r) * dim + d] = qv;
                    k[t * heads * dim + (h * 3 + r) * dim + d] = kv;
                }
            }
        for (size_t h = 0; h < heads; ++h)
            for (size_t d = 0; d < dim; ++d)
                v[t * heads * dim + h * dim + d] =
                    conv[t * qkv_n + 32 * dim + h * dim + d];
        for (size_t h = 0; h < heads; ++h) {
            float qn = 0.0f, kn = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                qn += q[t * heads * dim + h * dim + d] *
                      q[t * heads * dim + h * dim + d];
                kn += k[t * heads * dim + h * dim + d] *
                      k[t * heads * dim + h * dim + d];
            }
            const float qs = 1.0f / sqrtf(qn + 1e-6f);
            const float ks = 1.0f / sqrtf(kn + 1e-6f);
            for (size_t d = 0; d < dim; ++d) {
                q[t * heads * dim + h * dim + d] *= qs;
                k[t * heads * dim + h * dim + d] *= ks;
            }
            const float av = a[t * heads + h] +
                full_vector_scalar(model, layer->gdn.dt_bias, h,
                                   scratch, heads);
            decay[t * heads + h] =
            expf(-expf(full_vector_scalar(model, layer->gdn.A_log, h,
                                               scratch, heads)) *
                     log1pf(expf(av)));
            beta[t * heads + h] = 1.0f / (1.0f + expf(-b[t * heads + h]));
        }
    }
    {
        const int slot = q38_gdn_slot_for_layer(&state->storage.layout,
                                                layer_number);
        if (!q38_gdn_ref_run(
                q38_state_recurrent_slot(&state->storage, (uint32_t)slot), 1,
                tokens, q, k, v, decay, beta, 1.0f / sqrtf(128.0f), gdn_out))
            goto fail;
    }
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            float norm[128];
            memcpy(norm, gdn_out + (t * heads + h) * dim,
                   sizeof(norm));
            float nw[128];
            if (!full_decode_vector(model, layer->gdn.norm, nw, 128,
                                    error, error_len)) goto fail;
            full_rms(norm, nw, 128, false);
            for (size_t d = 0; d < dim; ++d)
                gdn_out[(t * heads + h) * dim + d] =
                    norm[d] * (1.0f / (1.0f + expf(-z[t * z_n + h * dim + d])));
        }
        if (!full_matvec(model, layer->gdn.out_proj,
                         gdn_out + t * heads * dim, Q38_GR_HIDDEN,
                         z_n, output + t * Q38_GR_HIDDEN,
                         scratch, error, error_len, "gdn_output_projection"))
            goto fail;
    }
    free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v);
    free(decay); free(beta); free(gdn_out);
    return true;
fail:
    free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v);
    free(decay); free(beta); free(gdn_out);
    return false;
}

static bool full_expert_down(const q38_gguf *model, const q38_tensor *tensor,
                             bool quantized, size_t expert, const float *x,
                             float *out, float *scratch, char *error,
                             size_t error_len) {
    for (size_t d = 0; d < Q38_GR_HIDDEN; ++d) {
        float value = 0.0f;
        for (size_t i = 0; i < 640; ++i) {
            const size_t row = quantized ? expert * 640u + i
                                         : expert * Q38_GR_HIDDEN + d;
            const size_t column = quantized ? d : i;
            const float weight = full_tensor_scalar(
                model, tensor, row, column, scratch, Q38_GR_HIDDEN);
            if (!isfinite(weight))
                return full_fail(error, error_len,
                                 "unsupported routed down tensor layout");
            value += weight * x[i];
        }
        out[d] = value;
    }
    return true;
}

static bool full_moe(const q38_gguf *model, const q38_layer_weights *layer,
                     bool quantized, const float *input, size_t tokens,
                     uint32_t layer_number, float *output, float *scratch,
                     q38_forward_diagnostics *diagnostics, char *error,
                     size_t error_len) {
    q38_moe_weights weights;
    if (!q38_moe_bind_layer(layer, quantized, &weights, error, error_len))
        return false;
    float *intermediate = calloc(640, sizeof(float));
    float *routed = calloc(Q38_GR_HIDDEN, sizeof(float));
    float *shared = calloc(Q38_GR_HIDDEN, sizeof(float));
    float *gate = calloc(640, sizeof(float));
    float *up = calloc(640, sizeof(float));
    if (!intermediate || !routed || !shared || !gate || !up) {
        free(intermediate); free(routed); free(shared); free(gate); free(up);
        return full_fail(error, error_len, "MoE activation allocation failed");
    }
    for (size_t t = 0; t < tokens; ++t) {
        const float *x = input + t * Q38_GR_HIDDEN;
        const q38_forward_dtype router_dtype =
            weights.router->type == 30 ? Q38_FORWARD_BF16
                                        : Q38_FORWARD_F32;
        if (layer_number == 9 && diagnostics &&
            diagnostics->pre_router_trace) {
            const q38_pre_router_trace pre_router = {
                .router_input = x,
                .router_input_count = Q38_MOE_HIDDEN,
                .gr_output = x,
                .gr_output_count = Q38_MOE_HIDDEN,
                .router = weights.router,
            };
            if (!diagnostics->pre_router_trace(
                    layer_number, &pre_router, diagnostics->trace_user,
                    error, error_len))
                goto fail;
        }
        if (!full_boundary_trace(layer_number, "router_input", x, 1,
                                     Q38_GR_HIDDEN, diagnostics, error,
                                     error_len))
            goto fail;
        float logits_pre_cast[Q38_MOE_EXPERTS];
        float logits_effective[Q38_MOE_EXPERTS];
        float probs_pre_cast[Q38_MOE_EXPERTS];
        float probs_effective[Q38_MOE_EXPERTS];
        float max_pre_cast = -INFINITY;
        float max_effective = -INFINITY;
        if (full_matrix_backend) {
            full_backend_context(weights.router, Q38_MOE_EXPERTS,
                                Q38_GR_HIDDEN, "moe_router");
            const double started = full_now_ms();
            if (!full_matrix_backend(
                    model, weights.router, x, Q38_MOE_EXPERTS,
                    Q38_GR_HIDDEN, logits_pre_cast, full_backend_user, error,
                    error_len)) {
                ++full_backend_declines;
                if (error && error_len && error[0] != '\0') goto fail;
                goto fail;
            }
            full_backend_rows += Q38_MOE_EXPERTS;
            if (!full_emit_stage(full_diagnostics, "moe_router",
                                 Q38_MOE_EXPERTS, 0, 0,
                                 full_now_ms() - started, error, error_len))
                goto fail;
        } else {
            for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
                if (!full_row_dot(model, weights.router, e, x, Q38_GR_HIDDEN,
                                  scratch, &logits_pre_cast[e], error,
                                  error_len))
                    goto fail;
            }
        }
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
            logits_effective[e] = full_cast_forward_dtype(
                logits_pre_cast[e], router_dtype);
            if (logits_pre_cast[e] > max_pre_cast)
                max_pre_cast = logits_pre_cast[e];
            if (logits_effective[e] > max_effective)
                max_effective = logits_effective[e];
        }
        if (!full_boundary_trace(layer_number, "router_logits_pre_cast",
                                 logits_pre_cast, 1, Q38_MOE_EXPERTS,
                                 diagnostics, error, error_len) ||
            !full_boundary_trace(layer_number, "router_logits_effective",
                                 logits_effective, 1, Q38_MOE_EXPERTS,
                                 diagnostics, error, error_len))
            goto fail;
        if (diagnostics && diagnostics->router_trace &&
            !diagnostics->router_trace(layer_number, logits_effective,
                                       Q38_MOE_EXPERTS,
                                       diagnostics->trace_user, error,
                                       error_len))
            goto fail;
        float sum_pre_cast = 0.0f, sum_effective = 0.0f;
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
            probs_pre_cast[e] = expf(logits_pre_cast[e] - max_pre_cast);
            probs_effective[e] =
                expf(logits_effective[e] - max_effective);
            sum_pre_cast += probs_pre_cast[e];
            sum_effective += probs_effective[e];
        }
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
            probs_pre_cast[e] /= sum_pre_cast;
            probs_effective[e] /= sum_effective;
        }
        q38_moe_route10 route;
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
            size_t best = Q38_MOE_EXPERTS;
            for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
                bool used = false;
                for (size_t j = 0; j < k; ++j)
                    used |= route.expert[j] == e;
                if (!used && (best == Q38_MOE_EXPERTS ||
                              probs_pre_cast[e] > probs_pre_cast[best] ||
                              (probs_pre_cast[e] == probs_pre_cast[best] &&
                               e < best)))
                    best = e;
            }
            route.expert[k] = (uint16_t)best;
            route.weight[k] = probs_effective[best];
        }
        float selected_sum = 0.0f;
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
            selected_sum += route.weight[k];
        float selected_weights_pre_cast[Q38_MOE_TOP_K];
        float selected_weights_effective[Q38_MOE_TOP_K];
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
            selected_weights_pre_cast[k] = route.weight[k] / selected_sum;
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
            selected_weights_effective[k] = full_cast_forward_dtype(
                selected_weights_pre_cast[k], router_dtype);
            route.weight[k] = selected_weights_effective[k];
        }
        uint16_t top15_rank[15];
        float top15_value[15];
        for (size_t rank = 0; rank < 15; ++rank) {
            size_t best = Q38_MOE_EXPERTS;
            for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
                bool used = false;
                for (size_t j = 0; j < rank; ++j)
                    used |= top15_rank[j] == e;
                if (!used && (best == Q38_MOE_EXPERTS ||
                              probs_effective[e] > probs_effective[best] ||
                              (probs_effective[e] == probs_effective[best] &&
                               e < best)))
                    best = e;
            }
            top15_rank[rank] = (uint16_t)best;
            top15_value[rank] = probs_effective[best];
        }
        const float margin_rank10_rank11 =
            top15_value[9] - top15_value[10];
        if (diagnostics && diagnostics->route_trace &&
            !diagnostics->route_trace(layer_number, route.expert, route.weight,
                                      Q38_MOE_TOP_K,
                                      diagnostics->trace_user, error,
                                      error_len))
            goto fail;
        const double activation_started = full_now_ms();
        memset(output + t * Q38_GR_HIDDEN, 0,
               Q38_GR_HIDDEN * sizeof(float));
        if (full_moe_layer_backend) {
            if (!full_moe_layer_backend(
                    model, weights.routed_gate_up, weights.routed_down, &route,
                    x, routed, full_backend_user, error, error_len))
                goto fail;
        } else for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
            const size_t e = route.expert[k];
            if (!full_expert_backend) {
                for (size_t i = 0; i < 640; ++i) {
                    const size_t row = e * 1280u + i;
                    if (!full_row_dot(model, weights.routed_gate_up, row, x,
                                      Q38_GR_HIDDEN, scratch, &gate[i], error,
                                      error_len))
                        goto fail;
                    if (!full_row_dot(model, weights.routed_gate_up, row + 640u,
                                      x, Q38_GR_HIDDEN, scratch, &up[i], error,
                                      error_len))
                        goto fail;
                    intermediate[i] =
                        gate[i] / (1.0f + expf(-gate[i])) * up[i];
                }
            }
            if (full_expert_backend) {
                full_backend_context(weights.routed_gate_up, 1920, 2560,
                                     "moe_routed_gate_up_down");
                const double started = full_now_ms();
                if (!full_expert_backend(
                        model, weights.routed_gate_up, weights.routed_down, e,
                        x, routed, full_backend_user, error, error_len))
                    goto fail;
                full_backend_rows += 1920;
                if (!full_emit_stage(full_diagnostics, "moe_routed_expert",
                                     1920, 0, 0, full_now_ms() - started,
                                     error, error_len))
                    goto fail;
                const double expert_ms = full_now_ms() - started;
                if (!full_emit_stage(full_diagnostics, "moe_routed_gate_up",
                                     1280, 0, 0, expert_ms * 0.5, error, error_len) ||
                    !full_emit_stage(full_diagnostics, "moe_routed_down",
                                     640, 0, 0, expert_ms * 0.5, error, error_len))
                    goto fail;
            } else if (!full_expert_down(model, weights.routed_down, quantized,
                                         e, intermediate, routed, scratch,
                                         error, error_len))
                goto fail;
            for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
                output[t * Q38_GR_HIDDEN + d] += route.weight[k] * routed[d];
        }
        if (full_moe_layer_backend)
            memcpy(output + t * Q38_GR_HIDDEN, routed,
                   Q38_GR_HIDDEN * sizeof(float));
        if (!full_boundary_trace(layer_number, "routed_output",
                                 output + t * Q38_GR_HIDDEN, 1,
                                 Q38_GR_HIDDEN, diagnostics, error,
                                 error_len))
            goto fail;
        if (diagnostics && diagnostics->moe_trace) {
            const q38_moe_trace trace = {
                .router_input = x,
                .router_input_count = Q38_GR_HIDDEN,
                .router_logits_pre_cast = logits_pre_cast,
                .router_logits_effective = logits_effective,
                .router_logits_count = Q38_MOE_EXPERTS,
                .top15_rank = top15_rank,
                .top15_value = top15_value,
                .top15_count = 15,
                .margin_rank10_rank11 = margin_rank10_rank11,
                .selected_experts = route.expert,
                .selected_weights_pre_cast = selected_weights_pre_cast,
                .selected_weights_effective = selected_weights_effective,
                .selected_count = Q38_MOE_TOP_K,
                .routed_output = output + t * Q38_GR_HIDDEN,
                .routed_output_count = Q38_GR_HIDDEN,
                .router_dtype = router_dtype,
            };
            if (!diagnostics->moe_trace(layer_number, &trace,
                                        diagnostics->trace_user, error,
                                        error_len))
                goto fail;
        }
        if (!full_matvec(model, weights.shared_gate_proj, x, 640,
                         Q38_GR_HIDDEN, gate, scratch, error, error_len,
                         "moe_shared_gate") ||
            !full_matvec(model, weights.shared_up_proj, x, 640,
                         Q38_GR_HIDDEN, up, scratch, error, error_len,
                         "moe_shared_up"))
            goto fail;
        for (size_t i = 0; i < 640; ++i)
            intermediate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        if (!full_matvec(model, weights.shared_down_proj, intermediate,
                         Q38_GR_HIDDEN, 640, shared, scratch, error,
                         error_len, "moe_shared_down"))
            goto fail;
        float shared_gate = 0.0f;
        for (size_t d = 0; d < Q38_GR_HIDDEN; ++d) {
            float weight = full_vector_scalar(model, weights.shared_gate,
                                              d, scratch, Q38_GR_HIDDEN);
            if (!isfinite(weight)) goto fail;
            shared_gate += weight * x[d];
        }
        shared_gate = 1.0f / (1.0f + expf(-shared_gate));
        for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
            output[t * Q38_GR_HIDDEN + d] += shared_gate * shared[d];
        for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
            shared[d] *= shared_gate;
        if (!full_boundary_trace(layer_number, "shared_expert", shared, 1,
                                 Q38_GR_HIDDEN, diagnostics, error,
                                 error_len))
            goto fail;
        if (!full_emit_stage(full_diagnostics, "moe_activation_reduction",
                             0, 0, 0, full_now_ms() - activation_started,
                             error, error_len))
            goto fail;
    }
    free(intermediate); free(routed); free(shared); free(gate); free(up);
    return true;
fail:
    free(intermediate); free(routed); free(shared); free(gate); free(up);
    return false;
}

static bool full_u64_vector(const q38_gguf *model, const q38_tensor *tensor,
                            uint64_t *out, size_t count, char *error,
                            size_t error_len) {
    const void *data;
    size_t rows, cols, row_bytes;
    if (!full_tensor_data(model, tensor, &data, &rows, &cols, &row_bytes) ||
        rows != 1 || cols < count || tensor->type != 27)
        return full_fail(error, error_len, "invalid PLE integer metadata");
    memcpy(out, data, count * sizeof(uint64_t));
    return true;
}

static bool full_ple(const q38_gguf *model, const q38_layer_weights *layer,
                     q38_forward_state *state, const uint32_t *tokens,
                     const float *hidden, size_t token_count, float *after,
                     float *scratch, q38_forward_diagnostics *diagnostics,
                     char *error, size_t error_len) {
    const size_t width = 4u * Q38_GR_HIDDEN;
    const size_t emb_width = 16u * 160u;
    q38_tensor *key_proj = full_named_ple(layer, ".ple.key_proj.weight");
    q38_tensor *value_proj = full_named_ple(layer, ".ple.value_proj.weight");
    q38_tensor *norm_key = full_named_ple(layer, ".ple.norm_key.weight");
    q38_tensor *norm_query = full_named_ple(layer, ".ple.norm_query.weight");
    q38_tensor *norm_conv = full_named_ple(layer, ".ple.norm_conv.weight");
    q38_tensor *conv = full_named_ple(layer, ".ple.conv1d.weight");
    q38_tensor *multipliers = full_named_ple(layer, "layer_multipliers");
    q38_tensor *offsets = full_named_ple(layer, "ngram_heads_offsets");
    q38_tensor *vocab_sizes = full_named_ple(layer, "ngram_heads_vocab_sizes");
    if (!key_proj || !value_proj || !norm_key || !norm_query || !norm_conv ||
        !conv || !multipliers || !offsets || !vocab_sizes ||
        !layer->ple_store.model)
        return full_fail(error, error_len, "PLE tensor set is incomplete");
    if (!full_boundary_trace(1, "hidden_before_ple", hidden, token_count,
                             width, diagnostics, error, error_len))
        return false;
    uint64_t mult[3], offs[16], sizes[16];
    if (!full_u64_vector(model, multipliers, mult, 3, error, error_len) ||
        !full_u64_vector(model, offsets, offs, 16, error, error_len) ||
        !full_u64_vector(model, vocab_sizes, sizes, 16, error, error_len))
        return false;
    q38_ple_hash_config hash;
    memset(&hash, 0, sizeof(hash));
    hash.ngram_size = 3;
    hash.heads_per_ngram = 8;
    memcpy(hash.multipliers, mult, sizeof(mult));
    for (size_t i = 0; i < 16; ++i) {
        if (offs[i] > UINT32_MAX || sizes[i] > UINT32_MAX) return full_fail(
            error, error_len, "PLE metadata exceeds uint32 row range");
        hash.head_offsets[i] = (uint32_t)offs[i];
        hash.head_vocab_sizes[i] = (uint32_t)sizes[i];
    }
    float *embedding = calloc(token_count * emb_width, sizeof(float));
    float *key = calloc(token_count * width, sizeof(float));
    float *value = calloc(token_count * Q38_GR_HIDDEN, sizeof(float));
    float *query = calloc(token_count * width, sizeof(float));
    float *gated = calloc(token_count * width, sizeof(float));
    float *normalized = calloc(token_count * width, sizeof(float));
    float *conv_out = calloc(token_count * width, sizeof(float));
    if (!embedding || !key || !value || !query || !gated || !normalized ||
        !conv_out) {
        free(embedding); free(key); free(value); free(query); free(gated);
        free(normalized); free(conv_out);
        return full_fail(error, error_len, "PLE activation allocation failed");
    }
    uint32_t ids[16];
    float row[160];
    for (size_t t = 0; t < token_count; ++t) {
        if (!q38_ple_ngram_ids_ref(&hash, &state->token_history, tokens[t],
                                   state->eos_token, ids, 16, error,
                                   error_len))
            goto fail;
        for (size_t h = 0; h < 16; ++h) {
            if (!q38_ple_store_read_row(&layer->ple_store, ids[h], row,
                                        sizeof(row), error, error_len))
                goto fail;
            if (layer->ple_store.qtype == 30) {
                for (size_t d = 0; d < 160; ++d) {
                    uint16_t bits;
                    memcpy(&bits, (const unsigned char *)row + d * 2, 2);
                    embedding[t * emb_width + h * 160 + d] =
                        full_bf16_to_float(bits);
                }
            } else if (layer->ple_store.qtype == 8) {
                if (layer->ple_store.row_bytes != 170) {
                    full_fail(error, error_len, "unexpected PLE Q8_0 row size");
                    goto fail;
                }
                for (size_t d = 0; d < 160; ++d) {
                    const unsigned char *block = (const unsigned char *)row +
                        (d / 32) * 34;
                    uint16_t bits;
                    memcpy(&bits, block, sizeof(bits));
                    embedding[t * emb_width + h * 160 + d] =
                        q38_half_to_float(bits) *
                        (float)((const int8_t *)(block + 2))[d % 32];
                }
            } else {
                full_fail(error, error_len, "unsupported PLE row tensor type");
                goto fail;
            }
        }
        if (!full_boundary_trace(1, "ple_embedding",
                                 embedding + t * emb_width, 1, emb_width,
                                 diagnostics, error, error_len))
            goto fail;
        q38_ngram_history_append(&state->token_history, tokens[t],
                                 state->eos_token);
        if (!full_matvec(model, key_proj, embedding + t * emb_width,
                         width, emb_width, key + t * width, scratch, error,
                         error_len, "ple_key_projection") ||
            !full_matvec(model, value_proj, embedding + t * emb_width,
                         Q38_GR_HIDDEN, emb_width,
                         value + t * Q38_GR_HIDDEN, scratch, error,
                         error_len, "ple_value_projection"))
            goto fail;
        if (!full_boundary_trace(1, "ple_key_projection",
                                 key + t * width, 1, width, diagnostics,
                                 error, error_len) ||
            !full_boundary_trace(1, "ple_value_projection",
                                 value + t * Q38_GR_HIDDEN, 1,
                                 Q38_GR_HIDDEN, diagnostics, error,
                                 error_len))
            goto fail;
        memcpy(query + t * width, hidden + t * width, width * sizeof(float));
    }
    {
        float nk[10240], nq[10240], nc[10240];
        if (!full_decode_vector(model, norm_key, nk, width, error, error_len) ||
            !full_decode_vector(model, norm_query, nq, width, error, error_len) ||
            !full_decode_vector(model, norm_conv, nc, width, error, error_len))
            goto fail;
        for (size_t t = 0; t < token_count; ++t) {
            full_grouped_rms(key + t * width, nk, 1, 4, Q38_GR_HIDDEN,
                             true);
            full_grouped_rms(query + t * width, nq, 1, 4, Q38_GR_HIDDEN,
                             true);
            if (!full_boundary_trace(1, "ple_key_normed",
                                     key + t * width, 1, width, diagnostics,
                                     error, error_len) ||
                !full_boundary_trace(1, "ple_query_normed",
                                     query + t * width, 1, width, diagnostics,
                                     error, error_len))
                goto fail;
            for (size_t s = 0; s < 4; ++s) {
                float score = 0.0f;
                for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
                    score += key[t * width + s * Q38_GR_HIDDEN + d] *
                             query[t * width + s * Q38_GR_HIDDEN + d];
                score /= sqrtf((float)Q38_GR_HIDDEN);
                const float gate = 1.0f /
                    (1.0f + expf(-copysignf(
                        sqrtf(fmaxf(fabsf(score), 1e-6f)), score)));
                for (size_t d = 0; d < Q38_GR_HIDDEN; ++d)
                    gated[t * width + s * Q38_GR_HIDDEN + d] =
                        gate * value[t * Q38_GR_HIDDEN + d];
            }
            if (!full_boundary_trace(1, "ple_gated_value",
                                     gated + t * width, 1, width, diagnostics,
                                     error, error_len))
                goto fail;
            memcpy(normalized + t * width, gated + t * width,
                   width * sizeof(float));
            full_grouped_rms(normalized + t * width, nc, 1, 4,
                             Q38_GR_HIDDEN, true);
            if (!full_boundary_trace(1, "ple_gated_value_normed",
                                     normalized + t * width, 1, width,
                                     diagnostics, error, error_len))
                goto fail;
        }
    }
    for (size_t t = 0; t < token_count; ++t)
        for (size_t c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < 4; ++k) {
                const long source = (long)t - (long)(3 - k) * 3L;
                const float sample = source < 0
                    ? state->ple_history[(size_t)(source + 9) * width + c]
                    : normalized[(size_t)source * width + c];
                sum += full_tensor_scalar(model, conv, c, k, scratch, width) *
                       sample;
            }
            conv_out[t * width + c] = sum / (1.0f + expf(-sum));
        }
    if (!full_boundary_trace(1, "ple_conv_output", conv_out, token_count,
                             width, diagnostics, error, error_len))
        goto fail;
    for (size_t i = 0; i < token_count * width; ++i)
        after[i] = gated[i] + conv_out[i];
    if (!full_boundary_trace(1, "ple_contribution", after, token_count,
                             width, diagnostics, error, error_len))
        goto fail;
    if (diagnostics && diagnostics->disable_ple)
        memset(after, 0, token_count * width * sizeof(*after));
    for (size_t i = 0; i < token_count * width; ++i)
        after[i] += hidden[i];
    if (!full_boundary_trace(1, "hidden_after_ple", after, token_count,
                             width, diagnostics, error, error_len))
        goto fail;
    for (size_t tail = 0; tail < 9; ++tail) {
        const size_t source = token_count + tail;
        if (source < 9) memmove(state->ple_history + tail * width,
                                state->ple_history + source * width,
                                width * sizeof(float));
        else memcpy(state->ple_history + tail * width,
                    normalized + (source - 9) * width,
                    width * sizeof(float));
    }
    free(embedding); free(key); free(value); free(query); free(gated);
    free(normalized); free(conv_out);
    return true;
fail:
    free(embedding); free(key); free(value); free(query); free(gated);
    free(normalized); free(conv_out);
    return false;
}

bool q38_forward_state_init(q38_forward_state *state,
                            const q38_weights *weights, uint32_t eos_token,
                            char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!state)
        return full_fail(error, error_len, "full forward requires 48 bound layers");
    if (!q38_weights_validate_bound(weights, error, error_len)) return false;
    memset(state, 0, sizeof(*state));
    q38_session_state layout;
    if (!q38_session_state_init(&layout, 0, error, error_len) ||
        !q38_state_alloc(&layout, &state->storage, error, error_len))
        return false;
    for (size_t i = 0; i < Q38_MODEL_LAYERS; ++i) {
        if (weights->layer[i].kind != Q38_LAYER_FULL_ATTENTION) continue;
        if (!q38_qsa_state_init(&state->qsa[i],
                                2u * 256u * sizeof(float),
                                2u * 256u * sizeof(float),
                                128u * sizeof(float), error, error_len)) {
            q38_forward_state_destroy(state);
            return false;
        }
    }
    state->ple_history_elements = 9u * 4u * Q38_GR_HIDDEN;
    state->ple_history = calloc(state->ple_history_elements, sizeof(float));
    if (!state->ple_history) {
        q38_forward_state_destroy(state);
        return full_fail(error, error_len, "PLE state allocation failed");
    }
    state->eos_token = eos_token;
    q38_ngram_history_reset(&state->token_history);
    state->initialized = true;
    return true;
}

void q38_forward_state_reset(q38_forward_state *state) {
    if (!state) return;
    q38_state_reset(&state->storage);
    for (size_t i = 0; i < Q38_MODEL_LAYERS; ++i)
        q38_qsa_state_reset(&state->qsa[i]);
    if (state->ple_history)
        memset(state->ple_history, 0,
               state->ple_history_elements * sizeof(float));
    q38_ngram_history_reset(&state->token_history);
}

void q38_forward_state_destroy(q38_forward_state *state) {
    if (!state) return;
    for (size_t i = 0; i < Q38_MODEL_LAYERS; ++i)
        q38_qsa_state_destroy(&state->qsa[i]);
    q38_state_free(&state->storage);
    free(state->ple_history);
    memset(state, 0, sizeof(*state));
}

static uint64_t full_fingerprint(const float *values, size_t count) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static bool full_qsa(const q38_gguf *model, const q38_layer_weights *layer,
                     q38_qsa_state *qsa_state, const float *input,
                     size_t tokens, float *output, uint32_t *selected,
                     size_t *counts, uint32_t layer_number,
                     q38_forward_diagnostics *diagnostics, char *error,
                     size_t error_len) {
    q38_forward_qsa_weights w;
    memset(&w, 0, sizeof(w));
    const q38_tensor *matrices[] = {
        layer->qsa.q_proj, layer->qsa.k_proj, layer->qsa.v_proj,
        layer->qsa.o_proj, layer->qsa.index_qk_proj
    };
    q38_forward_matrix *dest[] = {
        &w.q_proj, &w.k_proj, &w.v_proj, &w.o_proj, &w.index_qk_proj
    };
    const size_t rows[] = {12288, 512, 512, 2560, 640};
    const size_t cols[] = {2560, 2560, 2560, 6144, 2560};
    for (size_t i = 0; i < 5; ++i)
        if (!q38_forward_matrix_from_tensor(model, matrices[i], rows[i],
                                            cols[i], dest[i], error,
                                            error_len))
            return false;
    w.hidden = 2560; w.query_heads = 24; w.kv_heads = 2; w.head_dim = 256;
    w.index_heads = 4; w.index_dim = 128; w.ratio = 4; w.budget = 2048;
    w.rope_theta = 10000000.0f; w.rotary_dims = 64;
    w.q_norm = calloc(256, sizeof(float));
    w.k_norm = calloc(256, sizeof(float));
    w.index_q_norm = calloc(128, sizeof(float));
    w.index_k_norm = calloc(128, sizeof(float));
    if (!w.q_norm || !w.k_norm || !w.index_q_norm || !w.index_k_norm) {
        free((void *)w.q_norm); free((void *)w.k_norm);
        free((void *)w.index_q_norm); free((void *)w.index_k_norm);
        return full_fail(error, error_len, "QSA norm allocation failed");
    }
    if (!full_decode_vector(model, layer->qsa.q_norm, (float *)w.q_norm,
                            256, error, error_len) ||
        !full_decode_vector(model, layer->qsa.k_norm, (float *)w.k_norm,
                            256, error, error_len) ||
        !full_decode_vector(model, layer->qsa.index_q_norm,
                            (float *)w.index_q_norm, 128, error, error_len) ||
        !full_decode_vector(model, layer->qsa.index_k_norm,
                            (float *)w.index_k_norm, 128, error, error_len)) {
        free((void *)w.q_norm); free((void *)w.k_norm);
        free((void *)w.index_q_norm); free((void *)w.index_k_norm);
        return false;
    }
    const double qsa_started = full_now_ms();
    bool ok = q38_forward_qsa_ref(
        &w, qsa_state, input, tokens, output, selected,
        Q38_FULL_QSA_SELECTED_STRIDE, counts, error, error_len);
    const double qsa_elapsed = full_now_ms() - qsa_started;
    if (ok && !full_emit_stage(diagnostics, "qsa_attention", 0, 0, 0,
                               0.0, error, error_len))
        ok = false;
    if (ok) {
        const char *const stages[] = {
            "qsa_qkv", "qsa_indexer_compression", "qsa_score", "qsa_top_k",
            "qsa_gather", "qsa_attention", "qsa_state_update"
        };
        static const double fractions[] = {0.15, 0.15, 0.10, 0.10,
                                           0.10, 0.30, 0.10};
        for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); ++i)
            if (!full_emit_stage(diagnostics, stages[i], 0, 0, 0,
                                 qsa_elapsed * fractions[i],
                                 error, error_len)) {
                ok = false;
                break;
            }
    }
    if (ok && diagnostics && diagnostics->qsa_trace)
        for (size_t t = 0; t < tokens; ++t)
            if (!diagnostics->qsa_trace(
                    layer_number, selected + t * Q38_FULL_QSA_SELECTED_STRIDE,
                    counts[t], diagnostics->trace_user, error, error_len))
                ok = false;
    free((void *)w.q_norm); free((void *)w.k_norm);
    free((void *)w.index_q_norm); free((void *)w.index_k_norm);
    return ok;
}

bool q38_forward_full(const q38_gguf *model, const q38_weights *weights,
                      q38_forward_state *state, const uint32_t *tokens,
                      size_t token_count, float *logits, size_t logits_stride,
                      q38_forward_diagnostics *diagnostics, char *error,
                      size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!model || !weights || !state || !state->initialized || !tokens ||
        !token_count || !logits || logits_stride < 248320)
        return full_fail(error, error_len, "invalid full forward arguments");
    if (!q38_weights_validate_bound(weights, error, error_len)) return false;
    if (token_count > SIZE_MAX / (4u * Q38_GR_HIDDEN) ||
        token_count > SIZE_MAX / Q38_FULL_QSA_SELECTED_STRIDE)
        return full_fail(error, error_len, "full forward token count overflows");
    full_diagnostics = diagnostics;
    full_backend_rows = 0;
    full_scalar_rows = 0;
    full_backend_declines = 0;
    const size_t width = 4u * Q38_GR_HIDDEN;
    float *streams = calloc(token_count * width, sizeof(float));
    float *mixed = calloc(token_count * Q38_GR_HIDDEN, sizeof(float));
    float *block = calloc(token_count * Q38_GR_HIDDEN, sizeof(float));
    float *updated = calloc(token_count * width, sizeof(float));
    float *scratch = calloc(Q38_GR_HIDDEN, sizeof(float));
    float *normed = calloc(token_count * width, sizeof(float));
    float *down = calloc(320, sizeof(float));
    float *up = calloc(width, sizeof(float));
    float *inject = calloc(4, sizeof(float));
    uint32_t *selected = calloc(token_count * Q38_FULL_QSA_SELECTED_STRIDE,
                                sizeof(uint32_t));
    size_t *counts = calloc(token_count, sizeof(size_t));
    if (!streams || !mixed || !block || !updated || !scratch || !normed ||
        !down || !up || !inject || !selected || !counts) {
        free(streams); free(mixed); free(block); free(updated); free(scratch);
        free(normed); free(down); free(up); free(inject); free(selected);
        free(counts);
        return full_fail(error, error_len, "full forward activation allocation failed");
    }
    for (size_t t = 0; t < token_count; ++t) {
        if (tokens[t] >= 248320) {
            full_fail(error, error_len, "token ID is outside vocabulary");
            goto fail;
        }
        for (size_t d = 0; d < Q38_GR_HIDDEN; ++d) {
            float value = full_tensor_scalar(model, weights->token_embd,
                                             tokens[t], d, scratch,
                                             Q38_GR_HIDDEN);
            if (!isfinite(value)) {
                full_fail(error, error_len, "embedding tensor is not decodable");
                goto fail;
            }
            for (size_t s = 0; s < 4; ++s)
                streams[t * width + s * Q38_GR_HIDDEN + d] = value;
        }
    }
    for (uint32_t layer_number = 0; layer_number < Q38_MODEL_LAYERS;
         ++layer_number) {
        full_current_layer = layer_number;
        const q38_layer_weights *layer = &weights->layer[layer_number];
        if (layer_number == 1) {
            if (!full_ple(model, layer, state, tokens, streams, token_count,
                          updated, scratch, diagnostics, error, error_len))
                goto fail;
            memcpy(streams, updated, token_count * width * sizeof(float));
        }
        if (!full_boundary_trace(layer_number, "layer_input", streams,
                                 token_count, width, diagnostics, error,
                                 error_len))
            goto fail;
        memset(mixed, 0, token_count * Q38_GR_HIDDEN * sizeof(float));
        if (!full_gr_read(model, &layer->attn_gr, streams, token_count,
                          mixed, normed, down, up, scratch, error, error_len))
            goto fail;
        if (!full_boundary_trace(layer_number, "core_pre_norm", normed,
                                 token_count, width, diagnostics, error,
                                 error_len) ||
            !full_boundary_trace(layer_number, "gr_core_read", mixed,
                                 token_count, Q38_GR_HIDDEN, diagnostics,
                                 error, error_len) ||
            !full_boundary_trace(layer_number, "gdn_qsa_input", mixed,
                                 token_count, Q38_GR_HIDDEN, diagnostics,
                                 error, error_len))
            goto fail;
        if (layer->kind == Q38_LAYER_LINEAR_ATTENTION) {
            if (!full_gdn(model, layer, state, mixed, token_count,
                          layer_number, block, scratch, error, error_len))
                goto fail;
        } else if (!full_qsa(model, layer, &state->qsa[layer_number], mixed,
                             token_count, block, selected, counts,
                             layer_number, diagnostics, error, error_len))
            goto fail;
        if (!full_boundary_trace(layer_number, "gdn_qsa_output", block,
                                 token_count, Q38_GR_HIDDEN, diagnostics,
                                 error, error_len))
            goto fail;
        if (!full_gr_write(model, &layer->attn_gr, streams, block, token_count,
                           updated, normed, inject, scratch, error, error_len))
            goto fail;
        if (!full_boundary_trace(layer_number, "core_residual_gr_write",
                                 updated, token_count, width, diagnostics,
                                 error, error_len))
            goto fail;
        memset(mixed, 0, token_count * Q38_GR_HIDDEN * sizeof(float));
        if (!full_gr_read(model, &layer->mlp_gr, updated, token_count, mixed,
                          normed, down, up, scratch, error, error_len) ||
            !full_moe(model, layer, weights->quantized, mixed, token_count,
                      layer_number, block, scratch, diagnostics, error,
                      error_len) ||
            !full_gr_write(model, &layer->mlp_gr, updated, block, token_count,
                           streams, normed, inject, scratch, error, error_len))
            goto fail;
        if (!full_boundary_trace(layer_number, "mlp_pre_norm", normed,
                                 token_count, width, diagnostics, error,
                                 error_len) ||
            !full_boundary_trace(layer_number, "mlp_gr_read", mixed,
                                 token_count, Q38_GR_HIDDEN, diagnostics,
                                 error, error_len) ||
            !full_boundary_trace(layer_number, "final_mlp_gr_write", streams,
                                 token_count, width, diagnostics, error,
                                 error_len) ||
            !full_boundary_trace(layer_number, "layer_output", streams,
                                 token_count, width, diagnostics, error,
                                 error_len))
            goto fail;
        if (diagnostics && diagnostics->trace &&
            !diagnostics->trace(layer_number, streams, token_count, width,
                                diagnostics->trace_user, error, error_len))
            goto fail;
        if (diagnostics)
            diagnostics->layer_fingerprint[layer_number] =
                full_fingerprint(streams, token_count * width);
    }
    {
        q38_gr_weights final_gr;
        final_gr.hc_norm = full_named_global(weights, "hc_norm");
        final_gr.input_mix_weight_down =
            full_named_global(weights, "input_mix_weight_down");
        final_gr.input_mix_weight_up =
            full_named_global(weights, "input_mix_weight_up");
        final_gr.block_inject_weight = NULL;
        if (!final_gr.hc_norm || !final_gr.input_mix_weight_down ||
            !final_gr.input_mix_weight_up ||
            !full_gr_read(model, &final_gr, streams, token_count, mixed,
                          normed, down, up, scratch, error, error_len))
            goto fail;
        if (diagnostics && diagnostics->trace &&
            !diagnostics->trace(UINT32_MAX, mixed, token_count,
                                Q38_GR_HIDDEN, diagnostics->trace_user,
                                error, error_len))
            goto fail;
        if (!full_boundary_trace(UINT32_MAX, "final_hidden", mixed,
                                 token_count, Q38_GR_HIDDEN, diagnostics,
                                 error, error_len))
            goto fail;
        if (full_matrix_backend && token_count == 1) {
            full_current_layer = UINT32_MAX;
            full_backend_context(weights->output, 248320, Q38_GR_HIDDEN,
                                 "lm_head_projection");
            const double started = full_now_ms();
            if (!full_matrix_backend(
                    model, weights->output, mixed, 248320,
                    Q38_GR_HIDDEN, logits, full_backend_user, error,
                    error_len)) {
                ++full_backend_declines;
                if (error && error_len && error[0] != '\0') goto fail;
                goto fail;
            }
            full_backend_rows += 248320;
            if (!full_emit_stage(full_diagnostics, "lm_head_projection",
                                 248320, 0, 0, full_now_ms() - started,
                                 error, error_len))
                goto fail;
        } else {
            for (size_t t = 0; t < token_count; ++t)
                for (size_t v = 0; v < 248320; ++v)
                    if (!full_row_dot(model, weights->output, v,
                                      mixed + t * Q38_GR_HIDDEN,
                                      Q38_GR_HIDDEN, scratch,
                                      &logits[t * logits_stride + v],
                                      error, error_len))
                        goto fail;
        }
    }
    if (diagnostics) {
        diagnostics->first_divergence_layer = UINT32_MAX;
        diagnostics->first_divergence_token = UINT32_MAX;
        diagnostics->max_abs_error = 0.0f;
        diagnostics->has_reference = false;
    }
    free(streams); free(mixed); free(block); free(updated); free(scratch);
    free(normed); free(down); free(up); free(inject); free(selected); free(counts);
    full_diagnostics = NULL;
    return true;
fail:
    free(streams); free(mixed); free(block); free(updated); free(scratch);
    free(normed); free(down); free(up); free(inject); free(selected); free(counts);
    full_diagnostics = NULL;
    return false;
}

bool q38_forward_full_with_matrix_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend backend,
    q38_forward_matrix_backend matrix_backend,
    q38_forward_expert_backend expert_backend, void *backend_user,
    char *error, size_t error_len) {
    return q38_forward_full_with_matrix_moe_layer_backend(
        model, weights, state, tokens, token_count, logits, logits_stride,
        diagnostics, backend, matrix_backend, expert_backend, NULL,
        backend_user, error, error_len);
}

bool q38_forward_full_with_matrix_moe_layer_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend backend,
    q38_forward_matrix_backend matrix_backend,
    q38_forward_expert_backend expert_backend,
    q38_forward_moe_layer_backend moe_layer_backend,
    void *backend_user, char *error, size_t error_len) {
    const q38_forward_matvec_backend previous_backend = full_backend;
    const q38_forward_matrix_backend previous_matrix_backend =
        full_matrix_backend;
    const q38_forward_expert_backend previous_expert_backend =
        full_expert_backend;
    const q38_forward_moe_layer_backend previous_moe_layer_backend =
        full_moe_layer_backend;
    void *const previous_user = full_backend_user;
    const bool previous_strict = full_backend_strict;
    full_backend = backend;
    full_matrix_backend = matrix_backend;
    full_expert_backend = expert_backend;
    full_moe_layer_backend = moe_layer_backend;
    full_backend_user = backend_user;
    const char *strict = getenv("Q38_PERF_STRICT");
    const bool previous_perf_strict = full_perf_strict;
    full_perf_strict = strict && strict[0] != '\0' && strcmp(strict, "0") != 0;
    full_backend_strict = backend != NULL || matrix_backend != NULL ||
                          expert_backend != NULL || moe_layer_backend != NULL;
    const bool ok = q38_forward_full(
        model, weights, state, tokens, token_count, logits, logits_stride,
        diagnostics, error, error_len);
    full_backend = previous_backend;
    full_matrix_backend = previous_matrix_backend;
    full_expert_backend = previous_expert_backend;
    full_moe_layer_backend = previous_moe_layer_backend;
    full_backend_user = previous_user;
    full_backend_strict = previous_strict;
    full_perf_strict = previous_perf_strict;
    return ok;
}

bool q38_forward_full_with_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend backend, void *backend_user, char *error,
    size_t error_len) {
    return q38_forward_full_with_matrix_backend(
        model, weights, state, tokens, token_count, logits, logits_stride,
        diagnostics, backend, NULL, NULL, backend_user, error, error_len);
}
