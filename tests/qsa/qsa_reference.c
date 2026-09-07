#include "qsa_reference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

enum {
    HIDDEN = 2560,
    QUERY_HEADS = 24,
    KV_HEADS = 2,
    HEAD_DIM = 256,
    INDEX_HEADS = 4,
    INDEX_DIM = 128,
    RATIO = 4,
    ATTENTION_WIDTH = QUERY_HEADS * HEAD_DIM,
    Q_ROWS = QUERY_HEADS * HEAD_DIM * 2,
    K_ROWS = KV_HEADS * HEAD_DIM,
    INDEX_ROWS = (INDEX_HEADS + 1) * INDEX_DIM,
};

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static float bf16_to_float(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static float weight_at(const q38_test_qsa_weight *weight, size_t index) {
    if (weight->type == 30) {
        uint16_t bits;
        memcpy(&bits, weight->data + index * sizeof(bits), sizeof(bits));
        return bf16_to_float(bits);
    }
    if (weight->type == 0) {
        float value;
        memcpy(&value, weight->data + index * sizeof(value), sizeof(value));
        return value;
    }
    return NAN;
}

static void rms(float *values, const q38_test_qsa_weight *weight,
                size_t count);
static void rope(float *values, size_t rotary, size_t position, float theta);

bool q38_test_qsa_project(const q38_test_qsa_weight *weight,
                          const float *input, float *output,
                          char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!weight || !input || !output || !weight->data ||
        (weight->type != 0 && weight->type != 30) ||
        weight->rows == 0 || weight->cols == 0)
        return fail(error, error_len, "invalid QSA projection arguments");
    for (size_t row = 0; row < weight->rows; ++row) {
        float sum = 0.0f;
        for (size_t col = 0; col < weight->cols; ++col)
            sum += weight_at(weight, row * weight->cols + col) * input[col];
        output[row] = sum;
    }
    return true;
}

bool q38_test_qsa_build_state(
    const float *q_projection, const float *keys, const float *values,
    const float *index_projection, const q38_test_qsa_weight *k_norm,
    float *state_main_k, float *state_main_v, float *state_index_k,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q_projection || !keys || !values || !index_projection || !k_norm ||
        !state_main_k || !state_main_v || !state_index_k)
        return fail(error, error_len, "invalid QSA state arguments");
    memcpy(state_main_v, values, K_ROWS * sizeof(float));
    for (size_t head = 0; head < KV_HEADS; ++head) {
        memcpy(state_main_k + head * HEAD_DIM,
               keys + head * HEAD_DIM, HEAD_DIM * sizeof(float));
        rms(state_main_k + head * HEAD_DIM, k_norm, HEAD_DIM);
        rope(state_main_k + head * HEAD_DIM, 64, 0, 10000000.0f);
    }
    memcpy(state_index_k, index_projection + INDEX_HEADS * INDEX_DIM,
           INDEX_DIM * sizeof(float));
    (void)q_projection;
    return true;
}

static void rms(float *values, const q38_test_qsa_weight *weight,
                size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
        sum += (double)values[i] * values[i];
    const float scale = 1.0f / sqrtf((float)(sum / count) + 1.0e-6f);
    for (size_t i = 0; i < count; ++i)
        values[i] *= scale * (1.0f + weight_at(weight, i));
}

static void rope(float *values, size_t rotary, size_t position,
                 float theta) {
    const float base = powf(theta, -2.0f / (float)rotary);
    float frequency = 1.0f;
    for (size_t i = 0; i < rotary / 2; ++i) {
        const float angle = (float)position * frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float a = values[i];
        const float b = values[i + rotary / 2];
        values[i] = a * cosine - b * sine;
        values[i + rotary / 2] = a * sine + b * cosine;
        frequency *= base;
    }
}

static size_t select_prefix(
    const float *state_index_k, size_t visible,
    const float *query, const q38_test_qsa_weight *index_k_norm,
    uint32_t *selected, size_t stride) {
    const size_t complete = visible / RATIO;
    const size_t group_budget = 2048 / RATIO;
    const size_t groups = complete < group_budget ? complete : group_budget;
    const size_t tail = visible - complete * RATIO;
    if (groups * RATIO + tail > stride) return 0;
    float scores[512];
    size_t order[512];
    for (size_t group = 0; group < complete; ++group) {
        float total = 0.0f;
        for (size_t head = 0; head < INDEX_HEADS; ++head) {
            float key[INDEX_DIM] = {0};
            for (size_t d = 0; d < INDEX_DIM; ++d) {
                for (size_t j = 0; j < RATIO; ++j)
                    key[d] += state_index_k[(group * RATIO + j) *
                                              INDEX_DIM + d];
                key[d] /= (float)RATIO;
            }
            rms(key, index_k_norm, INDEX_DIM);
            rope(key, 64, group * RATIO, 10000000.0f);
            float dot = 0.0f;
            for (size_t d = 0; d < INDEX_DIM; ++d)
                dot += query[head * INDEX_DIM + d] * key[d];
            total += dot > 0.0f ? dot : 0.0f;
        }
        scores[group] = total / sqrtf((float)INDEX_DIM);
        order[group] = group;
    }
    for (size_t candidate = 1; candidate < complete; ++candidate) {
        const size_t value = order[candidate];
        size_t at = candidate;
        while (at > 0) {
            const size_t previous = order[at - 1];
            if (scores[previous] > scores[value] ||
                (scores[previous] == scores[value] && previous < value))
                break;
            --at;
        }
        for (size_t j = candidate; j > at; --j)
            order[j] = order[j - 1];
        order[at] = value;
    }
    size_t count = 0;
    for (size_t rank = 0; rank < groups; ++rank)
        for (size_t j = 0; j < RATIO; ++j)
            selected[count++] = (uint32_t)(order[rank] * RATIO + j);
    for (size_t row = complete * RATIO; row < visible; ++row)
        selected[count++] = (uint32_t)row;
    return count;
}

bool q38_test_qsa_attention(
    const float *q_projection, const float *index_projection,
    const float *state_main_k, const float *state_main_v,
    const float *state_index_k, size_t state_count,
    const q38_test_qsa_weight *q_norm,
    const q38_test_qsa_weight *index_q_norm,
    const q38_test_qsa_weight *index_k_norm,
    float *attention, uint32_t *selected, size_t selected_stride,
    size_t *selected_count, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q_projection || !index_projection || !state_main_k ||
        !state_main_v || !state_index_k || !state_count || !q_norm ||
        !index_q_norm || !index_k_norm || !attention || !selected ||
        !selected_count || selected_stride == 0)
        return fail(error, error_len, "invalid QSA post-projection arguments");

    float queries[ATTENTION_WIDTH];
    float index_queries[INDEX_HEADS * INDEX_DIM];
    for (size_t head = 0; head < QUERY_HEADS; ++head) {
        memcpy(queries + head * HEAD_DIM,
               q_projection + head * HEAD_DIM * 2,
               HEAD_DIM * sizeof(float));
        rms(queries + head * HEAD_DIM, q_norm, HEAD_DIM);
        rope(queries + head * HEAD_DIM, 64, 0, 10000000.0f);
    }
    for (size_t head = 0; head < INDEX_HEADS; ++head) {
        memcpy(index_queries + head * INDEX_DIM,
               index_projection + head * INDEX_DIM,
               INDEX_DIM * sizeof(float));
        rms(index_queries + head * INDEX_DIM, index_q_norm, INDEX_DIM);
        rope(index_queries + head * INDEX_DIM, 64, 0, 10000000.0f);
    }
    *selected_count = select_prefix(
        state_index_k, state_count, index_queries, index_k_norm, selected,
        selected_stride);
    if (!*selected_count)
        return fail(error, error_len, "QSA selection produced no rows");

    const size_t group = QUERY_HEADS / KV_HEADS;
    for (size_t head = 0; head < QUERY_HEADS; ++head) {
        const size_t kv_head = head / group;
        float maximum = -INFINITY;
        for (size_t row = 0; row < *selected_count; ++row) {
            const uint32_t selected_row = selected[row];
            const float *key = state_main_k +
                ((size_t)selected_row * KV_HEADS + kv_head) * HEAD_DIM;
            float score = 0.0f;
            for (size_t d = 0; d < HEAD_DIM; ++d)
                score += queries[head * HEAD_DIM + d] * key[d];
            score /= sqrtf((float)HEAD_DIM);
            if (score > maximum) maximum = score;
        }
        float denominator = 0.0f;
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            float numerator = 0.0f;
            for (size_t row = 0; row < *selected_count; ++row) {
                const uint32_t selected_row = selected[row];
                const float *key = state_main_k +
                    ((size_t)selected_row * KV_HEADS + kv_head) * HEAD_DIM;
                const float *value = state_main_v +
                    ((size_t)selected_row * KV_HEADS + kv_head) * HEAD_DIM;
                float score = 0.0f;
                for (size_t i = 0; i < HEAD_DIM; ++i)
                    score += queries[head * HEAD_DIM + i] * key[i];
                const float weight =
                    expf(score / sqrtf((float)HEAD_DIM) - maximum);
                denominator += d == 0 ? weight : 0.0f;
                numerator += weight * value[d];
            }
            const float gate = 1.0f /
                (1.0f + expf(-(q_projection[head * HEAD_DIM * 2 +
                                    HEAD_DIM + d])));
            attention[head * HEAD_DIM + d] = numerator / denominator * gate;
        }
    }
    return true;
}

bool q38_test_qsa_output_project(const q38_test_qsa_weight *o_proj,
                                  const float *attention, float *output,
                                  char *error, size_t error_len) {
    return q38_test_qsa_project(o_proj, attention, output, error, error_len);
}

bool q38_test_qsa_post(
    const float *q_projection, const float *index_projection,
    const float *state_main_k, const float *state_main_v,
    const float *state_index_k, size_t state_count,
    const q38_test_qsa_weight *q_norm, const q38_test_qsa_weight *k_norm,
    const q38_test_qsa_weight *index_q_norm,
    const q38_test_qsa_weight *index_k_norm, const q38_test_qsa_weight *o_proj,
    float *attention, uint32_t *selected, size_t selected_stride,
    size_t *selected_count, float *output, char *error, size_t error_len) {
    (void)k_norm;
    if (!q38_test_qsa_attention(
            q_projection, index_projection, state_main_k, state_main_v,
            state_index_k, state_count, q_norm, index_q_norm,
            index_k_norm, attention, selected, selected_stride,
            selected_count, error, error_len))
        return false;
    return q38_test_qsa_output_project(o_proj, attention, output, error,
                                        error_len);
}
