#include "q38_gdn_ref.h"

#include <string.h>

size_t q38_gdn_ref_state_elements(size_t sequence_count) {
    return sequence_count * (size_t)Q38_GDN_REF_VALUE_HEADS *
           (size_t)Q38_GDN_REF_HEAD_DIM * (size_t)Q38_GDN_REF_HEAD_DIM;
}

void q38_gdn_ref_reset(float *state, size_t sequence_count) {
    if (!state) return;
    memset(state, 0, q38_gdn_ref_state_elements(sequence_count) *
                       sizeof(*state));
}

static size_t state_offset(size_t sequence_index, size_t head, size_t row,
                           size_t column) {
    return (((sequence_index * (size_t)Q38_GDN_REF_VALUE_HEADS + head) *
             (size_t)Q38_GDN_REF_HEAD_DIM + row) *
            (size_t)Q38_GDN_REF_HEAD_DIM + column);
}

bool q38_gdn_ref_step(float *state, size_t sequence_count,
                      size_t sequence_index, const float *q,
                      const float *k, const float *v, const float *decay,
                      const float *beta, float scale, float *output) {
    if (!state || !q || !k || !v || !decay || !beta || !output ||
        sequence_index >= sequence_count)
        return false;

    float delta[Q38_GDN_REF_HEAD_DIM];
    for (size_t head = 0; head < Q38_GDN_REF_VALUE_HEADS; head++) {
        float *matrix = state + state_offset(sequence_index, head, 0, 0);
        const float *q_head = q + head * Q38_GDN_REF_HEAD_DIM;
        const float *k_head = k + head * Q38_GDN_REF_HEAD_DIM;
        const float *v_head = v + head * Q38_GDN_REF_HEAD_DIM;
        float *output_head = output + head * Q38_GDN_REF_HEAD_DIM;

        for (size_t row = 0; row < Q38_GDN_REF_HEAD_DIM; row++) {
            float *state_row = matrix + row * Q38_GDN_REF_HEAD_DIM;
            for (size_t column = 0; column < Q38_GDN_REF_HEAD_DIM;
                 column++)
                state_row[column] *= decay[head];
        }

        for (size_t column = 0; column < Q38_GDN_REF_HEAD_DIM; column++) {
            float value = 0.0f;
            for (size_t row = 0; row < Q38_GDN_REF_HEAD_DIM; row++)
                value += matrix[row * Q38_GDN_REF_HEAD_DIM + column] *
                         k_head[row];
            delta[column] = (v_head[column] - value) * beta[head];
        }

        for (size_t row = 0; row < Q38_GDN_REF_HEAD_DIM; row++) {
            float *state_row = matrix + row * Q38_GDN_REF_HEAD_DIM;
            for (size_t column = 0; column < Q38_GDN_REF_HEAD_DIM;
                 column++)
                state_row[column] += k_head[row] * delta[column];
        }

        for (size_t column = 0; column < Q38_GDN_REF_HEAD_DIM; column++) {
            float value = 0.0f;
            for (size_t row = 0; row < Q38_GDN_REF_HEAD_DIM; row++)
                value += matrix[row * Q38_GDN_REF_HEAD_DIM + column] *
                         q_head[row];
            output_head[column] = scale * value;
        }
    }
    return true;
}

bool q38_gdn_ref_run(float *state, size_t sequence_count, size_t token_count,
                     const float *q, const float *k, const float *v,
                     const float *decay, const float *beta, float scale,
                     float *output) {
    if (!state || !q || !k || !v || !decay || !beta || !output)
        return false;
    const size_t qkv_stride = (size_t)Q38_GDN_REF_VALUE_HEADS *
                              (size_t)Q38_GDN_REF_HEAD_DIM;
    const size_t gate_stride = (size_t)Q38_GDN_REF_VALUE_HEADS;
    for (size_t sequence = 0; sequence < sequence_count; sequence++) {
        for (size_t token = 0; token < token_count; token++) {
            const size_t token_index = sequence * token_count + token;
            if (!q38_gdn_ref_step(
                    state, sequence_count, sequence, q + token_index * qkv_stride,
                    k + token_index * qkv_stride, v + token_index * qkv_stride,
                    decay + token_index * gate_stride,
                    beta + token_index * gate_stride, scale,
                    output + token_index * qkv_stride))
                return false;
        }
    }
    return true;
}

void q38_gdn_ref_repeat_key_heads(const float *key_heads,
                                   float *value_heads) {
    if (!key_heads || !value_heads) return;
    for (size_t value_head = 0;
         value_head < Q38_GDN_REF_VALUE_HEADS; value_head++) {
        const size_t key_head = value_head / 3u;
        memcpy(value_heads + value_head * Q38_GDN_REF_HEAD_DIM,
               key_heads + key_head * Q38_GDN_REF_HEAD_DIM,
               Q38_GDN_REF_HEAD_DIM * sizeof(float));
    }
}
