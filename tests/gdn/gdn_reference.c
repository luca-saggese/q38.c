#include "gdn_reference.h"

#include "../../q38_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static float weight_scalar(const q38_test_gdn_weight *weight, size_t index) {
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
    if (weight->type == 8) {
        const size_t block = index / 32u;
        const size_t element = index % 32u;
        const unsigned char *payload = weight->data + block * 34u;
        uint16_t bits;
        memcpy(&bits, payload, sizeof(bits));
        return q38_half_to_float(bits) *
               (float)((const int8_t *)(payload + 2))[element];
    }
    return NAN;
}

static bool decode_row(const q38_test_gdn_weight *weight, size_t row,
                       float *decoded, char *error, size_t error_len) {
    if (!weight || !decoded || row >= weight->rows || !weight->data)
        return fail(error, error_len, "invalid GDN reference row");
    const unsigned char *row_data = weight->data;
    if (weight->type == 30) {
        row_data += row * weight->cols * sizeof(uint16_t);
        for (size_t i = 0; i < weight->cols; ++i) {
            uint16_t bits;
            memcpy(&bits, row_data + i * sizeof(bits), sizeof(bits));
            decoded[i] = bf16_to_float(bits);
        }
        return true;
    }
    if (weight->type == 0) {
        memcpy(decoded, row_data + row * weight->cols * sizeof(float),
               weight->cols * sizeof(float));
        return true;
    }
    if (weight->type == 8) {
        row_data += row * ((weight->cols / 32u) * 34u);
        for (size_t i = 0; i < weight->cols; ++i)
            decoded[i] = weight_scalar(
                &(q38_test_gdn_weight){.type = 8, .data = row_data},
                i);
        return true;
    }
    if (weight->type == 10 || weight->type == 12)
        return q38_quant_dequantize_row(
            weight->type, row_data + row * (weight->cols / 256u) *
                                      (weight->type == 10 ? 84u : 144u),
            weight->cols / 256u, decoded, weight->cols, error, error_len);
    return fail(error, error_len, "unsupported GDN reference weight type");
}

static bool matrix_project(const q38_test_gdn_weight *weight,
                           const float *input, float *output, char *error,
                           size_t error_len) {
    float *row = (float *)malloc(weight->cols * sizeof(float));
    if (!row) return fail(error, error_len, "GDN oracle row allocation failed");
    for (size_t r = 0; r < weight->rows; ++r) {
        if (!decode_row(weight, r, row, error, error_len)) {
            free(row);
            return false;
        }
        float sum = 0.0f;
        for (size_t c = 0; c < weight->cols; ++c)
            sum += row[c] * input[c];
        output[r] = sum;
    }
    free(row);
    return true;
}

static void rms_norm(float *values, const float *weight, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
        sum += (double)values[i] * values[i];
    const float scale =
        1.0f / sqrtf((float)(sum / (double)count) + 1.0e-6f);
    for (size_t i = 0; i < count; ++i)
        values[i] *= scale * weight[i];
}

bool q38_test_gdn_reference(
    const float *hidden, const float *history, const float *state,
    const q38_test_gdn_weight *qkv, const q38_test_gdn_weight *z,
    const q38_test_gdn_weight *a, const q38_test_gdn_weight *b,
    const q38_test_gdn_weight *conv, const q38_test_gdn_weight *a_log,
    const q38_test_gdn_weight *dt_bias, const q38_test_gdn_weight *norm,
    const q38_test_gdn_weight *out_proj, float *output, float *next_history,
    float *next_state, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !history || !state || !qkv || !z || !a || !b || !conv ||
        !a_log || !dt_bias || !norm || !out_proj || !output ||
        !next_history || !next_state)
        return fail(error, error_len, "invalid GDN reference arguments");

    float qkv_values[Q38_TEST_GDN_QKV];
    float z_values[Q38_TEST_GDN_Z];
    float a_values[Q38_TEST_GDN_HEADS];
    float b_values[Q38_TEST_GDN_HEADS];
    if (!matrix_project(qkv, hidden, qkv_values, error, error_len) ||
        !matrix_project(z, hidden, z_values, error, error_len) ||
        !matrix_project(a, hidden, a_values, error, error_len) ||
        !matrix_project(b, hidden, b_values, error, error_len))
        return false;

    float conv_values[Q38_TEST_GDN_QKV];
    for (size_t channel = 0; channel < Q38_TEST_GDN_QKV; ++channel) {
        float sum = 0.0f;
        for (size_t tap = 0; tap < Q38_TEST_GDN_KERNEL; ++tap) {
            const size_t source = tap;
            const float sample = source < Q38_TEST_GDN_HISTORY
                ? history[source * Q38_TEST_GDN_QKV + channel]
                : qkv_values[(source - Q38_TEST_GDN_HISTORY) *
                             Q38_TEST_GDN_QKV + channel];
            sum += weight_scalar(conv, channel * Q38_TEST_GDN_KERNEL + tap) *
                   sample;
        }
        conv_values[channel] = sum / (1.0f + expf(-sum));
    }
    for (size_t tail = 0; tail < Q38_TEST_GDN_HISTORY; ++tail) {
        const size_t source = 1u + tail;
        if (source < Q38_TEST_GDN_HISTORY)
            memcpy(next_history + tail * Q38_TEST_GDN_QKV,
                   history + source * Q38_TEST_GDN_QKV,
                   Q38_TEST_GDN_QKV * sizeof(float));
        else
            memcpy(next_history + tail * Q38_TEST_GDN_QKV,
                   qkv_values + (source - Q38_TEST_GDN_HISTORY) *
                                    Q38_TEST_GDN_QKV,
                   Q38_TEST_GDN_QKV * sizeof(float));
    }

    float q[Q38_TEST_GDN_HEADS * Q38_TEST_GDN_DIM];
    float k[Q38_TEST_GDN_HEADS * Q38_TEST_GDN_DIM];
    float v[Q38_TEST_GDN_HEADS * Q38_TEST_GDN_DIM];
    float decay[Q38_TEST_GDN_HEADS];
    float beta[Q38_TEST_GDN_HEADS];
    for (size_t head = 0; head < Q38_TEST_GDN_HEADS; ++head) {
        const size_t key_head = head / 3u;
        for (size_t d = 0; d < Q38_TEST_GDN_DIM; ++d) {
            q[head * Q38_TEST_GDN_DIM + d] =
                conv_values[key_head * Q38_TEST_GDN_DIM + d];
            k[head * Q38_TEST_GDN_DIM + d] =
                conv_values[Q38_TEST_GDN_KEY_HEADS * Q38_TEST_GDN_DIM +
                            key_head * Q38_TEST_GDN_DIM + d];
            v[head * Q38_TEST_GDN_DIM + d] =
                conv_values[2u * Q38_TEST_GDN_KEY_HEADS * Q38_TEST_GDN_DIM +
                            head * Q38_TEST_GDN_DIM + d];
        }
        float q_norm = 0.0f, k_norm = 0.0f;
        for (size_t d = 0; d < Q38_TEST_GDN_DIM; ++d) {
            q_norm += q[head * Q38_TEST_GDN_DIM + d] *
                      q[head * Q38_TEST_GDN_DIM + d];
            k_norm += k[head * Q38_TEST_GDN_DIM + d] *
                      k[head * Q38_TEST_GDN_DIM + d];
        }
        const float q_scale = 1.0f / sqrtf(q_norm + 1.0e-6f);
        const float k_scale = 1.0f / sqrtf(k_norm + 1.0e-6f);
        for (size_t d = 0; d < Q38_TEST_GDN_DIM; ++d) {
            q[head * Q38_TEST_GDN_DIM + d] *= q_scale;
            k[head * Q38_TEST_GDN_DIM + d] *= k_scale;
        }
        const float av = a_values[head] + weight_scalar(dt_bias, head);
        decay[head] = expf(-expf(weight_scalar(a_log, head)) *
                           log1pf(expf(av)));
        beta[head] = 1.0f / (1.0f + expf(-b_values[head]));
    }

    memcpy(next_state, state,
           Q38_TEST_GDN_HEADS * Q38_TEST_GDN_DIM * Q38_TEST_GDN_DIM *
               sizeof(float));
    float recurrent[Q38_TEST_GDN_HEADS * Q38_TEST_GDN_DIM];
    const float scale = 1.0f / sqrtf((float)Q38_TEST_GDN_DIM);
    for (size_t head = 0; head < Q38_TEST_GDN_HEADS; ++head) {
        float delta[Q38_TEST_GDN_DIM];
        float *matrix = next_state + head * Q38_TEST_GDN_DIM *
                                      Q38_TEST_GDN_DIM;
        for (size_t row = 0; row < Q38_TEST_GDN_DIM; ++row)
            for (size_t column = 0; column < Q38_TEST_GDN_DIM; ++column)
                matrix[row * Q38_TEST_GDN_DIM + column] *= decay[head];
        for (size_t column = 0; column < Q38_TEST_GDN_DIM; ++column) {
            float prediction = 0.0f;
            for (size_t row = 0; row < Q38_TEST_GDN_DIM; ++row)
                prediction += matrix[row * Q38_TEST_GDN_DIM + column] *
                              k[head * Q38_TEST_GDN_DIM + row];
            delta[column] =
                (v[head * Q38_TEST_GDN_DIM + column] - prediction) *
                beta[head];
        }
        for (size_t row = 0; row < Q38_TEST_GDN_DIM; ++row)
            for (size_t column = 0; column < Q38_TEST_GDN_DIM; ++column)
                matrix[row * Q38_TEST_GDN_DIM + column] +=
                    k[head * Q38_TEST_GDN_DIM + row] * delta[column];
        for (size_t column = 0; column < Q38_TEST_GDN_DIM; ++column) {
            float value = 0.0f;
            for (size_t row = 0; row < Q38_TEST_GDN_DIM; ++row)
                value += matrix[row * Q38_TEST_GDN_DIM + column] *
                         q[head * Q38_TEST_GDN_DIM + row];
            recurrent[head * Q38_TEST_GDN_DIM + column] = scale * value;
        }
    }

    float norm_values[Q38_TEST_GDN_DIM];
    for (size_t d = 0; d < Q38_TEST_GDN_DIM; ++d)
        norm_values[d] = weight_scalar(norm, d);
    for (size_t head = 0; head < Q38_TEST_GDN_HEADS; ++head)
        rms_norm(recurrent + head * Q38_TEST_GDN_DIM, norm_values,
                 Q38_TEST_GDN_DIM);
    for (size_t head = 0; head < Q38_TEST_GDN_HEADS; ++head)
        for (size_t d = 0; d < Q38_TEST_GDN_DIM; ++d)
            recurrent[head * Q38_TEST_GDN_DIM + d] *=
                1.0f / (1.0f + expf(-z_values[head * Q38_TEST_GDN_DIM + d]));

    if (!matrix_project(out_proj, recurrent, output, error, error_len))
        return false;
    return true;
}
