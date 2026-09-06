#include "moe_reference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static float bf16_to_float(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = ((bits >> 16) & 1u) + 0x7fffu;
    bits += rounding;
    return (uint16_t)(bits >> 16);
}

static float silu(float value) {
    return value / (1.0f + expf(-value));
}

static float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static float q2_block_value(const q38_q2_k_block *block, size_t element) {
    const size_t half = element / 128u;
    const size_t segment = (element % 128u) / 16u;
    const uint8_t packed =
        block->qs[half * 32u + (element % 32u)];
    const unsigned shift = (unsigned)(((element % 128u) / 32u) * 2u);
    const size_t scale_index = half * 8u + segment;
    const uint8_t scale = block->scales[scale_index];
    const float d = q38_half_to_float(block->d);
    const float dmin = q38_half_to_float(block->dmin);
    return d * (float)(scale & 0xfu) *
               (float)((packed >> shift) & 3u) -
           dmin * (float)(scale >> 4);
}

static float q2_row_dot(const q38_q2_k_block *rows, size_t row,
                        const float *input, size_t input_width) {
    const size_t blocks_per_row = input_width / Q38_QUANT_QK_K;
    float sum = 0.0f;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const q38_q2_k_block *payload =
            rows + row * blocks_per_row + block;
        for (size_t i = 0; i < Q38_QUANT_QK_K; ++i)
            sum += q2_block_value(payload, i) *
                   input[block * Q38_QUANT_QK_K + i];
    }
    return sum;
}

static float q2_transposed_column_dot(const q38_q2_k_block *rows,
                                      size_t column, const float *input,
                                      size_t input_width) {
    const size_t blocks_per_row = Q38_MOE_HIDDEN / Q38_QUANT_QK_K;
    float sum = 0.0f;
    for (size_t row = 0; row < input_width; ++row) {
        const q38_q2_k_block *payload =
            rows + row * blocks_per_row + column / Q38_QUANT_QK_K;
        sum += q2_block_value(payload, column % Q38_QUANT_QK_K) * input[row];
    }
    return sum;
}

bool q38_test_moe_router_bf16(
    const float *hidden, const uint16_t *router_bf16,
    q38_test_moe_route *route, float *logits_pre_cast,
    float *logits_effective, float *weights_pre_cast,
    float *weights_effective, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !router_bf16 || !route || !logits_pre_cast ||
        !logits_effective || !weights_pre_cast || !weights_effective)
        return fail(error, error_len, "invalid MoE router fixture");

    for (size_t expert = 0; expert < Q38_MOE_EXPERTS; ++expert) {
        float sum = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            sum += bf16_to_float(router_bf16[expert * Q38_MOE_HIDDEN + d]) *
                   hidden[d];
        logits_pre_cast[expert] = sum;
    }
    return q38_test_moe_select_logits(
        logits_pre_cast, route, logits_effective, weights_pre_cast,
        weights_effective, error, error_len);
}

bool q38_test_moe_select_logits(
    const float *logits_pre_cast, q38_test_moe_route *route,
    float *logits_effective, float *weights_pre_cast,
    float *weights_effective, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!logits_pre_cast || !route || !logits_effective ||
        !weights_pre_cast || !weights_effective)
        return fail(error, error_len, "invalid MoE logits fixture");
    float max_pre = -INFINITY;
    float max_effective = -INFINITY;
    for (size_t expert = 0; expert < Q38_MOE_EXPERTS; ++expert) {
        logits_effective[expert] =
            bf16_to_float(float_to_bf16(logits_pre_cast[expert]));
        if (logits_pre_cast[expert] > max_pre)
            max_pre = logits_pre_cast[expert];
        if (logits_effective[expert] > max_effective)
            max_effective = logits_effective[expert];
    }
    float pre_sum = 0.0f;
    float effective_sum = 0.0f;
    for (size_t expert = 0; expert < Q38_MOE_EXPERTS; ++expert) {
        weights_pre_cast[expert] = expf(logits_pre_cast[expert] - max_pre);
        weights_effective[expert] =
            expf(logits_effective[expert] - max_effective);
        pre_sum += weights_pre_cast[expert];
        effective_sum += weights_effective[expert];
    }
    for (size_t expert = 0; expert < Q38_MOE_EXPERTS; ++expert) {
        weights_pre_cast[expert] /= pre_sum;
        weights_effective[expert] /= effective_sum;
    }

    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        size_t best = Q38_MOE_EXPERTS;
        for (size_t expert = 0; expert < Q38_MOE_EXPERTS; ++expert) {
            bool used = false;
            for (size_t prior = 0; prior < k; ++prior)
                used |= route->expert[prior] == expert;
            if (used) continue;
            if (best == Q38_MOE_EXPERTS ||
                weights_pre_cast[expert] > weights_pre_cast[best] ||
                (weights_pre_cast[expert] == weights_pre_cast[best] &&
                 expert < best))
                best = expert;
        }
        route->expert[k] = (uint16_t)best;
        route->weight[k] = weights_effective[best];
    }

    float selected_sum = 0.0f;
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
        selected_sum += route->weight[k];
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        const float normalized = route->weight[k] / selected_sum;
        route->weight[k] = bf16_to_float(float_to_bf16(normalized));
    }
    return true;
}

bool q38_test_moe_expert_q2(
    const float *hidden, const q38_q2_k_block *gate_up,
    const q38_q2_k_block *down, float *output, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !gate_up || !down || !output)
        return fail(error, error_len, "invalid Q2 expert fixture");

    float intermediate[Q38_MOE_INTERMEDIATE];
    const size_t gate_rows = Q38_MOE_INTERMEDIATE;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        const float gate =
            q2_row_dot(gate_up, i, hidden, Q38_MOE_HIDDEN);
        const float up =
            q2_row_dot(gate_up, gate_rows + i, hidden, Q38_MOE_HIDDEN);
        intermediate[i] = silu(gate) * up;
    }
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        output[d] =
            q2_transposed_column_dot(down, d, intermediate,
                                     Q38_MOE_INTERMEDIATE);
    return true;
}

bool q38_test_moe_shared_f32(
    const float *hidden, const float *gate_proj, const float *up_proj,
    const float *down_proj, const float *gate_weight, float *output,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !gate_proj || !up_proj || !down_proj || !gate_weight ||
        !output)
        return fail(error, error_len, "invalid shared expert fixture");

    float intermediate[Q38_MOE_INTERMEDIATE];
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        float gate = 0.0f;
        float up = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
            gate += gate_proj[i * Q38_MOE_HIDDEN + d] * hidden[d];
            up += up_proj[i * Q38_MOE_HIDDEN + d] * hidden[d];
        }
        intermediate[i] = silu(gate) * up;
    }
    float gate_value = 0.0f;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        gate_value += gate_weight[d] * hidden[d];
    gate_value = sigmoid(gate_value);
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
        float value = 0.0f;
        for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
            value += down_proj[d * Q38_MOE_INTERMEDIATE + i] *
                     intermediate[i];
        output[d] = gate_value * value;
    }
    return true;
}

bool q38_test_moe_layer_q2(
    const float *hidden, const uint16_t *router_bf16,
    const q38_q2_k_block *selected_gate_up[Q38_MOE_TOP_K],
    const q38_q2_k_block *selected_down[Q38_MOE_TOP_K],
    const float *shared_gate_proj, const float *shared_up_proj,
    const float *shared_down_proj, const float *shared_gate_weight,
    float *output, q38_test_moe_route *route, char *error,
    size_t error_len) {
    float logits_pre[Q38_MOE_EXPERTS];
    float logits_effective[Q38_MOE_EXPERTS];
    float weights_pre[Q38_MOE_EXPERTS];
    float weights_effective[Q38_MOE_EXPERTS];
    if (!q38_test_moe_router_bf16(
            hidden, router_bf16, route, logits_pre, logits_effective,
            weights_pre, weights_effective, error, error_len))
        return false;

    memset(output, 0, Q38_MOE_HIDDEN * sizeof(float));
    float routed[Q38_MOE_HIDDEN];
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        if (!q38_test_moe_expert_q2(
                hidden, selected_gate_up[k], selected_down[k], routed,
                error, error_len))
            return false;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            output[d] += route->weight[k] * routed[d];
    }

    float shared[Q38_MOE_HIDDEN];
    if (!q38_test_moe_shared_f32(
            hidden, shared_gate_proj, shared_up_proj, shared_down_proj,
            shared_gate_weight, shared, error, error_len))
        return false;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        output[d] += shared[d];
    return true;
}
