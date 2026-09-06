#ifndef Q38_TEST_MOE_REFERENCE_H
#define Q38_TEST_MOE_REFERENCE_H

#include "../../q38_quant.h"
#include "../../q38_moe_ref.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_TEST_MOE_ROUTER_VALUES (Q38_MOE_EXPERTS * Q38_MOE_HIDDEN)
#define Q38_TEST_MOE_GATE_UP_BLOCKS \
    (2u * Q38_MOE_INTERMEDIATE * (Q38_MOE_HIDDEN / Q38_QUANT_QK_K))
#define Q38_TEST_MOE_DOWN_BLOCKS \
    (Q38_MOE_INTERMEDIATE * (Q38_MOE_HIDDEN / Q38_QUANT_QK_K))

typedef struct {
    uint16_t expert[Q38_MOE_TOP_K];
    float weight[Q38_MOE_TOP_K];
} q38_test_moe_route;

bool q38_test_moe_router_bf16(
    const float *hidden, const uint16_t *router_bf16,
    q38_test_moe_route *route, float *logits_pre_cast,
    float *logits_effective, float *weights_pre_cast,
    float *weights_effective, char *error, size_t error_len);

bool q38_test_moe_select_logits(
    const float *logits_pre_cast, q38_test_moe_route *route,
    float *logits_effective, float *weights_pre_cast,
    float *weights_effective, char *error, size_t error_len);

bool q38_test_moe_expert_q2(
    const float *hidden, const q38_q2_k_block *gate_up,
    const q38_q2_k_block *down, float *output, char *error,
    size_t error_len);

bool q38_test_moe_shared_f32(
    const float *hidden, const float *gate_proj, const float *up_proj,
    const float *down_proj, const float *gate_weight, float *output,
    char *error, size_t error_len);

bool q38_test_moe_layer_q2(
    const float *hidden, const uint16_t *router_bf16,
    const q38_q2_k_block *selected_gate_up[Q38_MOE_TOP_K],
    const q38_q2_k_block *selected_down[Q38_MOE_TOP_K],
    const float *shared_gate_proj, const float *shared_up_proj,
    const float *shared_down_proj, const float *shared_gate_weight,
    float *output, q38_test_moe_route *route, char *error,
    size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
