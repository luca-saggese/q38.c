#include "moe_reference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_zero(const float *values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (values[i] != 0.0f) return 0;
    return 1;
}

int main(void) {
    const size_t router_values = Q38_TEST_MOE_ROUTER_VALUES;
    const size_t gate_up_blocks = Q38_TEST_MOE_GATE_UP_BLOCKS;
    const size_t down_blocks = Q38_TEST_MOE_DOWN_BLOCKS;
    const size_t matrix_values =
        Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN;
    float *hidden = calloc(Q38_MOE_HIDDEN, sizeof(float));
    uint16_t *router = calloc(router_values, sizeof(uint16_t));
    q38_q2_k_block *gate_up = calloc(gate_up_blocks, sizeof(*gate_up));
    q38_q2_k_block *down = calloc(down_blocks, sizeof(*down));
    float *shared_gate = calloc(matrix_values, sizeof(float));
    float *shared_up = calloc(matrix_values, sizeof(float));
    float *shared_down = calloc(matrix_values, sizeof(float));
    float *shared_weight = calloc(Q38_MOE_HIDDEN, sizeof(float));
    float *output = calloc(Q38_MOE_HIDDEN, sizeof(float));
    float *logits = calloc(Q38_MOE_EXPERTS, sizeof(float));
    float *effective_logits = calloc(Q38_MOE_EXPERTS, sizeof(float));
    float *weights_pre = calloc(Q38_MOE_EXPERTS, sizeof(float));
    float *weights_effective = calloc(Q38_MOE_EXPERTS, sizeof(float));
    q38_test_moe_route route = {0};
    char error[256] = {0};
    int ok = hidden && router && gate_up && down && shared_gate && shared_up &&
             shared_down && shared_weight && output && logits &&
             effective_logits && weights_pre && weights_effective;
    if (!ok) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    ok = q38_test_moe_router_bf16(
        hidden, router, &route, logits, effective_logits, weights_pre,
        weights_effective, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "router oracle failed: %s\n", error);
        return 1;
    }
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        if (route.expert[k] != k || !isfinite(route.weight[k])) {
            fprintf(stderr, "router tie ordering failed at %zu\n", k);
            return 1;
        }
    }
    {
        float expert[Q38_MOE_HIDDEN];
        if (!q38_test_moe_expert_q2(hidden, gate_up, down, expert, error,
                                    sizeof(error))) {
            fprintf(stderr, "expert oracle failed: %s\n", error);
            return 1;
        }
    }
    {
        float shared[Q38_MOE_HIDDEN];
        if (!q38_test_moe_shared_f32(hidden, shared_gate, shared_up,
                                     shared_down, shared_weight, shared, error,
                                     sizeof(error))) {
            fprintf(stderr, "shared oracle failed: %s\n", error);
            return 1;
        }
    }

    const q38_q2_k_block *selected_gate[Q38_MOE_TOP_K];
    const q38_q2_k_block *selected_down[Q38_MOE_TOP_K];
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        selected_gate[k] = gate_up;
        selected_down[k] = down;
    }
    ok = q38_test_moe_layer_q2(
        hidden, router, selected_gate, selected_down, shared_gate, shared_up,
        shared_down, shared_weight, output, &route, error, sizeof(error));
    if (!ok || !check_zero(output, Q38_MOE_HIDDEN)) {
        fprintf(stderr, "layer oracle failed: %s output0=%g output1=%g\n",
                error, output[0], output[1]);
        return 1;
    }

    free(hidden);
    free(router);
    free(gate_up);
    free(down);
    free(shared_gate);
    free(shared_up);
    free(shared_down);
    free(shared_weight);
    free(output);
    free(logits);
    free(effective_logits);
    free(weights_pre);
    free(weights_effective);
    puts("test_moe_reference: zero-fixture and router tie goldens passed");
    return 0;
}
