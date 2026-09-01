#include "q38_moe_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    float *hidden = calloc(Q38_MOE_HIDDEN, sizeof(float));
    float *gate_up = calloc(2u * Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN,
                            sizeof(float));
    float *down = calloc(Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE, sizeof(float));
    float *output = calloc(Q38_MOE_HIDDEN, sizeof(float));
    if (!hidden || !gate_up || !down || !output) return 1;
    hidden[0] = 1.0f;
    gate_up[0] = 1.0f;
    gate_up[Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN] = 2.0f;
    down[0] = 3.0f;
    char error[128];
    if (!q38_moe_expert_ref(hidden, gate_up, down, output, error,
                            sizeof(error)) ||
        fabsf(output[0] - 4.3863515f) > 1e-4f) {
        fprintf(stderr, "expert reference failed: %s output=%g\n", error,
                output[0]);
        return 1;
    }
    float *shared_gate = calloc(Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN,
                                sizeof(float));
    float *shared_up = calloc(Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN,
                              sizeof(float));
    float *shared_down = calloc(Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE,
                                sizeof(float));
    float *shared_weight = calloc(Q38_MOE_HIDDEN, sizeof(float));
    if (!shared_gate || !shared_up || !shared_down || !shared_weight)
        return 1;
    shared_gate[0] = 1.0f; shared_up[0] = 2.0f; shared_down[0] = 3.0f;
    if (!q38_moe_shared_ref(hidden, shared_gate, shared_up, shared_down,
                            shared_weight, output, error, sizeof(error)) ||
        fabsf(output[0] - 2.1931757f) > 1e-4f) {
        fprintf(stderr, "shared expert failed: %s output=%g\n", error, output[0]);
        return 1;
    }
    free(shared_gate); free(shared_up); free(shared_down); free(shared_weight);
    free(hidden); free(gate_up); free(down); free(output);
    puts("test_m6_expert_ref: routed gate/up SiLU/down reference passed");
    return 0;
}
