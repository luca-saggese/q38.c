#include "q38_qsa.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    q38_tensor tensors[9];
    memset(tensors, 0, sizeof(tensors));
    q38_qsa_weights weights = {
        .q_proj = &tensors[0], .k_proj = &tensors[1], .v_proj = &tensors[2],
        .o_proj = &tensors[3], .q_norm = &tensors[4], .k_norm = &tensors[5],
        .index_qk_proj = &tensors[6], .index_q_norm = &tensors[7],
        .index_k_norm = &tensors[8],
    };
    char error[128];
    if (!q38_qsa_weights_validate(&weights, error, sizeof(error))) {
        fprintf(stderr, "QSA binding validation failed: %s\n", error);
        return 1;
    }
    weights.index_k_norm = NULL;
    if (q38_qsa_weights_validate(&weights, error, sizeof(error))) {
        fprintf(stderr, "incomplete QSA binding was accepted\n");
        return 1;
    }
    q38_qsa_state state;
    if (!q38_qsa_state_init(&state, 512, 512, 128, error, sizeof(error))) {
        fprintf(stderr, "QSA state init failed: %s\n", error);
        return 1;
    }
    state.position = 9;
    state.committed_tokens = 9;
    state.main_k.count = state.main_v.count = state.index_k.count = 9;
    q38_qsa_state_reset(&state);
    if (state.position != 0 || state.committed_tokens != 0 ||
        state.main_k.count || state.main_v.count || state.index_k.count) {
        fprintf(stderr, "QSA state reset failed\n");
        q38_qsa_state_destroy(&state);
        return 1;
    }
    q38_qsa_state_destroy(&state);
    puts("test_m5_qsa_binding: strict tensor family and separate state passed");
    return 0;
}
