#include "q38_forward.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    const float q_proj[] = {1,0, 0,1, 1,0, 0,1};
    const float k_proj[] = {1,0, 0,1};
    const float v_proj[] = {1,0, 0,1};
    const float o_proj[] = {1,0, 0,1};
    const float index_proj[] = {1,0, 0,1, 1,0, 0,1};
    const float norm[] = {1,1};
    q38_forward_qsa_weights w;
    memset(&w, 0, sizeof(w));
    w.q_proj = (q38_forward_matrix){q_proj, 4, 2, Q38_FORWARD_F32};
    w.k_proj = (q38_forward_matrix){k_proj, 2, 2, Q38_FORWARD_F32};
    w.v_proj = (q38_forward_matrix){v_proj, 2, 2, Q38_FORWARD_F32};
    w.o_proj = (q38_forward_matrix){o_proj, 2, 2, Q38_FORWARD_F32};
    w.index_qk_proj = (q38_forward_matrix){index_proj, 4, 2, Q38_FORWARD_F32};
    w.q_norm = norm; w.k_norm = norm; w.index_q_norm = norm; w.index_k_norm = norm;
    w.hidden = 2; w.query_heads = 1; w.kv_heads = 1; w.head_dim = 2;
    w.index_heads = 1; w.index_dim = 2; w.ratio = 2; w.budget = 2;
    w.rope_theta = 10000000.0f; w.rotary_dims = 2;
    q38_qsa_state state;
    char error[128];
    if (!q38_forward_qsa_state_init(&state, &w, error, sizeof(error)))
        return 1;
    const float hidden[] = {1,0, 0,1, 1,1};
    float output[6];
    uint32_t selected[9];
    size_t counts[3];
    if (!q38_forward_qsa_ref(&w, &state, hidden, 3, output,
                             selected, 3, counts,
                             error, sizeof(error)) ||
        counts[0] != 1 || counts[1] != 2 || counts[2] != 3 ||
        state.position != 3 || state.committed_tokens != 3) {
        fprintf(stderr, "reference forward failed: %s\n", error);
        q38_qsa_state_destroy(&state);
        return 1;
    }
    q38_qsa_state_destroy(&state);
    puts("test_forward_ref: file-backed-compatible QSA forward/decode graph passed");
    return 0;
}
