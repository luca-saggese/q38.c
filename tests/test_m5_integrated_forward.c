#include "q38_forward.h"
#include "q38_ple_ref.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const q38_ple_forward_config pc = {
        .hidden = 2, .streams = 1, .heads = 1, .row_width = 2,
        .kernel = 1, .dilation = 1, .eps = 1e-6f,
    };
    const float input[] = {1, 0, 0, 1};
    const float emb[] = {1, 2, 2, 1};
    const float pmat[] = {1, 0, 0, 1};
    const float pconv[] = {0.1f, 0.2f};
    const float pn[] = {1, 1};
    float history[2] = {0}, contribution[4], after_ple[4];
    char error[128];
    if (!q38_ple_forward_ref(&pc, input, 2, emb, pmat, pmat, pn, pn, pn,
                             pconv, history, contribution, after_ple, error,
                             sizeof(error)))
        return 1;

    const float qmat[] = {1,0, 0,1, 1,0, 0,1};
    const float kmat[] = {1,0, 0,1};
    const float vmat[] = {1,0, 0,1};
    const float omat[] = {1,0, 0,1};
    const float imat[] = {1,0, 0,1};
    q38_forward_qsa_weights qw;
    memset(&qw, 0, sizeof(qw));
    qw.q_proj = (q38_forward_matrix){qmat, 4, 2, Q38_FORWARD_F32};
    qw.k_proj = (q38_forward_matrix){kmat, 2, 2, Q38_FORWARD_F32};
    qw.v_proj = (q38_forward_matrix){vmat, 2, 2, Q38_FORWARD_F32};
    qw.o_proj = (q38_forward_matrix){omat, 2, 2, Q38_FORWARD_F32};
    qw.index_qk_proj = (q38_forward_matrix){imat, 4, 2, Q38_FORWARD_F32};
    qw.q_norm = pn; qw.k_norm = pn; qw.index_q_norm = pn; qw.index_k_norm = pn;
    qw.hidden = 2; qw.query_heads = 1; qw.kv_heads = 1; qw.head_dim = 2;
    qw.index_heads = 1; qw.index_dim = 2; qw.ratio = 2; qw.budget = 2;
    qw.rope_theta = 10000000.0f; qw.rotary_dims = 2;
    q38_qsa_state state;
    if (!q38_forward_qsa_state_init(&state, &qw, error, sizeof(error)))
        return 1;
    float qsa_out[4];
    uint32_t ids[4];
    size_t counts[2];
    if (!q38_forward_qsa_ref(&qw, &state, after_ple, 2, qsa_out, ids, 2,
                             counts, error, sizeof(error)) ||
        counts[0] != 1 || counts[1] != 2) {
        fprintf(stderr, "integrated forward failed: %s\n", error);
        return 1;
    }
    for (size_t i = 0; i < 4; ++i)
        if (!isfinite(qsa_out[i])) return 1;
    q38_qsa_state_destroy(&state);
    puts("test_m5_integrated_forward: layer-stage PLE then QSA ordering passed");
    return 0;
}
