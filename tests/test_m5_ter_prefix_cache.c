#include "q38_gdn_ref.h"
#include "q38_qsa.h"
#include "q38_session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int qsa_equal(const q38_qsa_state *a, const q38_qsa_state *b) {
    return a->position == b->position &&
           a->committed_tokens == b->committed_tokens &&
           a->pending_count == b->pending_count &&
           a->pending_position == b->pending_position &&
           a->main_k.count == b->main_k.count &&
           memcmp(a->main_k.data, b->main_k.data,
                  a->main_k.count * a->main_k.row_bytes) == 0 &&
           memcmp(a->main_v.data, b->main_v.data,
                  a->main_v.count * a->main_v.row_bytes) == 0 &&
           memcmp(a->index_k.data, b->index_k.data,
                  a->index_k.count * a->index_k.row_bytes) == 0;
}

int main(int argc, char **argv) {
    const size_t prefix = 1024, suffix = 17;
    const size_t state_n = q38_gdn_ref_state_elements(1);
    float *miss_gdn = calloc(state_n, sizeof(float));
    float *hit_gdn = calloc(state_n, sizeof(float));
    float *prefix_gdn = calloc(state_n, sizeof(float));
    float q[48*128] = {0}, k[48*128] = {0}, v[48*128] = {0};
    float decay[48], beta[48], out[48*128];
    for (size_t h = 0; h < 48; ++h) { decay[h] = 0.99f; beta[h] = 0.1f; }
    q38_qsa_state miss, hit;
    char error[128];
    if (!miss_gdn || !hit_gdn || !prefix_gdn ||
        !q38_qsa_state_init(&miss, 4, 4, 2, error, sizeof(error)))
        return 1;
    for (size_t n = 0; n < prefix + suffix; ++n) {
        q[0] = k[0] = (float)(n + 1); v[0] = 0.5f;
        if (!q38_gdn_ref_step(miss_gdn, 1, 0, q, k, v, decay, beta,
                              1.0f, out)) return 1;
        unsigned char kv[4] = {(unsigned char)n, 1, 2, 3};
        unsigned char ix[2] = {(unsigned char)n, 4};
        if (!q38_qsa_state_append(&miss, kv, kv, ix, 1, error,
                                  sizeof(error))) return 1;
    }
    q38_gdn_ref_reset(prefix_gdn, 1);
    for (size_t n = 0; n < prefix; ++n) {
        q[0] = k[0] = (float)(n + 1); v[0] = 0.5f;
        if (!q38_gdn_ref_step(prefix_gdn, 1, 0, q, k, v, decay, beta,
                              1.0f, out)) return 1;
    }
    memcpy(hit_gdn, prefix_gdn, state_n * sizeof(float));
    if (!q38_qsa_state_init(&hit, 4, 4, 2, error, sizeof(error))) return 1;
    for (size_t n = 0; n < prefix; ++n) {
        unsigned char kv[4] = {(unsigned char)n, 1, 2, 3};
        unsigned char ix[2] = {(unsigned char)n, 4};
        if (!q38_qsa_state_append(&hit, kv, kv, ix, 1, error,
                                  sizeof(error))) return 1;
    }
    q38_qsa_state restored;
    if (!q38_qsa_state_clone(&hit, &restored, error, sizeof(error)))
        return 1;
    for (size_t n = prefix; n < prefix + suffix; ++n) {
        unsigned char kv[4] = {(unsigned char)n, 1, 2, 3};
        unsigned char ix[2] = {(unsigned char)n, 4};
        if (!q38_qsa_state_append(&restored, kv, kv, ix, 1, error,
                                  sizeof(error))) return 1;
        q[0] = k[0] = (float)(n + 1); v[0] = 0.5f;
        if (!q38_gdn_ref_step(hit_gdn, 1, 0, q, k, v, decay, beta,
                              1.0f, out)) return 1;
    }
    if (!qsa_equal(&miss, &restored) ||
        memcmp(miss_gdn, hit_gdn, state_n * sizeof(float)) != 0) return 1;
    FILE *outf = argc > 1 ? fopen(argv[1], "w") : NULL;
    if (!outf) return 2;
    fprintf(outf, "{\"prefix\":%zu,\"suffix\":%zu,\"qsa_state\":\"equal\","
            "\"gdn_state\":\"equal\",\"pending_mod4\":%u,\"status\":\"pass\"}\n",
            prefix, suffix, restored.pending_count);
    fclose(outf);
    q38_qsa_state_destroy(&hit);
    q38_qsa_state_destroy(&restored);
    free(miss_gdn); free(hit_gdn); free(prefix_gdn);
    puts("test_m5_ter_prefix_cache: prefix hit/miss GDN and QSA state equivalence passed");
    return 0;
}
