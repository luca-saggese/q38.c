#include "q38_ple_ref.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_vec(const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (fabsf(a[i] - b[i]) > 2e-5f) return 0;
    return 1;
}

int main(void) {
    const q38_ple_forward_config c = {
        .hidden = 3, .streams = 2, .heads = 2, .row_width = 2,
        .kernel = 2, .dilation = 1, .eps = 1e-6f,
    };
    const float hidden[] = {1,2,3, 4,5,6, 2,1,0, 3,2,1};
    const float embedding[] = {1,2,3,4, 2,1,0,1};
    const float key_proj[] = {
        1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1,1,1,1, 1,0,1,0,
    };
    const float value_proj[] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float norm[] = {1,1,1,1,1,1};
    const float conv[] = {
        1,0,0,1,0,0,
        0,1,1,0,0,1,
    };
    float h1[6] = {0}, h2[6] = {0}, contribution[12], after[12];
    char error[128];
    if (!q38_ple_forward_ref(&c, hidden, 2, embedding, key_proj, value_proj,
                             norm, norm, norm, conv, h1, contribution, after,
                             error, sizeof(error))) {
        fprintf(stderr, "PLE injection failed: %s\n", error);
        return 1;
    }
    float chunk_contribution[12], chunk_after[12], first_after[6];
    if (!q38_ple_forward_ref(&c, hidden, 1, embedding, key_proj, value_proj,
                             norm, norm, norm, conv, h2, chunk_contribution,
                             first_after, error, sizeof(error)) ||
        !q38_ple_forward_ref(&c, hidden + 6, 1, embedding + 4, key_proj,
                             value_proj, norm, norm, norm, conv, h2,
                             chunk_contribution + 6, chunk_after, error,
                             sizeof(error)) ||
        !close_vec(after, first_after, 6) ||
        !close_vec(after + 6, chunk_after, 6)) {
        fprintf(stderr, "PLE injection chunk invariance failed: %s\n", error);
        return 1;
    }
    puts("test_m4_ple_injection_ref: grouped PLE gate, convolution, injection, and chunk state passed");
    return 0;
}
