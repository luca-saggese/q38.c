#include "q38_qsa_ref.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const float keys[] = {1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
    const float queries[] = {
        1, 0, 0, 0, 0, 1, 0, 0,
    };
    float scores[4], tokens[3];
    char error[128];
    if (!q38_qsa_index_scores_ref(keys, 3, queries, 1, 2, 4, 2, scores,
                                  error, sizeof(error))) {
        fprintf(stderr, "index score failed: %s\n", error);
        return 1;
    }
    if (fabsf(scores[0] - 2.0f) > 2e-5f ||
        fabsf(scores[1] - 2.0f) > 2e-5f) {
        fprintf(stderr, "unexpected pooled/ReLU scores: %g %g\n",
                scores[0], scores[1]);
        return 1;
    }
    if (!q38_qsa_expand_block_scores_ref(scores, 2, 3, 2, tokens, error,
                                         sizeof(error)) ||
        tokens[0] != scores[0] || tokens[1] != scores[0] ||
        tokens[2] != scores[1]) {
        fprintf(stderr, "incomplete causal tail expansion failed\n");
        return 1;
    }
    puts("test_m5_qsa_ref: block pooling, RMS normalization, ReLU scoring, tail passed");
    return 0;
}
