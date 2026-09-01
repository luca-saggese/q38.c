#include "q38_gr_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int close_to(float actual, float expected) {
    return fabsf(actual - expected) <= 3e-5f;
}

int main(void) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    float *residual = calloc(width, sizeof(float));
    float *gamma = malloc(width * sizeof(float));
    float *down = calloc(Q38_GR_RANK * width, sizeof(float));
    float *up = calloc(width * Q38_GR_RANK, sizeof(float));
    float *inject = calloc(Q38_GR_BRANCHES * width, sizeof(float));
    float *input = calloc(Q38_GR_HIDDEN, sizeof(float));
    float *updated = calloc(width, sizeof(float));
    float *block_output = calloc(Q38_GR_HIDDEN, sizeof(float));
    if (!residual || !gamma || !down || !up || !inject || !input ||
        !updated || !block_output)
        return 1;
    for (size_t i = 0; i < width; i++) gamma[i] = 1.0f;

    q38_gr_ref_params params = {gamma, down, up, inject};
    for (size_t i = 0; i < Q38_GR_HIDDEN; i++) block_output[i] = 0.25f;
    q38_gr_collapse(residual, block_output, &params, input, updated);
    for (size_t i = 0; i < Q38_GR_HIDDEN; i++)
        if (!close_to(input[i], 0.0f) ||
            !close_to(updated[i], 0.25f) ||
            !close_to(updated[Q38_GR_HIDDEN + i], 0.25f) ||
            !close_to(updated[2 * Q38_GR_HIDDEN + i], 0.25f) ||
            !close_to(updated[3 * Q38_GR_HIDDEN + i], 0.25f)) {
            fprintf(stderr, "zero GR smoke mismatch at %zu\n", i);
            return 1;
        }
    for (size_t i = 0; i < width; i++) updated[i] = 0.0f;
    for (size_t i = 0; i < Q38_GR_HIDDEN; i++) block_output[i] = 0.0f;

    residual[0] = 2.0f;
    residual[1] = -1.0f;
    residual[Q38_GR_HIDDEN] = -3.0f;
    residual[Q38_GR_HIDDEN + 1] = 0.5f;
    residual[2 * Q38_GR_HIDDEN] = 1.5f;
    residual[3 * Q38_GR_HIDDEN + 1] = -2.0f;

    down[0 * width + 0] = 1.2f;
    down[1 * width + 1] = -0.7f;
    down[2 * width + Q38_GR_HIDDEN] = 0.8f;
    down[3 * width + 3 * Q38_GR_HIDDEN + 1] = 1.1f;

    up[0 * Q38_GR_RANK + 0] = 0.9f;
    up[0 * Q38_GR_RANK + 2] = -0.4f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + 1] = -0.8f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + 2] = 0.6f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + 0] = -0.5f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + 3] = 0.7f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + 1] = 0.3f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + 3] = -0.2f;
    up[Q38_GR_RANK + 0] = -0.2f;
    up[Q38_GR_RANK + 3] = 0.4f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 1] = 0.6f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 2] = -0.3f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 0] = 0.7f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 3] = 0.1f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 1] = -0.9f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 3] = 0.5f;

    inject[0 * width + 0] = 0.4f;
    inject[0 * width + Q38_GR_HIDDEN] = -0.6f;
    inject[1 * width + 1] = -0.8f;
    inject[1 * width + 3 * Q38_GR_HIDDEN + 1] = 0.5f;
    inject[2 * width + Q38_GR_HIDDEN] = 0.3f;
    inject[3 * width + 3 * Q38_GR_HIDDEN + 1] = 1.1f;
    block_output[0] = 0.25f;
    block_output[1] = -0.125f;

    q38_gr_collapse(residual, block_output, &params, input, updated);

    /*
     * Independently evaluated from docs/qwen_gdn_semantics.md with
     * SiLU(x / 4) and 2*sigmoid(inject / 4), not from q38_gr_ref.c.
     */
    const float expected_input[] = {10.79133470f, 1.17167685f};
    const float expected_updated[][2] = {
        {2.49999696f, -1.24999848f},
        {-2.92896527f, 0.46448264f},
        {1.51157222f, -0.00578611f},
        {0.00000046f, -2.00000023f},
    };
    if (!close_to(input[0], expected_input[0]) ||
        !close_to(input[1], expected_input[1])) {
        fprintf(stderr, "external GR read golden mismatch: %g %g\n",
                input[0], input[1]);
        return 1;
    }
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; branch++) {
        if (!close_to(updated[branch * Q38_GR_HIDDEN], expected_updated[branch][0]) ||
            !close_to(updated[branch * Q38_GR_HIDDEN + 1],
                      expected_updated[branch][1])) {
            fprintf(stderr, "external GR write golden mismatch at branch %zu\n",
                    branch);
            return 1;
        }
    }
    free(residual); free(gamma); free(down); free(up); free(inject);
    free(input); free(updated); free(block_output);
    puts("test_m3_gr_ref: external non-zero GR SiLU/scaling goldens passed");
    return 0;
}
