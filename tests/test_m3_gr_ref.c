#include "q38_gr_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    float *residual = calloc(width, sizeof(float));
    float *gamma = malloc(width * sizeof(float));
    float *down = calloc(Q38_GR_RANK * width, sizeof(float));
    float *up = calloc(width * Q38_GR_RANK, sizeof(float));
    float *inject = calloc(Q38_GR_BRANCHES * width, sizeof(float));
    float *input = malloc(Q38_GR_HIDDEN * sizeof(float));
    float *updated = malloc(width * sizeof(float));
    float *block_output = malloc(Q38_GR_HIDDEN * sizeof(float));
    if (!residual || !gamma || !down || !up || !inject || !input ||
        !updated || !block_output) return 1;
    for (size_t i = 0; i < width; i++) gamma[i] = 1.0f;
    for (size_t i = 0; i < width; i++) inject[i] = 0.0f;
    for (size_t i = 0; i < Q38_GR_HIDDEN; i++) block_output[i] = 0.25f;
    q38_gr_ref_params params = {
        .gamma = gamma,
        .input_mix_down = down,
        .input_mix_up = up,
        .block_inject = inject,
    };
    q38_gr_collapse(residual, block_output, &params, input, updated);
    for (size_t i = 0; i < Q38_GR_HIDDEN; i++)
        if (fabsf(input[i]) > 1e-7f) {
            fprintf(stderr, "zero read gate mismatch at %zu\n", i);
            return 1;
        }
    const float expected_scale = 1.0f;
    for (size_t i = 0; i < width; i++)
        if (fabsf(updated[i] - (residual[i] + expected_scale * 0.25f)) > 1e-6f) {
            fprintf(stderr, "write gate mismatch at %zu\n", i);
            return 1;
        }
    free(residual); free(gamma); free(down); free(up); free(inject);
    free(input); free(updated); free(block_output);
    puts("test_m3_gr_ref: GR read/write/collapse golden vectors passed");
    return 0;
}
