#include "q38_ple_ref.h"
#include "q38_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_q2(void) {
    q38_q2_k_block blocks[10];
    memset(blocks, 0, sizeof(blocks));
    for (size_t b = 0; b < 10; ++b) {
        blocks[b].d = 0x3c00;
        for (size_t i = 0; i < sizeof(blocks[b].scales); ++i)
            blocks[b].scales[i] = 1;
        for (size_t i = 0; i < sizeof(blocks[b].qs); ++i)
            blocks[b].qs[i] = 0xe4;
    }

    float out[2560];
    char error[128];
    if (!q38_ple_decode_row_ref(Q38_QUANT_Q2_K, blocks, 2560, out,
                                2560, error, sizeof(error))) {
        fprintf(stderr, "Q2 PLE row decode failed: %s\n", error);
        return 1;
    }
    for (size_t i = 0; i < 2560; ++i) {
        float expected = (float)((i % 256) / 32 % 4);
        if (fabsf(out[i] - expected) > 1e-6f) {
            fprintf(stderr, "Q2 PLE row mismatch at %zu: %g expected %g\n",
                    i, out[i], expected);
            return 1;
        }
    }
    return 0;
}

static int check_q4(void) {
    q38_q4_k_block blocks[10];
    memset(blocks, 0, sizeof(blocks));
    for (size_t b = 0; b < 10; ++b) {
        blocks[b].d = 0x3c00;
        for (size_t i = 0; i < sizeof(blocks[b].scales); ++i)
            blocks[b].scales[i] = 1;
        for (size_t i = 0; i < sizeof(blocks[b].qs); ++i)
            blocks[b].qs[i] = 0x21;
    }

    float out[2560];
    char error[128];
    if (!q38_ple_decode_row_ref(Q38_QUANT_Q4_K, blocks, 2560, out,
                                2560, error, sizeof(error))) {
        fprintf(stderr, "Q4 PLE row decode failed: %s\n", error);
        return 1;
    }
    for (size_t i = 0; i < 2560; ++i) {
        float expected = ((i % 256) / 32) % 2 ? 2.0f : 1.0f;
        if (fabsf(out[i] - expected) > 1e-6f) {
            fprintf(stderr, "Q4 PLE row mismatch at %zu: %g expected %g\n",
                    i, out[i], expected);
            return 1;
        }
    }
    return 0;
}

static int check_arguments(void) {
    q38_q2_k_block block;
    float out[2560];
    char error[128];
    memset(&block, 0, sizeof(block));
    if (q38_ple_decode_row_ref(Q38_QUANT_Q2_K, &block, 2559, out,
                               2560, error, sizeof(error)) ||
        q38_ple_decode_row_ref(Q38_QUANT_Q2_K, &block, 2560, out,
                               2559, error, sizeof(error)) ||
        q38_ple_decode_row_ref(30, &block, 2560, out,
                               2560, error, sizeof(error))) {
        fprintf(stderr, "invalid PLE row arguments were accepted\n");
        return 1;
    }
    return 0;
}

int main(void) {
    if (check_q2() || check_q4() || check_arguments()) return 1;
    puts("test_m4_ple_row: Q2_K/Q4_K scalar row decode and validation passed");
    return 0;
}
