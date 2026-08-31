#include "q38_oracle.h"
#include "q38_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_close(float actual, float expected) {
    return fabsf(actual - expected) > 1e-6f;
}

static int check_q2(void) {
    q38_q2_k_block blocks[2];
    memset(blocks, 0, sizeof(blocks));
    for (size_t b = 0; b < 2; b++) {
        blocks[b].d = 0x3c00; /* 1 */
        for (size_t i = 0; i < 16; i++) blocks[b].scales[i] = 1;
        for (size_t i = 0; i < 64; i++) blocks[b].qs[i] = 0xe4;
    }
    float out[512];
    char error[128];
    if (!q38_quant_dequantize_row(Q38_QUANT_Q2_K, blocks, 2, out, 512,
                                  error, sizeof(error))) {
        fprintf(stderr, "Q2 decode failed: %s\n", error);
        return 1;
    }
    for (size_t b = 0; b < 2; b++)
        for (size_t i = 0; i < 256; i++)
            if (check_close(out[b * 256 + i], (float)((i / 32) % 4))) {
                fprintf(stderr, "Q2 boundary mismatch at %zu: %g\n",
                        b * 256 + i, out[b * 256 + i]);
                return 1;
            }
    q38_oracle_metrics metrics;
    q38_oracle_compare(out, out, 512, 1e-6f, &metrics);
    return metrics.max_abs != 0.0f || metrics.cosine != 1.0f;
}

static int check_q4(void) {
    q38_q4_k_block block;
    memset(&block, 0, sizeof(block));
    block.d = 0x3c00; /* 1 */
    for (size_t i = 0; i < sizeof(block.scales); i++) block.scales[i] = 1;
    for (size_t i = 0; i < sizeof(block.qs); i++) block.qs[i] = 0x21;
    float out[256];
    char error[128];
    if (!q38_quant_dequantize_row(Q38_QUANT_Q4_K, &block, 1, out, 256,
                                  error, sizeof(error))) {
        fprintf(stderr, "Q4 decode failed: %s\n", error);
        return 1;
    }
    for (size_t i = 0; i < 256; i++) {
        float expected = i < 32 ? 1.0f : i < 64 ? 2.0f :
                         i < 96 ? 1.0f : i < 128 ? 2.0f :
                         i < 160 ? 1.0f : i < 192 ? 2.0f :
                         i < 224 ? 1.0f : 2.0f;
        if (check_close(out[i], expected)) {
            fprintf(stderr, "Q4 boundary mismatch at %zu: %g expected %g\n",
                    i, out[i], expected);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    if (check_q2() || check_q4()) return 1;
    puts("test_m2_quant: Q2_K and Q4_K scalar boundary/oracle tests passed");
    return 0;
}
