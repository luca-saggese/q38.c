#include "q38_rope_ref.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    q38_rope_config config = {
        .theta = 10000000.0f,
        .freq_scale = 1.0f,
        .ext_factor = 0.0f,
        .attn_factor = 1.0f,
        .beta_fast = 0.0f,
        .beta_slow = 0.0f,
        .n_dims = 64,
        .sections = {11, 11, 10, 0},
        .interleaved = true,
    };
    float input[256], output[256], expected[256];
    for (size_t i = 0; i < 256; ++i) input[i] = (float)i / 17.0f;
    const int64_t positions[] = {32, 32, 32, 32};
    char error[128];
    if (!q38_rope_apply_ref(&config, positions, input, output, 256, error,
                            sizeof(error))) {
        fprintf(stderr, "RoPE reference failed: %s\n", error);
        return 1;
    }
    memcpy(expected, input, sizeof(expected));
    for (uint32_t i = 0; i < 64; i += 2) {
        const float angle = 32.0f *
            powf(powf(10000000.0f, -2.0f / 64.0f), i / 2);
        const float c = cosf(angle), s = sinf(angle);
        const size_t a = i / 2, b = a + 32;
        expected[a] = input[a] * c - input[b] * s;
        expected[b] = input[a] * s + input[b] * c;
    }
    for (size_t i = 0; i < 256; ++i) {
        if (fabsf(output[i] - expected[i]) > 2e-6f) {
            fprintf(stderr, "RoPE mismatch at %zu\n", i);
            return 1;
        }
    }
    if (memcmp(output + 64, input + 64, 192 * sizeof(float)) != 0) {
        fprintf(stderr, "unrotated dimensions changed\n");
        return 1;
    }
    puts("test_m5_rope_ref: partial interleaved mRoPE reference passed");
    return 0;
}
