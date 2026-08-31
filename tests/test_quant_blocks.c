#include "../to_be_deleted/gguf-tools/quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int check_type(ds4q_type type, int with_weights) {
    enum { rows = 512, cols = 256 };
    size_t row_size = ds4q_row_size(type, cols);
    float *src = malloc((size_t)rows * cols * sizeof(*src));
    float *weights = with_weights ? malloc((size_t)cols * sizeof(*weights)) : NULL;
    unsigned char *out = malloc((size_t)rows * row_size);
    if (!src || (with_weights && !weights) || !out) return 1;
    for (int c = 0; c < cols; c++) if (weights) weights[c] = 1.0f;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            src[(size_t)r * cols + c] =
                (float)((r * 17 + c * 31) % 101 - 50) / 13.0f;
    ds4q_quantize_init(type);
    for (int r = 0; r < rows; r++) {
        size_t written = ds4q_quantize_chunk(
            type, src, out, (int64_t)r * cols, 1, cols, weights);
        if (written != row_size) return 1;
    }
    for (size_t i = 0; i < (size_t)rows * row_size; i++) {
        if (out[i] != 0) {
            free(weights);
            free(out);
            free(src);
            return 0;
        }
    }
    free(weights);
    free(out);
    free(src);
    return 1;
}

int main(void) {
    if (check_type(DS4Q_TYPE_Q2_K, 0) ||
        check_type(DS4Q_TYPE_Q8_0, 0) ||
        check_type(DS4Q_TYPE_Q4_K, 0) ||
        check_type(DS4Q_TYPE_IQ2_XXS, 1)) {
        fprintf(stderr, "quant block or expert boundary test failed\n");
        return 1;
    }
    puts("test_quant_blocks: Q2/IQ2/Q4 donor block tests passed");
    return 0;
}
