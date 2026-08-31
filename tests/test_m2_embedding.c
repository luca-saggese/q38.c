#include "q38_gguf.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static uint64_t fnv1a(const uint8_t *data, size_t bytes) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s runtime-subset.gguf\n", argv[0]);
        return 1;
    }
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 0, &weights, error, sizeof(error))) {
        fprintf(stderr, "bind failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    static const uint32_t token_ids[] = {9419, 109266, 248045};
    const q38_tensor *embedding = weights.token_embd;
    if (!embedding || embedding->type != 30 || embedding->dim[1] != 2560) {
        fprintf(stderr, "unexpected embedding descriptor\n");
        q38_gguf_close(model);
        return 1;
    }
    const size_t row_bytes = (size_t)embedding->dim[1] * 2;
    printf("{\"gate\":\"M2-C07\",\"dtype\":\"BF16\",\"rows\":[");
    for (size_t i = 0; i < sizeof(token_ids) / sizeof(token_ids[0]); i++) {
        uint64_t offset = embedding->abs_offset +
            (uint64_t)token_ids[i] * row_bytes;
        if (offset > model->size || row_bytes > model->size - offset) {
            fprintf(stderr, "embedding row is outside mapped GGUF\n");
            q38_gguf_close(model);
            return 1;
        }
        if (i) putchar(',');
        printf("{\"token_id\":%u,\"elements\":2560,\"fnv1a64\":\"%016" PRIx64 "\"}",
               token_ids[i], fnv1a(model->map + offset, row_bytes));
    }
    puts("]}");
    q38_gguf_close(model);
    return 0;
}
