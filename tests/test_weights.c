#include "q38_gguf.h"
#include "q38_weights.h"

#include <stdio.h>

int main(int argc, char **argv) {
    uint32_t qtypes[Q38_MODEL_EXPERTS];
    for (uint32_t i = 0; i < Q38_MODEL_EXPERTS; i++) {
        qtypes[i] = (i == 1 || i == 511) ? 12 : 10;
    }
    q38_layer_expert_store mixed;
    if (!q38_expert_store_init_mixed(&mixed, qtypes) ||
        mixed.bank_count != 2 || mixed.bank[0].expert_count != 510 ||
        mixed.bank[1].expert_count != 2 ||
        mixed.loc[1].bank_id != mixed.loc[511].bank_id ||
        mixed.loc[1].local_index != 0 ||
        mixed.loc[511].local_index != 1) {
        fprintf(stderr, "mixed expert bank mapping failed\n");
        return 1;
    }
    q38_layer_expert_store uniform;
    if (!q38_expert_store_init_uniform(&uniform, 10) ||
        uniform.bank_count != 1 || uniform.bank[0].expert_count != 512 ||
        uniform.loc[511].bank_id != 0 ||
        uniform.loc[511].local_index != 511) {
        fprintf(stderr, "uniform expert bank mapping failed\n");
        return 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: test_weights subset.gguf\n");
        return 2;
    }
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 3, &weights, error, sizeof(error))) {
        fprintf(stderr, "bind failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    if (weights.bound_layers != 4 || weights.bound_tensor_count != 30 ||
        weights.layer[3].kind != Q38_LAYER_FULL_ATTENTION ||
        weights.layer[0].experts.loc[511].local_index != 511) {
        fprintf(stderr, "unexpected bound subset layout\n");
        q38_gguf_close(model);
        return 1;
    }
    q38_gguf_close(model);
    puts("test_weights: strict subset bind passed");
    return 0;
}
