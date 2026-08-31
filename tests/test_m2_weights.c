#include "q38_weights.h"

#include <stdio.h>
#include <string.h>

static void put_u64(unsigned char *map, uint64_t at, uint64_t value) {
    memcpy(map + at, &value, sizeof(value));
}

static void put_u32(unsigned char *map, uint64_t at, uint32_t value) {
    memcpy(map + at, &value, sizeof(value));
}

static void put_bool(unsigned char *map, uint64_t at, bool value) {
    map[at] = value ? 1 : 0;
}

static void setup_metadata(q38_gguf *model, q38_kv *kv, unsigned char *map) {
    static const char *keys[] = {
        "general.architecture", "q38.runtime_only", "q38.excluded_vision",
        "q38.excluded_mtp", "q38.max_layer", "q38.quantized"
    };
    static const uint32_t types[] = {8, 7, 7, 7, 4, 7};
    memset(model, 0, sizeof(*model));
    memset(map, 0, 128);
    model->map = map;
    model->size = 128;
    model->n_kv = 6;
    model->kv = kv;
    for (unsigned i = 0; i < 6; i++) {
        kv[i].key.ptr = keys[i];
        kv[i].key.len = strlen(keys[i]);
        kv[i].type = types[i];
    }
    put_u64(map, 0, 9);
    memcpy(map + 8, "qwen4_exp", 9);
    kv[0].value_pos = 0;
    put_bool(map, 24, true); kv[1].value_pos = 24;
    put_bool(map, 25, true); kv[2].value_pos = 25;
    put_bool(map, 26, true); kv[3].value_pos = 26;
    put_u32(map, 27, 0); kv[4].value_pos = 27;
    put_bool(map, 31, false); kv[5].value_pos = 31;
}

static void tensor(q38_tensor *out, const char *name, uint64_t rows,
                   uint64_t cols) {
    memset(out, 0, sizeof(*out));
    out->name.ptr = name;
    out->name.len = strlen(name);
    out->ndim = 2;
    out->dim[0] = rows;
    out->dim[1] = cols;
    out->type = 30;
}

int main(void) {
    q38_gguf model;
    q38_kv kv[6];
    unsigned char map[128];
    q38_tensor tensors[2];
    char error[256];
    setup_metadata(&model, kv, map);

    tensor(&tensors[0], "model.language_model.embed_tokens.weight", 248319, 2560);
    model.n_tensors = 1;
    model.tensors = tensors;
    q38_weights weights;
    if (q38_weights_bind_subset(&model, 0, &weights, error, sizeof(error)) ||
        strstr(error, "shape/type mismatch") == NULL) {
        fprintf(stderr, "wrong-shape tensor was not rejected precisely\n");
        return 1;
    }

    tensor(&tensors[0], "model.language_model.embed_tokens.weight", 248320, 2560);
    model.n_tensors = 1;
    if (q38_weights_bind_subset(&model, 0, &weights, error, sizeof(error)) ||
        strstr(error, "missing required tensor: lm_head.weight") == NULL) {
        fprintf(stderr, "missing tensor was not rejected precisely\n");
        return 1;
    }
    puts("test_m2_weights: strict missing and shape/type gates passed");
    return 0;
}
