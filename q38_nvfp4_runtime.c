#include "q38_nvfp4_runtime.h"

#include "q38_quant.h"
#include "q38_weights.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q38_NATIVE_BF16_TYPE 30u

struct q38_nvfp4_runtime_model {
    q38_nvfp4_pack *pack;
    q38_gguf *model;
    q38_tensor *owned_scale;
    char **owned_names;
    size_t owned_name_count;
    size_t owned_name_capacity;
};

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_nvfp4_path_is_pack(const char *path) {
    static const unsigned char magic[24] = {
        'Q', '3', '8', '_', 'N', 'V', 'F', 'P', '4', '_', 'P', 'A',
        'C', 'K', '_', 'V', '1'
    };
    if (!path) return false;
    FILE *file = fopen(path, "rb");
    if (!file) return false;
    unsigned char header[sizeof(magic)];
    const bool ok = fread(header, 1, sizeof(header), file) == sizeof(header) &&
                    memcmp(header, magic, sizeof(magic)) == 0;
    fclose(file);
    return ok;
}

static bool own_name(q38_nvfp4_runtime_model *runtime, const char *name,
                     size_t length, const char **out) {
    if (!runtime || !name || length > SIZE_MAX - 1) return false;
    if (runtime->owned_name_count == runtime->owned_name_capacity) {
        size_t next = runtime->owned_name_capacity
            ? runtime->owned_name_capacity * 2 : 128;
        char **grown = realloc(runtime->owned_names, next * sizeof(*grown));
        if (!grown) return false;
        runtime->owned_names = grown;
        runtime->owned_name_capacity = next;
    }
    char *copy = malloc(length + 1);
    if (!copy) return false;
    memcpy(copy, name, length);
    copy[length] = '\0';
    runtime->owned_names[runtime->owned_name_count++] = copy;
    *out = copy;
    return true;
}

static bool add_tensor(q38_nvfp4_runtime_model *runtime,
                       uint32_t *cursor, const char *name, size_t name_len,
                       uint32_t ndim, const uint64_t shape[4], uint32_t type,
                       const void *data, uint64_t bytes) {
    if (!runtime || !runtime->model || !cursor || !name ||
        *cursor >= runtime->model->n_tensors || ndim > Q38_MAX_DIMS)
        return false;
    q38_tensor *tensor = &runtime->model->tensors[(*cursor)++];
    memset(tensor, 0, sizeof(*tensor));
    tensor->name.ptr = name;
    tensor->name.len = name_len;
    tensor->ndim = ndim;
    memcpy(tensor->dim, shape, 4u * sizeof(shape[0]));
    tensor->type = type;
    tensor->data = data;
    tensor->bytes = bytes;
    tensor->elements = 1;
    for (uint32_t i = 0; i < ndim; ++i)
        tensor->elements *= shape[i];
    return true;
}

static uint32_t native_type(uint32_t dtype) {
    if (dtype == 1) return Q38_NATIVE_BF16_TYPE;
    if (dtype == 5) return 27;
    if (dtype == 3) return Q38_DTYPE_NVIDIA_FP8_E4M3;
    return dtype;
}

static bool add_expert_descriptors(q38_nvfp4_runtime_model *runtime,
                                   uint32_t *cursor, uint32_t layer,
                                   const q38_nvfp4_region_view regions[
                                       Q38_NVFP4_REGION_COUNT]) {
    const uint64_t gate_shape[4] = {512, 1280, 2560, 0};
    const uint64_t down_shape[4] = {512, 640, 2560, 0};
    const uint64_t gate_bytes = UINT64_C(512) * 640u * 1280u;
    const uint64_t down_bytes = UINT64_C(512) * 2560u * 320u;
    const uint64_t gate_stride = gate_bytes;
    const uint64_t down_stride = down_bytes;
    char name[256];
    const uint8_t *gate_base = (const uint8_t *)regions[
        Q38_NVFP4_REGION_WEIGHT].data;
    const uint8_t *down_base = gate_base;
    const uint64_t gate_layer_offset =
        (uint64_t)layer * gate_stride * 3u;
    const uint64_t down_layer_offset =
        (uint64_t)layer * down_stride * 3u;
    const uint8_t *gate_ptr = gate_base + gate_layer_offset;
    const uint8_t *down_ptr = down_base + down_layer_offset + down_stride * 2u;
    int written = snprintf(
        name, sizeof(name),
        "model.language_model.layers.%u.mlp.experts.gate_up_proj.weight",
        layer);
    if (written < 0 || (size_t)written >= sizeof(name)) return false;
    const char *owned = NULL;
    if (!own_name(runtime, name, (size_t)written, &owned) ||
        !add_tensor(runtime, cursor, owned, (size_t)written, 3, gate_shape,
                    Q38_QUANT_NVIDIA_NVFP4, gate_ptr, gate_bytes))
        return false;
    written = snprintf(
        name, sizeof(name),
        "model.language_model.layers.%u.mlp.experts.down_proj.weight",
        layer);
    if (written < 0 || (size_t)written >= sizeof(name)) return false;
    if (!own_name(runtime, name, (size_t)written, &owned) ||
        !add_tensor(runtime, cursor, owned, (size_t)written, 3, down_shape,
                    Q38_QUANT_NVIDIA_NVFP4, down_ptr, down_bytes))
        return false;
    return true;
}

bool q38_nvfp4_runtime_model_open(
    const char *pack_path, const char *source_root,
    q38_nvfp4_runtime_model **out, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!pack_path || !out)
        return fail(error, error_len, "invalid NVFP4 runtime model arguments");
    *out = NULL;
    q38_nvfp4_runtime_model *runtime = calloc(1, sizeof(*runtime));
    if (!runtime) return fail(error, error_len, "NVFP4 runtime allocation failed");
    if (!q38_nvfp4_pack_open(pack_path, source_root ? source_root : ".",
                             &runtime->pack, error, error_len))
        goto fail_open;
    if (q38_nvfp4_pack_is_source_backed(runtime->pack))
        return fail(error, error_len,
                    "production NVFP4 runtime requires a materialized pack");

    uint32_t bf16_count = 0, aux_count = 0, ple_count = 0;
    if (!q38_nvfp4_pack_get_bf16_count(runtime->pack, &bf16_count) ||
        !q38_nvfp4_pack_get_aux_count(runtime->pack, &aux_count) ||
        !q38_nvfp4_pack_get_ple_count(runtime->pack, &ple_count) ||
        ple_count < Q38_PLE_SHARD_COUNT) {
        fail(error, error_len, "native NVFP4 pack descriptor counts are invalid");
        goto fail_open;
    }
    uint32_t runtime_aux_count = 0;
    for (uint32_t i = 0; i < aux_count; ++i) {
        q38_nvfp4_view view;
        const char *name = NULL;
        uint32_t class_id = 0, ndim = 0;
        uint64_t shape[4] = {};
        if (!q38_nvfp4_pack_get_aux_view(
                runtime->pack, i, &view, &name, &class_id, &ndim, shape,
                error, error_len))
            goto fail_open;
        if (class_id == 3) runtime_aux_count++;
    }
    const uint32_t tensor_count =
        bf16_count + runtime_aux_count + Q38_PLE_SHARD_COUNT +
        Q38_NVFP4_LAYER_COUNT * 2u;
    runtime->model = calloc(1, sizeof(*runtime->model));
    if (!runtime->model || !(runtime->model->tensors =
                                 calloc(tensor_count,
                                        sizeof(*runtime->model->tensors))))
        goto fail_open;
    runtime->model->fd = -1;
    runtime->model->map = (const uint8_t *)(uintptr_t)1;
    runtime->model->n_tensors = tensor_count;
    runtime->model->native_nvfp4 = true;

    q38_nvfp4_region_view regions[Q38_NVFP4_REGION_COUNT];
    for (uint32_t region = 0; region < Q38_NVFP4_REGION_COUNT; ++region)
        if (!q38_nvfp4_pack_get_region_view(
                runtime->pack, region, &regions[region], error, error_len))
            goto fail_open;

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < bf16_count; ++i) {
        q38_nvfp4_view view;
        const char *name = NULL;
        uint32_t name_len = 0;
        uint32_t ndim = 0;
        uint64_t shape[4] = {};
        if (!q38_nvfp4_pack_get_bf16_view(
                runtime->pack, i, &view, &name, &ndim, shape,
                error, error_len) ||
            !q38_nvfp4_pack_get_bf16_name_length(
                runtime->pack, i, &name_len) ||
            !add_tensor(runtime, &cursor, name, name_len, ndim, shape,
                        native_type(view.dtype), view.data, view.bytes)) {
            if (!error[0]) snprintf(error, error_len,
                                    "cannot add native BF16 descriptor %u", i);
            goto fail_open;
        }
    }
    for (uint32_t i = 0; i < aux_count; ++i) {
        q38_nvfp4_view view;
        const char *name = NULL;
        uint32_t name_len = 0;
        uint32_t class_id = 0, ndim = 0;
        uint64_t shape[4] = {};
        if (!q38_nvfp4_pack_get_aux_view(
                runtime->pack, i, &view, &name, &class_id, &ndim,
                shape, error, error_len))
            goto fail_open;
        if (class_id != 3) continue;
        if (!q38_nvfp4_pack_get_aux_name_length(
                runtime->pack, i, &name_len) ||
            !add_tensor(runtime, &cursor, name, name_len, ndim, shape,
                        native_type(view.dtype), view.data, view.bytes)) {
            if (!error[0]) snprintf(error, error_len,
                                    "invalid native auxiliary descriptor %u (class=%u)",
                                    i, class_id);
            goto fail_open;
        }
    }
    for (uint32_t i = 0; i < ple_count; ++i) {
        q38_nvfp4_view view;
        const char *name = NULL;
        uint32_t shard_id = UINT32_MAX, ndim = 0;
        uint64_t shape[4] = {};
        if (!q38_nvfp4_pack_get_ple_tensor_view(
                runtime->pack, i, &view, &name, &shard_id, &ndim,
                shape, error, error_len))
            goto fail_open;
        if (shard_id == UINT32_MAX) {
            if (!name || !strstr(name, "ngram_embedding.weight_scale"))
                goto fail_open;
            runtime->owned_scale = calloc(1, sizeof(*runtime->owned_scale));
            if (!runtime->owned_scale) goto fail_open;
            runtime->owned_scale->name.ptr = name;
            runtime->owned_scale->name.len = strlen(name);
            runtime->owned_scale->ndim = ndim;
            memcpy(runtime->owned_scale->dim, shape,
                   4u * sizeof(shape[0]));
            runtime->owned_scale->type = native_type(view.dtype);
            runtime->owned_scale->data = view.data;
            runtime->owned_scale->bytes = view.bytes;
            continue;
        }
        if (shard_id >= Q38_PLE_SHARD_COUNT || !view.data) {
            if (!error[0]) snprintf(error, error_len,
                                    "invalid native PLE descriptor %u", i);
            goto fail_open;
        }
        char name_buf[256];
        int written = snprintf(
            name_buf, sizeof(name_buf),
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_%u.weight", shard_id);
        const char *owned = NULL;
        if (written < 0 || (size_t)written >= sizeof(name_buf) ||
            !own_name(runtime, name_buf, (size_t)written, &owned) ||
            !add_tensor(runtime, &cursor, owned, (size_t)written, ndim, shape,
                        native_type(view.dtype), view.data, view.bytes)) {
            if (!error[0]) snprintf(error, error_len,
                                    "cannot add native PLE descriptor %u", i);
            goto fail_open;
        }
    }
    for (uint32_t layer = 0; layer < Q38_NVFP4_LAYER_COUNT; ++layer)
        if (!add_expert_descriptors(runtime, &cursor, layer, regions)) {
            if (!error[0]) snprintf(error, error_len,
                                    "cannot add native expert descriptors for layer %u",
                                    layer);
            goto fail_open;
        }
    if (cursor != tensor_count || !runtime->owned_scale) {
        if (!error[0]) snprintf(error, error_len,
                                "native descriptor count mismatch (%u/%u)",
                                cursor, tensor_count);
        goto fail_open;
    }
    *out = runtime;
    return true;

fail_open:
    if (error && error_len && !error[0])
        snprintf(error, error_len, "cannot build native NVFP4 runtime model");
    q38_nvfp4_runtime_model_close(runtime);
    return false;
}

void q38_nvfp4_runtime_model_close(q38_nvfp4_runtime_model *runtime) {
    if (!runtime) return;
    if (runtime->model) {
        free(runtime->model->tensors);
        free(runtime->model);
    }
    free(runtime->owned_scale);
    for (size_t i = 0; i < runtime->owned_name_count; ++i)
        free(runtime->owned_names[i]);
    free(runtime->owned_names);
    q38_nvfp4_pack_close(runtime->pack);
    free(runtime);
}

q38_gguf *q38_nvfp4_runtime_model_gguf(
    q38_nvfp4_runtime_model *runtime) {
    return runtime ? runtime->model : NULL;
}

const q38_nvfp4_pack *q38_nvfp4_runtime_model_pack(
    const q38_nvfp4_runtime_model *runtime) {
    return runtime ? runtime->pack : NULL;
}

const q38_tensor *q38_nvfp4_runtime_model_ple_scale(
    const q38_nvfp4_runtime_model *runtime) {
    return runtime ? runtime->owned_scale : NULL;
}
