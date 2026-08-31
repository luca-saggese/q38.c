#include "q38_weights.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
}

static bool shape_is(const q38_tensor *tensor, const uint64_t *shape,
                     uint32_t ndim) {
    if (!tensor || tensor->ndim != ndim) return false;
    for (uint32_t i = 0; i < ndim; i++) {
        if (tensor->dim[i] != shape[i]) return false;
    }
    return true;
}

static bool name_is(const q38_tensor *tensor, const char *name) {
    size_t n = strlen(name);
    return tensor->name.len == n && memcmp(tensor->name.ptr, name, n) == 0;
}

static bool name_has(const q38_tensor *tensor, const char *part) {
    size_t n = strlen(part);
    if (n > tensor->name.len) return false;
    for (uint64_t i = 0; i + n <= tensor->name.len; i++) {
        if (memcmp(tensor->name.ptr + i, part, n) == 0) return true;
    }
    return false;
}

static bool layer_number(const q38_tensor *tensor, unsigned *layer) {
    const char marker[] = ".layers.";
    const size_t marker_len = sizeof(marker) - 1;
    for (uint64_t i = 0; i + marker_len < tensor->name.len; i++) {
        if (memcmp(tensor->name.ptr + i, marker, marker_len) != 0) continue;
        uint64_t p = i + marker_len;
        if (p == tensor->name.len || tensor->name.ptr[p] < '0' ||
            tensor->name.ptr[p] > '9') return false;
        unsigned value = 0;
        while (p < tensor->name.len && tensor->name.ptr[p] >= '0' &&
               tensor->name.ptr[p] <= '9') {
            unsigned digit = (unsigned)(tensor->name.ptr[p] - '0');
            if (value > 1000U) return false;
            value = value * 10U + digit;
            p++;
        }
        *layer = value;
        return true;
    }
    return false;
}

static q38_tensor *find(const q38_gguf *model, const char *name) {
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        if (name_is(&model->tensors[i], name)) return &model->tensors[i];
    }
    return NULL;
}

static bool bind_exact(const q38_gguf *model, const char *name,
                       const uint64_t *shape, uint32_t ndim, uint32_t type,
                       q38_tensor **out, char *error, size_t error_len) {
    q38_tensor *tensor = find(model, name);
    if (!tensor) {
        if (error && error_len > 0)
            snprintf(error, error_len, "missing required tensor: %s", name);
        return false;
    }
    if (tensor->type != type || !shape_is(tensor, shape, ndim)) {
        if (error && error_len > 0)
            snprintf(error, error_len, "shape/type mismatch: %s", name);
        return false;
    }
    *out = tensor;
    return true;
}

bool q38_expert_store_init_uniform(q38_layer_expert_store *store,
                                   uint32_t qtype) {
    if (!store) return false;
    memset(store, 0, sizeof(*store));
    store->bank_count = 1;
    store->bank[0].qtype = qtype;
    store->bank[0].expert_count = Q38_MODEL_EXPERTS;
    for (uint32_t expert = 0; expert < Q38_MODEL_EXPERTS; expert++) {
        store->loc[expert].bank_id = 0;
        store->loc[expert].local_index = (uint16_t)expert;
    }
    return true;
}

bool q38_expert_store_init_mixed(q38_layer_expert_store *store,
                                 const uint32_t qtypes[Q38_MODEL_EXPERTS]) {
    if (!store || !qtypes) return false;
    memset(store, 0, sizeof(*store));
    for (uint32_t expert = 0; expert < Q38_MODEL_EXPERTS; expert++) {
        uint32_t bank_id = 0;
        while (bank_id < store->bank_count &&
               store->bank[bank_id].qtype != qtypes[expert]) {
            bank_id++;
        }
        if (bank_id == store->bank_count) {
            if (store->bank_count == Q38_MAX_BANKS_PER_LAYER) return false;
            store->bank[bank_id].qtype = qtypes[expert];
            store->bank_count++;
        }
        q38_expert_bank *bank = &store->bank[bank_id];
        store->loc[expert].bank_id = (uint8_t)bank_id;
        store->loc[expert].local_index = (uint16_t)bank->expert_count;
        bank->expert_count++;
    }
    return true;
}

static bool validate_metadata(const q38_gguf *model, uint32_t max_layer,
                              bool *quantized, char *error, size_t error_len) {
    q38_str arch;
    uint32_t metadata_layer;
    bool runtime_only, excluded_vision, excluded_mtp;
    if (!q38_gguf_get_string(model, "general.architecture", &arch) ||
        arch.len != strlen("qwen4_exp") ||
        memcmp(arch.ptr, "qwen4_exp", arch.len) != 0 ||
        !q38_gguf_get_bool(model, "q38.runtime_only", &runtime_only) ||
        !runtime_only ||
        !q38_gguf_get_bool(model, "q38.excluded_vision", &excluded_vision) ||
        !excluded_vision ||
        !q38_gguf_get_bool(model, "q38.excluded_mtp", &excluded_mtp) ||
        !excluded_mtp ||
        !q38_gguf_get_u32(model, "q38.max_layer", &metadata_layer) ||
        metadata_layer != max_layer) {
        set_error(error, error_len, "runtime-only architecture metadata mismatch");
        return false;
    }
    *quantized = false;
    (void)q38_gguf_get_bool(model, "q38.quantized", quantized);
    return true;
}

static uint32_t expected_tensor_count(uint32_t max_layer) {
    uint32_t layers = max_layer + 1;
    uint32_t full = 0;
    for (uint32_t i = 0; i < layers; i++) {
        full += (i % 4 == 3) ? 9 : 9; /* one core family per layer */
    }
    uint32_t ple = max_layer >= 1 ? Q38_MAX_PLE_TENSORS - 23 : 0;
    /* 8 hyper-connection, 7 MoE/router, and one core family are represented
     * by the inventory; layer 2 additionally owns the 137 PLE tensors. */
    return 5 + full + layers * 15 + ple;
}

bool q38_weights_bind_subset(const q38_gguf *model, uint32_t max_layer,
                             q38_weights *out, char *error, size_t error_len) {
    static const uint64_t embedding_shape[] = {248320, 2560};
    static const uint64_t routed_down_source[] = {512, 2560, 640};
    static const uint64_t routed_down_transposed[] = {512, 640, 2560};
    static const uint64_t routed_gate_up_shape[] = {512, 1280, 2560};
    static const uint64_t router_shape[] = {512, 2560};
    static const uint64_t shared_shape[] = {640, 2560};
    static const uint64_t shared_down_shape[] = {2560, 640};
    static const uint64_t shared_gate_shape[] = {1, 2560};

    if (error && error_len > 0) error[0] = '\0';
    if (!model || !out || max_layer >= Q38_MODEL_LAYERS) {
        set_error(error, error_len, "invalid binder arguments");
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (!q38_model_config_validate(q38_model_config_default(), error, error_len) ||
        !validate_metadata(model, max_layer, &out->quantized, error, error_len)) {
        return false;
    }
    if (!bind_exact(model, "model.language_model.embed_tokens.weight",
                    embedding_shape, 2, 30, &out->token_embd,
                    error, error_len) ||
        !bind_exact(model, "lm_head.weight", embedding_shape, 2, 30,
                    &out->output, error, error_len)) {
        return false;
    }

    uint32_t bound = 2;
    const q38_model_config *config = q38_model_config_default();
    for (uint32_t layer = 0; layer <= max_layer; layer++)
        out->layer[layer].kind = config->layer_types[layer];
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        q38_tensor *tensor = &model->tensors[i];
        if (tensor == out->token_embd || tensor == out->output) continue;
        if (name_has(tensor, "model.language_model.hyper_connection_mixer.")) {
            if (out->global_tensor_count >= 8) {
                set_error(error, error_len, "global tensor table capacity exceeded");
                return false;
            }
            if (tensor->type != 30) {
                set_error(error, error_len, "global hyper-connection type mismatch");
                return false;
            }
            out->global_tensor[out->global_tensor_count++] = tensor;
            bound++;
            continue;
        }
        if (!name_has(tensor, "model.language_model.layers.")) {
            set_error(error, error_len, "unexpected non-text tensor in runtime artifact");
            return false;
        }
        unsigned layer = 0;
        if (!layer_number(tensor, &layer) ||
            layer > max_layer || layer >= Q38_MODEL_LAYERS) {
            set_error(error, error_len, "tensor layer is outside bound subset");
            return false;
        }
        q38_layer_weights *dst = &out->layer[layer];
        if (dst->tensor_count >= Q38_MAX_LAYER_TENSORS) {
            set_error(error, error_len, "layer tensor table capacity exceeded");
            return false;
        }
        if (tensor->type != 30 && tensor->type != 27 &&
            tensor->type != 8 && tensor->type != 10 && tensor->type != 16) {
            set_error(error, error_len, "unsupported tensor type in runtime artifact");
            return false;
        }
        if (name_has(tensor, ".mlp.experts.gate_up_proj")) {
            if (!shape_is(tensor, routed_gate_up_shape, 3) ||
                (out->quantized ? tensor->type != 10 : tensor->type != 30))
                set_error(error, error_len, "routed gate/up shape/type mismatch");
            else {
                if (!dst->experts.bank_count)
                    q38_expert_store_init_uniform(&dst->experts, tensor->type);
                else if (dst->experts.bank[0].qtype != tensor->type)
                    set_error(error, error_len, "routed expert quant type mismatch");
                dst->experts.bank[0].gate_up = tensor;
            }
        } else if (name_has(tensor, ".mlp.experts.down_proj")) {
            const uint64_t *shape = out->quantized
                ? routed_down_transposed : routed_down_source;
            if (!shape_is(tensor, shape, 3) ||
                (out->quantized ? tensor->type != 10 : tensor->type != 30))
                set_error(error, error_len, "routed down shape/type mismatch");
                else {
                    if (!dst->experts.bank_count)
                        q38_expert_store_init_uniform(&dst->experts, tensor->type);
                    dst->experts.bank[0].down = tensor;
                }
        } else if (name_has(tensor, ".mlp.gate.weight")) {
            if (!shape_is(tensor, router_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "router shape/type mismatch");
            else dst->router = tensor;
        } else if (name_has(tensor, ".shared_expert.gate_proj.weight") ||
                   name_has(tensor, ".shared_expert.up_proj.weight")) {
            if (!shape_is(tensor, shared_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "shared projection shape/type mismatch");
            else if (name_has(tensor, ".gate_proj."))
                dst->shared_gate_proj = tensor;
            else
                dst->shared_up_proj = tensor;
        } else if (name_has(tensor, ".shared_expert.down_proj.weight")) {
            if (!shape_is(tensor, shared_down_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "shared down shape/type mismatch");
            else dst->shared_down_proj = tensor;
        } else if (name_has(tensor, ".shared_expert_gate.weight")) {
            if (!shape_is(tensor, shared_gate_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "shared gate shape/type mismatch");
            else dst->shared_expert_gate = tensor;
        } else if (name_has(tensor, ".ple.")) {
            if (dst->ple_tensor_count >= Q38_MAX_PLE_TENSORS)
                set_error(error, error_len, "PLE tensor table capacity exceeded");
            else {
                bool source_integer = name_has(tensor, "layer_multipliers") ||
                    name_has(tensor, "ngram_heads_offsets") ||
                    name_has(tensor, "ngram_heads_vocab_sizes");
                bool row_quantized = out->quantized && tensor->ndim >= 2 &&
                    tensor->dim[tensor->ndim - 1] % 32 == 0;
                uint32_t expected_type = source_integer ? 27 :
                    (row_quantized ? 8 : 30);
                if (tensor->type != expected_type)
                    set_error(error, error_len, "PLE shape/type mismatch");
                else
                    dst->ple_tensor[dst->ple_tensor_count++] = tensor;
            }
        } else if (name_has(tensor, ".linear_attn.") ||
                   name_has(tensor, ".self_attn.") ||
                   name_has(tensor, "hyper_connection.")) {
            if (tensor->type != 30)
                set_error(error, error_len, "core projection type mismatch");
        } else {
            set_error(error, error_len, "unknown layer tensor role");
        }
        if (error && error[0]) return false;
        dst->tensor[dst->tensor_count++] = tensor;
        bound++;
    }

    uint32_t expected = expected_tensor_count(max_layer);
    /* The inventory is authoritative for role cardinality; this check catches
     * dropped tensors without baking source-shard ordering into the binder. */
    if (bound != expected) {
        if (error && error_len > 0)
            snprintf(error, error_len, "runtime tensor count mismatch: expected %u, got %u",
                     expected, bound);
        return false;
    }
    out->bound_layers = max_layer + 1;
    out->bound_tensor_count = bound;
    return true;
}
