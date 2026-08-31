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

static q38_tensor *find(const q38_gguf *model, const char *name) {
    q38_str needle = {name, strlen(name)};
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        q38_tensor *tensor = &model->tensors[i];
        if (tensor->name.len == needle.len &&
            memcmp(tensor->name.ptr, needle.ptr, needle.len) == 0) {
            return tensor;
        }
    }
    return NULL;
}

static bool bind_required(const q38_gguf *model, const char *name,
                          const uint64_t *shape, uint32_t ndim,
                          q38_tensor **out, char *error, size_t error_len) {
    q38_tensor *tensor = find(model, name);
    if (!tensor) {
        if (error && error_len > 0) {
            snprintf(error, error_len, "missing required tensor: %s", name);
        }
        return false;
    }
    if (tensor->type != 30 || !shape_is(tensor, shape, ndim)) {
        if (error && error_len > 0) {
            snprintf(error, error_len, "shape/type mismatch: %s", name);
        }
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

bool q38_weights_bind_subset(const q38_gguf *model, uint32_t max_layer,
                             q38_weights *out, char *error, size_t error_len) {
    static const uint64_t embedding_shape[] = {248320, 2560};
    static const uint64_t routed_down_shape[] = {512, 2560, 640};
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
    if (!bind_required(model, "model.language_model.embed_tokens.weight",
                       embedding_shape, 2, &out->token_embd,
                       error, error_len) ||
        !bind_required(model, "lm_head.weight", embedding_shape, 2,
                       &out->output, error, error_len)) {
        return false;
    }

    uint32_t expected = 2;
    for (uint32_t layer = 0; layer <= max_layer; layer++) {
        char name[160];
        q38_layer_weights *bound = &out->layer[layer];
        bound->kind = q38_model_config_default()->layer_types[layer];
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.gate.weight", layer);
        if (!bind_required(model, name, router_shape, 2, &bound->router,
                           error, error_len)) return false;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.experts.down_proj", layer);
        if (!q38_expert_store_init_uniform(&bound->experts, 30)) return false;
        if (!bind_required(model, name, routed_down_shape, 3,
                           &bound->experts.bank[0].down,
                           error, error_len)) return false;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.experts.gate_up_proj", layer);
        if (!bind_required(model, name, routed_gate_up_shape, 3,
                           &bound->experts.bank[0].gate_up,
                           error, error_len)) return false;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.shared_expert.gate_proj.weight",
                 layer);
        if (!bind_required(model, name, shared_shape, 2,
                           &bound->shared_gate_proj, error, error_len)) return false;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.shared_expert.up_proj.weight",
                 layer);
        if (!bind_required(model, name, shared_shape, 2,
                           &bound->shared_up_proj, error, error_len)) return false;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.shared_expert.down_proj.weight",
                 layer);
        if (!bind_required(model, name, shared_down_shape, 2,
                           &bound->shared_down_proj, error, error_len)) return false;
        snprintf(name, sizeof(name),
                 "model.language_model.layers.%u.mlp.shared_expert_gate.weight",
                 layer);
        if (!bind_required(model, name, shared_gate_shape, 2,
                           &bound->shared_expert_gate, error, error_len)) return false;
        expected += 7;
    }
    out->bound_layers = max_layer + 1;
    out->bound_tensor_count = expected;
    return true;
}
