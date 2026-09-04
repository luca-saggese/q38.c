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

static bool validate_core_tensor(const q38_tensor *tensor, char *error,
                                 size_t error_len) {
    static const uint64_t a_log[] = {48};
    static const uint64_t conv[] = {10240, 1, 4};
    static const uint64_t proj_48[] = {48, 2560};
    static const uint64_t qkv[] = {10240, 2560};
    static const uint64_t z[] = {6144, 2560};
    static const uint64_t norm[] = {128};
    static const uint64_t out[] = {2560, 6144};
    static const uint64_t index_qk[] = {640, 2560};
    static const uint64_t norm_256[] = {256};
    static const uint64_t proj_512[] = {512, 2560};
    static const uint64_t q_proj[] = {12288, 2560};
    const uint64_t *shape = NULL;
    uint32_t ndim = 0;
    if (name_has(tensor, ".linear_attn.A_log") ||
        name_has(tensor, ".linear_attn.dt_bias")) {
        shape = a_log; ndim = 1;
    } else if (name_has(tensor, ".linear_attn.conv1d.weight")) {
        shape = conv; ndim = 3;
    } else if (name_has(tensor, ".linear_attn.in_proj_a.weight") ||
               name_has(tensor, ".linear_attn.in_proj_b.weight")) {
        shape = proj_48; ndim = 2;
    } else if (name_has(tensor, ".linear_attn.in_proj_qkv.weight")) {
        shape = qkv; ndim = 2;
    } else if (name_has(tensor, ".linear_attn.in_proj_z.weight")) {
        shape = z; ndim = 2;
    } else if (name_has(tensor, ".linear_attn.norm.weight")) {
        shape = norm; ndim = 1;
    } else if (name_has(tensor, ".linear_attn.out_proj.weight")) {
        shape = out; ndim = 2;
    } else if (name_has(tensor, ".self_attn.indexer.index_qk_proj.weight")) {
        shape = index_qk; ndim = 2;
    } else if (name_has(tensor, ".self_attn.indexer.k_layernorm.weight") ||
               name_has(tensor, ".self_attn.indexer.q_layernorm.weight")) {
        shape = norm; ndim = 1;
    } else if (name_has(tensor, ".self_attn.k_norm.weight") ||
               name_has(tensor, ".self_attn.q_norm.weight")) {
        shape = norm_256; ndim = 1;
    } else if (name_has(tensor, ".self_attn.k_proj.weight") ||
               name_has(tensor, ".self_attn.v_proj.weight")) {
        shape = proj_512; ndim = 2;
    } else if (name_has(tensor, ".self_attn.o_proj.weight")) {
        shape = out; ndim = 2;
    } else if (name_has(tensor, ".self_attn.q_proj.weight")) {
        shape = q_proj; ndim = 2;
    } else {
        if (error && error_len)
            snprintf(error, error_len, "unknown core tensor role: %.*s",
                     (int)tensor->name.len, tensor->name.ptr);
        return false;
    }
    if (tensor->type != 30 || !shape_is(tensor, shape, ndim)) {
        if (error && error_len)
            snprintf(error, error_len, "core shape/type mismatch: %.*s",
                     (int)tensor->name.len, tensor->name.ptr);
        return false;
    }
    return true;
}

static bool validate_ple_tensor_set(const q38_layer_weights *layer,
                                    uint32_t layer_number, char *error,
                                    size_t error_len) {
    uint32_t shards = 0, key = 0, value = 0, norm_key = 0;
    uint32_t norm_query = 0, norm_conv = 0, conv = 0, multipliers = 0;
    uint32_t offsets = 0, sizes = 0;
    static const uint64_t key_shape[] = {10240, 2560};
    static const uint64_t value_shape[] = {2560, 2560};
    static const uint64_t norm_shape[] = {10240};
    static const uint64_t conv_shape[] = {10240, 1, 4};
    static const uint64_t vector3[] = {3};
    static const uint64_t vector16[] = {16};
    for (uint32_t i = 0; i < layer->ple_tensor_count; ++i) {
        const q38_tensor *t = layer->ple_tensor[i];
        if (name_has(t, ".ngram_embedding.shard_")) shards++;
        else if (name_has(t, ".ple.key_proj.weight")) {
            key++; if (!shape_is(t, key_shape, 2)) goto mismatch;
        } else if (name_has(t, ".ple.value_proj.weight")) {
            value++; if (!shape_is(t, value_shape, 2)) goto mismatch;
        } else if (name_has(t, ".ple.norm_key.weight")) {
            norm_key++; if (!shape_is(t, norm_shape, 1)) goto mismatch;
        } else if (name_has(t, ".ple.norm_query.weight")) {
            norm_query++; if (!shape_is(t, norm_shape, 1)) goto mismatch;
        } else if (name_has(t, ".ple.norm_conv.weight")) {
            norm_conv++; if (!shape_is(t, norm_shape, 1)) goto mismatch;
        } else if (name_has(t, ".ple.conv1d.weight")) {
            conv++; if (!shape_is(t, conv_shape, 3)) goto mismatch;
        } else if (name_has(t, "layer_multipliers")) {
            multipliers++; if (!shape_is(t, vector3, 1)) goto mismatch;
        } else if (name_has(t, "ngram_heads_offsets")) {
            offsets++; if (!shape_is(t, vector16, 1)) goto mismatch;
        } else if (name_has(t, "ngram_heads_vocab_sizes")) {
            sizes++; if (!shape_is(t, vector16, 1)) goto mismatch;
        } else goto mismatch;
    }
    if (shards != Q38_PLE_SHARD_COUNT || key != 1 || value != 1 ||
        norm_key != 1 || norm_query != 1 || norm_conv != 1 || conv != 1 ||
        multipliers != 1 || offsets != 1 || sizes != 1) goto mismatch;
    return true;
mismatch:
    if (error && error_len)
        snprintf(error, error_len, "layer %u PLE semantic tensor set mismatch",
                 layer_number);
    return false;
}

static bool validate_layer_complete(const q38_layer_weights *layer,
                                    uint32_t layer_number, char *error,
                                    size_t error_len) {
    uint32_t core = 0, hyper = 0, mlp = 0;
    for (uint32_t i = 0; i < layer->tensor_count; i++) {
        const q38_tensor *tensor = layer->tensor[i];
        if (name_has(tensor, ".linear_attn.") ||
            name_has(tensor, ".self_attn.")) core++;
        else if (name_has(tensor, ".attn_hyper_connection.") ||
                 name_has(tensor, ".mlp_hyper_connection.")) hyper++;
        else if (name_has(tensor, ".mlp.")) mlp++;
    }
    bool attn_gr = layer->attn_gr.block_inject_weight &&
        layer->attn_gr.hc_norm && layer->attn_gr.input_mix_weight_down &&
        layer->attn_gr.input_mix_weight_up;
    bool mlp_gr = layer->mlp_gr.block_inject_weight &&
        layer->mlp_gr.hc_norm && layer->mlp_gr.input_mix_weight_down &&
        layer->mlp_gr.input_mix_weight_up;
    if (core != 9 || hyper != 8 || mlp != 7 || !attn_gr || !mlp_gr ||
        !layer->router || !layer->shared_expert_gate ||
        !layer->shared_gate_proj || !layer->shared_up_proj ||
        !layer->shared_down_proj || layer->experts.bank_count != 1 ||
        !layer->experts.bank[0].gate_up || !layer->experts.bank[0].down) {
        if (error && error_len)
            snprintf(error, error_len,
            "layer %u required tensor set incomplete (core=%u hyper=%u mlp=%u router=%d gate=%d up=%d down=%d expert=%d)",
            layer_number, core, hyper, mlp, layer->router != NULL,
            layer->shared_expert_gate != NULL,
            layer->shared_gate_proj != NULL,
            layer->shared_up_proj != NULL,
            layer->experts.bank_count == 1 &&
                layer->experts.bank[0].gate_up &&
                layer->experts.bank[0].down);
        return false;
    }
    if (layer->kind == Q38_LAYER_LINEAR_ATTENTION) {
        if (layer->gdn.in_proj_qkv == NULL || layer->gdn.in_proj_z == NULL ||
            layer->gdn.in_proj_a == NULL || layer->gdn.in_proj_b == NULL ||
            layer->gdn.conv1d == NULL || layer->gdn.A_log == NULL ||
            layer->gdn.dt_bias == NULL || layer->gdn.norm == NULL ||
            layer->gdn.out_proj == NULL ||
            layer->qsa.q_proj != NULL || layer->qsa.k_proj != NULL) {
            if (error && error_len)
                snprintf(error, error_len,
                         "layer %u semantic GDN/QSA family mismatch",
                         layer_number);
            return false;
        }
    } else {
        if (layer->qsa.q_proj == NULL || layer->qsa.k_proj == NULL ||
            layer->qsa.v_proj == NULL || layer->qsa.o_proj == NULL ||
            layer->qsa.q_norm == NULL || layer->qsa.k_norm == NULL ||
            layer->qsa.index_qk_proj == NULL ||
            layer->qsa.index_q_norm == NULL ||
            layer->qsa.index_k_norm == NULL ||
            layer->gdn.in_proj_qkv != NULL) {
            if (error && error_len)
                snprintf(error, error_len,
                         "layer %u semantic QSA/GDN family mismatch",
                         layer_number);
            return false;
        }
    }
    if (layer->kind == Q38_LAYER_FULL_ATTENTION) {
        if (!q38_qsa_weights_validate(&layer->qsa, error, error_len))
            return false;
    }
    if (layer_number == 1 && !validate_ple_tensor_set(layer, layer_number,
                                                       error, error_len))
        return false;
    return true;
}

bool q38_weights_validate_bound(const q38_weights *weights, char *error,
                                size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!weights || weights->bound_layers != Q38_MODEL_LAYERS) {
        set_error(error, error_len,
                  "complete forward requires all 48 layers");
        return false;
    }
    if (!weights->token_embd || !weights->output) {
        set_error(error, error_len, "embedding/output tensor set is incomplete");
        return false;
    }
    if (weights->global_tensor_count != 3) {
        set_error(error, error_len, "global tensor set is incomplete");
        return false;
    }
    for (uint32_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer)
        if (!validate_layer_complete(&weights->layer[layer], layer, error,
                                     error_len))
            return false;
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
        metadata_layer < max_layer) {
        set_error(error, error_len, "runtime-only architecture metadata mismatch");
        return false;
    }
    *quantized = false;
    (void)q38_gguf_get_bool(model, "q38.quantized", quantized);
    return true;
}

static uint32_t expected_tensor_count(uint32_t max_layer) {
    uint32_t layers = max_layer + 1;
    const uint32_t common = 9 /* core */ + 8 /* GR */ + 7 /* MoE */;
    const uint32_t ple = max_layer >= 1
        ? Q38_PLE_SHARD_COUNT + Q38_PLE_AUX_TENSORS : 0;
    return 5 + layers * common + ple;
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
        if (!layer_number(tensor, &layer) || layer >= Q38_MODEL_LAYERS) {
            set_error(error, error_len, "tensor layer is outside bound subset");
            return false;
        }
        if (layer > max_layer) continue;
        q38_layer_weights *dst = &out->layer[layer];
        if (dst->tensor_count >= Q38_MAX_LAYER_TENSORS) {
            set_error(error, error_len, "layer tensor table capacity exceeded");
            return false;
        }
        if (tensor->type != 30 && tensor->type != 27 &&
            tensor->type != 8 && tensor->type != 10 && tensor->type != 12 &&
            tensor->type != 16) {
            set_error(error, error_len, "unsupported tensor type in runtime artifact");
            return false;
        }
        if (name_has(tensor, ".mlp.experts.gate_up_proj")) {
            if (!shape_is(tensor, routed_gate_up_shape, 3) ||
                (out->quantized ? (tensor->type != 10 && tensor->type != 12)
                                : tensor->type != 30))
                set_error(error, error_len, "routed gate/up shape/type mismatch");
            else {
                if (!dst->experts.bank_count)
                    q38_expert_store_init_uniform(&dst->experts, tensor->type);
                if (dst->experts.bank[0].qtype != tensor->type)
                    set_error(error, error_len, "routed expert quant type mismatch");
                else if (dst->experts.bank[0].gate_up)
                    set_error(error, error_len, "duplicate routed gate/up tensor");
                else
                    dst->experts.bank[0].gate_up = tensor;
            }
        } else if (name_has(tensor, ".mlp.experts.down_proj")) {
            const uint64_t *shape = out->quantized
                ? routed_down_transposed : routed_down_source;
            if (!shape_is(tensor, shape, 3) ||
                (out->quantized ? (tensor->type != 10 && tensor->type != 12)
                                : tensor->type != 30))
                set_error(error, error_len, "routed down shape/type mismatch");
            else {
                if (!dst->experts.bank_count)
                    q38_expert_store_init_uniform(&dst->experts, tensor->type);
                if (dst->experts.bank[0].qtype != tensor->type)
                    set_error(error, error_len, "routed expert quant type mismatch");
                else if (dst->experts.bank[0].down)
                    set_error(error, error_len, "duplicate routed down tensor");
                else
                    dst->experts.bank[0].down = tensor;
            }
        } else if (name_has(tensor, ".mlp.gate.weight")) {
            if (!shape_is(tensor, router_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "router shape/type mismatch");
            else if (dst->router)
                set_error(error, error_len, "duplicate router tensor");
            else dst->router = tensor;
        } else if (name_has(tensor, ".shared_expert.gate_proj.weight") ||
                   name_has(tensor, ".shared_expert.up_proj.weight")) {
            if (!shape_is(tensor, shared_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "shared projection shape/type mismatch");
            else if (name_has(tensor, ".gate_proj.")) {
                if (dst->shared_gate_proj)
                    set_error(error, error_len, "duplicate shared gate tensor");
                else dst->shared_gate_proj = tensor;
            } else {
                if (dst->shared_up_proj)
                    set_error(error, error_len, "duplicate shared up tensor");
                else dst->shared_up_proj = tensor;
            }
        } else if (name_has(tensor, ".shared_expert.down_proj.weight")) {
            if (!shape_is(tensor, shared_down_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "shared down shape/type mismatch");
            else if (dst->shared_down_proj)
                set_error(error, error_len, "duplicate shared down tensor");
            else dst->shared_down_proj = tensor;
        } else if (name_has(tensor, ".shared_expert_gate.weight")) {
            if (!shape_is(tensor, shared_gate_shape, 2) || tensor->type != 30)
                set_error(error, error_len, "shared gate shape/type mismatch");
            else if (dst->shared_expert_gate)
                set_error(error, error_len, "duplicate shared expert gate tensor");
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
            if (name_has(tensor, "hyper_connection.")) {
                static const uint64_t inject[] = {4, 10240};
                static const uint64_t hcnorm[] = {10240};
                static const uint64_t down[] = {320, 10240};
                static const uint64_t up[] = {10240, 320};
                q38_gr_weights *gr = name_has(tensor, ".attn_hyper_connection.")
                    ? &dst->attn_gr : name_has(tensor, ".mlp_hyper_connection.")
                    ? &dst->mlp_gr : NULL;
                const uint64_t *shape = NULL;
                q38_tensor **slot = NULL;
                uint32_t ndim = 2;
                if (!gr) {
                    set_error(error, error_len, "unknown GR tensor family");
                } else if (name_has(tensor, "block_inject_weight")) {
                    shape = inject;
                    slot = &gr->block_inject_weight;
                } else if (name_has(tensor, "hc_norm")) {
                    shape = hcnorm;
                    ndim = 1;
                    slot = &gr->hc_norm;
                } else if (name_has(tensor, "input_mix_weight_down")) {
                    shape = down;
                    slot = &gr->input_mix_weight_down;
                } else if (name_has(tensor, "input_mix_weight_up")) {
                    shape = up;
                    slot = &gr->input_mix_weight_up;
                } else {
                    set_error(error, error_len, "unknown GR tensor role");
                }
                if (!gr || !shape || !slot)
                    set_error(error, error_len, "unknown GR tensor family");
                else if (*slot)
                    set_error(error, error_len, "duplicate GR tensor");
                else if (tensor->type != 30 || !shape_is(tensor, shape, ndim))
                    set_error(error, error_len, "hyper-connection shape/type mismatch");
                else
                    *slot = tensor;
            } else {
                validate_core_tensor(tensor, error, error_len);
                if ((!error || !error[0]) &&
                    name_has(tensor, ".linear_attn.")) {
                    q38_tensor **slot = NULL;
                    if (name_has(tensor, ".in_proj_qkv.weight"))
                        slot = &dst->gdn.in_proj_qkv;
                    else if (name_has(tensor, ".in_proj_z.weight"))
                        slot = &dst->gdn.in_proj_z;
                    else if (name_has(tensor, ".in_proj_a.weight"))
                        slot = &dst->gdn.in_proj_a;
                    else if (name_has(tensor, ".in_proj_b.weight"))
                        slot = &dst->gdn.in_proj_b;
                    else if (name_has(tensor, ".conv1d.weight"))
                        slot = &dst->gdn.conv1d;
                    else if (name_has(tensor, ".A_log"))
                        slot = &dst->gdn.A_log;
                    else if (name_has(tensor, ".dt_bias"))
                        slot = &dst->gdn.dt_bias;
                    else if (name_has(tensor, ".norm.weight"))
                        slot = &dst->gdn.norm;
                    else if (name_has(tensor, ".out_proj.weight"))
                        slot = &dst->gdn.out_proj;
                    if (!slot)
                        set_error(error, error_len, "unknown GDN tensor role");
                    else if (*slot)
                        set_error(error, error_len, "duplicate GDN tensor");
                    else
                        *slot = tensor;
                } else if ((!error || !error[0]) &&
                           name_has(tensor, ".self_attn.")) {
                    q38_qsa_weights *qsa = &dst->qsa;
                    q38_tensor **slot = NULL;
                    if (name_has(tensor, ".q_proj.weight"))
                        slot = &qsa->q_proj;
                    else if (name_has(tensor, ".k_proj.weight"))
                        slot = &qsa->k_proj;
                    else if (name_has(tensor, ".v_proj.weight"))
                        slot = &qsa->v_proj;
                    else if (name_has(tensor, ".o_proj.weight"))
                        slot = &qsa->o_proj;
                    else if (name_has(tensor, ".q_norm.weight"))
                        slot = &qsa->q_norm;
                    else if (name_has(tensor, ".k_norm.weight"))
                        slot = &qsa->k_norm;
                    else if (name_has(tensor, ".indexer.index_qk_proj.weight"))
                        slot = &qsa->index_qk_proj;
                    else if (name_has(tensor, ".indexer.q_layernorm.weight"))
                        slot = &qsa->index_q_norm;
                    else if (name_has(tensor, ".indexer.k_layernorm.weight"))
                        slot = &qsa->index_k_norm;
                    if (!slot)
                        set_error(error, error_len, "unknown QSA tensor role");
                    else if (*slot)
                        set_error(error, error_len, "duplicate QSA tensor");
                    else
                        *slot = tensor;
                }
            }
        } else {
            set_error(error, error_len, "unknown layer tensor role");
        }
        if (error && error[0]) return false;
        dst->tensor[dst->tensor_count++] = tensor;
        bound++;
    }

    if (out->global_tensor_count != 3) {
        if (error && error_len > 0)
            snprintf(error, error_len, "global tensor set mismatch: expected 3, got %u",
                     out->global_tensor_count);
        return false;
    }
    for (uint32_t layer = 0; layer <= max_layer; layer++) {
        if (!validate_layer_complete(&out->layer[layer], layer, error, error_len))
            return false;
        if (out->layer[layer].kind == Q38_LAYER_LINEAR_ATTENTION) {
            const q38_gdn_weights *gdn = &out->layer[layer].gdn;
            if (!gdn->in_proj_qkv || !gdn->in_proj_z || !gdn->in_proj_a ||
                !gdn->in_proj_b || !gdn->conv1d || !gdn->A_log ||
                !gdn->dt_bias || !gdn->norm || !gdn->out_proj) {
                if (error && error_len)
                    snprintf(error, error_len,
                             "layer %u GDN tensor set incomplete", layer);
                return false;
            }
        }
    }
    if (max_layer >= 1 && out->layer[1].ple_tensor_count != 0) {
        static const char ple_prefix[] =
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_";
        if (!q38_ple_store_bind_gguf(
                model, ple_prefix, 128, 160, &out->layer[1].ple_store,
                error, error_len)) {
            return false;
        }
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
