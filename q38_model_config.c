#include "q38_model_config.h"

#include <stdio.h>

static const q38_model_config frozen_config = {
    .hidden_size = 2560,
    .num_hidden_layers = 48,
    .num_experts = 512,
    .num_experts_per_tok = 10,
    .moe_intermediate_size = 640,
    .shared_expert_intermediate_size = 640,
    .hc_count = 4,
    .hc_lowrank = 320,
    .linear_num_key_heads = 16,
    .linear_num_value_heads = 48,
    .linear_key_head_dim = 128,
    .linear_value_head_dim = 128,
    .linear_conv_kernel_dim = 4,
    .num_attention_heads = 24,
    .num_key_value_heads = 2,
    .head_dim = 256,
    .indexer_n_heads = 4,
    .indexer_kv_heads = 1,
    .indexer_head_dim = 128,
    .indexer_compress_ratio = 4,
    .indexer_budget = 2048,
    .ngram_size = 3,
    .ngram_vocab_size_base = 20000000,
    .heads_per_ngram = 8,
    .split_ngram_parts = 128,
    .ple_layer = 2,
    .vocab_size = 248320,
    .max_position_embeddings = 262144,
    .layer_types = {
        [3] = Q38_LAYER_FULL_ATTENTION,
        [7] = Q38_LAYER_FULL_ATTENTION,
        [11] = Q38_LAYER_FULL_ATTENTION,
        [15] = Q38_LAYER_FULL_ATTENTION,
        [19] = Q38_LAYER_FULL_ATTENTION,
        [23] = Q38_LAYER_FULL_ATTENTION,
        [27] = Q38_LAYER_FULL_ATTENTION,
        [31] = Q38_LAYER_FULL_ATTENTION,
        [35] = Q38_LAYER_FULL_ATTENTION,
        [39] = Q38_LAYER_FULL_ATTENTION,
        [43] = Q38_LAYER_FULL_ATTENTION,
        [47] = Q38_LAYER_FULL_ATTENTION,
    },
};

const q38_model_config *q38_model_config_default(void) {
    return &frozen_config;
}

static bool expect_u32(const char *name, uint32_t actual, uint32_t expected,
                       char *error, size_t error_len) {
    if (actual == expected) return true;
    if (error && error_len > 0) {
        snprintf(error, error_len, "%s: expected %u, got %u",
                 name, expected, actual);
    }
    return false;
}

bool q38_model_config_validate(const q38_model_config *config,
                               char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!config) {
        if (error && error_len > 0) {
            snprintf(error, error_len, "configuration is null");
        }
        return false;
    }
#define CHECK_FIELD(field, value) \
    if (!expect_u32(#field, config->field, (value), error, error_len)) return false
    CHECK_FIELD(hidden_size, 2560);
    CHECK_FIELD(num_hidden_layers, 48);
    CHECK_FIELD(num_experts, 512);
    CHECK_FIELD(num_experts_per_tok, 10);
    CHECK_FIELD(moe_intermediate_size, 640);
    CHECK_FIELD(shared_expert_intermediate_size, 640);
    CHECK_FIELD(hc_count, 4);
    CHECK_FIELD(hc_lowrank, 320);
    CHECK_FIELD(linear_num_key_heads, 16);
    CHECK_FIELD(linear_num_value_heads, 48);
    CHECK_FIELD(linear_key_head_dim, 128);
    CHECK_FIELD(linear_value_head_dim, 128);
    CHECK_FIELD(linear_conv_kernel_dim, 4);
    CHECK_FIELD(num_attention_heads, 24);
    CHECK_FIELD(num_key_value_heads, 2);
    CHECK_FIELD(head_dim, 256);
    CHECK_FIELD(indexer_n_heads, 4);
    CHECK_FIELD(indexer_kv_heads, 1);
    CHECK_FIELD(indexer_head_dim, 128);
    CHECK_FIELD(indexer_compress_ratio, 4);
    CHECK_FIELD(indexer_budget, 2048);
    CHECK_FIELD(ngram_size, 3);
    CHECK_FIELD(ngram_vocab_size_base, 20000000);
    CHECK_FIELD(heads_per_ngram, 8);
    CHECK_FIELD(split_ngram_parts, 128);
    CHECK_FIELD(ple_layer, 2);
    CHECK_FIELD(vocab_size, 248320);
    CHECK_FIELD(max_position_embeddings, 262144);
#undef CHECK_FIELD
    for (uint32_t i = 0; i < Q38_MODEL_LAYERS; i++) {
        q38_layer_kind expected =
            (i % 4 == 3) ? Q38_LAYER_FULL_ATTENTION
                         : Q38_LAYER_LINEAR_ATTENTION;
        if (config->layer_types[i] != expected) {
            if (error && error_len > 0) {
                snprintf(error, error_len,
                         "layer_types[%u]: expected %s", i,
                         expected == Q38_LAYER_FULL_ATTENTION
                             ? "full_attention" : "linear_attention");
            }
            return false;
        }
    }
    return true;
}
