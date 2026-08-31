#ifndef Q38_MODEL_CONFIG_H
#define Q38_MODEL_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_MODEL_LAYERS 48
#define Q38_MODEL_EXPERTS 512
#define Q38_MODEL_EXPERTS_PER_TOKEN 10

typedef enum {
    Q38_LAYER_LINEAR_ATTENTION = 0,
    Q38_LAYER_FULL_ATTENTION = 1,
} q38_layer_kind;

typedef struct {
    uint32_t hidden_size;
    uint32_t num_hidden_layers;
    uint32_t num_experts;
    uint32_t num_experts_per_tok;
    uint32_t moe_intermediate_size;
    uint32_t shared_expert_intermediate_size;
    uint32_t hc_count;
    uint32_t hc_lowrank;
    uint32_t linear_num_key_heads;
    uint32_t linear_num_value_heads;
    uint32_t linear_key_head_dim;
    uint32_t linear_value_head_dim;
    uint32_t linear_conv_kernel_dim;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t indexer_n_heads;
    uint32_t indexer_kv_heads;
    uint32_t indexer_head_dim;
    uint32_t indexer_compress_ratio;
    uint32_t indexer_budget;
    uint32_t ngram_size;
    uint32_t ngram_vocab_size_base;
    uint32_t heads_per_ngram;
    uint32_t split_ngram_parts;
    uint32_t ple_layer;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    q38_layer_kind layer_types[Q38_MODEL_LAYERS];
} q38_model_config;

const q38_model_config *q38_model_config_default(void);

/* Validate the frozen architecture, including all layer kinds. */
bool q38_model_config_validate(const q38_model_config *config,
                               char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
