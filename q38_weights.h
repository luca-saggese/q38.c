#ifndef Q38_WEIGHTS_H
#define Q38_WEIGHTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q38_gguf.h"
#include "q38_model_config.h"
#include "q38_ple.h"
#include "q38_qsa.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_MAX_BANKS_PER_LAYER 512
#define Q38_MAX_LAYER_TENSORS 192
#define Q38_MAX_PLE_TENSORS 160
#define Q38_PLE_SHARD_COUNT 128
#define Q38_PLE_AUX_TENSORS 9 /* key/value, three norms, conv, three hash arrays */

typedef struct {
    uint8_t bank_id;
    uint16_t local_index;
} q38_expert_location;

typedef struct {
    uint32_t qtype;
    q38_tensor *gate_up;
    q38_tensor *down;
    uint32_t expert_count;
} q38_expert_bank;

typedef struct {
    q38_expert_bank bank[Q38_MAX_BANKS_PER_LAYER];
    q38_expert_location loc[Q38_MODEL_EXPERTS];
    uint32_t bank_count;
} q38_layer_expert_store;

typedef struct {
    q38_tensor *block_inject_weight;
    q38_tensor *hc_norm;
    q38_tensor *input_mix_weight_down;
    q38_tensor *input_mix_weight_up;
} q38_gr_weights;

typedef struct {
    q38_tensor *in_proj_qkv;
    q38_tensor *in_proj_z;
    q38_tensor *in_proj_a;
    q38_tensor *in_proj_b;
    q38_tensor *conv1d;
    q38_tensor *A_log;
    q38_tensor *dt_bias;
    q38_tensor *norm;
    q38_tensor *out_proj;
} q38_gdn_weights;

typedef struct {
    q38_layer_kind kind;
    q38_tensor *router;
    q38_tensor *shared_expert_gate;
    q38_tensor *shared_gate_proj;
    q38_tensor *shared_up_proj;
    q38_tensor *shared_down_proj;
    q38_tensor *gdn_or_qsa[32];
    uint32_t gdn_or_qsa_count;
    q38_tensor *tensor[Q38_MAX_LAYER_TENSORS];
    uint32_t tensor_count;
    q38_tensor *ple_tensor[Q38_MAX_PLE_TENSORS];
    uint32_t ple_tensor_count;
    q38_ple_store ple_store;
    q38_layer_expert_store experts;
    q38_gr_weights attn_gr;
    q38_gr_weights mlp_gr;
    q38_gdn_weights gdn;
    q38_qsa_weights qsa;
} q38_layer_weights;

typedef struct {
    q38_tensor *token_embd;
    q38_layer_weights layer[Q38_MODEL_LAYERS];
    q38_tensor *output;
    q38_tensor *global_tensor[8];
    uint32_t global_tensor_count;
    uint32_t bound_layers;
    uint32_t bound_tensor_count;
    bool quantized;
} q38_weights;

/* Bind a runtime-only subset containing layers 0..max_layer, without forward
 * execution or persistent dequantized storage. */
bool q38_weights_bind_subset(const q38_gguf *model, uint32_t max_layer,
                             q38_weights *out, char *error, size_t error_len);

/* Validate that every tensor required by the complete execution graph is
 * bound before a forward state or activation is allocated. */
bool q38_weights_validate_bound(const q38_weights *weights, char *error,
                                size_t error_len);

/* Construct an O(1) uniform bank mapping for 512 experts. */
bool q38_expert_store_init_uniform(q38_layer_expert_store *store,
                                   uint32_t qtype);

/* Group experts by quant type while preserving an O(1) expert lookup. */
bool q38_expert_store_init_mixed(q38_layer_expert_store *store,
                                 const uint32_t qtypes[Q38_MODEL_EXPERTS]);

#ifdef __cplusplus
}
#endif

#endif
