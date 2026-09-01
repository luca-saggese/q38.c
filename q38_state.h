#ifndef Q38_STATE_H
#define Q38_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_STATE_MODEL_LAYERS 48u
#define Q38_GDN_LAYER_COUNT 36u
#define Q38_FULL_ATTENTION_LAYER_COUNT 12u
#define Q38_GDN_VALUE_HEADS 48u
#define Q38_GDN_HEAD_DIM 128u
#define Q38_GDN_CONV_KERNEL 4u
#define Q38_GDN_CONV_CHANNELS 10240u
#define Q38_GDN_SEQUENCE_COUNT 1u

#define Q38_GR_STATE_BRANCHES 4u
#define Q38_GR_STATE_HIDDEN 2560u

typedef enum {
    Q38_STATE_DTYPE_INVALID = 0,
    Q38_STATE_DTYPE_F32 = 1,
} q38_state_dtype;

/* Aggregate descriptor: each of slot_count GDN layers owns one identical
 * logical region, while elements/bytes describe the complete allocation. */
typedef struct {
    uint32_t sequence_count;
    uint32_t value_heads;
    uint32_t head_dim;
    uint32_t slot_count;
    q38_state_dtype dtype;
    uint64_t elements_per_slot;
    uint64_t bytes_per_slot;
    uint64_t elements;
    uint64_t bytes;
} q38_gdn_state_desc;

typedef struct {
    uint32_t sequence_count;
    uint32_t channels;
    uint32_t kernel;
    uint32_t history_tokens;
    uint32_t slot_count;
    q38_state_dtype dtype;
    uint64_t elements_per_slot;
    uint64_t bytes_per_slot;
    uint64_t elements;
    uint64_t bytes;
} q38_conv_history_desc;

/* GR is a forward activation, never persistent semantic session state. */
typedef struct {
    uint32_t sequence_count;
    uint32_t branches;
    uint32_t hidden_size;
    q38_state_dtype dtype;
    uint64_t elements;
    uint64_t bytes;
} q38_forward_workspace_desc;

typedef struct {
    uint64_t persistent_recurrent_state_bytes;
    uint64_t conv_history_bytes;
    uint64_t gr_workspace_bytes;
    uint64_t workspace_bytes;
    uint64_t persistent_bytes;
    uint64_t activation_bytes;
    uint64_t allocation_bytes;
} q38_state_memory;

/* Pointer-free and directly serializable logical session description. */
typedef struct {
    q38_gdn_state_desc recurrent;
    q38_conv_history_desc conv_history;
    q38_forward_workspace_desc gr_workspace;
    int8_t layer_to_gdn_slot[Q38_STATE_MODEL_LAYERS];
    q38_state_memory memory;
} q38_session_state;

typedef struct {
    q38_session_state layout;
    float *recurrent_state;
    float *conv_history;
    float *gr_workspace;
    unsigned char *workspace;
} q38_state_storage;

bool q38_session_state_init(q38_session_state *state,
                            uint64_t workspace_bytes,
                            char *error, size_t error_len);
bool q38_session_state_validate(const q38_session_state *state,
                                char *error, size_t error_len);

bool q38_state_alloc(const q38_session_state *layout,
                     q38_state_storage *storage,
                     char *error, size_t error_len);
void q38_state_reset(q38_state_storage *storage);
void q38_state_free(q38_state_storage *storage);

int q38_gdn_slot_for_layer(const q38_session_state *state, uint32_t layer);
float *q38_state_recurrent_slot(q38_state_storage *storage, uint32_t slot);
float *q38_state_conv_history_slot(q38_state_storage *storage, uint32_t slot);

#ifdef __cplusplus
}
#endif

#endif /* Q38_STATE_H */
