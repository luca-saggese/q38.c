#ifndef Q38_STATE_H
#define Q38_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These are the frozen logical GDN dimensions.  They are deliberately
 * independent of any CUDA tiling or transposed physical representation. */
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

/* Serializable logical description of the persistent recurrent matrix:
 * state[sequence, value_head, row, column]. */
typedef struct {
    uint32_t sequence_count;
    uint32_t value_heads;
    uint32_t head_dim;
    q38_state_dtype dtype;
    uint64_t elements;
    uint64_t bytes;
} q38_gdn_state_desc;

/* The convolution keeps kernel-1 prior activations between chunks in the
 * logical order history[sequence, history_token, channel]. */
typedef struct {
    uint32_t sequence_count;
    uint32_t channels;
    uint32_t kernel;
    uint32_t history_tokens;
    q38_state_dtype dtype;
    uint64_t elements;
    uint64_t bytes;
} q38_conv_history_desc;

/* GR has no recurrence, but its branch values are session-owned state in the
 * M3 logical layout so its footprint remains independently accountable. */
typedef struct {
    uint32_t sequence_count;
    uint32_t branches;
    uint32_t hidden_size;
    q38_state_dtype dtype;
    uint64_t elements;
    uint64_t bytes;
} q38_gr_state_desc;

typedef struct {
    uint64_t persistent_recurrent_state_bytes;
    uint64_t conv_history_bytes;
    uint64_t gr_state_bytes;
    uint64_t workspace_bytes;
    uint64_t persistent_bytes;
    uint64_t allocation_bytes;
} q38_state_memory;

/* This is the serializable session layout.  It intentionally contains no
 * pointers or backend handles. */
typedef struct {
    q38_gdn_state_desc recurrent;
    q38_conv_history_desc conv_history;
    q38_gr_state_desc gr;
    q38_state_memory memory;
} q38_session_state;

/* Runtime ownership is separate from q38_session_state so the latter can be
 * copied or serialized without carrying opaque allocation pointers. */
typedef struct {
    q38_session_state layout;
    float *recurrent_state;
    float *conv_history;
    float *gr_state;
    unsigned char *workspace;
} q38_state_storage;

/* Initialize and validate the fixed, single-sequence M3 layout. */
bool q38_session_state_init(q38_session_state *state,
                            uint64_t workspace_bytes,
                            char *error, size_t error_len);
bool q38_session_state_validate(const q38_session_state *state,
                                char *error, size_t error_len);

/* Allocate, reset, and release the four separately-accounted regions. */
bool q38_state_alloc(const q38_session_state *layout,
                     q38_state_storage *storage,
                     char *error, size_t error_len);
void q38_state_reset(q38_state_storage *storage);
void q38_state_free(q38_state_storage *storage);

#ifdef __cplusplus
}
#endif

#endif /* Q38_STATE_H */
