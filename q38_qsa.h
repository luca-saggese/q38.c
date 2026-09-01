#ifndef Q38_QSA_H
#define Q38_QSA_H

#include "q38_gguf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    q38_tensor *q_proj;
    q38_tensor *k_proj;
    q38_tensor *v_proj;
    q38_tensor *o_proj;
    q38_tensor *q_norm;
    q38_tensor *k_norm;
    q38_tensor *index_qk_proj;
    q38_tensor *index_q_norm;
    q38_tensor *index_k_norm;
} q38_qsa_weights;

typedef struct {
    uint8_t *data;
    size_t row_bytes;
    size_t capacity;
    size_t count;
} q38_qsa_cache;

typedef struct {
    q38_qsa_cache main_k;
    q38_qsa_cache main_v;
    q38_qsa_cache index_k;
    uint64_t position;
    uint64_t committed_tokens;
} q38_qsa_state;

bool q38_qsa_weights_validate(const q38_qsa_weights *weights,
                              char *error, size_t error_len);

bool q38_qsa_state_init(q38_qsa_state *state, size_t main_k_row_bytes,
                        size_t main_v_row_bytes, size_t index_k_row_bytes,
                        char *error, size_t error_len);
void q38_qsa_state_reset(q38_qsa_state *state);
void q38_qsa_state_destroy(q38_qsa_state *state);

#ifdef __cplusplus
}
#endif

#endif
