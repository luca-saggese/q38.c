#ifndef Q38_GDN_REF_H
#define Q38_GDN_REF_H

#include <stdbool.h>
#include <stddef.h>

#include "q38_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scalar F32 recurrence oracle.  Inputs to one step are logical
 * [value_head, dimension] planes; run() accepts
 * [sequence, token, value_head, dimension] arrays.
 */
#define Q38_GDN_REF_VALUE_HEADS Q38_GDN_VALUE_HEADS
#define Q38_GDN_REF_KEY_HEADS 16u
#define Q38_GDN_REF_HEAD_DIM Q38_GDN_HEAD_DIM

size_t q38_gdn_ref_state_elements(size_t sequence_count);
void q38_gdn_ref_reset(float *state, size_t sequence_count);

/*
 * Apply exactly the frozen recurrence:
 *
 *   Sbar = decay * S
 *   prediction = Sbar^T * k
 *   delta = (v - prediction) * beta
 *   S = Sbar + k * delta^T
 *   output = scale * S^T * q
 *
 * State is logical [sequence, value_head, row, column] with column stride 1.
 * decay and beta contain one scalar per value head.
 */
bool q38_gdn_ref_step(float *state, size_t sequence_count,
                      size_t sequence_index, const float *q,
                      const float *k, const float *v, const float *decay,
                      const float *beta, float scale, float *output);

/*
 * Run token-major logical arrays with layout
 * [sequence, token, value_head, dimension] for q/k/v and
 * [sequence, token, value_head] for decay/beta.
 */
bool q38_gdn_ref_run(float *state, size_t sequence_count, size_t token_count,
                     const float *q, const float *k, const float *v,
                     const float *decay, const float *beta, float scale,
                     float *output);

/* Exact Qwen mapping from 16 key heads to 48 value heads. */
void q38_gdn_ref_repeat_key_heads(const float *key_heads,
                                   float *value_heads);

#ifdef __cplusplus
}
#endif

#endif /* Q38_GDN_REF_H */
