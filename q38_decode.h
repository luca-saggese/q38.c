#ifndef Q38_DECODE_H
#define Q38_DECODE_H

#include "q38_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Run one committed token through the complete text graph and select the
 * lowest-ID greedy token from the resulting checkpoint-derived logits.
 */
bool q38_decode(const q38_gguf *model, const q38_weights *weights,
                q38_forward_state *state, uint32_t token, float *logits,
                size_t logits_stride, uint32_t *next_token,
                q38_forward_diagnostics *diagnostics, char *error,
                size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
