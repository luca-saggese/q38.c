#ifndef Q38_DECODE_H
#define Q38_DECODE_H

#include "q38_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_DECODE_VOCAB_SIZE 248320u

typedef struct {
    uint64_t position;
    uint64_t committed_tokens;
    uint32_t pending_count;
    uint64_t pending_position;
    size_t main_k_count;
    size_t main_v_count;
    size_t index_k_count;
    uint64_t main_k_hash;
    uint64_t main_v_hash;
    uint64_t index_k_hash;
} q38_decode_qsa_snapshot;

/*
 * State captured immediately after one input token has been committed. Hashes
 * cover the complete F32 regions, so a caller can persist compact state
 * evidence without copying the model state.
 */
typedef struct {
    size_t step;
    bool generated;
    uint32_t input_token;
    uint32_t next_token;
    uint64_t committed_tokens;
    uint64_t gdn_state_hash;
    uint64_t conv_history_hash;
    uint64_t ple_history_hash;
    bool finite;
    q38_decode_qsa_snapshot qsa[Q38_MODEL_LAYERS];
} q38_decode_step;

typedef bool (*q38_decode_trace)(const q38_decode_step *step, void *user,
                                 char *error, size_t error_len);

/*
 * Run one committed token through the complete text graph and select the
 * lowest-ID greedy token from the resulting checkpoint-derived logits.
 */
bool q38_decode(const q38_gguf *model, const q38_weights *weights,
                q38_forward_state *state, uint32_t token, float *logits,
                size_t logits_stride, uint32_t *next_token,
                q38_forward_diagnostics *diagnostics, char *error,
                size_t error_len);

/*
 * Consume a prompt token stream one token at a time, then emit exactly
 * generated_count greedy temperature-zero tokens. The callback observes every
 * committed prompt and generated input token, including complete state
 * hashes and QSA cache metadata. `prompt_count` must be nonzero.
 */
bool q38_decode_stream(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_decode_trace trace, void *trace_user, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
