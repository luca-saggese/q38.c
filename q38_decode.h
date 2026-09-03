#ifndef Q38_DECODE_H
#define Q38_DECODE_H

#include "q38_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_DECODE_VOCAB_SIZE 248320u

typedef struct {
    float min;
    float max;
    float mean;
    float rms;
    float max_abs;
    size_t finite_count;
    size_t nan_count;
    size_t inf_count;
    uint64_t checksum;
} q38_decode_stats;

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
    uint64_t pending_main_k_hash;
    uint64_t pending_main_v_hash;
    uint64_t pending_index_k_hash;
    q38_decode_stats main_k_stats;
    q38_decode_stats main_v_stats;
    q38_decode_stats index_k_stats;
} q38_decode_qsa_snapshot;

typedef enum {
    Q38_DECODE_TRACE_PROMPT_PREDICTION = 0,
    Q38_DECODE_TRACE_GENERATED_EMIT = 1,
    Q38_DECODE_TRACE_GENERATED_CONSUME = 2,
} q38_decode_trace_kind;

/*
 * Trace evidence for either a prediction or a generated-token protocol event.
 * For GENERATED_EMIT, no forward pass occurs: the emitted token is the
 * prompt prediction and state_committed is false. GENERATED_CONSUME records
 * the forward pass which consumes the previously emitted token and predicts
 * the next one.
 */
typedef struct {
    size_t step;
    q38_decode_trace_kind kind;
    bool generated;
    bool state_committed;
    uint32_t input_token;
    uint32_t next_token;
    uint32_t emitted_token;
    uint32_t consumed_token;
    uint64_t committed_tokens;
    uint64_t gdn_state_hash;
    uint64_t conv_history_hash;
    uint64_t ple_history_hash;
    uint64_t gdn_layer_hash[Q38_MODEL_LAYERS];
    uint64_t conv_layer_hash[Q38_MODEL_LAYERS];
    q38_decode_stats gdn_state_stats;
    q38_decode_stats conv_history_stats;
    q38_decode_stats ple_history_stats;
    q38_decode_stats gdn_layer_stats[Q38_MODEL_LAYERS];
    q38_decode_stats conv_layer_stats[Q38_MODEL_LAYERS];
    uint32_t ple_prev_token_1;
    uint32_t ple_prev_token_2;
    bool ple_have_prev_1;
    bool ple_have_prev_2;
    uint64_t logits_hash;
    uint32_t argmax;
    float argmax_value;
    float logits_min;
    float logits_max;
    float logits_mean;
    float logits_rms;
    float logits_max_abs;
    uint32_t top_ids[20];
    float top_values[20];
    bool logits_finite;
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
 * generated_count greedy temperature-zero tokens. The first generated token
 * is the final prompt argmax and is emitted without another forward pass.
 * Subsequent generated tokens consume the previously emitted token. The
 * callback observes prompt predictions, generated emissions, and generated
 * consuming forward passes explicitly, including complete state hashes and
 * QSA cache metadata. `prompt_count` must be nonzero.
 */
bool q38_decode_stream(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_decode_trace trace,     void *trace_user, char *error, size_t error_len);

bool q38_decode_stream_with_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend backend, void *backend_user,
    q38_decode_trace trace, void *trace_user, char *error, size_t error_len);

bool q38_decode_stream_with_matrix_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend row_backend,
    q38_forward_matrix_backend matrix_backend,
    q38_forward_expert_backend expert_backend, void *backend_user,
    q38_decode_trace trace, void *trace_user, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
