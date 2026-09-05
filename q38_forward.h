#ifndef Q38_FORWARD_H
#define Q38_FORWARD_H

#include "q38_qsa.h"
#include "q38_moe_ref.h"
#include "q38_session.h"
#include "q38_state.h"
#include "q38_weights.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Q38_FORWARD_F32 = 0,
    Q38_FORWARD_BF16 = 30,
} q38_forward_dtype;

typedef struct {
    const void *data;
    size_t rows;
    size_t cols;
    q38_forward_dtype dtype;
} q38_forward_matrix;

bool q38_forward_matrix_from_tensor(const q38_gguf *model,
                                    const q38_tensor *tensor, size_t rows,
                                    size_t cols, q38_forward_matrix *out,
                                    char *error, size_t error_len);

typedef struct {
    q38_forward_matrix q_proj;
    q38_forward_matrix k_proj;
    q38_forward_matrix v_proj;
    q38_forward_matrix o_proj;
    q38_forward_matrix index_qk_proj;
    const float *q_norm;
    const float *k_norm;
    const float *index_q_norm;
    const float *index_k_norm;
    size_t hidden;
    size_t query_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t index_heads;
    size_t index_dim;
    size_t ratio;
    size_t budget;
    float rope_theta;
    size_t rotary_dims;
} q38_forward_qsa_weights;

/* Allocate cache rows for the dynamic reference graph. */
bool q38_forward_qsa_state_init(q38_qsa_state *state,
                                const q38_forward_qsa_weights *weights,
                                char *error, size_t error_len);

/*
 * Run a text-only QSA layer.  Matrices are output-by-input and may point
 * directly into a read-only GGUF mmap.  The graph appends all projections
 * before reading the cache, but applies the causal prefix for each query.
 */
bool q38_forward_qsa_ref(const q38_forward_qsa_weights *weights,
                         q38_qsa_state *state, const float *hidden,
                         size_t token_count, float *output,
                         uint32_t *selected, size_t selected_stride,
                         size_t *selected_counts, char *error,
                         size_t error_len);

typedef struct {
    q38_state_storage storage;
    q38_qsa_state qsa[Q38_MODEL_LAYERS];
    q38_ngram_history token_history;
    float *ple_history;
    size_t ple_history_elements;
    uint32_t eos_token;
    bool initialized;
} q38_forward_state;

typedef struct q38_moe_trace {
    const float *router_input;
    size_t router_input_count;
    const float *router_logits_pre_cast;
    const float *router_logits_effective;
    size_t router_logits_count;
    const uint16_t *top15_rank;
    const float *top15_value;
    size_t top15_count;
    float margin_rank10_rank11;
    const uint16_t *selected_experts;
    const float *selected_weights_pre_cast;
    const float *selected_weights_effective;
    size_t selected_count;
    const float *routed_output;
    size_t routed_output_count;
    q38_forward_dtype router_dtype;
} q38_moe_trace;

typedef struct q38_pre_router_trace {
    const float *router_input;
    size_t router_input_count;
    const float *gr_output;
    size_t gr_output_count;
    const q38_tensor *router;
} q38_pre_router_trace;

typedef bool (*q38_forward_boundary_trace)(
    uint32_t layer, const char *boundary, const float *values,
    size_t token_count, size_t width, void *user, char *error,
    size_t error_len);

/*
 * Optional execution accounting for the full graph.  A stage record is
 * emitted after each matrix operation and distinguishes device rows from
 * scalar rows.  The callback is observational and never changes ordering.
 */
typedef struct {
    const char *name;
    uint64_t matrix_calls;
    uint64_t backend_rows;
    uint64_t scalar_rows;
    uint64_t backend_declines;
    double elapsed_ms;
    uint32_t layer;
    const char *logical_stage;
} q38_forward_stage_usage;

typedef bool (*q38_forward_stage_trace)(
    const q38_forward_stage_usage *usage, void *user, char *error,
    size_t error_len);

typedef void (*q38_forward_backend_context_trace)(
    uint32_t layer, const char *logical_stage, const q38_tensor *tensor,
    size_t rows, size_t cols, void *user);

typedef struct {
    /*
     * Diagnostic mode: compute and commit PLE history normally, but omit its
     * activation from the residual stream consumed by the remaining graph.
     */
    bool disable_ple;
    uint32_t first_divergence_layer;
    uint32_t first_divergence_token;
    float max_abs_error;
    uint64_t layer_fingerprint[Q38_MODEL_LAYERS];
    bool has_reference;
    bool (*trace)(uint32_t layer, const float *hidden, size_t token_count,
                  size_t width, void *user, char *error, size_t error_len);
    bool (*route_trace)(uint32_t layer, const uint16_t *experts,
                        const float *weights, size_t count, void *user,
                        char *error, size_t error_len);
    bool (*qsa_trace)(uint32_t layer, const uint32_t *selected, size_t count,
                      void *user, char *error, size_t error_len);
    void *trace_user;
    /*
     * The pointers in q38_moe_trace are valid only for the callback.  The
     * layer-2 probe uses them to compare the FP32 scalar path with the
     * router's effective output dtype without changing GGUF storage.
     */
    bool (*moe_trace)(uint32_t layer, const q38_moe_trace *trace,
                      void *user, char *error, size_t error_len);
    bool (*router_trace)(uint32_t layer, const float *logits, size_t count,
                         void *user, char *error, size_t error_len);
    bool (*pre_router_trace)(uint32_t layer,
                             const q38_pre_router_trace *trace, void *user,
                             char *error, size_t error_len);
    /*
     * Optional full-vector checkpoints for progressive divergence
     * localization.  The scalar forward remains authoritative; callbacks
     * only observe activation buffers while they are live.
     */
    q38_forward_boundary_trace boundary_trace;
    q38_forward_stage_trace stage_trace;
    q38_forward_backend_context_trace backend_context;
} q38_forward_diagnostics;

/* Optional diagnostic row-matvec backend.  It is strict when installed via
 * q38_forward_full_with_backend: a declined row is an execution error. */
typedef bool (*q38_forward_matvec_backend)(
    const q38_gguf *model, const q38_tensor *tensor, size_t row,
    const float *input, size_t cols, float *output, void *user, char *error,
    size_t error_len);

bool q38_forward_state_init(q38_forward_state *state,
                            const q38_weights *weights, uint32_t eos_token,
                            char *error, size_t error_len);
void q38_forward_state_reset(q38_forward_state *state);
void q38_forward_state_destroy(q38_forward_state *state);

/*
 * Execute the complete text graph against the file-backed GGUF tensors.
 * `logits` is token-major and must provide `logits_stride >= vocab_size`
 * floats per token.  No logits are synthesized: unsupported tensor formats,
 * missing PLE metadata, or invalid storage are reported as errors.
 */
bool q38_forward_full(const q38_gguf *model, const q38_weights *weights,
                      q38_forward_state *state, const uint32_t *tokens,
                      size_t token_count, float *logits, size_t logits_stride,
                      q38_forward_diagnostics *diagnostics, char *error,
                      size_t error_len);

typedef bool (*q38_forward_matrix_backend)(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t rows, size_t cols, float *output, void *user, char *error,
    size_t error_len);

typedef bool (*q38_forward_expert_backend)(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, size_t expert, const float *input, float *output,
    void *user, char *error, size_t error_len);

typedef bool (*q38_forward_moe_layer_backend)(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, const q38_moe_route10 *route,
    const float *host_input, float *host_output, void *user, char *error,
    size_t error_len);

bool q38_forward_full_with_matrix_moe_layer_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend row_backend,
    q38_forward_matrix_backend matrix_backend,
    q38_forward_expert_backend expert_backend,
    q38_forward_moe_layer_backend moe_layer_backend,
    void *backend_user, char *error, size_t error_len);

bool q38_forward_full_with_matrix_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend row_backend,
    q38_forward_matrix_backend matrix_backend,
    q38_forward_expert_backend expert_backend, void *backend_user,
    char *error, size_t error_len);

/*
 * Run the same graph with a matvec backend.  Backend refusal is fatal: this
 * entry point never silently re-enters the scalar implementation.  The
 * scalar q38_forward_full entry point remains the reference/oracle path.
 */
bool q38_forward_full_with_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend backend, void *backend_user, char *error,
    size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
