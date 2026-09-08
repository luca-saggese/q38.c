#ifndef Q38_SESSION_H
#define Q38_SESSION_H

#include "q38_decode.h"
#include "q38_directional_steering.h"
#include "q38_gguf.h"
#include "q38_nvfp4_runtime.h"
#include "q38_tokenizer.h"
#include "q38_weights.h"
#include "q38_session_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_forward_cuda_context q38_forward_cuda_context;

typedef struct {
    q38_gguf *model;
    q38_nvfp4_runtime_model *nvfp4_runtime;
    q38_tokenizer tokenizer;
    q38_weights weights;
    q38_forward_cuda_context *cuda;
    q38_forward_backend_config backend;
    q38_directional_steering steering;
    bool tokenizer_initialized;
} q38_runtime;

typedef struct {
    q38_runtime *runtime;
    q38_forward_state state;
    uint32_t ctx_size;
    uint64_t position;
    uint32_t *token_history;
    size_t token_count;
    size_t token_capacity;
    double ple_wait_at_injection_ms;
    float steering_ffn_scale;
    float steering_attn_scale;
    bool steering_override_set;
} q38_session;

typedef void (*q38_token_callback)(uint32_t token, const char *piece,
                                   size_t len, void *userdata);

bool q38_runtime_init(q38_runtime *runtime, const char *model_path,
                      const char *tokenizer_path, char *error,
                      size_t error_len);
bool q38_runtime_init_ex(q38_runtime *runtime, const char *model_path,
                         const char *source_root,
                         const char *tokenizer_path, char *error,
                         size_t error_len);
bool q38_runtime_preflight_native_nvfp4(
    const char *pack_path, const char *source_root,
    char *error, size_t error_len);
void q38_runtime_destroy(q38_runtime *runtime);
bool q38_runtime_load_directional_steering(
    q38_runtime *runtime, const char *path, float ffn_scale,
    float attn_scale, char *error, size_t error_len);

bool q38_session_create(q38_session *session, q38_runtime *runtime,
                        uint32_t ctx_size, char *error, size_t error_len);
void q38_session_reset(q38_session *session);
void q38_session_destroy(q38_session *session);
bool q38_session_set_directional_steering(
    q38_session *session, float ffn_scale, float attn_scale, char *error,
    size_t error_len);
void q38_session_clear_directional_steering_override(q38_session *session);

bool q38_session_prefill(
    q38_session *session, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace trace,
    void *trace_user, size_t *step_index, char *error, size_t error_len);
bool q38_session_prefill_reference(
    q38_session *session, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace trace,
    void *trace_user, size_t *step_index, char *error, size_t error_len);
bool q38_session_prefill_chunked(
    q38_session *session, const uint32_t *tokens, size_t token_count,
    size_t chunk_size, float *logits, size_t logits_stride,
    uint32_t *next_token, q38_forward_diagnostics *diagnostics,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    char *error, size_t error_len);

bool q38_session_eval(
    q38_session *session, uint32_t token, float *logits,
    size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace_kind trace_kind,
    uint32_t emitted_token, uint32_t consumed_token,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    char *error, size_t error_len);

bool q38_session_eval_timed(
    q38_session *session, uint32_t token, float *logits,
    size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace_kind trace_kind,
    uint32_t emitted_token, uint32_t consumed_token,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    q38_decode_timing *timing, q38_ple_scheduler_stats *ple_stats,
    char *error, size_t error_len);

bool q38_session_emit(
    const q38_session *session, const float *logits, uint32_t token,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    char *error, size_t error_len);

bool q38_session_stream_token(q38_session *session, uint32_t token,
                              q38_token_callback callback, void *userdata,
                              char *error, size_t error_len);
uint32_t q38_session_eos_token(const q38_session *session);
size_t q38_session_context_remaining(const q38_session *session);

#ifdef __cplusplus
}
#endif

#endif
