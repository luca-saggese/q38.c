#include "q38_session.h"
#include "q38_forward_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

void q38_ngram_history_reset(q38_ngram_history *history) {
    if (!history) return;
    history->prev_token_1 = 0;
    history->prev_token_2 = 0;
    history->have_prev_1 = false;
    history->have_prev_2 = false;
}

void q38_ngram_history_append(q38_ngram_history *history,
                              uint32_t token, uint32_t eos_token) {
    if (!history) return;
    if (token == eos_token) {
        history->prev_token_1 = eos_token;
        history->prev_token_2 = eos_token;
        history->have_prev_1 = true;
        history->have_prev_2 = true;
        return;
    }
    history->prev_token_2 = history->prev_token_1;
    history->have_prev_2 = history->have_prev_1;
    history->prev_token_1 = token;
    history->have_prev_1 = true;
}

void q38_ngram_history_context(const q38_ngram_history *history,
                               uint32_t current_token, uint32_t eos_token,
                               uint32_t context[3]) {
    context[0] = current_token;
    if (!history || !history->have_prev_1 || history->prev_token_1 == eos_token) {
        context[1] = eos_token;
        context[2] = eos_token;
        return;
    }
    context[1] = history->prev_token_1;
    context[2] = history->have_prev_2 ? history->prev_token_2 : eos_token;
}

static void runtime_zero(q38_runtime *runtime) {
    if (runtime) memset(runtime, 0, sizeof(*runtime));
}

bool q38_runtime_init(q38_runtime *runtime, const char *model_path,
                      const char *tokenizer_path, char *error,
                      size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !model_path || !tokenizer_path || !tokenizer_path[0])
        return fail(error, error_len, "invalid runtime initialization arguments");
    runtime_zero(runtime);
    runtime->model = q38_gguf_open(model_path, error, error_len);
    if (!runtime->model)
        goto fail_runtime;
    if (!q38_tokenizer_init(&runtime->tokenizer, tokenizer_path, NULL,
                            error, error_len))
        goto fail_runtime;
    runtime->tokenizer_initialized = true;
    if (!q38_weights_bind_subset(runtime->model, 47, &runtime->weights,
                                 error, error_len))
        goto fail_runtime;
    runtime->cuda = q38_forward_cuda_context_create(error, error_len);
    if (!runtime->cuda ||
        !q38_forward_cuda_enable_all_non_ple_residency(
            runtime->cuda, runtime->model, error, error_len) ||
        !q38_forward_cuda_prepare_lm_head(
            runtime->cuda, runtime->model, runtime->weights.output,
            error, error_len))
        goto fail_runtime;
    return true;

fail_runtime:
    q38_runtime_destroy(runtime);
    return false;
}

void q38_runtime_destroy(q38_runtime *runtime) {
    if (!runtime) return;
    q38_forward_cuda_context_destroy(runtime->cuda);
    runtime->cuda = NULL;
    q38_weights_release(&runtime->weights);
    if (runtime->tokenizer_initialized)
        q38_tokenizer_destroy(&runtime->tokenizer);
    runtime->tokenizer_initialized = false;
    q38_gguf_close(runtime->model);
    runtime->model = NULL;
}

static bool reserve_history(q38_session *session, size_t extra,
                            char *error, size_t error_len) {
    if (extra > SIZE_MAX - session->token_count)
        return fail(error, error_len, "session token history overflows");
    const size_t needed = session->token_count + extra;
    if (needed <= session->token_capacity) return true;
    size_t capacity = session->token_capacity ? session->token_capacity : 64;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    uint32_t *grown = (uint32_t *)realloc(
        session->token_history, capacity * sizeof(*grown));
    if (!grown) return fail(error, error_len, "session token history allocation failed");
    session->token_history = grown;
    session->token_capacity = capacity;
    return true;
}

static bool append_history(q38_session *session, uint32_t token,
                           char *error, size_t error_len) {
    if (!reserve_history(session, 1, error, error_len)) return false;
    session->token_history[session->token_count++] = token;
    session->position++;
    return true;
}

static void runtime_backend_context(uint32_t layer, const char *stage,
                                    const q38_tensor *tensor, size_t rows,
                                    size_t cols, void *user) {
    (void)tensor;
    (void)rows;
    (void)cols;
    q38_forward_cuda_set_stage_context(
        (q38_forward_cuda_context *)user, layer, stage);
}

bool q38_session_create(q38_session *session, q38_runtime *runtime,
                        uint32_t ctx_size, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!session || !runtime || !runtime->model || !runtime->cuda || !ctx_size)
        return fail(error, error_len, "invalid session creation arguments");
    memset(session, 0, sizeof(*session));
    session->runtime = runtime;
    session->ctx_size = ctx_size;
    if (!q38_forward_state_init(&session->state, &runtime->weights,
                                runtime->tokenizer.eos_id, error, error_len)) {
        memset(session, 0, sizeof(*session));
        return false;
    }
    return true;
}

void q38_session_reset(q38_session *session) {
    if (!session) return;
    q38_forward_state_reset(&session->state);
    session->position = 0;
    session->token_count = 0;
    session->ple_wait_at_injection_ms = 0.0;
}

void q38_session_destroy(q38_session *session) {
    if (!session) return;
    q38_forward_state_destroy(&session->state);
    free(session->token_history);
    memset(session, 0, sizeof(*session));
}

bool q38_session_eval(
    q38_session *session, uint32_t token, float *logits,
    size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace_kind trace_kind,
    uint32_t emitted_token, uint32_t consumed_token,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!session || !session->runtime || !logits || !next_token ||
        !step_index)
        return fail(error, error_len, "invalid session evaluation arguments");
    if (session->position >= session->ctx_size)
        return fail(error, error_len, "session context is full");
    if (diagnostics) {
        diagnostics->backend_context = runtime_backend_context;
        diagnostics->backend_context_user = session->runtime->cuda;
    }
    if (!q38_decode_step_with_matrix_moe_layer_backend(
            session->runtime->model, &session->runtime->weights,
            &session->state, token, logits, logits_stride, next_token,
            diagnostics, q38_forward_cuda_matvec_backend,
            q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend,
            q38_forward_cuda_moe_layer_q2_backend, session->runtime->cuda,
            trace_kind, emitted_token, consumed_token, (*step_index)++,
            trace, trace_user, error, error_len))
        return false;
    q38_ple_scheduler_stats ple = {0};
    if (q38_forward_state_get_ple_prefetch_stats(&session->state, &ple) &&
        ple.wait_ms > session->ple_wait_at_injection_ms)
        session->ple_wait_at_injection_ms = ple.wait_ms;
    return append_history(session, token, error, error_len);
}

bool q38_session_prefill(
    q38_session *session, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace trace,
    void *trace_user, size_t *step_index, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!session || !tokens || !token_count || !logits || !next_token ||
        !step_index)
        return fail(error, error_len, "invalid session prefill arguments");
    if (token_count > session->ctx_size)
        return fail(error, error_len, "prompt exceeds session context");
    q38_session_reset(session);
    for (size_t i = 0; i < token_count; ++i) {
        if (!q38_session_eval(
                session, tokens[i], logits, logits_stride, next_token,
                diagnostics, Q38_DECODE_TRACE_PROMPT_PREDICTION,
                UINT32_MAX, tokens[i], trace, trace_user, step_index,
                error, error_len))
            return false;
    }
    return true;
}

bool q38_session_emit(
    const q38_session *session, const float *logits, uint32_t token,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!session || !logits || !step_index)
        return fail(error, error_len, "invalid session emit arguments");
    return q38_decode_emit_trace(
        &session->state, logits, (*step_index)++, token, trace, trace_user,
        error, error_len);
}

bool q38_session_stream_token(q38_session *session, uint32_t token,
                              q38_token_callback callback, void *userdata,
                              char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!session || !session->runtime || !callback)
        return fail(error, error_len, "invalid token callback arguments");
    char *piece = NULL;
    size_t piece_len = 0;
    if (!q38_tokenizer_decode(&session->runtime->tokenizer, &token, 1,
                              &piece, &piece_len, error, error_len))
        return false;
    callback(token, piece, piece_len, userdata);
    free(piece);
    return true;
}

uint32_t q38_session_eos_token(const q38_session *session) {
    return session && session->runtime
        ? session->runtime->tokenizer.eos_id : UINT32_MAX;
}

size_t q38_session_context_remaining(const q38_session *session) {
    if (!session || session->position >= session->ctx_size) return 0;
    return (size_t)session->ctx_size - (size_t)session->position;
}
