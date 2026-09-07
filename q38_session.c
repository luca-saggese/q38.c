#include "q38_session.h"
#include "q38_diagnostics.h"
#include "q38_forward_cuda.h"
#include "q38_gr_ref.h"

#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

#if Q38_DIAGNOSTICS
static double session_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}
#else
#define session_now_ms() 0.0
#endif

static void runtime_zero(q38_runtime *runtime) {
    if (runtime) memset(runtime, 0, sizeof(*runtime));
}

static bool validate_lm_head_geometry(const q38_tensor *tensor,
                                      char *error, size_t error_len) {
    if (!tensor || tensor->ndim != 2 ||
        tensor->dim[0] != Q38_DECODE_VOCAB_SIZE ||
        tensor->dim[1] != Q38_GR_HIDDEN)
        return fail(error, error_len,
                    "LM-head geometry does not match decode dimensions");
    return true;
}

bool q38_runtime_init(q38_runtime *runtime, const char *model_path,
                      const char *tokenizer_path, char *error,
                      size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!runtime || !model_path || !tokenizer_path || !tokenizer_path[0])
        return fail(error, error_len, "invalid runtime initialization arguments");
    runtime_zero(runtime);
#if Q38_DIAGNOSTICS
    const double init_started = session_now_ms();
    double gguf_open_ms = 0.0;
    double tokenizer_ms = 0.0;
    double binding_ms = 0.0;
    double cuda_prepare_ms = 0.0;
#endif
    runtime->model = q38_gguf_open(model_path, error, error_len);
#if Q38_DIAGNOSTICS
    gguf_open_ms = session_now_ms() - init_started;
#endif
    if (!runtime->model)
        goto fail_runtime;
    const double tokenizer_started =
#if Q38_DIAGNOSTICS
        session_now_ms();
#else
        0.0;
#endif
    if (!q38_tokenizer_init(&runtime->tokenizer, tokenizer_path, NULL,
                            error, error_len))
        goto fail_runtime;
    runtime->tokenizer_initialized = true;
#if Q38_DIAGNOSTICS
    tokenizer_ms = session_now_ms() - tokenizer_started;
#endif
    const double binding_started =
#if Q38_DIAGNOSTICS
        session_now_ms();
#else
        0.0;
#endif
    if (!q38_weights_bind_subset(runtime->model, 47, &runtime->weights,
                                 error, error_len))
        goto fail_runtime;
#if Q38_DIAGNOSTICS
    binding_ms = session_now_ms() - binding_started;
#endif
    if (!validate_lm_head_geometry(runtime->weights.output, error, error_len))
        goto fail_runtime;
    const double cuda_prepare_started =
#if Q38_DIAGNOSTICS
        session_now_ms();
#else
        0.0;
#endif
    runtime->cuda = q38_forward_cuda_context_create(error, error_len);
    if (!runtime->cuda ||
        !q38_forward_cuda_enable_all_non_ple_residency(
            runtime->cuda, runtime->model, error, error_len) ||
        !q38_forward_cuda_prepare_lm_head(
            runtime->cuda, runtime->model, runtime->weights.output,
            error, error_len))
        goto fail_runtime;
#if Q38_DIAGNOSTICS
    cuda_prepare_ms = session_now_ms() - cuda_prepare_started;
    q38_forward_cuda_residency_stats startup_stats;
    q38_forward_cuda_get_residency_stats(runtime->cuda, &startup_stats);
    fprintf(stderr,
            "q38: startup_timing {\"gguf_open_ms\":%.3f,"
            "\"tokenizer_ms\":%.3f,\"binding_ms\":%.3f,"
            "\"cuda_prepare_ms\":%.3f,\"runtime_init_ms\":%.3f,"
            "\"residency_plan_ms\":%.3f,"
            "\"residency_device_alloc_ms\":%.3f,"
            "\"residency_source_copy_ms\":%.3f,"
            "\"residency_h2d_enqueue_ms\":%.3f,"
            "\"residency_d2d_enqueue_ms\":%.3f,"
            "\"residency_final_wait_ms\":%.3f,"
            "\"residency_other_ms\":%.3f,"
            "\"residency_planned_bytes\":%zu,"
            "\"residency_staged_bytes\":%zu,"
            "\"residency_h2d_bytes\":%zu,"
            "\"residency_planned_spans\":%" PRIu64 ","
            "\"residency_transfer_calls\":%" PRIu64 ","
            "\"residency_final_syncs\":%" PRIu64 ","
            "\"residency_stage_bytes\":%zu,"
            "\"residency_allocations\":%" PRIu64 ","
            "\"residency_allocated_bytes\":%zu,"
            "\"mincore_pages_before\":%" PRIu64 ","
            "\"mincore_pages_after\":%" PRIu64 ","
            "\"minor_faults_before\":%ld,\"minor_faults_after\":%ld,"
            "\"major_faults_before\":%ld,\"major_faults_after\":%ld,"
            "\"span_timings\":[",
            gguf_open_ms, tokenizer_ms, binding_ms, cuda_prepare_ms,
            session_now_ms() - init_started,
            startup_stats.residency_plan_ms,
            startup_stats.residency_device_alloc_ms,
            startup_stats.residency_source_copy_ms,
            startup_stats.residency_h2d_enqueue_ms,
            startup_stats.residency_d2d_enqueue_ms,
            startup_stats.residency_final_wait_ms,
            cuda_prepare_ms -
                startup_stats.residency_plan_ms -
                startup_stats.residency_device_alloc_ms -
                startup_stats.residency_source_copy_ms -
                startup_stats.residency_h2d_enqueue_ms -
                startup_stats.residency_d2d_enqueue_ms -
                startup_stats.residency_final_wait_ms,
            startup_stats.residency_planned_bytes,
            startup_stats.residency_staged_bytes,
            startup_stats.residency_h2d_bytes,
            startup_stats.residency_planned_spans,
            startup_stats.residency_transfer_calls,
            startup_stats.residency_final_syncs,
            startup_stats.residency_stage_bytes,
            startup_stats.residency_allocations,
            startup_stats.residency_allocated_bytes,
            startup_stats.residency_mincore_pages_before,
            startup_stats.residency_mincore_pages_after,
            startup_stats.residency_minor_faults_before,
            startup_stats.residency_minor_faults_after,
            startup_stats.residency_major_faults_before,
            startup_stats.residency_major_faults_after);
    for (size_t i = 0; i < startup_stats.residency_span_timing_count; ++i) {
        const q38_residency_span_timing *span =
            &startup_stats.residency_span_timings[i];
        fprintf(stderr, "%s{\"source_offset\":%" PRIu64
                ",\"bytes\":%zu,\"source_copy_ms\":%.3f,"
                "\"h2d_enqueue_ms\":%.3f}",
                i ? "," : "", span->source_offset, span->bytes,
                span->source_copy_ms, span->h2d_enqueue_ms);
    }
    fprintf(stderr, "]}\n");
#endif
    runtime->backend.matvec = q38_forward_cuda_matvec_backend;
    runtime->backend.matrix = q38_forward_cuda_matrix_backend;
    runtime->backend.matrix_batch = q38_forward_cuda_matrix_batch_backend;
    runtime->backend.gr_read = q38_forward_cuda_gr_read_backend;
    runtime->backend.gr_write = q38_forward_cuda_gr_write_backend;
    runtime->backend.expert = q38_forward_cuda_expert_backend;
    runtime->backend.moe_layer = q38_forward_cuda_moe_layer_q2_backend;
    runtime->backend.gdn_layer = q38_forward_cuda_gdn_layer_backend;
    /* Keep the experimental full-layer island out of production until its
     * complete-layer fixture and integration gates are green. */
    runtime->backend.decoder_layer_chain = NULL;
    runtime->backend.sync_state = q38_forward_cuda_sync_gdn_state;
    runtime->backend.qsa_qkv = q38_forward_cuda_qsa_qkv_backend;
    runtime->backend.qsa_chain = q38_forward_cuda_qsa_chain_backend;
    runtime->backend.user = runtime->cuda;
    return true;

fail_runtime:
    q38_runtime_destroy(runtime);
    return false;
}

void q38_runtime_destroy(q38_runtime *runtime) {
    if (!runtime) return;
    q38_directional_steering_destroy(&runtime->steering);
    q38_forward_cuda_context_destroy(runtime->cuda);
    runtime->cuda = NULL;
    q38_weights_release(&runtime->weights);
    if (runtime->tokenizer_initialized)
        q38_tokenizer_destroy(&runtime->tokenizer);
    runtime->tokenizer_initialized = false;
    q38_gguf_close(runtime->model);
    runtime->model = NULL;
}

bool q38_runtime_load_directional_steering(
    q38_runtime *runtime, const char *path, float ffn_scale,
    float attn_scale, char *error, size_t error_len) {
    if (!runtime)
        return fail(error, error_len, "invalid runtime steering arguments");
    if (!q38_directional_steering_load(
            &runtime->steering, path, ffn_scale, attn_scale, error,
            error_len))
        return false;
    if (!q38_forward_cuda_load_directional_steering(
            runtime->cuda, &runtime->steering, error, error_len)) {
        q38_directional_steering_destroy(&runtime->steering);
        return false;
    }
    q38_forward_cuda_set_directional_steering_scales(
        runtime->cuda, ffn_scale, attn_scale);
    return true;
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
    session->steering_ffn_scale = runtime->steering.ffn_scale;
    session->steering_attn_scale = runtime->steering.attn_scale;
    q38_forward_cuda_set_directional_steering_scales(
        runtime->cuda, session->steering_ffn_scale,
        session->steering_attn_scale);
    if (!q38_forward_state_init(&session->state, &runtime->weights,
                                runtime->tokenizer.eos_id, error, error_len)) {
        memset(session, 0, sizeof(*session));
        return false;
    }
    q38_forward_cuda_reset_gdn_state(runtime->cuda);
    return true;
}

void q38_session_reset(q38_session *session) {
    if (!session) return;
    const bool keep_override = session->steering_override_set;
    const float override_ffn = session->steering_ffn_scale;
    const float override_attn = session->steering_attn_scale;
    q38_forward_state_reset(&session->state);
    if (session->runtime && session->runtime->cuda)
        q38_forward_cuda_reset_gdn_state(session->runtime->cuda);
    session->position = 0;
    session->token_count = 0;
    session->ple_wait_at_injection_ms = 0.0;
    session->steering_override_set = keep_override;
    session->steering_ffn_scale = keep_override
        ? override_ffn
        : session->runtime ? session->runtime->steering.ffn_scale : 0.0f;
    session->steering_attn_scale = keep_override
        ? override_attn
        : session->runtime ? session->runtime->steering.attn_scale : 0.0f;
    if (session->runtime && session->runtime->cuda)
        q38_forward_cuda_set_directional_steering_scales(
            session->runtime->cuda, session->steering_ffn_scale,
            session->steering_attn_scale);
}

bool q38_session_set_directional_steering(
    q38_session *session, float ffn_scale, float attn_scale, char *error,
    size_t error_len) {
    if (!session || !session->runtime)
        return fail(error, error_len, "invalid session steering arguments");
    if (!isfinite(ffn_scale) || !isfinite(attn_scale) ||
        fabsf(ffn_scale) > 100.0f || fabsf(attn_scale) > 100.0f)
        return fail(error, error_len, "Q38 steering scale is out of range");
    if ((ffn_scale != 0.0f || attn_scale != 0.0f) &&
        !session->runtime->steering.directions)
        return fail(error, error_len,
                    "session steering requires a loaded Q38 direction");
    session->steering_ffn_scale = ffn_scale;
    session->steering_attn_scale = attn_scale;
    session->steering_override_set = true;
    q38_forward_cuda_set_directional_steering_scales(
        session->runtime->cuda, ffn_scale, attn_scale);
    return true;
}

void q38_session_clear_directional_steering_override(q38_session *session) {
    if (!session) return;
    session->steering_override_set = false;
    session->steering_ffn_scale =
        session->runtime ? session->runtime->steering.ffn_scale : 0.0f;
    session->steering_attn_scale =
        session->runtime ? session->runtime->steering.attn_scale : 0.0f;
    if (session->runtime && session->runtime->cuda)
        q38_forward_cuda_set_directional_steering_scales(
            session->runtime->cuda, session->steering_ffn_scale,
            session->steering_attn_scale);
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
    return q38_session_eval_timed(
        session, token, logits, logits_stride, next_token, diagnostics,
        trace_kind, emitted_token, consumed_token, trace, trace_user,
        step_index, NULL, NULL, error, error_len);
}

bool q38_session_eval_timed(
    q38_session *session, uint32_t token, float *logits,
    size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace_kind trace_kind,
    uint32_t emitted_token, uint32_t consumed_token,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    q38_decode_timing *timing, q38_ple_scheduler_stats *ple_stats,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!session || !session->runtime || !logits || !next_token ||
        !step_index)
        return fail(error, error_len, "invalid session evaluation arguments");
    if (session->position >= session->ctx_size)
        return fail(error, error_len, "session context is full");
    q38_forward_diagnostics local_diagnostics;
    if (!diagnostics) {
        memset(&local_diagnostics, 0, sizeof(local_diagnostics));
        diagnostics = &local_diagnostics;
    }
    diagnostics->qsa_qkv_backend = session->runtime->backend.qsa_qkv;
    diagnostics->qsa_qkv_backend_user = session->runtime->backend.user;
    diagnostics->qsa_chain_backend = session->runtime->backend.qsa_chain;
    diagnostics->qsa_chain_backend_user = session->runtime->backend.user;
    diagnostics->directional_steering = &session->runtime->steering;
    diagnostics->directional_steering_ffn_scale =
        session->steering_ffn_scale;
    diagnostics->directional_steering_attn_scale =
        session->steering_attn_scale;
    diagnostics->directional_steering_attn_device =
        session->runtime->cuda != NULL &&
        session->runtime->steering.directions != NULL;
    if (diagnostics) {
        diagnostics->backend_context = runtime_backend_context;
        diagnostics->backend_context_user = session->runtime->backend.user;
    }
    const double started = session_now_ms();
    if (!q38_decode_step_with_backend_config_timed(
            session->runtime->model, &session->runtime->weights,
            &session->state, token, logits, logits_stride, next_token,
            diagnostics, &session->runtime->backend,
            trace_kind, emitted_token, consumed_token, (*step_index)++, trace,
            trace_user, timing, error, error_len))
        return false;
    q38_ple_scheduler_stats ple = {0};
    if (q38_forward_state_get_ple_prefetch_stats(&session->state, &ple)) {
        if (ple_stats) *ple_stats = ple;
        if (ple.wait_ms > session->ple_wait_at_injection_ms)
            session->ple_wait_at_injection_ms = ple.wait_ms;
    }
    const bool committed = append_history(session, token, error, error_len);
    (void)started;
    return committed;
}

bool q38_session_prefill_reference(
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

bool q38_session_prefill(
    q38_session *session, const uint32_t *tokens, size_t token_count,
    float *logits, size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, q38_decode_trace trace,
    void *trace_user, size_t *step_index, char *error, size_t error_len) {
    return q38_session_prefill_reference(
        session, tokens, token_count, logits, logits_stride, next_token,
        diagnostics, trace, trace_user, step_index, error, error_len);
}

static bool session_argmax(const float *logits, size_t count,
                           uint32_t *token, char *error, size_t error_len) {
    if (!logits || !count || !token)
        return fail(error, error_len, "invalid prefill logits");
    size_t best = 0;
    float best_value = logits[0];
    if (!isfinite(best_value))
        return fail(error, error_len, "prefill logits contain NaN/Inf");
    for (size_t i = 1; i < count; ++i) {
        if (!isfinite(logits[i]))
            return fail(error, error_len, "prefill logits contain NaN/Inf");
        if (logits[i] > best_value) {
            best = i;
            best_value = logits[i];
        }
    }
    *token = (uint32_t)best;
    return true;
}

bool q38_session_prefill_chunked(
    q38_session *session, const uint32_t *tokens, size_t token_count,
    size_t chunk_size, float *logits, size_t logits_stride,
    uint32_t *next_token, q38_forward_diagnostics *diagnostics,
    q38_decode_trace trace, void *trace_user, size_t *step_index,
    char *error, size_t error_len) {
    (void)trace;
    (void)trace_user;
    if (error && error_len) error[0] = '\0';
    if (!session || !tokens || !token_count || !chunk_size || !logits ||
        logits_stride < Q38_DECODE_VOCAB_SIZE || !next_token || !step_index)
        return fail(error, error_len, "invalid chunked prefill arguments");
    if (token_count > session->ctx_size)
        return fail(error, error_len, "prompt exceeds session context");
    q38_session_reset(session);
    const size_t max_chunk = chunk_size < token_count ? chunk_size : token_count;
    if (max_chunk > SIZE_MAX / Q38_DECODE_VOCAB_SIZE ||
        max_chunk * Q38_DECODE_VOCAB_SIZE > SIZE_MAX / sizeof(float))
        return fail(error, error_len, "chunked prefill logits overflow");
    float *chunk_logits = calloc(
        max_chunk * Q38_DECODE_VOCAB_SIZE, sizeof(float));
    if (!chunk_logits)
        return fail(error, error_len, "chunked prefill logits allocation failed");
    q38_forward_backend_config prefill_backend = session->runtime->backend;
    prefill_backend.decoder_layer_chain = NULL;
    prefill_backend.qsa_chain = NULL;
    prefill_backend.qsa_qkv = NULL;
    if (diagnostics) {
        diagnostics->backend_context = runtime_backend_context;
        diagnostics->backend_context_user = session->runtime->backend.user;
        /*
         * Let the layer-major graph project Q/K/V through the same batched
         * matrix backend as the other resident projections.  The legacy
         * standalone QSA QKV path remains available to decode/reference
         * callers, but its large-token launch is not the chunked path.
         */
        diagnostics->qsa_qkv_backend = NULL;
        diagnostics->qsa_qkv_backend_user = NULL;
        diagnostics->qsa_chain_backend = NULL;
        diagnostics->qsa_chain_backend_user = NULL;
    }
    size_t offset = 0;
    while (offset < token_count) {
        const size_t count = (token_count - offset) < max_chunk
            ? token_count - offset : max_chunk;
        if (!q38_forward_full_with_backend_config(
                session->runtime->model, &session->runtime->weights,
                &session->state, tokens + offset, count, chunk_logits,
                Q38_DECODE_VOCAB_SIZE, diagnostics,
                &prefill_backend, error, error_len)) {
            free(chunk_logits);
            return false;
        }
        for (size_t i = 0; i < count; ++i)
            if (!append_history(session, tokens[offset + i], error,
                                error_len)) {
                free(chunk_logits);
                return false;
            }
        memcpy(logits, chunk_logits + (count - 1) * Q38_DECODE_VOCAB_SIZE,
               Q38_DECODE_VOCAB_SIZE * sizeof(float));
        offset += count;
        (*step_index) += count;
    }
    /*
     * Multi-token prefill uses the host GDN implementation because the
     * single-token CUDA island cannot process a batch.  Seed its persistent
     * device state before the first single-token decode switches to GDN-C3.
     */
    if (max_chunk > 1 &&
        !q38_forward_cuda_load_gdn_state(
            &session->state, session->runtime->cuda, error, error_len)) {
        free(chunk_logits);
        return false;
    }
    if (!q38_forward_cuda_load_qsa_state(
            &session->state, session->runtime->cuda, error, error_len)) {
        free(chunk_logits);
        return false;
    }
    const bool ok = session_argmax(
        logits, Q38_DECODE_VOCAB_SIZE, next_token, error, error_len);
    free(chunk_logits);
    return ok;
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
