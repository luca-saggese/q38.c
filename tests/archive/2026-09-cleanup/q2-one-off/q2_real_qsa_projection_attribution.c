#include "q38_forward_cuda.h"
#include "q38_session.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ATTR_VOCAB = Q38_DECODE_VOCAB_SIZE,
    ATTR_CTX = 8192,
    ATTR_GENERATED = 128,
    ATTR_FIRST_MEASURED = 16,
    ATTR_LAYERS = Q38_MODEL_LAYERS,
};

typedef struct {
    uint64_t qkv_calls;
    uint64_t qkv_fast;
    uint64_t qkv_legacy;
    uint64_t qkv_fallback;
    uint64_t output_fast;
    uint64_t output_legacy;
    double q_projection_ms;
    double k_projection_ms;
    double v_projection_ms;
    double qkv_ms;
    double output_projection_ms;
    double indexer_ms;
} layer_stats;

typedef struct {
    layer_stats layers[ATTR_LAYERS];
    uint64_t callbacks;
    uint64_t dispatches;
    uint64_t syncs;
    uint64_t allocations;
    uint64_t non_ple_upload_bytes;
    uint64_t non_ple_misses;
    uint64_t fallback_events;
    double core_ms;
    double argmax_ms;
    double ple_wait_ms;
    size_t measured_count;
    bool measuring;
    bool use_fast_qkv;
} mode_probe;

static bool projection_trace(
    const q38_forward_qsa_projection_trace *trace, void *opaque,
    char *error, size_t error_len) {
    (void)error;
    (void)error_len;
    mode_probe *probe = opaque;
    if (!probe || !trace || !probe->measuring || trace->layer >= ATTR_LAYERS)
        return true;
    layer_stats *stats = &probe->layers[trace->layer];
    stats->qkv_calls++;
    if (trace->qkv_backend_used) {
        stats->qkv_fast++;
    } else {
        stats->qkv_legacy++;
    }
    if (trace->qkv_fallback) stats->qkv_fallback++;
    if (trace->output_projection_backend_used)
        stats->output_fast++;
    else
        stats->output_legacy++;
    stats->q_projection_ms += trace->q_projection_ms;
    stats->k_projection_ms += trace->k_projection_ms;
    stats->v_projection_ms += trace->v_projection_ms;
    stats->qkv_ms += trace->qkv_projection_ms;
    stats->output_projection_ms += trace->output_projection_ms;
    stats->indexer_ms += trace->indexer_projection_ms;
    return true;
}

static void telemetry(
    const q38_forward_cuda_telemetry *event, void *opaque) {
    mode_probe *probe = opaque;
    if (!probe || !event || !probe->measuring) return;
    probe->callbacks++;
    probe->dispatches++;
    probe->syncs += event->sync_count;
    probe->allocations += event->allocation_count;
    if (!event->ple_file_backed_access) {
        probe->non_ple_upload_bytes += event->upload_bytes;
        if (event->non_ple_residency_miss) probe->non_ple_misses++;
    }
    if (event->fallback_path &&
        strcmp(event->fallback_path, "gguf_host_upload") == 0)
        probe->fallback_events++;
}

static uint64_t hash_logits(const float *logits) {
    const unsigned char *bytes = (const unsigned char *)logits;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < ATTR_VOCAB * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool finite_logits(const float *logits) {
    for (size_t i = 0; i < ATTR_VOCAB; ++i)
        if (!isfinite(logits[i])) return false;
    return true;
}

static uint32_t argmax(const float *logits) {
    size_t best = 0;
    for (size_t i = 1; i < ATTR_VOCAB; ++i)
        if (logits[i] > logits[best]) best = i;
    return (uint32_t)best;
}

static double max_abs_diff(const float *a, const float *b) {
    double result = 0.0;
    for (size_t i = 0; i < ATTR_VOCAB; ++i) {
        const double diff = fabs((double)a[i] - (double)b[i]);
        if (diff > result) result = diff;
    }
    return result;
}

static void init_diagnostics(q38_forward_diagnostics *diagnostics,
                             mode_probe *probe) {
    memset(diagnostics, 0, sizeof(*diagnostics));
    diagnostics->qsa_projection_trace = projection_trace;
    diagnostics->trace_user = probe;
}

static uint64_t sum_qkv_fast(const mode_probe *probe) {
    uint64_t total = 0;
    for (size_t i = 0; i < ATTR_LAYERS; ++i)
        total += probe->layers[i].qkv_fast;
    return total;
}

static uint64_t sum_qkv_legacy(const mode_probe *probe) {
    uint64_t total = 0;
    for (size_t i = 0; i < ATTR_LAYERS; ++i)
        total += probe->layers[i].qkv_legacy;
    return total;
}

static uint64_t sum_qkv_fallback(const mode_probe *probe) {
    uint64_t total = 0;
    for (size_t i = 0; i < ATTR_LAYERS; ++i)
        total += probe->layers[i].qkv_fallback;
    return total;
}

static uint64_t sum_output_fast(const mode_probe *probe) {
    uint64_t total = 0;
    for (size_t i = 0; i < ATTR_LAYERS; ++i)
        total += probe->layers[i].output_fast;
    return total;
}

static uint64_t sum_output_legacy(const mode_probe *probe) {
    uint64_t total = 0;
    for (size_t i = 0; i < ATTR_LAYERS; ++i)
        total += probe->layers[i].output_legacy;
    return total;
}

static bool run_mode(
    q38_runtime *runtime, const q38_token_batch *prompt, bool use_fast_qkv,
    const uint32_t *expected_tokens, const float *baseline_logits,
    float *capture_logits, size_t baseline_stride, mode_probe *probe,
    uint32_t *generated,
    double *max_diff, uint64_t *logits_mismatch_count, char *error,
    size_t error_len) {
    q38_session session = {0};
    float *logits = calloc(ATTR_VOCAB, sizeof(*logits));
    q38_forward_diagnostics diagnostics;
    size_t step_index = 0;
    uint32_t next_token = 0;
    bool ok = false;

    if (!logits ||
        !q38_session_create(&session, runtime, ATTR_CTX, error, error_len))
        goto cleanup;
    memset(&diagnostics, 0, sizeof(diagnostics));
    if (!q38_session_prefill_chunked(
            &session, prompt->tokens, prompt->token_count, 128, logits,
            ATTR_VOCAB, &next_token, &diagnostics, NULL, NULL, &step_index,
            error, error_len))
        goto cleanup;

    diagnostics.qsa_projection_trace = projection_trace;
    diagnostics.trace_user = probe;
    if (use_fast_qkv) {
        diagnostics.qsa_qkv_backend = q38_forward_cuda_qsa_qkv_backend;
        diagnostics.qsa_qkv_backend_user = runtime->cuda;
    }
    q38_forward_cuda_set_telemetry_observer(runtime->cuda, telemetry, probe);
    generated[0] = next_token;
    if (capture_logits)
        memcpy(capture_logits, logits, ATTR_VOCAB * sizeof(*logits));
    if (expected_tokens && generated[0] != expected_tokens[0])
        (*logits_mismatch_count)++;
    if (baseline_logits && max_diff) {
        const double diff = max_abs_diff(logits, baseline_logits);
        if (diff > *max_diff) *max_diff = diff;
    }
    if (!q38_session_emit(&session, logits, next_token, NULL, NULL,
                          &step_index, error, error_len))
        goto cleanup_observer;

    for (size_t index = 1; index < ATTR_GENERATED; ++index) {
        const uint32_t input = expected_tokens
            ? expected_tokens[index - 1] : generated[index - 1];
        q38_decode_timing timing = {0};
        q38_ple_scheduler_stats ple = {0};
        probe->measuring = index >= ATTR_FIRST_MEASURED;
        if (probe->measuring)
            probe->measured_count++;
        if (!q38_session_eval_timed(
                &session, input, logits, ATTR_VOCAB, &next_token,
                &diagnostics, Q38_DECODE_TRACE_GENERATED_CONSUME, next_token,
                input, NULL, NULL, &step_index, &timing, &ple, error,
                error_len))
            goto cleanup_observer;
        if (probe->measuring) {
            probe->core_ms += timing.forward_core_ms;
            probe->argmax_ms += timing.argmax_ms;
            if (ple.wait_ms > probe->ple_wait_ms)
                probe->ple_wait_ms = ple.wait_ms;
        }
        generated[index] = next_token;
        if (capture_logits)
            memcpy(capture_logits + index * baseline_stride, logits,
                   ATTR_VOCAB * sizeof(*logits));
        if (expected_tokens && next_token != expected_tokens[index])
            (*logits_mismatch_count)++;
        if (baseline_logits && max_diff) {
            const double diff = max_abs_diff(
                logits, baseline_logits + index * baseline_stride);
            if (diff > *max_diff) *max_diff = diff;
        }
        if (!expected_tokens && generated[index] ==
                                  q38_session_eos_token(&session))
            break;
    }
    ok = true;

cleanup_observer:
    q38_forward_cuda_set_telemetry_observer(runtime->cuda, NULL, NULL);
cleanup:
    q38_session_destroy(&session);
    free(logits);
    return ok;
}

static void write_mode(FILE *out, const mode_probe *probe,
                       const char *name, const char *qkv_backend) {
    fprintf(out, "\"%s\":{\"qkv_backend\":\"%s\","
            "\"core_forward_mean_ms\":%.6f,\"argmax_mean_ms\":%.6f,"
            "\"q2_fast_qkv_calls_per_token\":%.6f,"
            "\"legacy_qkv_calls_per_token\":%.6f,"
            "\"fallback_qkv_calls_per_token\":%.6f,"
            "\"q2_fast_output_projection_calls_per_token\":%.6f,"
            "\"legacy_output_projection_calls_per_token\":%.6f,"
            "\"non_ple_upload_bytes\":%" PRIu64
            ",\"non_ple_residency_misses\":%" PRIu64
            ",\"fallback_events\":%" PRIu64
            ",\"ple_critical_stall_ms\":%.6f,\"layers\":[",
            name, qkv_backend, probe->core_ms / probe->measured_count,
            probe->argmax_ms / probe->measured_count,
            (double)sum_qkv_fast(probe) / probe->measured_count,
            (double)sum_qkv_legacy(probe) / probe->measured_count,
            (double)sum_qkv_fallback(probe) / probe->measured_count,
            (double)sum_output_fast(probe) / probe->measured_count,
            (double)sum_output_legacy(probe) / probe->measured_count,
            probe->non_ple_upload_bytes, probe->non_ple_misses,
            probe->fallback_events, probe->ple_wait_ms);
    bool first = true;
    for (size_t layer = 0; layer < ATTR_LAYERS; ++layer) {
        const layer_stats *s = &probe->layers[layer];
        if (!s->qkv_calls) continue;
        fprintf(out, "%s{\"layer\":%zu,\"qkv_backend\":\"%s\","
                "\"qkv_ms\":%.6f,\"q_projection_ms\":%.6f,"
                "\"k_projection_ms\":%.6f,\"v_projection_ms\":%.6f,"
                "\"output_projection_backend\":\"%s\","
                "\"output_projection_ms\":%.6f,\"indexer_projection_ms\":%.6f,"
                "\"qkv_calls\":%" PRIu64 ",\"qkv_fast_calls\":%" PRIu64
                ",\"qkv_legacy_calls\":%" PRIu64
                ",\"qkv_fallback_calls\":%" PRIu64 "}",
                first ? "" : ",", layer,
                s->qkv_fast ? qkv_backend : "legacy_qkv",
                s->qkv_ms / probe->measured_count,
                s->q_projection_ms / probe->measured_count,
                s->k_projection_ms / probe->measured_count,
                s->v_projection_ms / probe->measured_count,
                s->output_fast ? "fast_matrix_batch" : "legacy_matrix",
                s->output_projection_ms / probe->measured_count,
                s->indexer_ms / probe->measured_count, s->qkv_calls,
                s->qkv_fast, s->qkv_legacy, s->qkv_fallback);
        first = false;
    }
    fputs("]}", out);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    const char *tokenizer_path = argc > 2 ? argv[2] : "/home/lvx/q38model";
    const char *prompt_text = argc > 3 ? argv[3] :
        "Explain in simple terms why the sky appears blue during the day and red near sunset.";
    const char *artifact_path =
        "artifacts/perf/q2_real_qsa_projection_attribution.json";
    char error[256] = {0};
    q38_runtime runtime = {0};
    q38_token_batch prompt = {0};
    mode_probe production = {0};
    mode_probe forced_fast = {0};
    uint32_t production_tokens[ATTR_GENERATED] = {0};
    uint32_t forced_tokens[ATTR_GENERATED] = {0};
    float *baseline_logits =
        calloc(ATTR_GENERATED * ATTR_VOCAB, sizeof(float));
    double max_diff = 0.0;
    uint64_t logits_mismatch_count = 0;
    bool ok = false;

    if (!baseline_logits ||
        !q38_runtime_init(&runtime, model_path, tokenizer_path, error,
                          sizeof(error)) ||
        !q38_tokenizer_encode(&runtime.tokenizer, prompt_text, false, &prompt,
                              error, sizeof(error))) {
        fprintf(stderr, "attribution setup failed: %s\n",
                error[0] ? error : "allocation failure");
        goto cleanup;
    }

    /*
     * The first pass is the exact current q38_session_eval dispatch.
     * The session installs both the Q2 QKV and matrix-batch backends.
     */
    if (!run_mode(&runtime, &prompt, false, NULL, NULL, baseline_logits,
                  ATTR_VOCAB, &production, production_tokens, NULL,
                  &logits_mismatch_count, error, sizeof(error))) {
        fprintf(stderr, "production attribution failed: %s\n", error);
        goto cleanup;
    }

    /*
     * Re-run the same prompt and token inputs with the Q2 QKV backend
     * explicitly installed.  This is a dispatch-equivalence control.
     */
    if (!run_mode(&runtime, &prompt, true, production_tokens, baseline_logits,
                  NULL, ATTR_VOCAB, &forced_fast, forced_tokens, &max_diff,
                  &logits_mismatch_count, error, sizeof(error))) {
        fprintf(stderr, "forced QKV attribution failed: %s\n", error);
        goto cleanup;
    }

    {
        FILE *out = fopen(artifact_path, "w");
        if (!out) {
            fprintf(stderr, "cannot open %s\n", artifact_path);
            goto cleanup;
        }
        fprintf(out, "{\"format\":\"q2-real-qsa-projection-attribution-v1\","
                "\"model\":\"%s\",\"prompt_tokens\":%u,"
                "\"generated_tokens\":%u,\"measured_first\":%u,"
                "\"measured_last\":%u,\"measured_count\":%u,"
                "\"production\":{",
                model_path, prompt.token_count, ATTR_GENERATED,
                ATTR_FIRST_MEASURED, ATTR_GENERATED - 1,
                ATTR_GENERATED - ATTR_FIRST_MEASURED);
        write_mode(out, &production, "session_eval_current",
                   "q38_forward_cuda_qsa_qkv_backend");
        fputs("},\"forced_qkv_backend\":{", out);
        write_mode(out, &forced_fast, "session_eval_qkv_forced",
                   "q38_forward_cuda_qsa_qkv_backend");
        fprintf(out, "},\"invariants\":{\"same_generated_ids\":%s,"
                "\"logits_mismatch_count\":%" PRIu64
                ",\"max_abs_logits_diff\":%.9g,"
                "\"production_non_ple_upload_bytes\":%" PRIu64
                ",\"forced_non_ple_upload_bytes\":%" PRIu64
                ",\"production_residency_misses\":%" PRIu64
                ",\"forced_residency_misses\":%" PRIu64
                ",\"production_fallback_events\":%" PRIu64
                ",\"forced_fallback_events\":%" PRIu64
                ",\"production_ple_critical_stall_ms\":%.6f"
                ",\"forced_ple_critical_stall_ms\":%.6f},"
                "\"dispatch_conclusion\":\"current q38_session_eval "
                "installs qsa_qkv_backend and matrix_batch_backend\"}\n",
                memcmp(production_tokens, forced_tokens,
                       sizeof(production_tokens)) == 0 ? "true" : "false",
                logits_mismatch_count, max_diff,
                production.non_ple_upload_bytes, forced_fast.non_ple_upload_bytes,
                production.non_ple_misses, forced_fast.non_ple_misses,
                production.fallback_events, forced_fast.fallback_events,
                production.ple_wait_ms, forced_fast.ple_wait_ms);
        fclose(out);
    }
    printf("artifact: %s\n", artifact_path);
    printf("production QKV calls/token: fast=%.3f legacy=%.3f fallback=%.3f\n",
           (double)sum_qkv_fast(&production) / production.measured_count,
           (double)sum_qkv_legacy(&production) / production.measured_count,
           (double)sum_qkv_fallback(&production) / production.measured_count);
    printf("forced QKV calls/token:     fast=%.3f legacy=%.3f fallback=%.3f\n",
           (double)sum_qkv_fast(&forced_fast) / forced_fast.measured_count,
           (double)sum_qkv_legacy(&forced_fast) / forced_fast.measured_count,
           (double)sum_qkv_fallback(&forced_fast) / forced_fast.measured_count);
    printf("production output calls/token: fast=%.3f legacy=%.3f\n",
           (double)sum_output_fast(&production) / production.measured_count,
           (double)sum_output_legacy(&production) / production.measured_count);
    printf("QKV-forced max abs logits diff: %.9g\n", max_diff);
    ok = true;

cleanup:
    q38_forward_cuda_set_telemetry_observer(runtime.cuda, NULL, NULL);
    q38_token_batch_free(&prompt);
    q38_runtime_destroy(&runtime);
    free(baseline_logits);
    return ok ? 0 : 1;
}
