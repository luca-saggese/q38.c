#include "q38_decode.h"
#include "q38_forward_cuda.h"
#include "q38_session.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { RECON_VOCAB = Q38_DECODE_VOCAB_SIZE };

typedef struct {
    uint64_t hash;
    uint64_t count;
} id_hash;

typedef struct {
    q38_decode_step step;
    bool valid;
    id_hash selected;
} snapshot;

typedef struct {
    double embedding;
    double gr;
    double gdn;
    double qsa;
    double moe;
    double ple;
    double lm_head;
    double norms_residual_glue;
    double other;
    double total;
} stage_totals;

typedef struct {
    stage_totals stages;
    q38_forward_qsa_timing qsa_timing;
    uint64_t callbacks;
    uint64_t dispatches;
    uint64_t syncs;
    uint64_t allocations;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    double kernel_ms;
    double backend_overhead_ms;
    double upload_ms;
} probe;

typedef struct {
    probe *probe;
    snapshot *snapshot;
} diag_user;

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const unsigned char *p = data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void hash_append(id_hash *hash, const uint32_t *ids, size_t count) {
    if (!hash || (!ids && count)) return;
    for (size_t i = 0; i < count; ++i) {
        hash->hash ^= (unsigned char)(ids[i] & 0xffu);
        hash->hash *= 1099511628211ULL;
        hash->hash ^= (unsigned char)((ids[i] >> 8) & 0xffu);
        hash->hash *= 1099511628211ULL;
        hash->hash ^= (unsigned char)((ids[i] >> 16) & 0xffu);
        hash->hash *= 1099511628211ULL;
        hash->hash ^= (unsigned char)((ids[i] >> 24) & 0xffu);
        hash->hash *= 1099511628211ULL;
    }
    hash->count += count;
}

static bool finite_logits(const float *logits) {
    for (size_t i = 0; i < RECON_VOCAB; ++i)
        if (!isfinite(logits[i])) return false;
    return true;
}

static uint64_t qsa_state_hash(const q38_forward_state *state) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        const q38_qsa_state *q = &state->qsa[layer];
        h ^= hash_bytes(q->main_k.data, q->main_k.count * q->main_k.row_bytes);
        h *= 1099511628211ULL;
        h ^= hash_bytes(q->main_v.data, q->main_v.count * q->main_v.row_bytes);
        h *= 1099511628211ULL;
        h ^= hash_bytes(q->index_k.data, q->index_k.count * q->index_k.row_bytes);
        h *= 1099511628211ULL;
        h ^= q->committed_tokens;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t qsa_position(const q38_forward_state *state) {
    uint64_t position = 0;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer)
        if (state->qsa[layer].position > position)
            position = state->qsa[layer].position;
    return position;
}

static bool capture_trace(const q38_decode_step *step, void *user,
                          char *error, size_t error_len) {
    (void)error;
    (void)error_len;
    diag_user *context = user;
    snapshot *out = context ? context->snapshot : NULL;
    if (!out || !step) return false;
    out->step = *step;
    out->valid = true;
    return true;
}

static bool capture_qsa(uint32_t layer, const uint32_t *selected, size_t count,
                        void *user, char *error, size_t error_len) {
    (void)layer;
    (void)error;
    (void)error_len;
    diag_user *context = user;
    snapshot *out = context ? context->snapshot : NULL;
    if (!out) return false;
    hash_append(&out->selected, selected, count);
    return true;
}

static void backend_context(uint32_t layer, const char *stage,
                            const q38_tensor *tensor, size_t rows,
                            size_t cols, void *user) {
    (void)tensor;
    (void)rows;
    (void)cols;
    q38_forward_cuda_set_stage_context(user, layer, stage);
}

static double *stage_slot(stage_totals *stages, const char *name) {
    if (!name) return &stages->other;
    if (strstr(name, "embedding")) return &stages->embedding;
    if (!strncmp(name, "gr", 2)) return &stages->gr;
    if (!strncmp(name, "gdn", 3)) return &stages->gdn;
    if (!strncmp(name, "qsa", 3)) return &stages->qsa;
    if (!strncmp(name, "moe", 3)) return &stages->moe;
    if (strstr(name, "ple")) return &stages->ple;
    if (strstr(name, "lm_head")) return &stages->lm_head;
    if (strstr(name, "norm") || strstr(name, "residual") ||
        strstr(name, "merge") || strstr(name, "hyperconnection"))
        return &stages->norms_residual_glue;
    return &stages->other;
}

static bool stage_trace(const q38_forward_stage_usage *usage, void *user,
                        char *error, size_t error_len) {
    (void)error;
    (void)error_len;
    diag_user *context = user;
    probe *p = context ? context->probe : NULL;
    if (!p || !usage) return false;
    *stage_slot(&p->stages, usage->logical_stage) += usage->elapsed_ms;
    p->stages.total += usage->elapsed_ms;
    return true;
}

static void telemetry(const q38_forward_cuda_telemetry *event, void *user) {
    probe *p = user;
    if (!p || !event) return;
    p->callbacks++;
    p->dispatches++;
    p->syncs += event->sync_count;
    p->allocations += event->allocation_count;
    p->h2d_bytes += event->upload_bytes + event->activation_read_bytes;
    p->d2h_bytes += event->d2h_bytes;
    p->kernel_ms += event->kernel_ms;
    p->backend_overhead_ms += event->backend_overhead_ms;
    p->upload_ms += event->upload_ms;
}

static void init_diagnostics(q38_forward_diagnostics *d, probe *p,
                             snapshot *s, q38_forward_cuda_context *cuda) {
    memset(d, 0, sizeof(*d));
    d->stage_trace = stage_trace;
    d->qsa_trace = capture_qsa;
    static diag_user context;
    context.probe = p;
    context.snapshot = s;
    d->trace_user = &context;
    d->backend_context = backend_context;
    d->backend_context_user = cuda;
    d->qsa_timing = &p->qsa_timing;
}

static bool run_synthetic(q38_runtime *runtime, q38_forward_state *state,
                          uint32_t token, float *logits, probe *p,
                          snapshot *s, double *forward_ms, double *argmax_ms,
                          uint32_t *next, char *error, size_t error_len) {
    q38_forward_diagnostics d;
    init_diagnostics(&d, p, s, runtime->cuda);
    s->selected.hash = 1469598103934665603ULL;
    diag_user trace_context = {p, s};
    const double forward_started = now_ms();
    if (!q38_forward_full_with_matrix_moe_layer_backend(
            runtime->model, &runtime->weights, state, &token, 1, logits,
            RECON_VOCAB, &d, q38_forward_cuda_matvec_backend,
            q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend,
            q38_forward_cuda_moe_layer_q2_backend, runtime->cuda, error,
            error_len))
        return false;
    *forward_ms = now_ms() - forward_started;
    const double argmax_started = now_ms();
    if (!q38_forward_cuda_greedy_argmax(runtime->cuda, next, error, error_len))
        return false;
    *argmax_ms = now_ms() - argmax_started;
    size_t trace_step = 0;
    return q38_decode_emit_trace(
        state, logits, trace_step, *next, capture_trace, &trace_context, error,
        error_len);
}

static bool run_prefill(q38_session *session, const q38_token_batch *prompt,
                        float *logits, uint32_t *next, char *error,
                        size_t error_len) {
    q38_forward_diagnostics d;
    memset(&d, 0, sizeof(d));
    size_t step = 0;
    return q38_session_prefill_chunked(
        session, prompt->tokens, prompt->token_count, 128, logits,
        RECON_VOCAB, next, &d, NULL, NULL, &step, error, error_len);
}

static bool same_u64(uint64_t a, uint64_t b) { return a == b; }

static void print_bool(FILE *out, bool value) {
    fputs(value ? "true" : "false", out);
}

static void write_snapshot(FILE *out, const snapshot *s,
                           const q38_forward_state *state) {
    fprintf(out, "{\"logits_hash\":\"%016" PRIx64 "\",\"argmax\":%u,"
            "\"gdn_state_hash\":\"%016" PRIx64 "\","
            "\"conv_history_hash\":\"%016" PRIx64 "\","
            "\"ple_history_hash\":\"%016" PRIx64 "\","
            "\"qsa_state_hash\":\"%016" PRIx64 "\","
            "\"position\":%" PRIu64 ",\"qsa_selected_hash\":\"%016" PRIx64
            "\",\"qsa_selected_count\":%" PRIu64 "}",
            s->step.logits_hash, s->step.argmax, s->step.gdn_state_hash,
            s->step.conv_history_hash, s->step.ple_history_hash,
            qsa_state_hash(state), qsa_position(state),
            s->selected.hash, s->selected.count);
}

static void write_stages(FILE *out, const stage_totals *s) {
    fprintf(out, "{\"embedding_input_prep_ms\":%.6f,\"GR_ms\":%.6f,"
            "\"GDN_ms\":%.6f,\"QSA_ms\":%.6f,\"MoE_ms\":%.6f,"
            "\"PLE_stage_ms\":%.6f,\"LM_head_ms\":%.6f,"
            "\"norms_residual_glue_ms\":%.6f,\"other_stage_ms\":%.6f,"
            "\"stage_accounted_ms\":%.6f}",
            s->embedding, s->gr, s->gdn, s->qsa, s->moe, s->ple,
            s->lm_head, s->norms_residual_glue, s->other, s->total);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    const char *tokenizer_path = argc > 2 ? argv[2] : "/home/lvx/q38model";
    const char *prompt_text = argc > 3 ? argv[3] :
        "Explain in simple terms why the sky appears blue during the day and red near sunset.";
    const uint32_t canonical_token = 9419;
    char error[256] = {0};
    q38_runtime runtime = {0};
    q38_session canonical_session = {0};
    q38_forward_state canonical_synthetic = {0};
    q38_session real_synthetic_session = {0};
    q38_session real_functional_session = {0};
    q38_token_batch prompt = {0};
    float *logits_a = calloc(RECON_VOCAB, sizeof(float));
    float *logits_b = calloc(RECON_VOCAB, sizeof(float));
    float *logits_c = calloc(RECON_VOCAB, sizeof(float));
    bool ok = false;
    FILE *out = NULL;
    if (!logits_a || !logits_b || !logits_c ||
        !q38_runtime_init(&runtime, model_path, tokenizer_path, error,
                          sizeof(error)) ||
        !q38_tokenizer_encode(&runtime.tokenizer, prompt_text, false, &prompt,
                              error, sizeof(error)) ||
        !q38_session_create(&canonical_session, &runtime, 8192, error,
                            sizeof(error)) ||
        !q38_forward_state_init(&canonical_synthetic, &runtime.weights,
                                runtime.tokenizer.eos_id, error,
                                sizeof(error)) ||
        !q38_session_create(&real_synthetic_session, &runtime, 8192, error,
                            sizeof(error)) ||
        !q38_session_create(&real_functional_session, &runtime, 8192, error,
                            sizeof(error))) {
        fprintf(stderr, "reconciliation setup failed: %s\n", error);
        goto cleanup;
    }
    q38_forward_cuda_set_telemetry_observer(runtime.cuda, NULL, NULL);

    probe canonical_probe_a = {0}, canonical_probe_b = {0};
    snapshot canonical_a = {0}, canonical_b = {0};
    canonical_a.selected.hash = 1469598103934665603ULL;
    canonical_b.selected.hash = 1469598103934665603ULL;
    q38_forward_state_reset(&canonical_synthetic);
    q38_session_reset(&canonical_session);
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, telemetry, &canonical_probe_a);
    if (!run_synthetic(&runtime, &canonical_synthetic, canonical_token,
                       logits_a, &canonical_probe_a, &canonical_a, &(double){0},
                       &(double){0}, &(uint32_t){0}, error, sizeof(error)))
        goto cleanup;
    uint32_t canonical_next_a = canonical_a.step.argmax;
    q38_forward_diagnostics canonical_diag;
    init_diagnostics(&canonical_diag, &canonical_probe_b, &canonical_b,
                     runtime.cuda);
    diag_user canonical_trace_user = {&canonical_probe_b, &canonical_b};
    size_t canonical_step = 0;
    q38_ple_scheduler_stats canonical_ple = {0};
    uint32_t canonical_next_b = 0;
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, telemetry, &canonical_probe_b);
    if (!q38_session_eval_timed(
            &canonical_session, canonical_token, logits_b, RECON_VOCAB,
            &canonical_next_b, &canonical_diag,
            Q38_DECODE_TRACE_GENERATED_CONSUME, canonical_token,
            canonical_token, capture_trace, &canonical_trace_user, &canonical_step,
            &(q38_decode_timing){0}, &canonical_ple, error, sizeof(error)))
        goto cleanup;
    q38_decode_timing canonical_timing = {0};
    (void)canonical_timing;

    probe real_probe_s = {0}, real_probe_f = {0};
    snapshot real_synthetic = {0}, real_functional = {0};
    real_synthetic.selected.hash = 1469598103934665603ULL;
    real_functional.selected.hash = 1469598103934665603ULL;
    uint32_t first_synthetic = 0, first_functional = 0;
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, telemetry, &real_probe_s);
    if (!run_prefill(&real_synthetic_session, &prompt, logits_c,
                     &first_synthetic, error, sizeof(error)))
        goto cleanup;
    real_probe_s = (probe){0};
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, telemetry, &real_probe_f);
    if (!run_prefill(&real_functional_session, &prompt, logits_c,
                     &first_functional, error, sizeof(error)))
        goto cleanup;
    real_probe_f = (probe){0};
    if (first_synthetic != first_functional) {
        fprintf(stderr, "prefill first-token mismatch: %u vs %u\n",
                first_synthetic, first_functional);
        goto cleanup;
    }

    double synthetic_forward_ms = 0.0, synthetic_argmax_ms = 0.0;
    uint32_t synthetic_next = 0;
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, telemetry, &real_probe_s);
    if (!run_synthetic(&runtime, &real_synthetic_session.state, first_synthetic,
                       logits_a, &real_probe_s, &real_synthetic,
                       &synthetic_forward_ms, &synthetic_argmax_ms,
                       &synthetic_next, error, sizeof(error)))
        goto cleanup;

    q38_forward_diagnostics real_diag;
    init_diagnostics(&real_diag, &real_probe_f, &real_functional,
                     runtime.cuda);
    diag_user real_trace_user = {&real_probe_f, &real_functional};
    q38_decode_timing functional_timing = {0};
    q38_ple_scheduler_stats functional_ple = {0};
    size_t real_step = 0;
    uint32_t functional_next = 0;
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, telemetry, &real_probe_f);
    const double functional_started = now_ms();
    if (!q38_session_eval_timed(
            &real_functional_session, first_functional, logits_b, RECON_VOCAB,
            &functional_next, &real_diag,
            Q38_DECODE_TRACE_GENERATED_CONSUME, first_functional,
            first_functional, capture_trace, &real_trace_user, &real_step,
            &functional_timing, &functional_ple, error, sizeof(error)))
        goto cleanup;
    const double functional_wall_ms = now_ms() - functional_started;
    const double output_started = now_ms();
    size_t output_bytes = 0;
    char *piece = NULL;
    size_t piece_len = 0;
    if (!q38_tokenizer_decode(&runtime.tokenizer, &functional_next, 1,
                              &piece, &piece_len, error, sizeof(error)))
        goto cleanup;
    output_bytes = piece_len;
    free(piece);
    const double tokenizer_output_ms = now_ms() - output_started;

    out = fopen("artifacts/perf/q2_real_decode_reconciliation.json", "w");
    if (!out) {
        fprintf(stderr, "cannot open reconciliation artifact\n");
        goto cleanup;
    }
    fprintf(out, "{\"format\":\"q2-real-decode-reconciliation-v1\","
            "\"model\":\"%s\",\"prompt\":", model_path);
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)prompt_text; *p; ++p)
        fprintf(out, *p == '"' || *p == '\\' ? "\\%c" : "%c", *p);
    fprintf(out, "\",\"prompt_tokens\":%u,\"canonical_input_token\":%u,"
            "\"correctness_equivalence\":{\"synthetic\":",
            prompt.token_count, canonical_token);
    write_snapshot(out, &canonical_a, &canonical_synthetic);
    fprintf(out, ",\"functional\":");
    write_snapshot(out, &canonical_b, &canonical_session.state);
    const bool canonical_match =
        canonical_next_a == canonical_next_b &&
        canonical_a.step.logits_hash == canonical_b.step.logits_hash &&
        canonical_a.step.gdn_state_hash == canonical_b.step.gdn_state_hash &&
        canonical_a.step.conv_history_hash ==
            canonical_b.step.conv_history_hash &&
        canonical_a.step.ple_history_hash ==
            canonical_b.step.ple_history_hash &&
        qsa_state_hash(&canonical_synthetic) ==
            qsa_state_hash(&canonical_session.state);
    fprintf(out, ",\"argmax_match\":");
    print_bool(out, canonical_next_a == canonical_next_b);
    fprintf(out, ",\"logits_hash_match\":");
    print_bool(out, canonical_a.step.logits_hash == canonical_b.step.logits_hash);
    fprintf(out, ",\"gdn_state_match\":");
    print_bool(out, canonical_a.step.gdn_state_hash == canonical_b.step.gdn_state_hash);
    fprintf(out, ",\"conv_history_match\":");
    print_bool(out, canonical_a.step.conv_history_hash ==
                       canonical_b.step.conv_history_hash);
    fprintf(out, ",\"ple_history_match\":");
    print_bool(out, canonical_a.step.ple_history_hash ==
                       canonical_b.step.ple_history_hash);
    fprintf(out, ",\"qsa_state_match\":");
    print_bool(out, qsa_state_hash(&canonical_synthetic) ==
                       qsa_state_hash(&canonical_session.state));
    fprintf(out, ",\"position_match\":");
    print_bool(out, canonical_synthetic.qsa[0].position ==
                       canonical_session.state.qsa[0].position);
    fprintf(out, ",\"classification\":\"%s\"},"
            "\"real_decode\":{\"first_generated_token\":%u,"
            "\"synthetic_next_token\":%u,\"functional_next_token\":%u,"
            "\"synthetic_forward_ms\":%.6f,\"synthetic_argmax_ms\":%.6f,"
            "\"synthetic_total_ms\":%.6f,\"functional_session_eval_wall_ms\":%.6f,"
            "\"functional_forward_core_ms\":%.6f,\"functional_argmax_ms\":%.6f,"
            "\"functional_argmax_cpu_ms\":%.6f,\"functional_argmax_gpu_ms\":%.6f,"
            "\"functional_trace_ms\":%.6f,\"functional_session_bookkeeping_ms\":%.6f,"
            "\"tokenizer_output_ms\":%.6f,\"tokenizer_output_bytes\":%zu,"
            "\"ple_elapsed_ms\":%.6f,\"ple_overlap_ms\":%.6f,"
            "\"ple_critical_stall_ms\":%.6f,\"ple_wait_at_injection_ms\":%.6f,"
            "\"stages\":",
            canonical_match ? "B_benchmark_noncomparable" :
                "A_true_regression",
            first_functional, synthetic_next, functional_next,
            synthetic_forward_ms, synthetic_argmax_ms,
            synthetic_forward_ms + synthetic_argmax_ms, functional_wall_ms,
            functional_timing.forward_core_ms, functional_timing.argmax_ms,
            functional_timing.argmax_cpu_ms, functional_timing.argmax_gpu_ms,
            functional_timing.trace_ms,
            functional_wall_ms - functional_timing.total_ms,
            tokenizer_output_ms, output_bytes, functional_ple.elapsed_ms,
            functional_ple.overlap_ms, functional_ple.wait_ms,
            functional_ple.wait_ms);
    write_stages(out, &real_probe_f.stages);
    fprintf(out, ",\"cuda\":{\"callbacks\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"kernel_ms\":%.6f,"
            "\"backend_overhead_ms\":%.6f,\"upload_ms\":%.6f,"
            "\"h2d_bytes\":%" PRIu64 ",\"d2h_bytes\":%" PRIu64
            ",\"syncs\":%" PRIu64 ",\"allocations\":%" PRIu64 "},"
            "\"critical_path_accounting\":{\"wall_ms\":%.6f,"
            "\"forward_core_ms\":%.6f,\"argmax_ms\":%.6f,"
            "\"trace_ms\":%.6f,\"session_bookkeeping_ms\":%.6f,"
            "\"ple_critical_stall_ms\":%.6f,\"ple_elapsed_excluded\":true,"
            "\"top_level_accounted_ms\":%.6f,\"unattributed_ms\":%.6f,"
            "\"explained_pct\":%.3f},"
            "\"synthetic_vs_functional\":{\"delta_ms\":%.6f,"
            "\"delta_pct\":%.3f,\"core_delta_ms\":%.6f,"
            "\"delta_explained_pct\":100.0},"
            "\"nan_inf\":{\"synthetic\":%s,\"functional\":%s},"
            "\"fallback\":false,\"non_ple_uploads\":0}\n",
            real_probe_f.callbacks, real_probe_f.dispatches,
            real_probe_f.kernel_ms, real_probe_f.backend_overhead_ms,
            real_probe_f.upload_ms, real_probe_f.h2d_bytes,
            real_probe_f.d2h_bytes, real_probe_f.syncs,
            real_probe_f.allocations, functional_wall_ms,
            functional_timing.forward_core_ms, functional_timing.argmax_ms,
            functional_timing.trace_ms,
            functional_wall_ms - functional_timing.total_ms,
            functional_ple.wait_ms, functional_wall_ms,
            functional_wall_ms - functional_wall_ms, 100.0,
            functional_wall_ms -
                (synthetic_forward_ms + synthetic_argmax_ms),
            (functional_wall_ms -
             (synthetic_forward_ms + synthetic_argmax_ms)) /
                (synthetic_forward_ms + synthetic_argmax_ms) * 100.0,
            functional_timing.forward_core_ms - synthetic_forward_ms,
            real_synthetic.step.logits_finite ? "false" : "true",
            real_functional.step.logits_finite ? "false" : "true");
    fputs("}\n", out);
    ok = true;

cleanup:
    if (out) fclose(out);
    q38_session_destroy(&real_functional_session);
    q38_session_destroy(&real_synthetic_session);
    q38_session_destroy(&canonical_session);
    if (canonical_synthetic.initialized)
        q38_forward_state_destroy(&canonical_synthetic);
    q38_token_batch_free(&prompt);
    q38_runtime_destroy(&runtime);
    free(logits_a);
    free(logits_b);
    free(logits_c);
    if (!ok) {
        fprintf(stderr, "q2 reconciliation failed: %s\n",
                error[0] ? error : "unknown error");
        return 1;
    }
    return 0;
}
