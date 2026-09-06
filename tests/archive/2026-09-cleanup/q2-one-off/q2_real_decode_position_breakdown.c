#include "q38_forward_cuda.h"
#include "q38_session.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    BREAKDOWN_VOCAB = Q38_DECODE_VOCAB_SIZE,
    BREAKDOWN_CTX = 8192,
    BREAKDOWN_GENERATED = 128,
    BREAKDOWN_FIRST_MEASURED = 16,
};

typedef struct {
    double embedding;
    double gr;
    double gdn;
    double qsa;
    double qsa_projection;
    double qsa_attention;
    double qsa_other;
    double moe;
    double ple;
    double lm_head;
    double norms_residual;
    double other;
    double total;
} stage_totals;

typedef struct {
    stage_totals stages;
    q38_forward_qsa_timing qsa;
    uint64_t callbacks;
    uint64_t dispatches;
    uint64_t syncs;
    uint64_t allocations;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    double kernel_ms;
    double backend_overhead_ms;
    double upload_ms;
    size_t current_token;
    bool measuring;
} breakdown_probe;

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static bool stage_is(const char *name, const char *prefix) {
    return name && prefix && strncmp(name, prefix, strlen(prefix)) == 0;
}

static double *stage_slot(stage_totals *stages, const char *name) {
    if (!name) return &stages->other;
    if (strstr(name, "embedding")) return &stages->embedding;
    if (stage_is(name, "gr")) return &stages->gr;
    if (stage_is(name, "gdn")) return &stages->gdn;
    if (stage_is(name, "qsa")) return &stages->qsa;
    if (stage_is(name, "moe")) return &stages->moe;
    if (strstr(name, "ple")) return &stages->ple;
    if (strstr(name, "lm_head")) return &stages->lm_head;
    if (strstr(name, "norm") || strstr(name, "residual") ||
        strstr(name, "merge") || strstr(name, "hyperconnection"))
        return &stages->norms_residual;
    return &stages->other;
}

static bool stage_trace(const q38_forward_stage_usage *usage, void *opaque,
                        char *error, size_t error_len) {
    (void)error;
    (void)error_len;
    breakdown_probe *probe = opaque;
    if (!probe || !usage) return false;
    if (!probe->measuring) return true;
    *stage_slot(&probe->stages, usage->logical_stage) += usage->elapsed_ms;
    if (stage_is(usage->logical_stage, "qsa")) {
        if (stage_is(usage->logical_stage, "qsa_qkv"))
            probe->stages.qsa_projection += usage->elapsed_ms;
        else if (stage_is(usage->logical_stage, "qsa_attention"))
            probe->stages.qsa_attention += usage->elapsed_ms;
        else
            probe->stages.qsa_other += usage->elapsed_ms;
    }
    probe->stages.total += usage->elapsed_ms;
    return true;
}

static void backend_context(uint32_t layer, const char *stage,
                            const q38_tensor *tensor, size_t rows,
                            size_t cols, void *opaque) {
    (void)layer;
    (void)tensor;
    (void)rows;
    (void)cols;
    q38_forward_cuda_set_stage_context(opaque, layer, stage);
}

static void telemetry(const q38_forward_cuda_telemetry *event, void *opaque) {
    breakdown_probe *probe = opaque;
    if (!probe || !event || !probe->measuring) return;
    probe->callbacks++;
    probe->dispatches++;
    probe->syncs += event->sync_count;
    probe->allocations += event->allocation_count;
    probe->h2d_bytes += event->upload_bytes + event->activation_read_bytes;
    probe->d2h_bytes += event->d2h_bytes;
    probe->kernel_ms += event->kernel_ms;
    probe->backend_overhead_ms += event->backend_overhead_ms;
    probe->upload_ms += event->upload_ms;
}

static void init_diagnostics(q38_forward_diagnostics *diagnostics,
                             breakdown_probe *probe,
                             q38_forward_cuda_context *cuda) {
    memset(diagnostics, 0, sizeof(*diagnostics));
    diagnostics->stage_trace = stage_trace;
    diagnostics->backend_context = backend_context;
    diagnostics->backend_context_user = cuda;
    diagnostics->trace_user = probe;
    diagnostics->qsa_timing = &probe->qsa;
}

static void print_stage_json(FILE *out, const stage_totals *stages) {
    fprintf(out, "\"stages\":{\"embedding_input_prep_ms\":%.6f,"
            "\"GR_ms\":%.6f,\"GDN_ms\":%.6f,\"QSA_ms\":%.6f,"
            "\"MoE_ms\":%.6f,\"PLE_stage_ms\":%.6f,"
            "\"LM_head_ms\":%.6f,\"norms_residual_ms\":%.6f,"
            "\"other_ms\":%.6f,\"stage_accounted_ms\":%.6f}",
           stages->embedding, stages->gr, stages->gdn, stages->qsa,
           stages->moe, stages->ple, stages->lm_head,
           stages->norms_residual, stages->other, stages->total);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    const char *tokenizer_path = argc > 2 ? argv[2] : "/home/lvx/q38model";
    const char *prompt_text = argc > 3 ? argv[3] :
        "Explain in simple terms why the sky appears blue during the day and red near sunset.";
    const char *artifact_path = "artifacts/perf/q2_real_decode_position_breakdown.json";
    char error[256] = {0};
    q38_runtime runtime = {0};
    q38_session session = {0};
    q38_token_batch prompt = {0};
    float *logits = NULL;
    uint32_t generated[BREAKDOWN_GENERATED] = {0};
    breakdown_probe probe = {0};
    q38_forward_diagnostics prefill_diagnostics;
    q38_forward_diagnostics decode_diagnostics;
    size_t step_index = 0;
    uint32_t next_token = 0;
    size_t generated_count = 0;
    size_t measured_count = 0;
    double core_ms = 0.0;
    double argmax_ms = 0.0;
    double ple_stall_ms = 0.0;
    bool ok = false;

    logits = calloc(BREAKDOWN_VOCAB, sizeof(*logits));
    if (!logits ||
        !q38_runtime_init(&runtime, model_path, tokenizer_path, error,
                          sizeof(error)) ||
        !q38_tokenizer_encode(&runtime.tokenizer, prompt_text, false, &prompt,
                              error, sizeof(error)) ||
        !q38_session_create(&session, &runtime, BREAKDOWN_CTX, error,
                            sizeof(error))) {
        fprintf(stderr, "breakdown setup failed: %s\n",
                error[0] ? error : "allocation failure");
        goto cleanup;
    }

    memset(&prefill_diagnostics, 0, sizeof(prefill_diagnostics));
    if (!q38_session_prefill_chunked(
            &session, prompt.tokens, prompt.token_count, 128, logits,
            BREAKDOWN_VOCAB, &next_token, &prefill_diagnostics, NULL, NULL,
            &step_index, error, sizeof(error))) {
        fprintf(stderr, "breakdown prefill failed: %s\n", error);
        goto cleanup;
    }

    generated[0] = next_token;
    generated_count = 1;
    if (!q38_session_emit(&session, logits, next_token, NULL, NULL,
                          &step_index, error, sizeof(error))) {
        fprintf(stderr, "breakdown first emission failed: %s\n", error);
        goto cleanup;
    }

    init_diagnostics(&decode_diagnostics, &probe, runtime.cuda);
    q38_forward_cuda_set_telemetry_observer(runtime.cuda, telemetry, &probe);
    while (generated_count < BREAKDOWN_GENERATED &&
           generated[generated_count - 1] !=
               q38_session_eos_token(&session)) {
        const size_t token_index = generated_count;
        const uint32_t input = generated[token_index - 1];
        q38_decode_timing timing = {0};
        q38_ple_scheduler_stats ple = {0};

        probe.current_token = token_index;
        probe.measuring = token_index >= BREAKDOWN_FIRST_MEASURED;
        if (probe.measuring) {
            memset(&probe.qsa, 0, sizeof(probe.qsa));
            measured_count++;
        }
        if (!q38_session_eval_timed(
                &session, input, logits, BREAKDOWN_VOCAB, &next_token,
                &decode_diagnostics, Q38_DECODE_TRACE_GENERATED_CONSUME,
                next_token, input, NULL, NULL, &step_index, &timing, &ple,
                error, sizeof(error))) {
            fprintf(stderr, "breakdown decode failed at token %zu: %s\n",
                    token_index, error);
            goto cleanup;
        }
        if (probe.measuring) {
            core_ms += timing.forward_core_ms;
            argmax_ms += timing.argmax_ms;
            if (ple.wait_ms > ple_stall_ms) ple_stall_ms = ple.wait_ms;
        }
        generated[generated_count++] = next_token;
    }
    q38_forward_cuda_set_telemetry_observer(runtime.cuda, NULL, NULL);
    if (!measured_count) {
        fprintf(stderr, "breakdown produced no measured decode tokens\n");
        goto cleanup;
    }

    {
        FILE *out = fopen(artifact_path, "w");
        const double mean_core = core_ms / measured_count;
        const double critical_stage_ms =
            (probe.stages.total - probe.stages.ple) / measured_count;
        const double unattributed_core_ms = mean_core - critical_stage_ms;
        if (!out) {
            fprintf(stderr, "cannot open %s\n", artifact_path);
            goto cleanup;
        }
        fprintf(out, "{\"format\":\"q2-real-decode-position-breakdown-v1\","
                "\"model\":\"%s\",\"prompt_tokens\":%u,"
                "\"generated_tokens\":%zu,\"measured_first\":%u,"
                "\"measured_last\":%zu,\"measured_count\":%zu,"
                "\"core_forward_median_unavailable\":true,"
                "\"core_forward_sum_ms\":%.6f,"
                "\"core_forward_mean_ms\":%.6f,"
                "\"argmax_mean_ms\":%.6f,\"ple_critical_stall_ms\":%.6f,"
                "\"stage_accounted_mean_ms\":%.6f,"
                "\"critical_stage_accounted_mean_ms\":%.6f,"
                "\"unattributed_core_mean_ms\":%.6f,",
                model_path, prompt.token_count, generated_count,
                BREAKDOWN_FIRST_MEASURED, generated_count - 1,
                measured_count, core_ms, mean_core,
                argmax_ms / measured_count, ple_stall_ms,
                probe.stages.total / measured_count, critical_stage_ms,
                unattributed_core_ms);
        print_stage_json(out, &probe.stages);
        fprintf(out, ",\"qsa_stage_breakdown\":{\"total_ms\":%.6f,"
                "\"projection_ms\":%.6f,\"attention_ms\":%.6f,"
                "\"other_ms\":%.6f},"
                "\"cuda\":{\"callbacks\":%" PRIu64 ",\"dispatches\":%" PRIu64
                ",\"kernel_ms\":%.6f,\"backend_overhead_ms\":%.6f,"
                "\"upload_ms\":%.6f,\"h2d_bytes\":%" PRIu64
                ",\"d2h_bytes\":%" PRIu64 ",\"syncs\":%" PRIu64
                ",\"allocations\":%" PRIu64 "},"
                "\"note\":\"PLE stage elapsed is asynchronous and excluded "
                "from critical_stage_accounted_mean_ms; CUDA telemetry is "
                "side accounting and is not summed into the stage total\"}\n",
                probe.stages.qsa / measured_count,
                probe.stages.qsa_projection / measured_count,
                probe.stages.qsa_attention / measured_count,
                probe.stages.qsa_other / measured_count,
                probe.callbacks / measured_count, probe.dispatches / measured_count,
                probe.kernel_ms / measured_count,
                probe.backend_overhead_ms / measured_count,
                probe.upload_ms / measured_count,
                probe.h2d_bytes / measured_count, probe.d2h_bytes / measured_count,
                probe.syncs / measured_count, probe.allocations / measured_count);
        fclose(out);
    }

    printf("measured tokens:       %zu (indices %u..%zu)\n",
           measured_count, BREAKDOWN_FIRST_MEASURED, generated_count - 1);
    printf("core forward mean:     %.3f ms/token\n",
           core_ms / measured_count);
    printf("argmax mean:           %.3f ms/token\n",
           argmax_ms / measured_count);
    printf("PLE critical stall:    %.3f ms/token\n", ple_stall_ms);
    printf("stage breakdown mean:\n");
    printf("  embedding/input:     %.3f ms\n",
           probe.stages.embedding / measured_count);
    printf("  GR:                   %.3f ms\n",
           probe.stages.gr / measured_count);
    printf("  GDN:                  %.3f ms\n",
           probe.stages.gdn / measured_count);
    printf("  QSA:                  %.3f ms\n",
           probe.stages.qsa / measured_count);
    printf("    projections:        %.3f ms\n",
           probe.stages.qsa_projection / measured_count);
    printf("    attention:          %.3f ms\n",
           probe.stages.qsa_attention / measured_count);
    printf("    other/index/compress %.3f ms\n",
           probe.stages.qsa_other / measured_count);
    printf("  MoE:                  %.3f ms\n",
           probe.stages.moe / measured_count);
    printf("  norms/residual:       %.3f ms\n",
           probe.stages.norms_residual / measured_count);
    printf("  LM head:              %.3f ms\n",
           probe.stages.lm_head / measured_count);
    printf("  PLE stage (async):    %.3f ms\n",
           probe.stages.ple / measured_count);
    printf("  other:                %.3f ms\n",
           probe.stages.other / measured_count);
    printf("  stage accounted:      %.3f ms\n",
           probe.stages.total / measured_count);
    printf("  critical accounted:   %.3f ms\n",
           (probe.stages.total - probe.stages.ple) / measured_count);
    printf("  other/unattributed:   %.3f ms\n",
           core_ms / measured_count -
               (probe.stages.total - probe.stages.ple) / measured_count);
    printf("CUDA side accounting (not added to stage total):\n");
    printf("  kernel busy:          %.3f ms\n",
           probe.kernel_ms / measured_count);
    printf("  backend overhead:     %.3f ms\n",
           probe.backend_overhead_ms / measured_count);
    printf("  upload:               %.3f ms\n",
           probe.upload_ms / measured_count);
    printf("  syncs/dispatches:     %.3f / %.3f per token\n",
           (double)probe.syncs / measured_count,
           (double)probe.dispatches / measured_count);
    printf("artifact: %s\n", artifact_path);
    ok = true;

cleanup:
    q38_forward_cuda_set_telemetry_observer(runtime.cuda, NULL, NULL);
    q38_session_destroy(&session);
    q38_token_batch_free(&prompt);
    q38_runtime_destroy(&runtime);
    free(logits);
    return ok ? 0 : 1;
}
