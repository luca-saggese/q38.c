#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_profile.h"
#include "q38_gguf.h"
#include "q38_weights.h"

#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
    double gdn, lm_head, moe, qsa, ple;
    double moe_router, moe_routed_expert, moe_shared_gate;
    double moe_shared_up, moe_shared_down, moe_routed_gate_up;
    double moe_routed_down, moe_activation_reduction;
    double qsa_qkv, qsa_indexer_compression, qsa_score, qsa_top_k;
    double qsa_gather, qsa_attention, qsa_state_update;
} stages;

static q38_forward_cuda_context *backend_cuda;
static q38_profile *stage_profile;
static FILE *residency_progress;

static void residency_progress_observer(const char *group,
                                        const q38_tensor *tensor,
                                        size_t cumulative_bytes,
                                        size_t free_bytes, size_t total_bytes,
                                        void *user) {
    (void)user;
    if (!residency_progress) return;
    char name[128], smi[256] = "";
    {
        size_t n = tensor && tensor->name.len < sizeof(name) - 1
            ? (size_t)tensor->name.len : sizeof(name) - 1;
        if (tensor && tensor->name.ptr && n) memcpy(name, tensor->name.ptr, n);
        name[n] = '\0';
    }
    FILE *pipe = popen("nvidia-smi --query-gpu=memory.used,memory.free,memory.total "
                       "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        if (fgets(smi, sizeof(smi), pipe)) {
            size_t n = strlen(smi);
            while (n && (smi[n - 1] == '\n' || smi[n - 1] == '\r')) smi[--n] = '\0';
        }
        pclose(pipe);
    }
    unsigned long pages = 0;
    FILE *statm = fopen("/proc/self/statm", "r");
    if (statm) { (void)fscanf(statm, "%*lu %lu", &pages); fclose(statm); }
    unsigned long available = 0;
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo) {
        char key[32]; unsigned long value; char unit[16];
        while (fscanf(meminfo, "%31s %lu %15s", key, &value, unit) == 3)
            if (strcmp(key, "MemAvailable:") == 0) { available = value * 1024; break; }
        fclose(meminfo);
    }
    fprintf(residency_progress,
            "{\"group\":\"%s\",\"tensor_name\":\"%s\",\"cumulative_resident_bytes\":%zu,"
            "\"cuda_free_bytes\":%zu,\"cuda_total_bytes\":%zu,\"rss_bytes\":%lu,"
            "\"mem_available_bytes\":%lu,\"nvidia_smi\":\"%s\"}\n",
            group ? group : "unknown", name, cumulative_bytes, free_bytes,
            total_bytes, pages * (unsigned long)sysconf(_SC_PAGESIZE), available, smi);
    fflush(residency_progress);
}

static void backend_context(uint32_t layer, const char *stage,
                            const q38_tensor *tensor, size_t rows,
                            size_t cols, void *user) {
    (void)tensor; (void)rows; (void)cols;
    (void)user;
    q38_forward_cuda_set_stage_context(backend_cuda, layer, stage);
}

static void telemetry_observer(const q38_forward_cuda_telemetry *telemetry,
                               void *user) {
    q38_profile_record_cuda_telemetry((q38_profile *)user, telemetry);
}

static bool stage_trace(const q38_forward_stage_usage *u, void *user,
                        char *error, size_t error_len) {
    stages *s = (stages *)user;
    if (!u || !s) {
        if (error && error_len) snprintf(error, error_len, "invalid stage");
        return false;
    }
    double *dst = NULL;
    if (strstr(u->name, "lm_head")) dst = &s->lm_head;
    else if (strstr(u->name, "gdn")) dst = &s->gdn;
    else if (strstr(u->name, "moe_router")) dst = &s->moe_router;
    else if (strstr(u->name, "moe_routed_expert")) dst = &s->moe_routed_expert;
    else if (strstr(u->name, "moe_routed_gate_up")) dst = &s->moe_routed_gate_up;
    else if (strstr(u->name, "moe_routed_down")) dst = &s->moe_routed_down;
    else if (strstr(u->name, "moe_activation_reduction")) dst = &s->moe_activation_reduction;
    else if (strstr(u->name, "moe_shared_gate")) dst = &s->moe_shared_gate;
    else if (strstr(u->name, "moe_shared_up")) dst = &s->moe_shared_up;
    else if (strstr(u->name, "moe_shared_down")) dst = &s->moe_shared_down;
    else if (strstr(u->name, "moe")) dst = &s->moe;
    else if (strstr(u->name, "qsa_qkv")) dst = &s->qsa_qkv;
    else if (strstr(u->name, "qsa_indexer_compression")) dst = &s->qsa_indexer_compression;
    else if (strstr(u->name, "qsa_score")) dst = &s->qsa_score;
    else if (strstr(u->name, "qsa_top_k")) dst = &s->qsa_top_k;
    else if (strstr(u->name, "qsa_gather")) dst = &s->qsa_gather;
    else if (strstr(u->name, "qsa_attention")) dst = &s->qsa_attention;
    else if (strstr(u->name, "qsa_state_update")) dst = &s->qsa_state_update;
    if (u->name && strstr(u->name, "qsa")) s->qsa += u->elapsed_ms;
    else if (strstr(u->name, "ple")) dst = &s->ple;
    if (dst) *dst += u->elapsed_ms;
    if (stage_profile &&
        !q38_profile_stage_trace(u, stage_profile, error, error_len))
        return false;
    return true;
}

static size_t argmax(const float *x, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    char error[256] = {0};
    q38_gguf *model = q38_gguf_open(path, error, sizeof(error));
    if (!model) { fprintf(stderr, "%s\n", error); return 1; }
    q38_weights weights;
    q38_forward_cuda_context *cuda =
        q38_forward_cuda_context_create(error, sizeof(error));
    float *logits1 = calloc(248320, sizeof(float));
    float *logits2 = calloc(248320, sizeof(float));
    q38_forward_state state;
    memset(&state, 0, sizeof(state));
    q38_profile profile;
    q38_profile_init(&profile);
    residency_progress = fopen("artifacts/m7/all_non_ple_residency_progress.jsonl",
                               "w");
    if (!residency_progress) return 1;
    bool all_non_ple_enabled = false;
    if (!cuda || !logits1 || !logits2 ||
        !q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error[0] ? error : "setup failed");
        return 1;
    }
    setenv("Q38_FORCE_ALL_NON_PLE_RESIDENCY", "1", 1);
    q38_forward_cuda_set_residency_progress_observer(
        cuda, residency_progress_observer, NULL);
    all_non_ple_enabled =
        q38_forward_cuda_enable_all_non_ple_residency(
            cuda, model, error, sizeof(error));
    if (!all_non_ple_enabled)
        fprintf(stderr, "all-non-PLE residency disabled: %s\n", error);
    if (!q38_forward_cuda_prepare_lm_head(cuda, model, weights.output, error,
                                          sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    q38_forward_cuda_set_telemetry_observer(cuda, telemetry_observer, &profile);
    backend_cuda = cuda;
    if (!q38_forward_state_init(&state, &weights, 248044, error,
                                sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    FILE *artifact = fopen(argc > 2 ? argv[2] :
                           "artifacts/m7/canonical_cold_warm.jsonl", "w");
    if (!artifact) return 1;
    const uint32_t token = 9419;
    for (int run = 1; run <= 6; ++run) {
        stages s = {0};
        q38_profile_destroy(&profile);
        q38_profile_init(&profile);
        q38_profile_set_token_count(&profile, 1);
        stage_profile = &profile;
        q38_forward_cuda_set_telemetry_observer(cuda, telemetry_observer, &profile);
        q38_forward_diagnostics d;
        memset(&d, 0, sizeof(d));
        d.stage_trace = stage_trace;
        d.backend_context = backend_context;
        d.trace_user = &s;
        float *logits = run == 1 ? logits1 : logits2;
        q38_forward_cuda_residency_stats before, after;
        q38_forward_cuda_get_residency_stats(cuda, &before);
        double start = now_ms();
        bool ok = q38_forward_full_with_matrix_backend(
            model, &weights, &state, &token, 1, logits, 248320, &d,
            q38_forward_cuda_matvec_backend, q38_forward_cuda_matrix_backend,
            q38_forward_cuda_expert_backend, cuda, error, sizeof(error));
        double wall = now_ms() - start;
        q38_forward_cuda_get_residency_stats(cuda, &after);
        size_t best = argmax(logits, 248320);
        char *telemetry = calloc(8 * 1024 * 1024, 1);
        if (!telemetry || !q38_profile_json(&profile, telemetry, 8 * 1024 * 1024)) {
            free(telemetry); fclose(artifact); return 1;
        }
        printf(        "{\"run\":%d,\"cold\":%s,\"all_non_ple_resident\":%s,"
        "\"persistent_resident_bytes\":%zu,\"persistent_resident_tensors\":%" PRIu64 ","
        "\"persistent_expected_bytes\":%zu,\"persistent_expected_tensors\":%" PRIu64 ","
        "\"persistent_coverage_ok\":%s,\"persistent_duplicate_tensors\":%" PRIu64 ","
        "\"persistent_ple_entries\":%" PRIu64 ",\"wall_ms\":%.6f,"
        "\"matrix_upload_bytes\":%zu,\"matrix_upload_ms\":null,"
        "\"lm_head_upload_bytes\":0,\"lm_head_allocation_count\":0,"
        "\"allocations\":null,\"cuda_allocations\":null,"
               "\"resident_hits\":%" PRIu64 ",\"resident_misses\":%" PRIu64
               ",\"GDN_ms\":%.6f,\"LM_head_ms\":%.6f,\"MoE_ms\":%.6f,"
               "\"QSA_ms\":%.6f,\"PLE_ms\":%.6f,\"argmax\":%zu,"
               "\"moe_router_ms\":%.6f,\"moe_routed_expert_ms\":%.6f,"
               "\"moe_shared_gate_ms\":%.6f,\"moe_shared_up_ms\":%.6f,"
               "\"moe_shared_down_ms\":%.6f,\"moe_routed_gate_up_ms\":%.6f,"
               "\"moe_routed_down_ms\":%.6f,\"moe_activation_reduction_ms\":%.6f,"
               "\"qsa_qkv_ms\":%.6f,\"qsa_indexer_compression_ms\":%.6f,"
               "\"qsa_score_ms\":%.6f,\"qsa_top_k_ms\":%.6f,\"qsa_gather_ms\":%.6f,"
               "\"qsa_attention_ms\":%.6f,\"qsa_state_update_ms\":%.6f,"
               "\"telemetry\":%s,"
               "\"correctness\":%s,\"stable_device_pointer\":%s}\n",
               run, run == 1 ? "true" : "false",
               after.all_non_ple_resident ? "true" : "false",
               after.persistent_resident_bytes, after.persistent_resident_tensors,
               after.persistent_expected_bytes, after.persistent_expected_tensors,
               after.persistent_coverage_ok ? "true" : "false",
               after.persistent_duplicate_tensors, after.persistent_ple_entries,
               wall, after.matrix_upload_bytes - before.matrix_upload_bytes,
               after.resident_hits - before.resident_hits,
               after.resident_misses - before.resident_misses, s.gdn,
               s.lm_head, s.moe, s.qsa, s.ple, best, s.moe_router,
               s.moe_routed_expert, s.moe_shared_gate, s.moe_shared_up,
               s.moe_shared_down, s.moe_routed_gate_up, s.moe_routed_down,
               s.moe_activation_reduction, s.qsa_qkv, s.qsa_indexer_compression,
               s.qsa_score, s.qsa_top_k, s.qsa_gather, s.qsa_attention,
               s.qsa_state_update, telemetry, ok ? "true" : "false",
               after.lm_head_device_pointer ? "true" : "false");
        fprintf(artifact,
                "{\"run\":%d,\"cold\":%s,\"all_non_ple_resident\":%s,"
                "\"persistent_resident_bytes\":%zu,\"persistent_resident_tensors\":%" PRIu64 ","
                "\"persistent_expected_bytes\":%zu,\"persistent_expected_tensors\":%" PRIu64 ","
                "\"persistent_coverage_ok\":%s,\"persistent_duplicate_tensors\":%" PRIu64 ","
                "\"persistent_ple_entries\":%" PRIu64 ",\"persistent_hits\":%" PRIu64 ","
                "\"persistent_misses\":%" PRIu64 ",\"resident_hits\":%" PRIu64 ","
                "\"resident_misses\":%" PRIu64 ",\"wall_ms\":%.6f,\"telemetry\":%s,"
                "\"moe\":{\"router_ms\":%.6f,\"routed_gate_up_ms\":%.6f,"
                "\"routed_down_ms\":%.6f,\"shared_gate_ms\":%.6f,"
                "\"shared_up_ms\":%.6f,\"shared_down_ms\":%.6f,"
                "\"activation_reduction_ms\":%.6f},\"qsa\":{\"q_k_v_ms\":%.6f,"
                "\"indexer_compression_ms\":%.6f,\"score_ms\":%.6f,"
                "\"top_k_ms\":%.6f,\"gather_ms\":%.6f,\"attention_ms\":%.6f,"
                "\"state_update_ms\":%.6f}}\n", run, run == 1 ? "true" : "false",
                all_non_ple_enabled ? "true" : "false",
                after.persistent_resident_bytes, after.persistent_resident_tensors,
                after.persistent_expected_bytes, after.persistent_expected_tensors,
                after.persistent_coverage_ok ? "true" : "false",
                after.persistent_duplicate_tensors, after.persistent_ple_entries,
                after.persistent_hits - before.persistent_hits,
                after.persistent_misses - before.persistent_misses,
                after.resident_hits - before.resident_hits,
                after.resident_misses - before.resident_misses,
                wall, telemetry, s.moe_router, s.moe_routed_gate_up,
                s.moe_routed_down, s.moe_shared_gate, s.moe_shared_up,
                s.moe_shared_down, s.moe_activation_reduction, s.qsa_qkv,
                s.qsa_indexer_compression, s.qsa_score, s.qsa_top_k,
                s.qsa_gather, s.qsa_attention, s.qsa_state_update);
        free(telemetry);
        if (!ok) { fprintf(stderr, "%s\n", error); return 1; }
        if (run < 6) q38_forward_state_reset(&state);
    }
    if (argmax(logits1, 248320) != argmax(logits2, 248320) ||
        memcmp(logits1, logits2, 248320 * sizeof(float)) != 0) {
        fprintf(stderr, "cold/warm logits mismatch\n");
        return 1;
    }
    free(logits1); free(logits2);
    fclose(artifact);
    q38_forward_state_destroy(&state);
    q38_profile_destroy(&profile);
    q38_forward_cuda_context_destroy(cuda);
    if (residency_progress) fclose(residency_progress);
    q38_gguf_close(model);
    return 0;
}
