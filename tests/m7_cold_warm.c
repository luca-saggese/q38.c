#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_moe.h"
#include "q38_profile.h"
#include "q38_gguf.h"
#include "q38_weights.h"

#include <cuda_runtime_api.h>

#include <math.h>
#include <inttypes.h>
#include <limits.h>
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
static size_t peak_cuda_bytes;
static unsigned long peak_rss_bytes;
static unsigned long min_mem_available_bytes = ULONG_MAX;
static FILE *hidden_capture;
static bool hidden_captured;

static bool capture_moe_hidden(uint32_t layer, const q38_moe_trace *trace,
                               void *user, char *error, size_t error_len) {
    (void)layer;
    (void)user;
    if (hidden_captured) return true;
    if (!trace || !trace->router_input ||
        trace->router_input_count != 2560) {
        if (error && error_len)
            snprintf(error, error_len, "invalid CUDA MoE hidden capture");
        return false;
    }
    if (!hidden_capture ||
        fwrite(trace->router_input, sizeof(float), trace->router_input_count,
               hidden_capture) != trace->router_input_count) {
        if (error && error_len)
            snprintf(error, error_len, "failed to write CUDA MoE hidden");
        return false;
    }
    fflush(hidden_capture);
    hidden_captured = true;
    return true;
}

static void sample_memory(void) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        size_t used = total_bytes - free_bytes;
        if (used > peak_cuda_bytes) peak_cuda_bytes = used;
    }
    unsigned long pages = 0;
    FILE *statm = fopen("/proc/self/statm", "r");
    if (statm) {
        (void)fscanf(statm, "%*lu %lu", &pages);
        fclose(statm);
    }
    unsigned long rss = pages * (unsigned long)sysconf(_SC_PAGESIZE);
    if (rss > peak_rss_bytes) peak_rss_bytes = rss;
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo) {
        char key[32], unit[16];
        unsigned long value;
        while (fscanf(meminfo, "%31s %lu %15s", key, &value, unit) == 3) {
            if (strcmp(key, "MemAvailable:") == 0) {
                unsigned long bytes = value * 1024;
                if (bytes < min_mem_available_bytes)
                    min_mem_available_bytes = bytes;
                break;
            }
        }
        fclose(meminfo);
    }
}

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
    sample_memory();
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

static size_t routed_selected[Q38_MODEL_LAYERS];

static bool route_trace(uint32_t layer, const uint16_t *experts,
                        const float *weights, size_t count, void *user,
                        char *error, size_t error_len) {
    (void)experts;
    (void)weights;
    (void)user;
    (void)error;
    (void)error_len;
    if (layer < Q38_MODEL_LAYERS) {
        routed_selected[layer] = count;
        q38_forward_cuda_record_route(backend_cuda, layer, count);
    }
    return true;
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
    hidden_capture = fopen("artifacts/post_m8_opt/shared_hidden_layer0.bin", "wb");
    if (!hidden_capture) return 1;
    const uint32_t token = 9419;
    const bool diagnostic_only = getenv("Q38_Q2_DIAGNOSTIC") != NULL;
    const int run_count = diagnostic_only ? 2 : 6;
    for (int run = 1; run <= run_count; ++run) {
        stages s = {0};
        memset(routed_selected, 0, sizeof(routed_selected));
        q38_profile_destroy(&profile);
        q38_profile_init(&profile);
        q38_profile_set_token_count(&profile, 1);
        stage_profile = &profile;
        q38_forward_cuda_set_telemetry_observer(cuda, telemetry_observer, &profile);
        q38_forward_diagnostics d;
        memset(&d, 0, sizeof(d));
        d.stage_trace = stage_trace;
        d.backend_context = backend_context;
        d.moe_trace = capture_moe_hidden;
        d.route_trace = route_trace;
        d.trace_user = &s;
        float *logits = run == 1 ? logits1 : logits2;
        q38_forward_cuda_residency_stats before, after;
        q38_forward_cuda_get_residency_stats(cuda, &before);
        uint64_t before_fast[Q38_MODEL_LAYERS] = {0};
        uint64_t before_legacy[Q38_MODEL_LAYERS] = {0};
        for (uint32_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer)
            q38_forward_cuda_get_expert_layer_calls(
                cuda, layer, &before_fast[layer], &before_legacy[layer]);
        double start = now_ms();
        bool ok = q38_forward_full_with_matrix_backend(
            model, &weights, &state, &token, 1, logits, 248320, &d,
            q38_forward_cuda_matvec_backend, q38_forward_cuda_matrix_backend,
            q38_forward_cuda_expert_backend, cuda, error, sizeof(error));
        sample_memory();
        uint32_t gpu_argmax_token = 0;
        float gpu_argmax_ms = 0.0f;
        bool gpu_argmax_ok = q38_forward_cuda_greedy_argmax(
            cuda, &gpu_argmax_token, error, sizeof(error));
        q38_forward_cuda_residency_stats argmax_stats;
        q38_forward_cuda_get_residency_stats(cuda, &argmax_stats);
        gpu_argmax_ms = argmax_stats.gpu_argmax_kernel_ms;
        double wall = now_ms() - start;
        q38_forward_cuda_get_residency_stats(cuda, &after);
        if (run == 2) {
            printf("{\"q2_expert_diagnostic\":{"
                   "\"routed_layers_executed\":%" PRIu64
                   ",\"selected_experts_total\":%" PRIu64
                   ",\"q2_gate_up_fast_calls\":%" PRIu64
                   ",\"q2_gate_up_legacy_calls\":%" PRIu64
                   ",\"q2_gate_up_fallback_calls\":%" PRIu64
                   ",\"q2_down_calls\":%" PRIu64
                   ",\"q2_gate_up_fast_total_kernel_ms\":%.6f"
                   ",\"q2_gate_up_legacy_total_kernel_ms\":%.6f"
                   ",\"q2_down_total_kernel_ms\":%.6f"
                   ",\"expert_backend_total_wall_ms\":%.6f"
                   ",\"expert_host_sync_count\":%" PRIu64
                   ",\"expert_H2D_bytes\":%" PRIu64
                   ",\"expert_D2H_bytes\":%" PRIu64 "}}\n",
                   after.routed_layers_executed -
                       before.routed_layers_executed,
                   after.selected_experts_total -
                       before.selected_experts_total,
                   after.q2_gate_up_fast_calls -
                       before.q2_gate_up_fast_calls,
                   after.q2_gate_up_legacy_calls -
                       before.q2_gate_up_legacy_calls,
                   after.q2_gate_up_fallback_calls -
                       before.q2_gate_up_fallback_calls,
                   after.q2_down_calls - before.q2_down_calls,
                   after.q2_gate_up_fast_total_kernel_ms -
                       before.q2_gate_up_fast_total_kernel_ms,
                   after.q2_gate_up_legacy_total_kernel_ms -
                       before.q2_gate_up_legacy_total_kernel_ms,
                   after.q2_down_total_kernel_ms -
                       before.q2_down_total_kernel_ms,
                   after.expert_backend_total_wall_ms -
                       before.expert_backend_total_wall_ms,
                   after.expert_host_sync_count - before.expert_host_sync_count,
                   after.expert_H2D_bytes - before.expert_H2D_bytes,
                   after.expert_D2H_bytes - before.expert_D2H_bytes);
            for (uint32_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
                uint64_t fast = 0, legacy = 0;
                q38_forward_cuda_get_expert_layer_calls(
                    cuda, layer, &fast, &legacy);
                fast -= before_fast[layer];
                legacy -= before_legacy[layer];
                if (routed_selected[layer] || fast || legacy)
                    printf("{\"layer\":%u,\"selected_count\":%zu,"
                           "\"fast_gate_up_calls\":%" PRIu64
                           ",\"legacy_gate_up_calls\":%" PRIu64 "}\n",
                           layer, routed_selected[layer], fast, legacy);
            }
        }
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
        "\"non_ple_upload_bytes_per_token\":%zu,"
        "\"non_ple_residency_miss\":%" PRIu64 ","
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
               "\"telemetry\":%s,\"gpu_argmax_kernel_ms\":%.6f,"
               "\"gpu_argmax_token\":%u,\"gpu_argmax_ok\":%s,"
               "\"peak_cuda_bytes\":%zu,\"peak_rss_bytes\":%lu,"
               "\"mem_available_min_bytes\":%lu,"
               "\"correctness\":%s,\"stable_device_pointer\":%s}\n",
               run, run == 1 ? "true" : "false",
               after.all_non_ple_resident ? "true" : "false",
               after.persistent_resident_bytes, after.persistent_resident_tensors,
               after.persistent_expected_bytes, after.persistent_expected_tensors,
               after.persistent_coverage_ok ? "true" : "false",
               after.persistent_duplicate_tensors, after.persistent_ple_entries,
               wall, after.matrix_upload_bytes - before.matrix_upload_bytes,
               after.non_ple_upload_bytes_per_token,
               after.non_ple_residency_miss,
               after.resident_hits - before.resident_hits,
               after.resident_misses - before.resident_misses, s.gdn,
               s.lm_head, s.moe, s.qsa, s.ple, best, s.moe_router,
               s.moe_routed_expert, s.moe_shared_gate, s.moe_shared_up,
               s.moe_shared_down, s.moe_routed_gate_up, s.moe_routed_down,
               s.moe_activation_reduction, s.qsa_qkv, s.qsa_indexer_compression,
               s.qsa_score, s.qsa_top_k, s.qsa_gather, s.qsa_attention,
               s.qsa_state_update, telemetry, gpu_argmax_ms,
               gpu_argmax_token, gpu_argmax_ok ? "true" : "false",
               peak_cuda_bytes, peak_rss_bytes, min_mem_available_bytes,
               ok ? "true" : "false",
               after.lm_head_device_pointer ? "true" : "false");
        fprintf(artifact,
                "{\"run\":%d,\"cold\":%s,\"all_non_ple_resident\":%s,"
                "\"persistent_resident_bytes\":%zu,\"persistent_resident_tensors\":%" PRIu64 ","
                "\"persistent_expected_bytes\":%zu,\"persistent_expected_tensors\":%" PRIu64 ","
                "\"persistent_coverage_ok\":%s,\"persistent_duplicate_tensors\":%" PRIu64 ","
                "\"persistent_ple_entries\":%" PRIu64 ",\"persistent_hits\":%" PRIu64 ","
                "\"persistent_misses\":%" PRIu64 ",\"resident_hits\":%" PRIu64 ","
                "\"resident_misses\":%" PRIu64 ",\"wall_ms\":%.6f,"
                "\"gpu_argmax_kernel_ms\":%.6f,\"gpu_argmax_token\":%u,"
                "\"gpu_argmax_ok\":%s,\"peak_cuda_bytes\":%zu,"
                "\"peak_rss_bytes\":%lu,\"mem_available_min_bytes\":%lu,"
                "\"telemetry\":%s,"
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
                wall, gpu_argmax_ms, gpu_argmax_token,
                gpu_argmax_ok ? "true" : "false", peak_cuda_bytes,
                peak_rss_bytes, min_mem_available_bytes, telemetry,
                s.moe_router, s.moe_routed_gate_up,
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
    if (hidden_capture) fclose(hidden_capture);
    q38_gguf_close(model);
    return 0;
}
