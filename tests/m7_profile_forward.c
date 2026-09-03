#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_decode.h"
#include "q38_gguf.h"
#include "q38_memory.h"
#include "q38_platform.h"
#include "q38_profile.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    q38_profile *profile;
    uint64_t matrix_calls, backend_rows, scalar_rows, declines;
    double row_ms, launch_ms, qsa_ms;
} profile_context;

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static bool stage_trace(const q38_forward_stage_usage *usage, void *user,
                        char *error, size_t error_len) {
    profile_context *context = (profile_context *)user;
    if (!context || !usage) {
        if (error && error_len) snprintf(error, error_len, "invalid profile stage");
        return false;
    }

    if (!q38_profile_stage_trace(usage, context->profile, error, error_len))
        return false;
    context->matrix_calls += usage->matrix_calls;
    context->backend_rows += usage->backend_rows;
    context->scalar_rows += usage->scalar_rows;
    context->declines += usage->backend_declines;
    if (usage->name && strstr(usage->name, "qsa"))
        context->qsa_ms += usage->elapsed_ms;
    else if (usage->backend_rows || usage->scalar_rows)
        context->row_ms += usage->elapsed_ms;
    else
        context->launch_ms += usage->elapsed_ms;
    return true;
}

static bool qsa_trace(uint32_t layer, const uint32_t *selected, size_t count,
                      void *user, char *error, size_t error_len) {
    profile_context *context = (profile_context *)user;
    return context && q38_profile_qsa_trace(layer, selected, count,
                                             context->profile, error,
                                             error_len);
}

static void allocation_observer(size_t bytes, void *user) {
    q38_profile_record_runtime_allocation((q38_profile *)user, bytes);
}

static bool write_file(const char *path, const char *text) {
    FILE *file = fopen(path, "w");
    if (!file) return false;
    const bool ok = fputs(text, file) >= 0 && fclose(file) == 0;
    if (!ok) fclose(file);
    return ok;
}

static bool write_platform(const char *path, const q38_platform_info *p) {
    char json[2048];
    snprintf(json, sizeof(json),
             "{\"format\":\"q38-m7-platform-v1\",\"cuda_device_count\":%d,"
             "\"cuda_device\":%d,\"compute_capability\":\"%d.%d\","
             "\"device_name\":\"%s\",\"cuda_total_bytes\":%" PRIu64
             ",\"cuda_free_bytes\":%" PRIu64 ",\"mem_total_bytes\":%" PRIu64
             ",\"mem_available_bytes\":%" PRIu64 ",\"driver_version\":\"%s\","
             "\"runtime_version\":\"%s\"}\n",
             p->cuda_device_count, p->cuda_device, p->cc_major, p->cc_minor,
             p->device_name, p->cuda_total_bytes, p->cuda_free_bytes,
             p->mem_total_bytes, p->mem_available_bytes, p->driver_version,
             p->runtime_version);
    return write_file(path, json);
}

static bool write_memory(const char *path, const q38_memory_snapshot *before,
                         const q38_memory_snapshot *after) {
    char b[1024], a[1024], json[2300];
    q38_memory_snapshot_json(before, b, sizeof(b));
    q38_memory_snapshot_json(after, a, sizeof(a));
    snprintf(json, sizeof(json),
             "{\"format\":\"q38-m7-memory-v1\",\"before\":%s,\"after\":%s}\n",
             b, a);
    return write_file(path, json);
}

static bool write_profile_files(const char *dir, double wall_ms, uint32_t argmax,
                                const q38_memory_snapshot *memory,
                                const q38_profile *profile,
                                const profile_context *context) {
    char path[512], telemetry[4096], memory_json[1024], bench[6500],
        decode[5200];
    if (!q38_profile_json(profile, telemetry, sizeof(telemetry))) return false;
    q38_memory_snapshot_json(memory, memory_json, sizeof(memory_json));
    snprintf(path, sizeof(path), "%s/baseline_bench.json", dir);
    snprintf(bench, sizeof(bench),
             "{\"format\":\"q38-m7-bench-v1\",\"case\":\"S\","
             "\"prompt_tokens\":1,\"decode_tokens\":1,\"runs\":1,"
             "\"wall_ms\":%.9g,\"tokens_per_second\":%.9g,\"argmax\":%u,"
             "\"memory\":%s,\"telemetry\":%s,\"matrix_calls\":%" PRIu64
             ",\"backend_rows\":%" PRIu64 ",\"scalar_rows\":%" PRIu64
             ",\"backend_declines\":%" PRIu64
             ",\"row_matvec_dequant_ms\":%.9g,"
             "\"launch_overhead_proxy_ms\":%.9g,"
             "\"unattributed_gpu_ms\":%.9g}\n",
             wall_ms, wall_ms > 0.0 ? 1000.0 / wall_ms : 0.0, argmax,
             memory_json, telemetry, context->matrix_calls,
             context->backend_rows, context->scalar_rows, context->declines,
             context->row_ms, context->launch_ms,
             fmax(0.0, profile->cuda_elapsed_ms -
                  context->row_ms - context->launch_ms -
                  context->qsa_ms));
    if (!write_file(path, bench)) return false;
    snprintf(path, sizeof(path), "%s/decode_profile.json", dir);
    snprintf(decode, sizeof(decode),
             "{\"format\":\"q38-m7-decode-profile-v1\","
             "\"classification\":{\"row_matvec_dequant\":"
             "\"backend_rows + scalar_rows\","
             "\"launch_overhead\":\"stages with no row work; proxy only\"},"
             "\"matrix_calls\":%" PRIu64 ",\"backend_rows\":%" PRIu64
             ",\"scalar_rows\":%" PRIu64 ",\"backend_declines\":%" PRIu64
             ",\"row_matvec_dequant_ms\":%.9g,"
             "\"launch_overhead_proxy_ms\":%.9g,\"unattributed_gpu_ms\":%.9g,"
             "\"telemetry\":%s}\n",
             context->matrix_calls, context->backend_rows, context->scalar_rows,
             context->declines, context->row_ms, context->launch_ms,
             fmax(0.0, profile->cuda_elapsed_ms -
                  context->row_ms - context->launch_ms -
                  context->qsa_ms), telemetry);
    return write_file(path, decode);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    const char *dir = argc > 2 ? argv[2] : "artifacts/m7";
    char error[256], reason[256], path[512];
    q38_platform_info platform;
    if (q38_platform_probe(&platform, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "platform: %s\n", reason);
        return 1;
    }
    if (mkdir(dir, 0755) != 0 && errno != EEXIST) return 1;
    snprintf(path, sizeof(path), "%s/baseline_platform.json", dir);
    if (!write_platform(path, &platform)) return 1;
    q38_gguf *model = q38_gguf_open(model_path, error, sizeof(error));
    if (!model) { fprintf(stderr, "open: %s\n", error); return 1; }
    q38_weights weights;
    q38_forward_state state;
    q38_forward_cuda_context *cuda = NULL;
    q38_profile profile;
    float *logits = NULL;
    bool ok = q38_weights_bind_subset(model, 47, &weights, error, sizeof(error)) &&
              q38_forward_state_init(&state, &weights, 248044, error, sizeof(error));
    if (!ok) goto cleanup_model;
    q38_memory_tracker tracker;
    q38_memory_snapshot before, after;
    q38_memory_tracker_init(&tracker);
    q38_memory_capture(&tracker, "before_forward", model->size, model->size, 0, &before);
    q38_profile_init(&profile);
    profile_context context = {.profile = &profile};
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.stage_trace = stage_trace;
    diagnostics.qsa_trace = qsa_trace;
    diagnostics.trace_user = &context;
    cuda = q38_forward_cuda_context_create(error, sizeof(error));
    if (!cuda) { fprintf(stderr, "cuda: %s\n", error); ok = false; goto cleanup_state; }
    ok = ok && q38_forward_cuda_prepare_lm_head(
        cuda, model, weights.output, error, sizeof(error));
    if (!ok) { fprintf(stderr, "lm-head residency: %s\n", error); goto cleanup_state; }
    (void)q38_profile_cuda_init(&profile);
    q38_forward_cuda_set_allocation_observer(cuda, allocation_observer,
                                             &profile);
    (void)q38_profile_cuda_begin(&profile, Q38_PROFILE_LM_HEAD,
                                 q38_forward_cuda_stream(cuda));
    logits = (float *)calloc(Q38_DECODE_VOCAB_SIZE, sizeof(float));
    const uint32_t token = 9419;
    const double start = now_ms();
    ok = logits && q38_forward_full_with_matrix_backend(
        model, &weights, &state, &token, 1, logits, Q38_DECODE_VOCAB_SIZE,
        &diagnostics, q38_forward_cuda_matvec_backend,
        q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend, cuda,
        error, sizeof(error));
    const double wall_ms = now_ms() - start;
    (void)q38_profile_cuda_end(&profile, Q38_PROFILE_LM_HEAD,
                               q38_forward_cuda_stream(cuda));
    q38_memory_capture(&tracker, "after_forward", model->size, model->size, 0, &after);
    if (!ok) fprintf(stderr, "forward: %s\n", error);
    size_t argmax = 0;
    if (ok) for (size_t i = 1; i < Q38_DECODE_VOCAB_SIZE; ++i)
        if (logits[i] > logits[argmax]) argmax = i;
    snprintf(path, sizeof(path), "%s/baseline_memory.json", dir);
    ok = ok && write_memory(path, &before, &after) &&
         write_profile_files(dir, wall_ms, (uint32_t)argmax, &after,
                             &profile, &context);
    free(logits);
    q38_profile_destroy(&profile);
    q38_forward_cuda_context_destroy(cuda);
cleanup_state:
    q38_forward_state_destroy(&state);
cleanup_model:
    q38_gguf_close(model);
    return ok ? 0 : 1;
}
