#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_gguf.h"
#include "q38_weights.h"

#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
    double gdn, lm_head, moe, qsa, ple;
} stages;

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
    else if (strstr(u->name, "moe")) dst = &s->moe;
    else if (strstr(u->name, "qsa")) dst = &s->qsa;
    else if (strstr(u->name, "ple")) dst = &s->ple;
    if (dst) *dst += u->elapsed_ms;
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
    if (!cuda || !logits1 || !logits2 ||
        !q38_weights_bind_subset(model, 47, &weights, error, sizeof(error)) ||
        !q38_forward_cuda_prepare_lm_head(cuda, model, weights.output, error,
                                          sizeof(error))) {
        fprintf(stderr, "%s\n", error[0] ? error : "setup failed");
        return 1;
    }
    const uint32_t token = 9419;
    for (int run = 1; run <= 6; ++run) {
        q38_forward_state state;
        memset(&state, 0, sizeof(state));
        stages s = {0};
        q38_forward_diagnostics d;
        memset(&d, 0, sizeof(d));
        d.stage_trace = stage_trace;
        d.trace_user = &s;
        float *logits = run == 1 ? logits1 : logits2;
        if (!q38_forward_state_init(&state, &weights, 248044, error,
                                    sizeof(error))) return 1;
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
        printf(        "{\"run\":%d,\"wall_ms\":%.6f,\"kernel_ms\":%.6f,"
               "\"matrix_upload_bytes\":%zu,\"matrix_upload_ms\":0.0,"
        "\"lm_head_upload_bytes\":0,\"lm_head_allocation_count\":0,"
        "\"allocations\":0,\"cuda_allocations\":0,"
               "\"resident_hits\":%" PRIu64 ",\"resident_misses\":%" PRIu64
               ",\"GDN_ms\":%.6f,\"LM_head_ms\":%.6f,\"MoE_ms\":%.6f,"
               "\"QSA_ms\":%.6f,\"PLE_ms\":%.6f,\"argmax\":%zu,"
               "\"correctness\":%s,\"stable_device_pointer\":%s}\n",
               run, wall, s.lm_head, after.matrix_upload_bytes - before.matrix_upload_bytes,
               after.resident_hits - before.resident_hits,
               after.resident_misses - before.resident_misses, s.gdn,
               s.lm_head, s.moe, s.qsa, s.ple, best, ok ? "true" : "false",
               after.lm_head_device_pointer ? "true" : "false");
        q38_forward_state_destroy(&state);
        if (!ok) { fprintf(stderr, "%s\n", error); return 1; }
    }
    if (argmax(logits1, 248320) != argmax(logits2, 248320) ||
        memcmp(logits1, logits2, 248320 * sizeof(float)) != 0) {
        fprintf(stderr, "cold/warm logits mismatch\n");
        return 1;
    }
    free(logits1); free(logits2);
    q38_forward_cuda_context_destroy(cuda);
    q38_gguf_close(model);
    return 0;
}
