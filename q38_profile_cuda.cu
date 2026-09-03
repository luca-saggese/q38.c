#include "q38_profile.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#if defined(Q38_ENABLE_NVTX) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define Q38_PROFILE_HAS_NVTX 1
#endif

typedef struct {
    cudaEvent_t start[Q38_PROFILE_SUBSYSTEM_COUNT];
    cudaEvent_t stop[Q38_PROFILE_SUBSYSTEM_COUNT];
    unsigned char active[Q38_PROFILE_SUBSYSTEM_COUNT];
} q38_profile_cuda_state;

int q38_profile_cuda_init(q38_profile *profile) {
    if (!profile) return 1;
    q38_profile_cuda_destroy(profile);
    q38_profile_cuda_state *state =
        (q38_profile_cuda_state *)calloc(1, sizeof(*state));
    if (!state) return 1;
    for (int i = 0; i < Q38_PROFILE_SUBSYSTEM_COUNT; ++i) {
        if (cudaEventCreate(&state->start[i]) != cudaSuccess ||
            cudaEventCreate(&state->stop[i]) != cudaSuccess) {
            for (int j = 0; j <= i; ++j) {
                if (state->start[j]) cudaEventDestroy(state->start[j]);
                if (state->stop[j]) cudaEventDestroy(state->stop[j]);
            }
            free(state);
            return 1;
        }
    }
    profile->cuda_state = state;
    return 0;
}

void q38_profile_cuda_destroy(q38_profile *profile) {
    if (!profile || !profile->cuda_state) return;
    q38_profile_cuda_state *state =
        (q38_profile_cuda_state *)profile->cuda_state;
    for (int i = 0; i < Q38_PROFILE_SUBSYSTEM_COUNT; ++i) {
        if (state->start[i]) cudaEventDestroy(state->start[i]);
        if (state->stop[i]) cudaEventDestroy(state->stop[i]);
    }
    free(state);
    profile->cuda_state = NULL;
}

int q38_profile_cuda_begin(q38_profile *profile,
                           q38_profile_subsystem subsystem, void *stream) {
    if (!profile || !profile->cuda_state ||
        subsystem < 0 || subsystem >= Q38_PROFILE_SUBSYSTEM_COUNT)
        return 1;
    q38_profile_cuda_state *state =
        (q38_profile_cuda_state *)profile->cuda_state;
    if (state->active[subsystem]) return 1;
    cudaError_t status = cudaEventRecord(
        state->start[subsystem], (cudaStream_t)stream);
    if (status == cudaSuccess) state->active[subsystem] = 1;
    return (int)status;
}

int q38_profile_cuda_end(q38_profile *profile,
                         q38_profile_subsystem subsystem, void *stream) {
    if (!profile || !profile->cuda_state ||
        subsystem < 0 || subsystem >= Q38_PROFILE_SUBSYSTEM_COUNT)
        return 1;
    q38_profile_cuda_state *state =
        (q38_profile_cuda_state *)profile->cuda_state;
    if (!state->active[subsystem]) return 1;
    cudaError_t status = cudaEventRecord(
        state->stop[subsystem], (cudaStream_t)stream);
    if (status == cudaSuccess) status = cudaEventSynchronize(state->stop[subsystem]);
    float elapsed = 0.0f;
    if (status == cudaSuccess)
        status = cudaEventElapsedTime(&elapsed, state->start[subsystem],
                                      state->stop[subsystem]);
    if (status == cudaSuccess)
        profile->subsystem[subsystem].elapsed_ms += elapsed;
    state->active[subsystem] = 0;
    return (int)status;
}

void q38_profile_nvtx_push(const char *name) {
#if defined(Q38_PROFILE_HAS_NVTX)
    nvtx3::nvtxRangePushA(name ? name : "");
#else
    (void)name;
#endif
}

void q38_profile_nvtx_pop(void) {
#if defined(Q38_PROFILE_HAS_NVTX)
    nvtx3::nvtxRangePop();
#endif
}
