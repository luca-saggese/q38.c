#include "q38_moe_cuda.h"
#include "q38_moe_ref.h"

#include <cuda_runtime.h>

#include <stdio.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

__global__ static void router_kernel(const float *hidden, size_t tokens,
                                     const float *router, float *logits) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_MOE_EXPERTS;
    if (i >= total) return;
    const size_t token = i / Q38_MOE_EXPERTS;
    const size_t expert = i % Q38_MOE_EXPERTS;
    float value = 0.0f;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        value += hidden[token * Q38_MOE_HIDDEN + d] *
                 router[expert * Q38_MOE_HIDDEN + d];
    logits[i] = value;
}

extern "C" bool q38_moe_cuda_router(const float *device_hidden,
                                    size_t token_count,
                                    const float *device_router,
                                    float *device_logits, cudaStream_t stream,
                                    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!device_hidden || !token_count || !device_router || !device_logits ||
        token_count > SIZE_MAX / Q38_MOE_EXPERTS)
        return fail(error, error_len, "invalid CUDA MoE router arguments");
    const size_t total = token_count * Q38_MOE_EXPERTS;
    router_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        device_hidden, token_count, device_router, device_logits);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}
