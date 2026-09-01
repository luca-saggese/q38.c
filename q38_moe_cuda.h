#ifndef Q38_MOE_CUDA_H
#define Q38_MOE_CUDA_H

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Naive reference router projection. The output is token-major F32 logits. */
bool q38_moe_cuda_router(const float *device_hidden, size_t token_count,
                         const float *device_router, float *device_logits,
                         cudaStream_t stream, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
