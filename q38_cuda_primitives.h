#ifndef Q38_CUDA_PRIMITIVES_H
#define Q38_CUDA_PRIMITIVES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#include "q38_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

bool q38_cuda_dequantize_row(uint32_t type, const void *blocks,
                             size_t block_count, float *out,
                             cudaStream_t stream, char *error,
                             size_t error_len);

bool q38_cuda_rms_norm(const float *input, const float *weight, float *output,
                       size_t elements, float epsilon, cudaStream_t stream,
                       char *error, size_t error_len);
bool q38_cuda_silu(const float *input, float *output, size_t elements,
                   cudaStream_t stream, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
