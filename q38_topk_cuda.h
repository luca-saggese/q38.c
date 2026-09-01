#ifndef Q38_TOPK_CUDA_H
#define Q38_TOPK_CUDA_H

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool q38_topk_cuda(const float *device_scores, size_t row_count,
                   size_t score_count, size_t k, uint32_t *device_indices,
                   cudaStream_t stream, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
