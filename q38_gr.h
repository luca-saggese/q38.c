#ifndef Q38_GR_H
#define Q38_GR_H

#include <stdbool.h>
#include <stddef.h>

#include <cuda_runtime_api.h>

#include "q38_gr_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float normalize_ms;
    float down_projection_ms;
    float up_projection_ms;
    float branch_merge_ms;
    float injection_ms;
} q38_cuda_gr_timing;

bool q38_cuda_gr_collapse(const float *residual, const float *gamma,
                          const float *input_mix_down,
                          const float *input_mix_up,
                          const float *block_inject,
                          const float *block_output, float *input,
                          float *updated, cudaStream_t stream, char *error,
                          size_t error_len);

bool q38_cuda_gr_collapse_timed(
    const float *residual, const float *gamma,
    const float *input_mix_down, const float *input_mix_up,
    const float *block_inject, const float *block_output, float *input,
    float *updated, cudaStream_t stream, q38_cuda_gr_timing *timing,
    char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
