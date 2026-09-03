#ifndef Q38_CUDA_TIMING_H
#define Q38_CUDA_TIMING_H

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
    uint64_t launch_count;
    uint64_t allocation_count;
    uint64_t allocation_bytes;
    float elapsed_ms;
    int active;
} q38_cuda_timing;

int q38_cuda_timing_init(q38_cuda_timing *timing);
void q38_cuda_timing_destroy(q38_cuda_timing *timing);
int q38_cuda_timing_begin(q38_cuda_timing *timing, cudaStream_t stream);
int q38_cuda_timing_end(q38_cuda_timing *timing, cudaStream_t stream);
void q38_cuda_timing_record_launch(q38_cuda_timing *timing);
void q38_cuda_timing_record_allocation(q38_cuda_timing *timing,
                                       size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* Q38_CUDA_TIMING_H */
