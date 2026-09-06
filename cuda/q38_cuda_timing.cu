#include "q38_cuda_timing.h"

#include <string.h>

int q38_cuda_timing_init(q38_cuda_timing *timing) {
    if (!timing) return 1;
    memset(timing, 0, sizeof(*timing));
    cudaError_t status = cudaEventCreate(&timing->start);
    if (status != cudaSuccess) return (int)status;
    status = cudaEventCreate(&timing->stop);
    if (status != cudaSuccess) {
        cudaEventDestroy(timing->start);
        memset(timing, 0, sizeof(*timing));
        return (int)status;
    }
    return 0;
}

void q38_cuda_timing_destroy(q38_cuda_timing *timing) {
    if (!timing) return;
    if (timing->start) cudaEventDestroy(timing->start);
    if (timing->stop) cudaEventDestroy(timing->stop);
    memset(timing, 0, sizeof(*timing));
}

int q38_cuda_timing_begin(q38_cuda_timing *timing, cudaStream_t stream) {
    if (!timing || !timing->start || timing->active) return 1;
    cudaError_t status = cudaEventRecord(timing->start, stream);
    if (status == cudaSuccess) timing->active = 1;
    return (int)status;
}

int q38_cuda_timing_end(q38_cuda_timing *timing, cudaStream_t stream) {
    if (!timing || !timing->stop || !timing->active) return 1;
    cudaError_t status = cudaEventRecord(timing->stop, stream);
    if (status != cudaSuccess) return (int)status;
    status = cudaEventSynchronize(timing->stop);
    if (status == cudaSuccess)
        status = cudaEventElapsedTime(&timing->elapsed_ms, timing->start,
                                      timing->stop);
    timing->active = 0;
    return (int)status;
}

void q38_cuda_timing_record_launch(q38_cuda_timing *timing) {
    if (timing) timing->launch_count++;
}

void q38_cuda_timing_record_allocation(q38_cuda_timing *timing,
                                       size_t bytes) {
    if (!timing) return;
    timing->allocation_count++;
    timing->allocation_bytes += bytes;
}
