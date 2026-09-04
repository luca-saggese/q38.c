#include "q38_topk_cuda.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdio.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

__global__ static void topk_kernel(const float *scores, size_t rows,
                                   size_t count, size_t k, uint32_t *indices) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float *source = scores + row * count;
    uint32_t *output = indices + row * k;
    for (size_t out = 0; out < k; ++out) {
        size_t best = count;
        for (size_t i = 0; i < count; ++i) {
            bool already = false;
            for (size_t j = 0; j < out; ++j)
                already = already || output[j] == i;
            if (already) continue;
            if (best == count || source[i] > source[best] ||
                (source[i] == source[best] && i < best))
                best = i;
        }
        output[out] = (uint32_t)best;
    }
}

__global__ static void argmax_kernel(const float *scores, size_t rows,
                                     size_t count, uint32_t *indices) {
    __shared__ uint32_t best_ids[256];
    __shared__ float best_values[256];
    const unsigned lane = threadIdx.x;
    const size_t row = (size_t)blockIdx.x;
    if (row >= rows) return;
    const float *source = scores + row * count;
    size_t best = count;
    for (size_t i = lane; i < count; i += blockDim.x) {
        if (isnan(source[i])) continue;
        if (best == count || source[i] > source[best] ||
            (source[i] == source[best] && i < best))
            best = i;
    }
    best_ids[lane] = best == count ? UINT32_MAX : (uint32_t)best;
    best_values[lane] = best == count ? -INFINITY : source[best];
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (lane < stride) {
            const uint32_t other = best_ids[lane + stride];
            if (other != UINT32_MAX &&
                (best_ids[lane] == UINT32_MAX ||
                 best_values[lane + stride] > best_values[lane] ||
                 (best_values[lane + stride] == best_values[lane] &&
                  other < best_ids[lane]))) {
                best_ids[lane] = other;
                best_values[lane] = best_values[lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0) indices[row] = best_ids[0];
}

extern "C" bool q38_topk_cuda(const float *device_scores, size_t row_count,
                              size_t score_count, size_t k,
                              uint32_t *device_indices, cudaStream_t stream,
                              char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!device_scores || !row_count || !score_count || !k ||
        k > score_count || score_count > UINT32_MAX || !device_indices)
        return fail(error, error_len, "invalid CUDA top-k arguments");
    topk_kernel<<<(unsigned)((row_count + 255) / 256), 256, 0, stream>>>(
        device_scores, row_count, score_count, k, device_indices);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_argmax_cuda(const float *device_scores, size_t row_count,
                                size_t score_count, uint32_t *device_indices,
                                cudaStream_t stream, char *error,
                                size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!device_scores || !row_count || !score_count ||
        row_count > (size_t)UINT_MAX || score_count > UINT32_MAX ||
        !device_indices)
        return fail(error, error_len, "invalid CUDA argmax arguments");
    argmax_kernel<<<(unsigned)row_count, 256, 0, stream>>>(
        device_scores, row_count, score_count, device_indices);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    return true;
}
