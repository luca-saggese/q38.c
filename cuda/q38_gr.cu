#include "q38_gr.h"

#include <stdio.h>

__global__ static void gr_normalize_kernel(const float *residual,
                                            const float *gamma,
                                            float *normalized) {
    const unsigned branch = blockIdx.x;
    const unsigned channel = threadIdx.x;
    __shared__ float sum;
    if (channel == 0) sum = 0.0f;
    __syncthreads();
    for (unsigned i = channel; i < Q38_GR_HIDDEN; i += blockDim.x)
        atomicAdd(&sum, residual[branch * Q38_GR_HIDDEN + i] *
                         residual[branch * Q38_GR_HIDDEN + i]);
    __syncthreads();
    const float scale = rsqrtf(sum / Q38_GR_HIDDEN + 1e-6f);
    for (unsigned i = channel; i < Q38_GR_HIDDEN; i += blockDim.x)
        normalized[branch * Q38_GR_HIDDEN + i] =
            residual[branch * Q38_GR_HIDDEN + i] * scale *
            gamma[branch * Q38_GR_HIDDEN + i];
}

__global__ static void gr_down_kernel(const float *normalized, const float *down,
                                      float *bottleneck) {
    const unsigned rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= Q38_GR_RANK) return;
    float value = 0.0f;
    for (unsigned i = 0; i < Q38_GR_BRANCHES * Q38_GR_HIDDEN; i++)
        value += down[rank * Q38_GR_BRANCHES * Q38_GR_HIDDEN + i] * normalized[i];
    value /= (float)Q38_GR_HC_COUNT;
    bottleneck[rank] = value / (1.0f + expf(-value));
}

__global__ static void gr_up_kernel(const float *normalized, const float *up,
                                    const float *bottleneck, float *gates) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q38_GR_BRANCHES * Q38_GR_HIDDEN) return;
    float value = 0.0f;
    for (unsigned rank = 0; rank < Q38_GR_RANK; rank++)
        value += up[i * Q38_GR_RANK + rank] * bottleneck[rank];
    gates[i] = 1.0f / (1.0f + expf(-value));
}

__global__ static void gr_read_kernel(const float *normalized, const float *gates,
                                      float *input) {
    const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= Q38_GR_HIDDEN) return;
    float value = 0.0f;
    for (unsigned branch = 0; branch < Q38_GR_BRANCHES; branch++)
        value += gates[branch * Q38_GR_HIDDEN + channel] *
                 normalized[branch * Q38_GR_HIDDEN + channel];
    input[channel] = value / Q38_GR_BRANCHES;
}

__global__ static void gr_write_kernel(const float *residual,
                                       const float *normalized,
                                       const float *inject,
                                       const float *block_output,
                                       float *updated) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Q38_GR_BRANCHES * Q38_GR_HIDDEN) return;
    const unsigned branch = index / Q38_GR_HIDDEN;
    float value = 0.0f;
    for (unsigned i = 0; i < Q38_GR_BRANCHES * Q38_GR_HIDDEN; i++)
        value += inject[branch * Q38_GR_BRANCHES * Q38_GR_HIDDEN + i] *
                 normalized[i];
    const float scale =
        2.0f / (1.0f + expf(-value / (float)Q38_GR_HC_COUNT));
    updated[index] = residual[index] + scale * block_output[index % Q38_GR_HIDDEN];
}

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

extern "C" bool q38_cuda_gr_collapse(const float *residual, const float *gamma,
                                      const float *input_mix_down,
                                      const float *input_mix_up,
                                      const float *block_inject,
                                      const float *block_output, float *input,
                                      float *updated, cudaStream_t stream,
                                      char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!residual || !gamma || !input_mix_down || !input_mix_up ||
        !block_inject || !block_output || !input || !updated) {
        set_error(error, error_len, "invalid CUDA GR arguments");
        return false;
    }
    float *normalized = nullptr, *bottleneck = nullptr, *gates = nullptr;
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    if (cudaMalloc(&normalized, width * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&bottleneck, Q38_GR_RANK * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&gates, width * sizeof(float)) != cudaSuccess) {
        cudaFree(normalized); cudaFree(bottleneck); cudaFree(gates);
        set_error(error, error_len, "CUDA GR workspace allocation failed");
        return false;
    }
    gr_normalize_kernel<<<Q38_GR_BRANCHES, 256, 0, stream>>>(residual, gamma,
                                                               normalized);
    gr_down_kernel<<<2, 256, 0, stream>>>(normalized, input_mix_down, bottleneck);
    gr_up_kernel<<<(width + 255) / 256, 256, 0, stream>>>(
        normalized, input_mix_up, bottleneck, gates);
    gr_read_kernel<<<(Q38_GR_HIDDEN + 255) / 256, 256, 0, stream>>>(
        normalized, gates, input);
    gr_write_kernel<<<(width + 255) / 256, 256, 0, stream>>>(
        residual, normalized, block_inject, block_output, updated);
    cudaError_t status = cudaGetLastError();
    cudaFree(normalized); cudaFree(bottleneck); cudaFree(gates);
    if (status != cudaSuccess) {
        if (error && error_len) snprintf(error, error_len, "CUDA GR launch failed: %s",
                                         cudaGetErrorString(status));
        return false;
    }
    return true;
}
