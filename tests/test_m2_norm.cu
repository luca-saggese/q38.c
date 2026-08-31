#include "q38_cuda_primitives.h"
#include "q38_oracle.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int run_test() {
    const size_t n = 513; /* non-multiple tail */
    std::vector<float> input(n), weight(n), expected_norm(n), actual_norm(n);
    std::vector<float> expected_silu(n), actual_silu(n);
    for (size_t i = 0; i < n; i++) {
        input[i] = (float)((int)(i % 37) - 18) * 0.125f;
        weight[i] = 0.75f + (float)(i % 11) * 0.03125f;
    }
    input[0] = 0.0f;
    input[1] = 1000.0f;
    input[2] = -1000.0f;
    q38_oracle_rms_norm(input.data(), weight.data(), expected_norm.data(), n,
                        1e-5f);
    q38_oracle_silu(input.data(), expected_silu.data(), n);
    float *device_input = nullptr, *device_weight = nullptr;
    float *device_norm = nullptr, *device_silu = nullptr;
    if (cudaMalloc(&device_input, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_weight, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_norm, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_silu, n * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA allocation failed\n");
        cudaFree(device_input); cudaFree(device_weight);
        cudaFree(device_norm); cudaFree(device_silu);
        return 1;
    }
    cudaMemcpy(device_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    char error[256];
    if (!q38_cuda_rms_norm(device_input, device_weight, device_norm, n, 1e-5f,
                           nullptr, error, sizeof(error)) ||
        !q38_cuda_silu(device_input, device_silu, n, nullptr, error,
                       sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(actual_norm.data(), device_norm, n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(actual_silu.data(), device_silu, n * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "CUDA primitive failed: %s\n", error);
        cudaFree(device_input); cudaFree(device_weight);
        cudaFree(device_norm); cudaFree(device_silu);
        return 1;
    }
    cudaFree(device_input); cudaFree(device_weight);
    cudaFree(device_norm); cudaFree(device_silu);
    q38_oracle_metrics norm_metrics, silu_metrics;
    q38_oracle_compare(expected_norm.data(), actual_norm.data(), n, 1e-6f,
                       &norm_metrics);
    q38_oracle_compare(expected_silu.data(), actual_silu.data(), n, 1e-6f,
                       &silu_metrics);
    if (norm_metrics.max_abs > 2e-6f || silu_metrics.max_abs > 2e-6f) {
        std::fprintf(stderr, "primitive parity failed: RMSNorm max=%g SiLU max=%g\n",
                     norm_metrics.max_abs, silu_metrics.max_abs);
        return 1;
    }
    return 0;
}

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "M2-C05: CUDA device unavailable: %s\n",
                     cudaGetErrorString(status));
        return 2;
    }
    if (run_test()) return 1;
    std::puts("test_m2_norm: RMSNorm and SiLU CUDA/oracle parity passed");
    return 0;
}
