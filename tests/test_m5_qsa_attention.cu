#include "../q38_qsa_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

int main() {
    const float k[] = {1, 0, 2, 0, 3, 0};
    const float v[] = {10, 1, 20, 2, 30, 3};
    const float q[] = {1, 0, 0, 1};
    const uint32_t ids[] = {2, 0};
    float *dk = nullptr, *dv = nullptr, *dq = nullptr, *do_ = nullptr;
    uint32_t *di = nullptr;
    float *sk = nullptr, *sv = nullptr;
    if (cudaMalloc(&dk, sizeof(k)) != cudaSuccess ||
        cudaMalloc(&dv, sizeof(v)) != cudaSuccess ||
        cudaMalloc(&dq, sizeof(q)) != cudaSuccess ||
        cudaMalloc(&do_, sizeof(q)) != cudaSuccess ||
        cudaMalloc(&di, sizeof(ids)) != cudaSuccess ||
        cudaMalloc(&sk, 2 * 2 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sv, 2 * 2 * sizeof(float)) != cudaSuccess)
        return 1;
    cudaMemcpy(dk, k, sizeof(k), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, sizeof(v), cudaMemcpyHostToDevice);
    cudaMemcpy(dq, q, sizeof(q), cudaMemcpyHostToDevice);
    cudaMemcpy(di, ids, sizeof(ids), cudaMemcpyHostToDevice);
    char error[128];
    if (!q38_qsa_cuda_gather_attention(dk, dv, 3, 1, 2, di, 2, sk, sv, dq,
                                       1, 2, do_, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "selected attention failed: %s\n", error);
        return 1;
    }
    float got[4];
    cudaMemcpy(got, do_, sizeof(got), cudaMemcpyDeviceToHost);
    const float a = std::exp(3.0f / std::sqrt(2.0f));
    const float b = std::exp(1.0f / std::sqrt(2.0f));
    const float expected0 = (30.0f * a + 10.0f * b) / (a + b);
    const float expected1 = (3.0f * a + 1.0f * b) / (a + b);
    if (std::fabs(got[0] - expected0) > 1e-3f ||
        std::fabs(got[1] - expected1) > 1e-3f ||
        std::fabs(got[2] - 20.0f) > 1e-3f ||
        std::fabs(got[3] - 2.0f) > 1e-3f) {
        std::fprintf(stderr, "selected attention output mismatch\n");
        return 1;
    }
    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(do_); cudaFree(di);
    cudaFree(sk); cudaFree(sv);
    std::puts("test_m5_qsa_attention: selected gather and dense attention passed");
    return 0;
}
