#include "q38_gr.h"
#include "q38_gr_ref.h"
#include "q38_oracle.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    std::vector<float> residual(width), gamma(width, 1.0f);
    std::vector<float> down(Q38_GR_RANK * width, 0.0f);
    std::vector<float> up(width * Q38_GR_RANK, 0.0f);
    std::vector<float> inject(Q38_GR_BRANCHES * width, 0.0f);
    std::vector<float> block_output(Q38_GR_HIDDEN);
    std::vector<float> expected_input(Q38_GR_HIDDEN), expected_updated(width);
    std::vector<float> actual_input(Q38_GR_HIDDEN), actual_updated(width);
    for (size_t i = 0; i < width; i++) residual[i] = (float)((int)(i % 23) - 11) / 17.0f;
    for (size_t i = 0; i < Q38_GR_HIDDEN; i++) block_output[i] = 0.125f;
    q38_gr_ref_params params = {gamma.data(), down.data(), up.data(), inject.data()};
    q38_gr_collapse(residual.data(), block_output.data(), &params,
                    expected_input.data(), expected_updated.data());
    float *dr = nullptr, *dg = nullptr, *dd = nullptr, *du = nullptr;
    float *di = nullptr, *dbo = nullptr, *dinput = nullptr, *dupdated = nullptr;
    if (cudaMalloc(&dr, width * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dg, width * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dd, down.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&du, up.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&di, inject.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dbo, block_output.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dinput, expected_input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dupdated, expected_updated.size() * sizeof(float)) != cudaSuccess)
        return 1;
    cudaMemcpy(dr, residual.data(), width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dg, gamma.data(), width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dd, down.data(), down.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(du, up.data(), up.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(di, inject.data(), inject.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dbo, block_output.data(), block_output.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_gr_collapse(dr, dg, dd, du, di, dbo, dinput, dupdated,
                                   nullptr, error, sizeof(error)) &&
              cudaDeviceSynchronize() == cudaSuccess &&
              cudaMemcpy(actual_input.data(), dinput,
                         actual_input.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(actual_updated.data(), dupdated,
                         actual_updated.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dr); cudaFree(dg); cudaFree(dd); cudaFree(du);
    cudaFree(di); cudaFree(dbo); cudaFree(dinput); cudaFree(dupdated);
    if (!ok) {
        std::fprintf(stderr, "CUDA GR failed: %s\n", error);
        return 1;
    }
    q38_oracle_metrics a, b;
    q38_oracle_compare(expected_input.data(), actual_input.data(),
                       expected_input.size(), 1e-6f, &a);
    q38_oracle_compare(expected_updated.data(), actual_updated.data(),
                       expected_updated.size(), 1e-6f, &b);
    if (a.max_abs > 3e-5f || b.max_abs > 3e-5f) return 1;
    std::puts("test_m3_gr_cuda: non-fused CUDA GR matches scalar oracle");
    return 0;
}
