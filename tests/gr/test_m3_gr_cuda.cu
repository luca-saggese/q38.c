#include "q38_gr.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

static void make_fixture(std::vector<float> &residual, std::vector<float> &gamma,
                         std::vector<float> &down, std::vector<float> &up,
                         std::vector<float> &inject,
                         std::vector<float> &block_output) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    residual.assign(width, 0.0f);
    gamma.assign(width, 1.0f);
    down.assign(Q38_GR_RANK * width, 0.0f);
    up.assign(width * Q38_GR_RANK, 0.0f);
    inject.assign(Q38_GR_BRANCHES * width, 0.0f);
    block_output.assign(Q38_GR_HIDDEN, 0.0f);
    residual[0] = 2.0f;
    residual[1] = -1.0f;
    residual[Q38_GR_HIDDEN] = -3.0f;
    residual[Q38_GR_HIDDEN + 1] = 0.5f;
    residual[2 * Q38_GR_HIDDEN] = 1.5f;
    residual[3 * Q38_GR_HIDDEN + 1] = -2.0f;
    down[0 * width + 0] = 1.2f;
    down[1 * width + 1] = -0.7f;
    down[2 * width + Q38_GR_HIDDEN] = 0.8f;
    down[3 * width + 3 * Q38_GR_HIDDEN + 1] = 1.1f;
    up[0 * Q38_GR_RANK + 0] = 0.9f;
    up[0 * Q38_GR_RANK + 2] = -0.4f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + 1] = -0.8f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + 2] = 0.6f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + 0] = -0.5f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + 3] = 0.7f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + 1] = 0.3f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + 3] = -0.2f;
    up[Q38_GR_RANK + 0] = -0.2f;
    up[Q38_GR_RANK + 3] = 0.4f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 1] = 0.6f;
    up[Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 2] = -0.3f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 0] = 0.7f;
    up[2 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 3] = 0.1f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 1] = -0.9f;
    up[3 * Q38_GR_HIDDEN * Q38_GR_RANK + Q38_GR_RANK + 3] = 0.5f;
    inject[0 * width + 0] = 0.4f;
    inject[0 * width + Q38_GR_HIDDEN] = -0.6f;
    inject[1 * width + 1] = -0.8f;
    inject[1 * width + 3 * Q38_GR_HIDDEN + 1] = 0.5f;
    inject[2 * width + Q38_GR_HIDDEN] = 0.3f;
    inject[3 * width + 3 * Q38_GR_HIDDEN + 1] = 1.1f;
    block_output[0] = 0.25f;
    block_output[1] = -0.125f;
}

static bool close_to(float actual, float expected) {
    return std::fabs(actual - expected) <= 3e-5f;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    std::vector<float> residual, gamma, down, up, inject, block_output;
    make_fixture(residual, gamma, down, up, inject, block_output);
    std::vector<float> actual_input(Q38_GR_HIDDEN);
    std::vector<float> actual_updated(width);
    const float expected_input[] = {10.79133470f, 1.17167685f};
    const float expected_updated[][2] = {
        {2.49999696f, -1.24999848f},
        {-2.92896527f, 0.46448264f},
        {1.51157222f, -0.00578611f},
        {0.00000046f, -2.00000023f},
    };
    float *dr = nullptr, *dg = nullptr, *dd = nullptr, *du = nullptr;
    float *di = nullptr, *dbo = nullptr, *dinput = nullptr, *dupdated = nullptr;
    auto alloc = [](float **p, size_t n) {
        return cudaMalloc(p, n * sizeof(float)) == cudaSuccess;
    };
    if (!alloc(&dr, residual.size()) || !alloc(&dg, gamma.size()) ||
        !alloc(&dd, down.size()) || !alloc(&du, up.size()) ||
        !alloc(&di, inject.size()) || !alloc(&dbo, block_output.size()) ||
        !alloc(&dinput, actual_input.size()) || !alloc(&dupdated, actual_updated.size()))
        return 1;
    bool ok = cudaMemcpy(dr, residual.data(), residual.size() * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(dg, gamma.data(), gamma.size() * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(dd, down.data(), down.size() * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(du, up.data(), up.size() * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(di, inject.data(), inject.size() * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(dbo, block_output.data(),
                         block_output.size() * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    char error[256] = {};
    std::vector<float> zero_residual(width, 0.0f);
    std::vector<float> zero_down(down.size(), 0.0f);
    std::vector<float> zero_up(up.size(), 0.0f);
    std::vector<float> zero_inject(inject.size(), 0.0f);
    std::vector<float> zero_block(Q38_GR_HIDDEN, 0.125f);
    ok = ok &&
         cudaMemcpy(dr, zero_residual.data(), zero_residual.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(dd, zero_down.data(), zero_down.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(du, zero_up.data(), zero_up.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(di, zero_inject.data(), zero_inject.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(dbo, zero_block.data(), zero_block.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         q38_cuda_gr_collapse(dr, dg, dd, du, di, dbo, dinput, dupdated,
                              nullptr, error, sizeof(error)) &&
         cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(actual_input.data(), dinput,
                    actual_input.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(actual_updated.data(), dupdated,
                    actual_updated.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) {
        for (size_t i = 0; i < Q38_GR_HIDDEN; i++)
            if (!close_to(actual_input[i], 0.0f))
                ok = false;
        for (size_t i = 0; i < width; i++)
            if (!close_to(actual_updated[i], 0.125f))
                ok = false;
    }
    ok = ok &&
         cudaMemcpy(dr, residual.data(), residual.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(dd, down.data(), down.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(du, up.data(), up.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(di, inject.data(), inject.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(dbo, block_output.data(), block_output.size() * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && q38_cuda_gr_collapse(dr, dg, dd, du, di, dbo, dinput, dupdated,
                                    nullptr, error, sizeof(error)) &&
         cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(actual_input.data(), dinput,
                    actual_input.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(actual_updated.data(), dupdated,
                    actual_updated.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dr); cudaFree(dg); cudaFree(dd); cudaFree(du);
    cudaFree(di); cudaFree(dbo); cudaFree(dinput); cudaFree(dupdated);
    if (!ok) {
        std::fprintf(stderr, "CUDA GR failed: %s\n", error);
        return 1;
    }
    if (!close_to(actual_input[0], expected_input[0]) ||
        !close_to(actual_input[1], expected_input[1])) {
        std::fprintf(stderr, "external CUDA GR read golden mismatch\n");
        return 1;
    }
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; branch++)
        if (!close_to(actual_updated[branch * Q38_GR_HIDDEN],
                      expected_updated[branch][0]) ||
            !close_to(actual_updated[branch * Q38_GR_HIDDEN + 1],
                      expected_updated[branch][1])) {
            std::fprintf(stderr, "external CUDA GR write golden mismatch\n");
            return 1;
        }
    std::puts("test_m3_gr_cuda: CUDA matches independent non-zero GR goldens");
    return 0;
}
