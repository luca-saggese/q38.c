#include "../q38_moe_cuda.h"
#include "../q38_moe_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const size_t tokens = 1;
    std::vector<float> hidden(Q38_MOE_HIDDEN, 0.0f);
    std::vector<float> router(Q38_MOE_EXPERTS * Q38_MOE_HIDDEN, 0.0f);
    hidden[0] = 1.0f;
    router[3 * Q38_MOE_HIDDEN] = 2.0f;
    float *dh = nullptr, *dr = nullptr, *dl = nullptr;
    const size_t logits_bytes = Q38_MOE_EXPERTS * sizeof(float);
    if (cudaMalloc(&dh, hidden.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dr, router.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dl, logits_bytes) != cudaSuccess)
        return 1;
    cudaMemcpy(dh, hidden.data(), hidden.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dr, router.data(), router.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[128];
    if (!q38_moe_cuda_router(dh, tokens, dr, dl, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    std::vector<float> actual(Q38_MOE_EXPERTS);
    cudaMemcpy(actual.data(), dl, logits_bytes, cudaMemcpyDeviceToHost);
    if (std::fabs(actual[3] - 2.0f) > 1e-5f || actual[2] != 0.0f)
        return 1;
    cudaFree(dh); cudaFree(dr); cudaFree(dl);
    std::puts("test_m6_moe_cuda: naive router projection passed");
    return 0;
}
