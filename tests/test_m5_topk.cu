#include "../q38_topk_cuda.h"
#include "../q38_topk_ref.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

int main() {
    const float host[] = {
        1.0f, 4.0f, 4.0f, -1.0f, 3.0f, 0.0f, 4.0f, 2.0f,
        0.0f, 0.0f, 8.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f,
    };
    uint32_t expected[6];
    char error[128];
    if (!q38_topk_select_ref(host, 8, 3, expected, error, sizeof(error)) ||
        !q38_topk_select_ref(host + 8, 8, 3, expected + 3, error,
                             sizeof(error)))
        return 1;
    float *scores = nullptr;
    uint32_t *indices = nullptr;
    if (cudaMalloc(&scores, sizeof(host)) != cudaSuccess ||
        cudaMalloc(&indices, sizeof(expected)) != cudaSuccess)
        return 1;
    cudaMemcpy(scores, host, sizeof(host), cudaMemcpyHostToDevice);
    if (!q38_topk_cuda(scores, 2, 8, 3, indices, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    uint32_t actual[6];
    cudaMemcpy(actual, indices, sizeof(actual), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 6; ++i)
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "top-k mismatch at %zu\n", i);
            return 1;
        }
    uint32_t *argmax = nullptr;
    if (cudaMalloc(&argmax, 2 * sizeof(*argmax)) != cudaSuccess ||
        !q38_argmax_cuda(scores, 2, 8, argmax, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    uint32_t actual_argmax[2];
    cudaMemcpy(actual_argmax, argmax, sizeof(actual_argmax),
               cudaMemcpyDeviceToHost);
    if (actual_argmax[0] != 1 || actual_argmax[1] != 2) {
        std::fprintf(stderr, "argmax tie-breaking mismatch\n");
        return 1;
    }
    cudaFree(scores); cudaFree(indices);
    cudaFree(argmax);
    std::puts("test_m5_topk: deterministic scalar/CUDA top-k passed");
    return 0;
}
