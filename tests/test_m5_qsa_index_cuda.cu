#include "../q38_qsa_cuda.h"
#include "../q38_qsa_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const float keys[] = {1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
    const float queries[] = {1, 0, 0, 0, 0, 1, 0, 0};
    float expected[2];
    char error[128];
    if (!q38_qsa_index_scores_ref(keys, 3, queries, 1, 2, 4, 2,
                                   expected, error, sizeof(error)))
        return 1;
    float *dk = nullptr, *dq = nullptr, *ds = nullptr;
    if (cudaMalloc(&dk, sizeof(keys)) != cudaSuccess ||
        cudaMalloc(&dq, sizeof(queries)) != cudaSuccess ||
        cudaMalloc(&ds, sizeof(expected)) != cudaSuccess)
        return 1;
    cudaMemcpy(dk, keys, sizeof(keys), cudaMemcpyHostToDevice);
    cudaMemcpy(dq, queries, sizeof(queries), cudaMemcpyHostToDevice);
    if (!q38_qsa_cuda_index_scores(dk, 3, dq, 1, 2, 4, 2, ds, 0, error,
                                   sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    float actual[2];
    cudaMemcpy(actual, ds, sizeof(actual), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 2; ++i)
        if (std::fabs(actual[i] - expected[i]) > 2e-4f) {
            std::fprintf(stderr, "index score mismatch at %zu\n", i);
            return 1;
        }
    cudaFree(dk); cudaFree(dq); cudaFree(ds);
    std::puts("test_m5_qsa_index_cuda: naive CUDA indexer matches scalar");
    return 0;
}
