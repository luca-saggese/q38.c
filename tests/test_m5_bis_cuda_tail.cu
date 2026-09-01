#include "../q38_qsa_cuda.h"
#include "../q38_qsa_ref.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const size_t lengths[] = {1,2,3,4,5,6,7,8,127,128,129,511,512,513,
                              2047,2048,2049,4095,4096,4097};
    char error[128];
    for (size_t n : lengths) {
        std::vector<float> keys(n * 4), query(4), expected((n + 3) / 4);
        for (size_t i = 0; i < keys.size(); ++i) keys[i] = float((i * 11) % 23) / 7.0f;
        query[0] = 1.0f;
        if (!q38_qsa_index_scores_ref(keys.data(), n, query.data(), 1, 1, 4,
                                      4, expected.data(), error, sizeof(error)))
            return 1;
        float *dk = nullptr, *dq = nullptr, *ds = nullptr;
        if (cudaMalloc(&dk, keys.size()*sizeof(float)) != cudaSuccess ||
            cudaMalloc(&dq, query.size()*sizeof(float)) != cudaSuccess ||
            cudaMalloc(&ds, expected.size()*sizeof(float)) != cudaSuccess) return 1;
        cudaMemcpy(dk, keys.data(), keys.size()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dq, query.data(), query.size()*sizeof(float), cudaMemcpyHostToDevice);
        if (!q38_qsa_cuda_index_scores(dk, n, dq, 1, 1, 4, 4, ds, 0,
                                       error, sizeof(error)) ||
            cudaDeviceSynchronize() != cudaSuccess) return 1;
        std::vector<float> actual(expected.size());
        cudaMemcpy(actual.data(), ds, actual.size()*sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < expected.size(); ++i)
            if (std::fabs(actual[i] - expected[i]) > 2e-4f) return 1;
        cudaFree(dk); cudaFree(dq); cudaFree(ds);
    }
    std::puts("test_m5_bis_cuda_tail: CUDA QSA tail and boundary stress passed");
    return 0;
}
