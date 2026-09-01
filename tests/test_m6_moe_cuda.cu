#include "../q38_moe_cuda.h"
#include "../q38_moe_ref.h"
#include "../q38_quant.h"

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
    q38_moe_route10 route;
    if (!q38_moe_cuda_route(dh, 1, dr, dl, &route, 0, error,
                            sizeof(error))) {
        std::fprintf(stderr, "route failed: %s\n", error);
        return 1;
    }
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
        if (route.expert[k] != (k == 0 ? 3 : (k <= 3 ? k - 1 : k)) ||
            route.weight[k] <= 0.0f) {
            std::fprintf(stderr, "route mismatch k=%zu id=%u weight=%g\n", k,
                         route.expert[k], route.weight[k]);
            return 1;
        }
    std::vector<q38_q2_k_block> gate_up(1280 * 10), down(2560 * 3);
    for (auto &block : gate_up) { block.d = 0x3c00; block.dmin = 0; block.scales[0] = 1; }
    for (auto &block : down) { block.d = 0x3c00; block.dmin = 0; block.scales[0] = 1; }
    void *dgu = nullptr, *dd = nullptr;
    float *do_ = nullptr;
    if (cudaMalloc(&dgu, gate_up.size()*sizeof(q38_q2_k_block)) != cudaSuccess ||
        cudaMalloc(&dd, down.size()*sizeof(q38_q2_k_block)) != cudaSuccess ||
        cudaMalloc(&do_, Q38_MOE_HIDDEN*sizeof(float)) != cudaSuccess) return 1;
    cudaMemcpy(dgu, gate_up.data(), gate_up.size()*sizeof(q38_q2_k_block), cudaMemcpyHostToDevice);
    cudaMemcpy(dd, down.data(), down.size()*sizeof(q38_q2_k_block), cudaMemcpyHostToDevice);
    if (!q38_moe_cuda_expert_q2(dgu, dd, dh, do_, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "q2 failed: %s\n", error);
        return 1;
    }
    std::vector<float> expert(Q38_MOE_HIDDEN);
    cudaMemcpy(expert.data(), do_, expert.size()*sizeof(float), cudaMemcpyDeviceToHost);
    for (float v : expert) if (v != 0.0f) {
        std::fprintf(stderr, "q2 output %g\n", v);
        return 1;
    }
    cudaFree(dgu); cudaFree(dd); cudaFree(do_);
    cudaFree(dh); cudaFree(dr); cudaFree(dl);
    std::puts("test_m6_moe_cuda: naive router projection passed");
    return 0;
}
