#include "../q38_qsa_cuda.h"
#include "../q38_rope_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

int main() {
    constexpr size_t cols = 8, tokens = 2, q_rows = 4, k_rows = 3, v_rows = 2;
    std::vector<uint16_t> q(q_rows * cols), k(k_rows * cols), v(v_rows * cols);
    std::vector<float> input(tokens * cols);
    for (size_t i = 0; i < input.size(); ++i) input[i] = (float)(i + 1);
    for (size_t i = 0; i < q.size(); ++i) q[i] = f32_to_bf16((float)(i % cols == i / cols ? 1 : 0));
    for (size_t i = 0; i < k.size(); ++i) k[i] = f32_to_bf16((float)(i % cols == i / cols ? 1 : 0));
    for (size_t i = 0; i < v.size(); ++i) v[i] = f32_to_bf16((float)(i % cols == i / cols ? 1 : 0));
    uint16_t *dq = nullptr, *dk = nullptr, *dv = nullptr;
    float *di = nullptr, *oq = nullptr, *ok = nullptr, *ov = nullptr;
    if (cudaMalloc(&dq, q.size() * sizeof(*dq)) != cudaSuccess ||
        cudaMalloc(&dk, k.size() * sizeof(*dk)) != cudaSuccess ||
        cudaMalloc(&dv, v.size() * sizeof(*dv)) != cudaSuccess ||
        cudaMalloc(&di, input.size() * sizeof(*di)) != cudaSuccess ||
        cudaMalloc(&oq, q_rows * tokens * sizeof(*oq)) != cudaSuccess ||
        cudaMalloc(&ok, k_rows * tokens * sizeof(*ok)) != cudaSuccess ||
        cudaMalloc(&ov, v_rows * tokens * sizeof(*ov)) != cudaSuccess) return 1;
    cudaMemcpy(dq, q.data(), q.size() * sizeof(*dq), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * sizeof(*dk), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * sizeof(*dv), cudaMemcpyHostToDevice);
    cudaMemcpy(di, input.data(), input.size() * sizeof(*di), cudaMemcpyHostToDevice);
    char error[128];
    if (!q38_qsa_cuda_project_main(dq, q_rows, dk, k_rows, dv, v_rows, cols,
                                   di, tokens, oq, ok, ov, 0, error,
                                   sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "projection failed: %s\n", error);
        return 1;
    }
    std::vector<float> got(q_rows * tokens);
    cudaMemcpy(got.data(), oq, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t r = 0; r < q_rows; ++r)
            if (std::fabs(got[t * q_rows + r] - input[t * cols + r]) > 0.02f)
                return 1;
    float host[64], expected[64];
    for (size_t i = 0; i < 64; ++i) host[i] = expected[i] = (float)i / 11.0f;
    q38_rope_config cfg = {.theta = 10000000.0f, .n_dims = 64,
                           .sections = {11, 11, 10, 0}, .interleaved = true};
    const int64_t position[] = {7, 7, 7, 7};
    if (!q38_rope_apply_ref(&cfg, position, host, expected, 64, error,
                            sizeof(error)))
        return 1;
    float *dr = nullptr;
    cudaMalloc(&dr, sizeof(host));
    cudaMemcpy(dr, host, sizeof(host), cudaMemcpyHostToDevice);
    if (!q38_qsa_cuda_apply_rope(dr, 1, 1, 64, 64, 7,
                                 cfg.sections, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    cudaMemcpy(host, dr, sizeof(host), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 64; ++i)
        if (std::fabs(host[i] - expected[i]) > 2e-4f) return 1;
    cudaFree(dr); cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(di);
    cudaFree(oq); cudaFree(ok); cudaFree(ov);
    std::puts("test_m5_qsa_cuda: main QKV projection and RoPE match reference");
    return 0;
}
