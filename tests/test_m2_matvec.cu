#include "q38_cuda_primitives.h"
#include "q38_oracle.h"
#include "q38_quant.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static unsigned next_byte(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 24) & 255u;
}

static float bf16_to_float_host(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static int run_q2() {
    const size_t rows = 3, cols = 512, blocks_per_row = 2;
    std::vector<unsigned char> weights(rows * blocks_per_row *
                                       Q38_QUANT_Q2_K_BLOCK_BYTES);
    unsigned state = 0x42u;
    for (unsigned char &value : weights) value = (unsigned char)next_byte(&state);
    std::vector<float> input(cols), expected(rows), actual(rows);
    for (float &value : input) value = (float)((int)next_byte(&state) - 128) / 37.0f;
    for (size_t row = 0; row < rows; row++) {
        std::vector<float> decoded(cols);
        char error[128];
        if (!q38_quant_dequantize_row(
                Q38_QUANT_Q2_K,
                weights.data() + row * blocks_per_row * Q38_QUANT_Q2_K_BLOCK_BYTES,
                blocks_per_row, decoded.data(), cols, error, sizeof(error)))
            return 1;
        for (size_t col = 0; col < cols; col++) expected[row] += decoded[col] * input[col];
    }
    void *dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dw, weights.size()) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, actual.size() * sizeof(float)) != cudaSuccess)
        return 1;
    cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_q2_matvec(dw, rows, cols, dx, dy, nullptr, error,
                                 sizeof(error)) &&
              cudaDeviceSynchronize() == cudaSuccess &&
              cudaMemcpy(actual.data(), dy, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    if (!ok) {
        std::fprintf(stderr, "Q2 matvec failed: %s\n", error);
        return 1;
    }
    q38_oracle_metrics metrics;
    q38_oracle_compare(expected.data(), actual.data(), rows, 1e-5f, &metrics);
    return metrics.max_abs > 2e-4f;
}

static int run_bf16() {
    const size_t rows = 4, cols = 37;
    std::vector<uint16_t> weights(rows * cols);
    std::vector<float> input(cols), expected(rows), actual(rows);
    unsigned state = 0x99u;
    for (uint16_t &value : weights) value = (uint16_t)next_byte(&state) << 8;
    for (float &value : input) value = (float)((int)next_byte(&state) - 128) / 19.0f;
    for (size_t row = 0; row < rows; row++)
        for (size_t col = 0; col < cols; col++)
            expected[row] += bf16_to_float_host(weights[row * cols + col]) *
                input[col];
    float *dx = nullptr, *dy = nullptr;
    uint16_t *dw = nullptr;
    if (cudaMalloc(&dw, weights.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, actual.size() * sizeof(float)) != cudaSuccess)
        return 1;
    cudaMemcpy(dw, weights.data(), weights.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_bf16_matvec(dw, rows, cols, dx, dy, nullptr, error,
                                   sizeof(error)) &&
              cudaDeviceSynchronize() == cudaSuccess &&
              cudaMemcpy(actual.data(), dy, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
    if (!ok) {
        std::fprintf(stderr, "BF16 matvec failed: %s\n", error);
        return 1;
    }
    q38_oracle_metrics metrics;
    q38_oracle_compare(expected.data(), actual.data(), rows, 1e-6f, &metrics);
    return metrics.max_abs > 2e-4f;
}

int main() {
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) return 2;
    if (run_q2() || run_bf16()) return 1;
    std::puts("test_m2_matvec: Q2 expert and BF16 core matvec parity passed");
    return 0;
}
