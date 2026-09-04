#include "../q38_moe_cuda.h"
#include "../q38_moe_ref.h"
#include "../q38_quant.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

static void fill_block(q38_q4_k_block *block, size_t index) {
    block->d = 0x3c00;
    block->dmin = 0;
    for (size_t i = 0; i < sizeof(block->scales); ++i)
        block->scales[i] = (uint8_t)(1 + ((index + i) % 15));
    for (size_t i = 0; i < sizeof(block->qs); ++i)
        block->qs[i] = (uint8_t)((index * 37 + i * 11) & 0xff);
}

static bool decode_matrix(const std::vector<q38_q4_k_block> &blocks,
                          size_t rows, std::vector<float> *dense) {
    dense->assign(rows * Q38_MOE_HIDDEN, 0.0f);
    char error[128];
    for (size_t row = 0; row < rows; ++row) {
        if (!q38_quant_dequantize_row(
                Q38_QUANT_Q4_K, blocks.data() + row * 10, 10,
                dense->data() + row * Q38_MOE_HIDDEN, Q38_MOE_HIDDEN,
                error, sizeof(error))) {
            std::fprintf(stderr, "Q4 decode failed: %s\n", error);
            return false;
        }
    }
    return true;
}

static bool decode_down(const std::vector<q38_q4_k_block> &blocks,
                        std::vector<float> *dense) {
    dense->assign(Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE, 0.0f);
    std::vector<float> row(Q38_MOE_HIDDEN);
    char error[128];
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        if (!q38_quant_dequantize_row(
                Q38_QUANT_Q4_K, blocks.data() + i * 10, 10, row.data(),
                row.size(), error, sizeof(error))) {
            std::fprintf(stderr, "Q4 down decode failed: %s\n", error);
            return false;
        }
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            (*dense)[d * Q38_MOE_INTERMEDIATE + i] = row[d];
    }
    return true;
}

int main() {
    std::vector<float> hidden(Q38_MOE_HIDDEN);
    for (size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = std::sin((float)i * 0.013f) * 0.25f;
    std::vector<q38_q4_k_block> gate_up(2 * Q38_MOE_INTERMEDIATE * 10);
    std::vector<q38_q4_k_block> down(Q38_MOE_INTERMEDIATE * 10);
    for (size_t i = 0; i < gate_up.size(); ++i) fill_block(&gate_up[i], i);
    for (size_t i = 0; i < down.size(); ++i) fill_block(&down[i], i + gate_up.size());
    std::vector<float> gate_up_dense, down_dense;
    if (!decode_matrix(gate_up, 2 * Q38_MOE_INTERMEDIATE, &gate_up_dense) ||
        !decode_down(down, &down_dense))
        return 1;
    std::vector<float> expected(Q38_MOE_HIDDEN);
    char error[128];
    if (!q38_moe_expert_ref(hidden.data(), gate_up_dense.data(),
                            down_dense.data(), expected.data(), error,
                            sizeof(error))) {
        std::fprintf(stderr, "reference failed: %s\n", error);
        return 1;
    }
    q38_q4_k_block *dgu = nullptr, *ddown = nullptr;
    float *dh = nullptr, *dout = nullptr, *dmid = nullptr;
    if (cudaMalloc(&dgu, gate_up.size() * sizeof(q38_q4_k_block)) != cudaSuccess ||
        cudaMalloc(&ddown, down.size() * sizeof(q38_q4_k_block)) != cudaSuccess ||
        cudaMalloc(&dh, hidden.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dout, expected.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dmid, Q38_MOE_INTERMEDIATE * sizeof(float)) != cudaSuccess)
        return 1;
    cudaMemcpy(dgu, gate_up.data(), gate_up.size() * sizeof(q38_q4_k_block),
               cudaMemcpyHostToDevice);
    cudaMemcpy(ddown, down.data(), down.size() * sizeof(q38_q4_k_block),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dh, hidden.data(), hidden.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    if (!q38_moe_cuda_expert_q4_workspace(
            dgu, ddown, dh, dout, dmid, 0, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "Q4 kernel failed: %s\n", error);
        return 1;
    }
    std::vector<float> actual(expected.size());
    cudaMemcpy(actual.data(), dout, actual.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    float max_error = 0.0f, max_relative_error = 0.0f;
    size_t max_index = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error_value = fabsf(actual[i] - expected[i]);
        const float relative_error =
            error_value / fmaxf(1.0f, fabsf(expected[i]));
        if (error_value > max_error) {
            max_error = error_value;
            max_index = i;
        }
        max_relative_error = fmaxf(max_relative_error, relative_error);
    }
    cudaFree(dgu); cudaFree(ddown); cudaFree(dh); cudaFree(dout); cudaFree(dmid);
    if (max_relative_error > 2e-5f) {
        std::fprintf(stderr, "Q4 CUDA/reference mismatch: max error %g at %zu "
                     "relative=%g (actual=%g expected=%g)\n", max_error,
                     max_index, max_relative_error, actual[max_index],
                     expected[max_index]);
        return 1;
    }
    std::puts("test_m8_q4_moe_cuda: Q4_K expert CUDA/reference parity passed");
    return 0;
}
