#include "../q38_cuda_primitives.h"
#include "../q38_gdn.h"
#include "../q38_moe_cuda.h"
#include "../q38_moe_ref.h"
#include "../q38_quant.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static bool sync_copy(float *host, const float *device, size_t count) {
    return cudaDeviceSynchronize() == cudaSuccess &&
           cudaMemcpy(host, device, count * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

static bool close_enough(const std::vector<float> &a,
                         const std::vector<float> &b, float tolerance,
                         const char *name) {
    float maximum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        maximum = std::fmax(maximum, std::fabs(a[i] - b[i]));
    if (maximum > tolerance) {
        std::fprintf(stderr, "%s mismatch: max_abs=%g\n", name, maximum);
        return false;
    }
    return true;
}

static bool matvec_tests() {
    const size_t rows = 3, cols = 32;
    std::vector<float> input(cols), expected(rows), actual(rows);
    for (size_t i = 0; i < cols; ++i) input[i] = (float)(i % 7) - 2.0f;

    std::vector<uint16_t> bf16_weights(rows * cols);
    for (size_t i = 0; i < bf16_weights.size(); ++i)
        bf16_weights[i] = bf16((float)((int)(i % 11) - 5) / 7.0f);
    for (size_t r = 0; r < rows; ++r) {
        expected[r] = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            uint32_t bits = (uint32_t)bf16_weights[r * cols + c] << 16;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            expected[r] += value * input[c];
        }
    }
    uint16_t *dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dw, bf16_weights.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, actual.size() * sizeof(float)) != cudaSuccess)
        return false;
    cudaMemcpy(dw, bf16_weights.data(), bf16_weights.size() * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_bf16_matvec(dw, rows, cols, dx, dy, 0, error,
                                   sizeof(error));
    ok = ok && sync_copy(actual.data(), dy, rows);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    if (!ok || !close_enough(expected, actual, 2e-4f, "BF16 matvec"))
        return false;

    std::vector<q38_gdn_q8_0_block> q8(rows);
    for (size_t r = 0; r < rows; ++r) {
        q8[r].d = 0x3400u; /* IEEE FP16 0.25, as required by Q8_0. */
        for (size_t c = 0; c < 32; ++c) q8[r].qs[c] = (int8_t)(c - 12 + r);
    }
    expected.assign(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            expected[r] += 0.25f * (float)q8[r].qs[c] * input[c];
    void *dq8 = nullptr;
    if (cudaMalloc(&dq8, q8.size() * sizeof(q8[0])) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, actual.size() * sizeof(float)) != cudaSuccess)
        return false;
    cudaMemcpy(dq8, q8.data(), q8.size() * sizeof(q8[0]),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    ok = q38_cuda_gdn_project(Q38_GDN_WEIGHT_Q8_0, dq8, rows, cols, dx, 1,
                              dy, 0, error, sizeof(error));
    ok = ok && sync_copy(actual.data(), dy, rows);
    cudaFree(dq8);
    cudaFree(dx);
    cudaFree(dy);
    return ok && close_enough(expected, actual, 2e-4f, "Q8 matvec");
}

static bool q2_expert_test() {
    const size_t gate_rows = 1280, down_rows = 640, cols = 2560;
    const size_t blocks_per_row = cols / Q38_QUANT_QK_K;
    std::vector<q38_q2_k_block> gate_up(gate_rows * blocks_per_row);
    std::vector<q38_q2_k_block> down(down_rows * blocks_per_row);
    for (q38_q2_k_block &block : gate_up) {
        block.d = 0x3c00u; /* IEEE FP16 1.0. */
        block.dmin = 0;
        std::memset(block.scales, 1, sizeof(block.scales));
        std::memset(block.qs, 0x55, sizeof(block.qs));
    }
    for (q38_q2_k_block &block : down) {
        block.d = 0x3800u; /* IEEE FP16 0.5. */
        block.dmin = 0;
        std::memset(block.scales, 1, sizeof(block.scales));
        std::memset(block.qs, 0x55, sizeof(block.qs));
    }
    std::vector<float> hidden(cols);
    for (size_t d = 0; d < cols; ++d)
        hidden[d] = (float)((int)(d % 13) - 6) / 17.0f;
    std::vector<float> gate_dense(gate_rows * cols);
    std::vector<float> down_dense(Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE);
    char error[256];
    for (size_t r = 0; r < gate_rows; ++r)
        if (!q38_quant_dequantize_row(
                Q38_QUANT_Q2_K, gate_up.data() + r * blocks_per_row,
                blocks_per_row, gate_dense.data() + r * cols, cols, error,
                sizeof(error)))
            return false;
    std::vector<float> row(cols);
    for (size_t r = 0; r < down_rows; ++r) {
        if (!q38_quant_dequantize_row(
                Q38_QUANT_Q2_K, down.data() + r * blocks_per_row,
                blocks_per_row, row.data(), cols, error, sizeof(error)))
            return false;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            down_dense[d * Q38_MOE_INTERMEDIATE + r] = row[d];
    }
    std::vector<float> expected(Q38_MOE_HIDDEN), actual(Q38_MOE_HIDDEN);
    if (!q38_moe_expert_ref(hidden.data(), gate_dense.data(), down_dense.data(),
                            expected.data(), error, sizeof(error)))
        return false;
    void *dgu = nullptr, *ddown = nullptr;
    float *dh = nullptr, *dout = nullptr;
    bool ok =
        cudaMalloc(&dgu, gate_up.size() * sizeof(gate_up[0])) == cudaSuccess &&
        cudaMalloc(&ddown, down.size() * sizeof(down[0])) == cudaSuccess &&
        cudaMalloc(&dh, hidden.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&dout, actual.size() * sizeof(float)) == cudaSuccess;
    if (ok) {
        cudaMemcpy(dgu, gate_up.data(), gate_up.size() * sizeof(gate_up[0]),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(ddown, down.data(), down.size() * sizeof(down[0]),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dh, hidden.data(), hidden.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        ok = q38_moe_cuda_expert_q2(dgu, ddown, dh, dout, 0, error,
                                    sizeof(error)) &&
             sync_copy(actual.data(), dout, actual.size());
    }
    cudaFree(dgu);
    cudaFree(ddown);
    cudaFree(dh);
    cudaFree(dout);
    return ok && close_enough(expected, actual, 2e-3f, "Q2 expert");
}

int main() {
    if (!matvec_tests() || !q2_expert_test()) return 1;
    std::puts("test_m6_gpu_forward: BF16/Q8 matvec and Q2 expert parity passed");
    return 0;
}
