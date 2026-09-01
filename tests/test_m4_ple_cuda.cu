#include "../q38_ple_cuda.h"
#include "../q38_ple_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

static int check_cuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return 0;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return 1;
}

template <typename Block>
static void init_blocks(Block *blocks, size_t count);

template <>
void init_blocks(q38_q2_k_block *blocks, size_t count) {
    std::memset(blocks, 0, count * sizeof(*blocks));
    for (size_t b = 0; b < count; ++b) {
        blocks[b].d = 0x3c00;
        for (size_t i = 0; i < sizeof(blocks[b].scales); ++i)
            blocks[b].scales[i] = 1;
        for (size_t i = 0; i < sizeof(blocks[b].qs); ++i)
            blocks[b].qs[i] = 0xe4;
    }
}

template <>
void init_blocks(q38_q4_k_block *blocks, size_t count) {
    std::memset(blocks, 0, count * sizeof(*blocks));
    for (size_t b = 0; b < count; ++b) {
        blocks[b].d = 0x3c00;
        for (size_t i = 0; i < sizeof(blocks[b].scales); ++i)
            blocks[b].scales[i] = 1;
        for (size_t i = 0; i < sizeof(blocks[b].qs); ++i)
            blocks[b].qs[i] = 0x21;
    }
}

template <typename Block>
static int check_type(uint32_t qtype) {
    constexpr size_t rows = 3;
    constexpr size_t row_width = 2560;
    constexpr size_t blocks_per_row = row_width / Q38_QUANT_QK_K;
    Block host_table[rows * blocks_per_row];
    init_blocks(host_table, rows * blocks_per_row);
    const uint32_t host_ids[] = {2, 0, 2, 1};
    constexpr size_t id_count = sizeof(host_ids) / sizeof(host_ids[0]);
    float host_expected[id_count * row_width];
    char error[128];

    for (size_t i = 0; i < id_count; ++i) {
        if (!q38_ple_decode_row_ref(
                qtype, host_table + host_ids[i] * blocks_per_row, row_width,
                host_expected + i * row_width, row_width, error,
                sizeof(error))) {
            std::fprintf(stderr, "scalar reference failed: %s\n", error);
            return 1;
        }
    }

    Block *device_table = nullptr;
    uint32_t *device_ids = nullptr;
    float *device_rows = nullptr;
    if (check_cuda(cudaMalloc(&device_table, sizeof(host_table)),
                   "cudaMalloc table") ||
        check_cuda(cudaMalloc(&device_ids, sizeof(host_ids)),
                   "cudaMalloc IDs") ||
        check_cuda(cudaMalloc(&device_rows, sizeof(host_expected)),
                   "cudaMalloc rows") ||
        check_cuda(cudaMemcpy(device_table, host_table, sizeof(host_table),
                              cudaMemcpyHostToDevice),
                   "copy table") ||
        check_cuda(cudaMemcpy(device_ids, host_ids, sizeof(host_ids),
                              cudaMemcpyHostToDevice),
                   "copy IDs")) {
        cudaFree(device_table);
        cudaFree(device_ids);
        cudaFree(device_rows);
        return 1;
    }

    if (!q38_ple_cuda_lookup_rows(
            qtype, device_table, rows, row_width, device_ids, id_count,
            device_rows, 0, error, sizeof(error))) {
        std::fprintf(stderr, "CUDA PLE lookup failed: %s\n", error);
        cudaFree(device_table);
        cudaFree(device_ids);
        cudaFree(device_rows);
        return 1;
    }
    if (check_cuda(cudaDeviceSynchronize(), "CUDA PLE lookup synchronize")) {
        cudaFree(device_table);
        cudaFree(device_ids);
        cudaFree(device_rows);
        return 1;
    }

    float host_actual[id_count * row_width];
    if (check_cuda(cudaMemcpy(host_actual, device_rows, sizeof(host_actual),
                              cudaMemcpyDeviceToHost),
                   "copy rows")) {
        cudaFree(device_table);
        cudaFree(device_ids);
        cudaFree(device_rows);
        return 1;
    }
    for (size_t i = 0; i < sizeof(host_actual) / sizeof(host_actual[0]); ++i) {
        if (std::fabs(host_actual[i] - host_expected[i]) > 1e-6f) {
            std::fprintf(stderr, "CUDA/scalar mismatch at %zu: %g vs %g\n",
                         i, host_actual[i], host_expected[i]);
            cudaFree(device_table);
            cudaFree(device_ids);
            cudaFree(device_rows);
            return 1;
        }
    }

    cudaFree(device_table);
    cudaFree(device_ids);
    cudaFree(device_rows);
    return 0;
}

int main() {
    if (check_type<q38_q2_k_block>(Q38_QUANT_Q2_K) ||
        check_type<q38_q4_k_block>(Q38_QUANT_Q4_K)) {
        return 1;
    }
    std::puts("test_m4_ple_cuda: CUDA row lookup matches scalar Q2/Q4 oracle");
    return 0;
}
