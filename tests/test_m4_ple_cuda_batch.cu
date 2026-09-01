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

static int run_type(uint32_t qtype, size_t block_bytes) {
    constexpr size_t table_rows = 4;
    constexpr size_t row_width = 2560;
    constexpr size_t blocks_per_row = row_width / Q38_QUANT_QK_K;
    const uint32_t host_ids[] = {2, 0, 2, 1, 0, 3, 2};
    constexpr size_t id_count = sizeof(host_ids) / sizeof(host_ids[0]);

    void *host_table = std::calloc(table_rows * blocks_per_row, block_bytes);
    if (!host_table) return 1;
    for (size_t row = 0; row < table_rows; ++row) {
        std::memset((char *)host_table + row * blocks_per_row * block_bytes,
                    (int)(0x20 + row), blocks_per_row * block_bytes);
    }
    float host_expected[id_count * row_width];
    char error[128];
    for (size_t i = 0; i < id_count; ++i) {
        if (!q38_ple_decode_row_ref(
                qtype,
                (const char *)host_table +
                    (size_t)host_ids[i] * blocks_per_row * block_bytes,
                row_width, host_expected + i * row_width, row_width, error,
                sizeof(error))) {
            std::fprintf(stderr, "scalar decode failed: %s\n", error);
            std::free(host_table);
            return 1;
        }
    }

    void *device_table = nullptr;
    float *device_rows = nullptr;
    if (check_cuda(cudaMalloc(&device_table,
                              table_rows * blocks_per_row * block_bytes),
                   "cudaMalloc table") ||
        check_cuda(cudaMalloc(&device_rows, sizeof(host_expected)),
                   "cudaMalloc output") ||
        check_cuda(cudaMemcpy(device_table, host_table,
                              table_rows * blocks_per_row * block_bytes,
                              cudaMemcpyHostToDevice),
                   "copy table")) {
        cudaFree(device_table);
        cudaFree(device_rows);
        std::free(host_table);
        return 1;
    }

    q38_ple_cuda_lookup_stats stats;
    if (!q38_ple_cuda_lookup_rows_dedup(
            qtype, device_table, table_rows, row_width, host_ids, id_count,
            device_rows, 0, &stats, error, sizeof(error))) {
        std::fprintf(stderr, "deduplicated CUDA lookup failed: %s\n", error);
        cudaFree(device_table);
        cudaFree(device_rows);
        std::free(host_table);
        return 1;
    }
    if (stats.input_count != id_count || stats.unique_count != 4) {
        std::fprintf(stderr, "unexpected dedup stats: %zu/%zu\n",
                     stats.input_count, stats.unique_count);
        cudaFree(device_table);
        cudaFree(device_rows);
        std::free(host_table);
        return 1;
    }
    float host_actual[id_count * row_width];
    if (check_cuda(cudaMemcpy(host_actual, device_rows, sizeof(host_actual),
                              cudaMemcpyDeviceToHost),
                   "copy output")) {
        cudaFree(device_table);
        cudaFree(device_rows);
        std::free(host_table);
        return 1;
    }
    for (size_t i = 0; i < sizeof(host_actual) / sizeof(host_actual[0]); ++i) {
        if (std::fabs(host_actual[i] - host_expected[i]) > 1e-6f) {
            std::fprintf(stderr, "deduplicated mismatch at %zu: %g vs %g\n",
                         i, host_actual[i], host_expected[i]);
            cudaFree(device_table);
            cudaFree(device_rows);
            std::free(host_table);
            return 1;
        }
    }
    cudaFree(device_table);
    cudaFree(device_rows);
    std::free(host_table);
    return 0;
}

int main() {
    if (run_type(Q38_QUANT_Q2_K, sizeof(q38_q2_k_block)) ||
        run_type(Q38_QUANT_Q4_K, sizeof(q38_q4_k_block))) {
        return 1;
    }
    std::puts("test_m4_ple_cuda_batch: deduplicated CUDA lookup preserves IDs and output order");
    return 0;
}
