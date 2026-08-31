#include "q38_cuda_primitives.h"
#include "q38_oracle.h"
#include "q38_quant.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static unsigned next_byte(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 24) & 255u;
}

template <typename Block>
static void fill_blocks(std::vector<Block> &blocks, unsigned seed) {
    unsigned state = seed;
    unsigned char *bytes = reinterpret_cast<unsigned char *>(blocks.data());
    for (size_t i = 0; i < blocks.size() * sizeof(Block); i++) bytes[i] = next_byte(&state);
}

static int check_type(uint32_t type, size_t block_count) {
    const size_t block_bytes = type == Q38_QUANT_Q2_K
        ? Q38_QUANT_Q2_K_BLOCK_BYTES : Q38_QUANT_Q4_K_BLOCK_BYTES;
    std::vector<unsigned char> blocks(block_count * block_bytes);
    unsigned state = 0x12345678u + type;
    for (unsigned char &value : blocks) value = (unsigned char)next_byte(&state);
    std::vector<float> expected(block_count * Q38_QUANT_QK_K);
    std::vector<float> actual(expected.size());
    char error[256];
    if (!q38_quant_dequantize_row(type, blocks.data(), block_count,
                                  expected.data(), expected.size(),
                                  error, sizeof(error))) {
        std::fprintf(stderr, "CPU decode failed: %s\n", error);
        return 1;
    }
    void *device_blocks = nullptr;
    float *device_out = nullptr;
    if (cudaMalloc(&device_blocks, blocks.size()) != cudaSuccess ||
        cudaMalloc(&device_out, actual.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA allocation failed\n");
        cudaFree(device_blocks);
        cudaFree(device_out);
        return 1;
    }
    cudaMemcpy(device_blocks, blocks.data(), blocks.size(), cudaMemcpyHostToDevice);
    if (!q38_cuda_dequantize_row(type, device_blocks, block_count, device_out,
                                 nullptr, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(actual.data(), device_out, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "CUDA decode failed: %s\n", error);
        cudaFree(device_blocks);
        cudaFree(device_out);
        return 1;
    }
    cudaFree(device_blocks);
    cudaFree(device_out);
    q38_oracle_metrics metrics;
    q38_oracle_compare(expected.data(), actual.data(), actual.size(), 1e-6f,
                       &metrics);
    if (metrics.max_abs > 1e-6f || metrics.rms > 1e-7f) {
        std::fprintf(stderr, "CUDA parity failed type=%u max=%g rms=%g\n",
                     type, metrics.max_abs, metrics.rms);
        return 1;
    }
    return 0;
}

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "M2-C04: CUDA device unavailable: %s\n",
                     cudaGetErrorString(status));
        return 2;
    }
    if (check_type(Q38_QUANT_Q2_K, 2) || check_type(Q38_QUANT_Q4_K, 2))
        return 1;
    std::puts("test_m2_cuda: CUDA Q2_K/Q4_K dequant parity passed");
    return 0;
}
