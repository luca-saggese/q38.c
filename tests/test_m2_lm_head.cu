#include "q38_cuda_primitives.h"
#include "q38_gguf.h"
#include "q38_oracle.h"
#include "q38_weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

static float bf16_to_float(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static uint64_t fnv1a(const uint8_t *data, size_t bytes) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; i++) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        std::fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 0, &weights, error, sizeof(error))) {
        std::fprintf(stderr, "bind failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    const uint32_t token_ids[] = {0, 9419, 109266, 248045, 248046, 248319};
    const size_t rows = sizeof(token_ids) / sizeof(token_ids[0]);
    const size_t cols = 2560, row_bytes = cols * sizeof(uint16_t);
    std::vector<uint16_t> host_weights(rows * cols);
    std::vector<float> hidden(cols), expected(rows), actual(rows);
    for (size_t i = 0; i < cols; i++)
        hidden[i] = (float)((int)(i % 31) - 15) * 0.0078125f +
                    (float)(i % 7) * 0.00003125f;
    const q38_tensor *output = weights.output;
    for (size_t r = 0; r < rows; r++) {
        uint64_t offset = output->abs_offset + (uint64_t)token_ids[r] * row_bytes;
        std::memcpy(host_weights.data() + r * cols, model->map + offset, row_bytes);
        for (size_t c = 0; c < cols; c++)
            expected[r] += bf16_to_float(host_weights[r * cols + c]) * hidden[c];
    }
    uint16_t *device_weights = nullptr;
    float *device_hidden = nullptr, *device_logits = nullptr;
    if (cudaMalloc(&device_weights, host_weights.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&device_hidden, hidden.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_logits, actual.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA allocation failed\n");
        cudaFree(device_weights); cudaFree(device_hidden); cudaFree(device_logits);
        q38_gguf_close(model);
        return 1;
    }
    cudaMemcpy(device_weights, host_weights.data(),
               host_weights.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_hidden, hidden.data(), hidden.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    if (!q38_cuda_bf16_matvec(device_weights, rows, cols, device_hidden,
                              device_logits, nullptr, error, sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(actual.data(), device_logits, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "LM-head CUDA probe failed: %s\n", error);
        cudaFree(device_weights); cudaFree(device_hidden); cudaFree(device_logits);
        q38_gguf_close(model);
        return 1;
    }
    cudaFree(device_weights); cudaFree(device_hidden); cudaFree(device_logits);
    q38_oracle_metrics metrics;
    q38_oracle_compare(expected.data(), actual.data(), rows, 1e-6f, &metrics);
    if (metrics.max_abs > 2e-4f) {
        std::fprintf(stderr, "LM-head parity failed: max=%g rms=%g\n",
                     metrics.max_abs, metrics.rms);
        q38_gguf_close(model);
        return 1;
    }
    std::vector<size_t> order(rows);
    for (size_t i = 0; i < rows; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return actual[a] > actual[b]; });
    std::printf("{\"gate\":\"M2-C08\",\"dtype\":\"BF16\",\"top_order\":[");
    for (size_t i = 0; i < rows; i++) {
        if (i) std::putchar(',');
        std::printf("%u", token_ids[order[i]]);
    }
    std::printf("],\"rows\":[");
    for (size_t i = 0; i < rows; i++) {
        uint64_t offset = output->abs_offset + (uint64_t)token_ids[i] * row_bytes;
        if (i) std::putchar(',');
        std::printf("{\"token_id\":%u,\"fnv1a64\":\"%016" PRIx64 "\"}",
                    token_ids[i], fnv1a(model->map + offset, row_bytes));
    }
    puts("]}");
    q38_gguf_close(model);
    return 0;
}
