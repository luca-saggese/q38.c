#include "../q38_cuda_primitives.h"
#include "../q38_gguf.h"
#include "../q38_weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static float bf16(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0f;
    cudaEventElapsedTime(&value, start, stop);
    return value;
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) return 1;
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 0, &weights, error, sizeof(error)) ||
        !weights.output || weights.output->type != 30) {
        q38_gguf_close(model);
        return 1;
    }
    const size_t rows = 248320, cols = 2560;
    const size_t matrix_bytes = rows * cols * sizeof(uint16_t);
    const uint16_t *host_weights =
        (const uint16_t *)(model->map + weights.output->abs_offset);
    std::vector<float> input(cols), actual(rows), expected(8);
    for (size_t i = 0; i < cols; ++i)
        input[i] = (float)((int)(i % 31) - 15) * 0.0078125f;
    const size_t checks[] = {0, 1, 9419, 109266, 248045, 248046, 248319, 100000};
    for (size_t i = 0; i < 8; ++i) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c)
            sum += bf16(host_weights[checks[i] * cols + c]) * input[c];
        expected[i] = sum;
    }
    uint16_t *device_weights = nullptr;
    float *device_input = nullptr, *device_output = nullptr;
    if (cudaMalloc(&device_weights, matrix_bytes) != cudaSuccess ||
        cudaMalloc(&device_input, cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_output, rows * sizeof(float)) != cudaSuccess)
        return 1;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    cudaMemcpyAsync(device_weights, host_weights, matrix_bytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_input, input.data(), cols * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    const double h2d_ms = elapsed_ms(start, stop);
    const unsigned geometries[] = {128, 256, 512};
    for (unsigned threads : geometries) {
        for (int warmup = 0; warmup < 1; ++warmup)
            if (!q38_cuda_bf16_matvec_configured(
                    device_weights, rows, cols, device_input, device_output,
                    threads, stream, error, sizeof(error)))
                return 1;
        cudaStreamSynchronize(stream);
        double samples[10];
        for (int iteration = 0; iteration < 10; ++iteration) {
            cudaEventRecord(start, stream);
            if (!q38_cuda_bf16_matvec_configured(
                    device_weights, rows, cols, device_input, device_output,
                    threads, stream, error, sizeof(error)))
                return 1;
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            samples[iteration] = elapsed_ms(start, stop);
        }
        std::sort(samples, samples + 10);
        cudaEventRecord(start, stream);
        cudaMemcpyAsync(actual.data(), device_output, rows * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        const double d2h_ms = elapsed_ms(start, stop);
        float max_error = 0.0f;
        for (size_t i = 0; i < 8; ++i)
            max_error = std::max(max_error,
                                 std::abs(actual[checks[i]] - expected[i]));
        printf("{\"threads\":%u,\"rows\":%zu,\"cols\":%zu,"
               "\"matrix_bytes\":%zu,\"input_bytes\":%zu,\"output_bytes\":%zu,"
               "\"h2d_ms\":%.6f,"
               "\"median_kernel_ms\":%.6f,\"p95_kernel_ms\":%.6f,"
               "\"effective_GBps\":%.6f,\"d2h_ms\":%.6f,\"sync_ms\":0.0,"
               "\"max_abs\":%.9g,"
               "\"grid_x\":%zu,\"rows_per_block\":1}\n",
               threads, rows, cols, matrix_bytes, cols * sizeof(float),
               rows * sizeof(float), h2d_ms, samples[5], samples[9],
               (double)matrix_bytes / (samples[5] * 1e6), d2h_ms, max_error,
               rows);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(device_weights);
    cudaFree(device_input);
    cudaFree(device_output);
    q38_gguf_close(model);
    return 0;
}
