#include "q38_directional_steering.h"
#include "q38_forward_cuda.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool check_cuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static void fill_direction(float *row, uint32_t layer) {
    double norm2 = 0.0;
    for (uint32_t i = 0; i < Q38_DIRECTIONAL_STEERING_HIDDEN; ++i) {
        const float value = std::sin((float)(layer + 1u) * 0.013f *
                                     (float)(i + 1u)) +
                            0.25f * std::cos((float)(i + 3u) * 0.007f);
        row[i] = value;
        norm2 += (double)value * value;
    }
    const float inverse_norm = 1.0f / std::sqrt((float)norm2);
    for (uint32_t i = 0; i < Q38_DIRECTIONAL_STEERING_HIDDEN; ++i)
        row[i] *= inverse_norm;
}

int main() {
    int devices = 0;
    if (!check_cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount") ||
        devices == 0) {
        std::fprintf(stderr, "test_q38_directional_steering_cuda: no CUDA device\n");
        return 2;
    }

    const size_t direction_bytes = Q38_DIRECTIONAL_STEERING_BYTES;
    float *directions = (float *)std::malloc(direction_bytes);
    if (!directions) return 1;
    for (uint32_t layer = 0; layer < Q38_DIRECTIONAL_STEERING_LAYERS; ++layer)
        fill_direction(directions +
                           (size_t)layer * Q38_DIRECTIONAL_STEERING_HIDDEN,
                       layer);

    q38_directional_steering steering;
    std::memset(&steering, 0, sizeof(steering));
    steering.directions = directions;
    steering.layers = Q38_DIRECTIONAL_STEERING_LAYERS;
    steering.hidden_size = Q38_DIRECTIONAL_STEERING_HIDDEN;
    char error[256] = {0};
    q38_forward_cuda_context *context =
        q38_forward_cuda_context_create(error, sizeof(error));
    if (!context) {
        std::fprintf(stderr, "test_q38_directional_steering_cuda: skipped: %s\n",
                     error);
        std::free(directions);
        return 2;
    }
    if (!q38_forward_cuda_load_directional_steering(
            context, &steering, error, sizeof(error))) {
        std::fprintf(stderr, "CUDA steering setup failed: %s\n", error);
        q38_forward_cuda_context_destroy(context);
        std::free(directions);
        return 1;
    }

    constexpr size_t rows = 3;
    const size_t value_count = rows * Q38_DIRECTIONAL_STEERING_HIDDEN;
    const size_t value_bytes = value_count * sizeof(float);
    float *input = (float *)std::malloc(value_bytes);
    float *expected = (float *)std::malloc(value_bytes);
    float *actual = (float *)std::malloc(value_bytes);
    float *device_values = nullptr;
    if (!input || !expected || !actual ||
        !check_cuda(cudaMalloc((void **)&device_values, value_bytes),
                    "cudaMalloc")) {
        std::free(input);
        std::free(expected);
        std::free(actual);
        q38_forward_cuda_context_destroy(context);
        std::free(directions);
        return 1;
    }
    for (size_t i = 0; i < value_count; ++i)
        input[i] = std::sin((float)(i + 5u) * 0.011f) * 0.5f +
                   std::cos((float)(i + 7u) * 0.003f) * 0.25f;

    const float scales[] = {-2.0f, -1.0f, -0.5f, 0.0f,
                            0.5f, 1.0f, 2.0f};
    const uint32_t layers[] = {0u, 17u, 47u};
    for (float scale : scales) {
        for (uint32_t layer : layers) {
            std::memcpy(expected, input, value_bytes);
            q38_directional_steering_apply_cpu(
                expected, rows, &steering, layer, scale);
            if (!check_cuda(cudaMemcpy(
                                device_values, input, value_bytes,
                                cudaMemcpyHostToDevice),
                            "cudaMemcpy H2D") ||
                !q38_forward_cuda_apply_directional_steering(
                    context, device_values, layer,
                    Q38_DIRECTIONAL_STEERING_HIDDEN, rows, scale,
                    error, sizeof(error)) ||
                !check_cuda(cudaStreamSynchronize(
                                (cudaStream_t)q38_forward_cuda_stream(context)),
                            "cudaStreamSynchronize") ||
                !check_cuda(cudaMemcpy(
                                actual, device_values, value_bytes,
                                cudaMemcpyDeviceToHost),
                            "cudaMemcpy D2H")) {
                std::fprintf(stderr, "CUDA steering execution failed: %s\n",
                             error);
                cudaFree(device_values);
                std::free(input);
                std::free(expected);
                std::free(actual);
                q38_forward_cuda_context_destroy(context);
                std::free(directions);
                return 1;
            }
            for (size_t i = 0; i < value_count; ++i) {
                if (std::fabs(actual[i] - expected[i]) > 2.0e-4f) {
                    std::fprintf(stderr,
                                 "CUDA steering mismatch layer=%u scale=%g "
                                 "index=%zu expected=%g actual=%g\n",
                                 layer, scale, i, expected[i], actual[i]);
                    cudaFree(device_values);
                    std::free(input);
                    std::free(expected);
                    std::free(actual);
                    q38_forward_cuda_context_destroy(context);
                    std::free(directions);
                    return 1;
                }
            }
        }
    }

    cudaFree(device_values);
    std::free(input);
    std::free(expected);
    std::free(actual);
    q38_forward_cuda_context_destroy(context);
    std::free(directions);
    std::puts("test_q38_directional_steering_cuda: all tests passed");
    return 0;
}
