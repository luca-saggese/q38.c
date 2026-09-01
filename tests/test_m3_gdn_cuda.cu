#include "q38_gdn.h"
#include "q38_cuda_primitives.h"
#include "q38_oracle.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float bf16_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static int compare(const std::vector<float> &expected,
                   const std::vector<float> &actual, const char *name,
                   float tolerance) {
    q38_oracle_metrics metrics;
    q38_oracle_compare(expected.data(), actual.data(), expected.size(),
                       1e-6f, &metrics);
    if (metrics.max_abs > tolerance) {
        std::fprintf(stderr, "%s mismatch: max_abs=%g\n", name,
                     metrics.max_abs);
        return 0;
    }
    return 1;
}

static int projection_tests() {
    const size_t rows = 2, cols = 4, tokens = 3;
    const std::vector<uint16_t> weights = {
        bf16(1.0f), bf16(-2.0f), bf16(0.5f), bf16(3.0f),
        bf16(-1.0f), bf16(0.25f), bf16(2.0f), bf16(-0.5f),
    };
    const std::vector<float> input = {
        1.0f, 2.0f, -1.0f, 0.5f,
        -2.0f, 0.25f, 3.0f, 1.5f,
        0.5f, -1.0f, 2.0f, -3.0f,
    };
    std::vector<float> expected(tokens * rows), actual(tokens * rows);
    for (size_t t = 0; t < tokens; t++)
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                expected[t * rows + r] +=
                    bf16_float(weights[r * cols + c]) *
                    input[t * cols + c];

    uint16_t *dw = nullptr;
    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dw, weights.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, actual.size() * sizeof(float)) != cudaSuccess)
        return 0;
    cudaMemcpy(dw, weights.data(), weights.size() * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_gdn_project(
        Q38_GDN_WEIGHT_BF16, dw, rows, cols, dx, tokens, dy, nullptr, error,
        sizeof(error));
    ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(actual.data(), dy, actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    if (!ok) {
        std::fprintf(stderr, "BF16 GDN projection failed: %s\n", error);
        return 0;
    }
    if (!compare(expected, actual, "BF16 projection", 2e-5f)) return 0;

    const size_t q8_rows = 2, q8_cols = 32;
    std::vector<q38_gdn_q8_0_block> q8(q8_rows);
    for (size_t r = 0; r < q8_rows; r++) {
        q8[r].d = 0x3800u; /* IEEE FP16 0.5, as used by Q8_0 blocks. */
        for (size_t c = 0; c < 32; c++) q8[r].qs[c] = (int8_t)(c + 1 + r);
    }
    std::vector<float> q8_input(q8_cols), q8_expected(q8_rows),
        q8_actual(q8_rows);
    for (size_t c = 0; c < q8_cols; c++) q8_input[c] = (float)(c % 5) - 2.0f;
    for (size_t r = 0; r < q8_rows; r++)
        for (size_t c = 0; c < q8_cols; c++)
            q8_expected[r] += 0.5f * (float)q8[r].qs[c] * q8_input[c];
    void *q8w = nullptr;
    float *q8x = nullptr, *q8y = nullptr;
    if (cudaMalloc(&q8w, q8.size() * sizeof(q8[0])) != cudaSuccess ||
        cudaMalloc(&q8x, q8_input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&q8y, q8_actual.size() * sizeof(float)) != cudaSuccess)
        return 0;
    cudaMemcpy(q8w, q8.data(), q8.size() * sizeof(q8[0]),
               cudaMemcpyHostToDevice);
    cudaMemcpy(q8x, q8_input.data(), q8_input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    ok = q38_cuda_gdn_project(
        Q38_GDN_WEIGHT_Q8_0, q8w, q8_rows, q8_cols, q8x, 1, q8y, nullptr,
        error, sizeof(error));
    ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(q8_actual.data(), q8y, q8_actual.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(q8w);
    cudaFree(q8x);
    cudaFree(q8y);
    if (!ok) {
        std::fprintf(stderr, "Q8 GDN projection failed: %s\n", error);
        return 0;
    }
    if (!compare(q8_expected, q8_actual, "Q8 projection", 2e-5f))
        return 0;

    /*
     * Q2 arithmetic is covered by M2's scalar/CUDA parity test; this
     * zero-block case additionally proves that the GDN dispatch reaches the
     * existing Q2 matvec helper without changing the token-major contract.
     */
    q38_q2_k_block q2 = {};
    std::vector<float> q2_input(Q38_QUANT_QK_K, 1.0f), q2_actual(1);
    void *q2w = nullptr;
    float *q2x = nullptr, *q2y = nullptr;
    if (cudaMalloc(&q2w, sizeof(q2)) != cudaSuccess ||
        cudaMalloc(&q2x, q2_input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&q2y, sizeof(float)) != cudaSuccess)
        return 0;
    cudaMemcpy(q2w, &q2, sizeof(q2), cudaMemcpyHostToDevice);
    cudaMemcpy(q2x, q2_input.data(), q2_input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    ok = q38_cuda_gdn_project(
        Q38_QUANT_Q2_K, q2w, 1, Q38_QUANT_QK_K, q2x, 1, q2y, nullptr,
        error, sizeof(error));
    ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(q2_actual.data(), q2y, sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(q2w);
    cudaFree(q2x);
    cudaFree(q2y);
    if (!ok) {
        std::fprintf(stderr, "Q2 GDN projection failed: %s\n", error);
        return 0;
    }
    return q2_actual[0] == 0.0f;
}

static void host_conv(const std::vector<float> &input, size_t tokens,
                      size_t channels, const std::vector<float> &kernel,
                      size_t kernel_size, std::vector<float> &history,
                      std::vector<float> &output) {
    const size_t history_tokens = kernel_size - 1u;
    for (size_t t = 0; t < tokens; t++) {
        for (size_t c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < kernel_size; tap++) {
                const size_t source = history_tokens + t -
                                      (kernel_size - 1u - tap);
                const float sample = source < history_tokens
                    ? history[source * channels + c]
                    : input[(source - history_tokens) * channels + c];
                sum += kernel[tap * channels + c] * sample;
            }
            output[t * channels + c] = sum;
        }
    }
    std::vector<float> combined(history_tokens * channels + input.size());
    std::memcpy(combined.data(), history.data(),
                history.size() * sizeof(float));
    std::memcpy(combined.data() + history.size(), input.data(),
                input.size() * sizeof(float));
    for (size_t h = 0; h < history_tokens; h++)
        std::memcpy(history.data() + h * channels,
                    combined.data() + (tokens + h) * channels,
                    channels * sizeof(float));
}

static int convolution_tests() {
    const size_t channels = 3, kernel_size = 4, tokens = 5;
    const std::vector<float> input = {
        1.0f, -2.0f, 0.5f, 2.0f, 1.0f, -1.5f, -0.5f, 3.0f, 2.5f,
        4.0f, -1.0f, 1.5f, -2.0f, 0.25f, 3.5f,
    };
    const std::vector<uint16_t> kernel_bf16 = {
        bf16(1.0f), bf16(-0.5f), bf16(0.25f),
        bf16(0.5f), bf16(2.0f), bf16(-1.0f),
        bf16(-0.25f), bf16(0.75f), bf16(1.5f),
        bf16(2.0f), bf16(-1.25f), bf16(0.5f),
    };
    std::vector<float> kernel(kernel_bf16.size());
    for (size_t i = 0; i < kernel.size(); i++)
        kernel[i] = bf16_float(kernel_bf16[i]);
    std::vector<float> expected_raw(tokens * channels), expected_history(
        (kernel_size - 1u) * channels, 0.0f);
    host_conv(input, tokens, channels, kernel, kernel_size, expected_history,
              expected_raw);
    std::vector<float> expected_silu(expected_raw.size());
    q38_oracle_silu(expected_raw.data(), expected_silu.data(),
                    expected_silu.size());

    uint16_t *dk = nullptr;
    float *dx = nullptr, *dh = nullptr, *dy = nullptr, *ds = nullptr;
    if (cudaMalloc(&dk, kernel_bf16.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dh, expected_history.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, expected_raw.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ds, expected_silu.size() * sizeof(float)) != cudaSuccess)
        return 0;
    std::vector<float> zero_history(expected_history.size(), 0.0f);
    cudaMemcpy(dk, kernel_bf16.data(),
               kernel_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dh, zero_history.data(), zero_history.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_gdn_conv(
        Q38_GDN_WEIGHT_BF16, dk, dx, tokens, channels, kernel_size, dh, dy,
        nullptr, error, sizeof(error));
    ok = ok && q38_cuda_silu(dy, ds, expected_silu.size(), nullptr, error,
                             sizeof(error)) &&
         cudaDeviceSynchronize() == cudaSuccess;
    std::vector<float> actual_raw(expected_raw.size()),
        actual_silu(expected_silu.size()), actual_history(expected_history.size());
    ok = ok && cudaMemcpy(actual_raw.data(), dy,
                          actual_raw.size() * sizeof(float),
                          cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(actual_silu.data(), ds,
                    actual_silu.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(actual_history.data(), dh,
                    actual_history.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) {
        std::fprintf(stderr, "GDN convolution failed: %s\n", error);
        cudaFree(dk); cudaFree(dx); cudaFree(dh); cudaFree(dy); cudaFree(ds);
        return 0;
    }
    if (!compare(expected_raw, actual_raw, "causal convolution", 2e-5f) ||
        !compare(expected_silu, actual_silu, "convolution SiLU order", 2e-5f) ||
        !compare(expected_history, actual_history, "convolution history", 0.0f)) {
        cudaFree(dk); cudaFree(dx); cudaFree(dh); cudaFree(dy); cudaFree(ds);
        return 0;
    }

    std::vector<float> silu_history(expected_history.size(), 0.0f);
    float *dh2 = nullptr, *ds2 = nullptr;
    if (cudaMalloc(&dh2, silu_history.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ds2, expected_silu.size() * sizeof(float)) != cudaSuccess)
        return 0;
    cudaMemcpy(dh2, silu_history.data(), silu_history.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    ok = q38_cuda_gdn_conv_silu(
        Q38_GDN_WEIGHT_BF16, dk, dx, tokens, channels, kernel_size, dh2, ds2,
        nullptr, error, sizeof(error)) &&
         cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(actual_silu.data(), ds2, actual_silu.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dh2);
    cudaFree(ds2);
    cudaFree(dk); cudaFree(dx); cudaFree(dh); cudaFree(dy); cudaFree(ds);
    if (!ok) {
        std::fprintf(stderr, "GDN convolution+SiLU failed: %s\n", error);
        return 0;
    }
    return compare(expected_silu, actual_silu, "sequenced convolution+SiLU",
                   2e-5f);
}

static int short_chunk_history_test() {
    const size_t channels = 3, kernel_size = 4, tokens = 1;
    const std::vector<float> input = {4.0f, 5.0f, 6.0f};
    const std::vector<float> kernel(kernel_size * channels, 1.0f);
    std::vector<float> expected_history = {1.0f, 2.0f, 3.0f,
                                           4.0f, 5.0f, 6.0f,
                                           7.0f, 8.0f, 9.0f};
    std::vector<float> expected_output(tokens * channels);
    host_conv(input, tokens, channels, kernel, kernel_size, expected_history,
              expected_output);
    uint16_t kernel_bf16[kernel_size * channels];
    for (size_t i = 0; i < kernel_size * channels; i++)
        kernel_bf16[i] = bf16(1.0f);
    float *dk = nullptr, *dx = nullptr, *dh = nullptr, *dy = nullptr;
    if (cudaMalloc(&dk, sizeof(kernel_bf16)) != cudaSuccess ||
        cudaMalloc(&dx, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dh, expected_history.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&dy, expected_output.size() * sizeof(float)) !=
            cudaSuccess)
        return 0;
    cudaMemcpy(dk, kernel_bf16, sizeof(kernel_bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    const std::vector<float> initial_history = {1.0f, 2.0f, 3.0f,
                                                4.0f, 5.0f, 6.0f,
                                                7.0f, 8.0f, 9.0f};
    cudaMemcpy(dh, initial_history.data(), initial_history.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_gdn_conv(
        Q38_GDN_WEIGHT_BF16, dk, dx, tokens, channels, kernel_size, dh, dy,
        nullptr, error, sizeof(error));
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
    std::vector<float> actual_history(expected_history.size());
    ok = ok && cudaMemcpy(actual_history.data(), dh,
                          actual_history.size() * sizeof(float),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dk);
    cudaFree(dx);
    cudaFree(dh);
    cudaFree(dy);
    if (!ok) {
        std::fprintf(stderr, "short-chunk history update failed: %s\n", error);
        return 0;
    }
    return compare(expected_history, actual_history, "short-chunk history",
                   0.0f);
}

static int qkv_layout_test() {
    const size_t tokens = 1;
    std::vector<float> qkv(Q38_GDN_QKV_CHANNELS);
    for (size_t i = 0; i < qkv.size(); i++) qkv[i] = (float)i;
    std::vector<float> q(Q38_GDN_KEY_CHANNELS), k(Q38_GDN_KEY_CHANNELS),
        v(Q38_GDN_VALUE_CHANNELS), repeated(Q38_GDN_VALUE_CHANNELS);
    float *dqkv = nullptr, *dq = nullptr, *dk = nullptr, *dv = nullptr,
          *dr = nullptr;
    if (cudaMalloc(&dqkv, qkv.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dq, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dk, k.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dv, v.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dr, repeated.size() * sizeof(float)) != cudaSuccess)
        return 0;
    cudaMemcpy(dqkv, qkv.data(), qkv.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    char error[256];
    bool ok = q38_cuda_gdn_split_qkv(dqkv, tokens, dq, dk, dv, nullptr, error,
                                     sizeof(error)) &&
              q38_cuda_gdn_repeat_key_heads(dk, tokens, dr, nullptr, error,
                                             sizeof(error)) &&
              cudaDeviceSynchronize() == cudaSuccess &&
              cudaMemcpy(q.data(), dq, q.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(k.data(), dk, k.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(v.data(), dv, v.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(repeated.data(), dr, repeated.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dqkv); cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dr);
    if (!ok) {
        std::fprintf(stderr, "GDN QKV layout kernel failed: %s\n", error);
        return 0;
    }
    if (q.front() != 0.0f || q.back() != (float)(Q38_GDN_KEY_CHANNELS - 1u) ||
        k.front() != (float)Q38_GDN_KEY_CHANNELS ||
        v.front() != (float)(2u * Q38_GDN_KEY_CHANNELS) ||
        repeated[3u * Q38_GDN_HEAD_DIM] !=
            (float)(Q38_GDN_KEY_CHANNELS + Q38_GDN_HEAD_DIM))
        return 0;
    return 1;
}

int main() {
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "M3-C07: CUDA device unavailable: %s\n",
                     cudaGetErrorString(status));
        return 2;
    }
    if (!projection_tests() || !convolution_tests() ||
        !short_chunk_history_test() || !qkv_layout_test())
        return 1;
    std::puts("test_m3_gdn_cuda: projection dispatch, causal history, QKV layout, and SiLU ordering passed");
    return 0;
}
