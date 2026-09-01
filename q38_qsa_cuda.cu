#include "q38_qsa_cuda.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

__device__ static float bf16_to_float(uint16_t value) {
    return __int_as_float((int)((uint32_t)value << 16));
}

__global__ static void project_kernel(const uint16_t *weights, size_t rows,
                                      size_t cols, const float *input,
                                      size_t tokens, float *output) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = rows * tokens;
    if (index >= elements) return;
    const size_t token = index / rows;
    const size_t row = index % rows;
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col)
        sum += bf16_to_float(weights[row * cols + col]) *
               input[token * cols + col];
    output[index] = sum;
}

__global__ static void rope_kernel(float *tensor, size_t tokens, size_t heads,
                                   size_t head_dim, size_t rotary_dims,
                                   int64_t position, uint32_t s0, uint32_t s1,
                                   uint32_t s2, uint32_t s3) {
    const size_t pair = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t pairs = rotary_dims / 2;
    const size_t total = tokens * heads * pairs;
    if (pair >= total) return;
    const size_t p = pair % pairs;
    const size_t head = (pair / pairs) % heads;
    const size_t token = pair / (pairs * heads);
    (void)s0;
    (void)s1;
    (void)s2;
    (void)s3;
    float theta = (float)position;
    const float theta_scale = powf(10000000.0f, -2.0f / (float)rotary_dims);
    for (size_t i = 0; i < p; ++i) theta *= theta_scale;
    const float angle = theta;
    const float c = cosf(angle), s = sinf(angle);
    const size_t base = (token * heads + head) * head_dim;
    const size_t a = base + p;
    const size_t b = base + rotary_dims / 2 + p;
    const float x = tensor[a], y = tensor[b];
    tensor[a] = x * c - y * s;
    tensor[b] = x * s + y * c;
}

extern "C" bool q38_qsa_cuda_project_main(
    const uint16_t *q_proj, size_t q_rows, const uint16_t *k_proj,
    size_t k_rows, const uint16_t *v_proj, size_t v_rows, size_t cols,
    const float *device_input, size_t token_count, float *device_q,
    float *device_k, float *device_v, cudaStream_t stream, char *error,
    size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!q_proj || !k_proj || !v_proj || !cols || !device_input ||
        !token_count || !device_q || !device_k || !device_v) {
        return fail(error, error_len, "invalid CUDA QSA projection arguments");
    }
    if (q_rows > SIZE_MAX / token_count || k_rows > SIZE_MAX / token_count ||
        v_rows > SIZE_MAX / token_count) {
        return fail(error, error_len, "CUDA QSA projection size overflows");
    }
    const size_t q_elements = q_rows * token_count;
    const size_t k_elements = k_rows * token_count;
    const size_t v_elements = v_rows * token_count;
    project_kernel<<<(unsigned)((q_elements + 255) / 256), 256, 0, stream>>>(
        q_proj, q_rows, cols, device_input, token_count, device_q);
    project_kernel<<<(unsigned)((k_elements + 255) / 256), 256, 0, stream>>>(
        k_proj, k_rows, cols, device_input, token_count, device_k);
    project_kernel<<<(unsigned)((v_elements + 255) / 256), 256, 0, stream>>>(
        v_proj, v_rows, cols, device_input, token_count, device_v);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_qsa_cuda_apply_rope(
    float *device_tensor, size_t token_count, size_t head_count,
    size_t head_dim, size_t rotary_dims, int64_t position,
    const uint32_t sections[4], cudaStream_t stream, char *error,
    size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!device_tensor || !token_count || !head_count || !head_dim ||
        rotary_dims == 0 || rotary_dims > head_dim || rotary_dims % 2 != 0 ||
        !sections || sections[0] + sections[1] + sections[2] + sections[3] == 0)
        return fail(error, error_len, "invalid CUDA QSA RoPE arguments");
    const size_t pairs = rotary_dims / 2;
    if (token_count > SIZE_MAX / head_count ||
        token_count * head_count > SIZE_MAX / pairs)
        return fail(error, error_len, "CUDA QSA RoPE size overflows");
    const size_t total = token_count * head_count * pairs;
    rope_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        device_tensor, token_count, head_count, head_dim, rotary_dims, position,
        sections[0], sections[1], sections[2], sections[3]);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}
