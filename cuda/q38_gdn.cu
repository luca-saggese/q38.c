#include "q38_gdn.h"

#include "q38_cuda_primitives.h"

#include <limits.h>
#include <stdio.h>

__device__ static float gdn_half_to_float(uint16_t bits) {
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x3ffu;
    uint32_t value;
    if (!exponent) {
        if (!fraction) value = sign;
        else {
            exponent = 1;
            while (!(fraction & 0x400u)) {
                fraction <<= 1;
                exponent--;
            }
            fraction &= 0x3ffu;
            value = sign | ((exponent + 112u) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (fraction << 13);
    } else {
        value = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    return __uint_as_float(value);
}

__device__ static float gdn_q8_value(const q38_gdn_q8_0_block *block,
                                      unsigned element) {
    return gdn_half_to_float(block->d) * (float)block->qs[element];
}

__global__ static void gdn_dense_project_kernel(uint32_t weight_type,
                                                const void *weights,
                                                size_t rows, size_t cols,
                                                const float *input,
                                                size_t tokens, float *output) {
    constexpr unsigned warp_count = 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const size_t index = (size_t)blockIdx.x;
    const size_t total = rows * tokens;
    if (index >= total) return;
    const size_t token = index / rows;
    const size_t row = index % rows;
    const float *x = input + token * cols;
    float sum = 0.0f;
    if (weight_type == Q38_GDN_WEIGHT_F32) {
        const float *w = (const float *)weights + row * cols;
        for (size_t col = threadIdx.x; col < cols; col += blockDim.x)
            sum += w[col] * x[col];
    } else {
        const size_t blocks_per_row = cols / 32u;
        const q38_gdn_q8_0_block *w =
            (const q38_gdn_q8_0_block *)weights + row * blocks_per_row;
        for (size_t col = threadIdx.x; col < cols; col += blockDim.x)
            sum += gdn_q8_value(w + col / 32u, (unsigned)(col % 32u)) *
                   x[col];
    }
    __shared__ float warp_sums[warp_count];
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0f;
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) output[index] = sum;
    }
}

__device__ static float gdn_conv_weight(uint32_t type, const void *kernel,
                                        size_t index) {
    if (type == Q38_GDN_WEIGHT_F32)
        return ((const float *)kernel)[index];
    return __uint_as_float((uint32_t)((const uint16_t *)kernel)[index] << 16);
}

__global__ static void gdn_conv_kernel(uint32_t kernel_type,
                                       const void *kernel,
                                       const float *input, size_t tokens,
                                       size_t channels, size_t kernel_size,
                                       const float *history, float *output) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * channels;
    if (index >= total) return;
    const size_t token = index / channels;
    const size_t channel = index % channels;
    const size_t history_tokens = kernel_size - 1u;
    float sum = 0.0f;
    for (size_t tap = 0; tap < kernel_size; tap++) {
        /*
         * Frozen source convention:
         * out[t] = sum_k w[k] * x[t - (K-1-k)].
         * The prior history is the prefix of the logical concatenation.
         */
        const size_t current = history_tokens + token;
        const size_t distance = kernel_size - 1u - tap;
        const size_t source = current - distance;
        const float sample = source < history_tokens
            ? history[source * channels + channel]
            : input[(source - history_tokens) * channels + channel];
        sum += gdn_conv_weight(kernel_type, kernel,
                               tap * channels + channel) * sample;
    }
    output[index] = sum;
}

__global__ static void gdn_conv_history_tail_kernel(
    const float *input, size_t tokens, size_t channels, size_t kernel_size,
    float *history) {
    const size_t history_tokens = kernel_size - 1u;
    const size_t channel =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    /*
     * A short chunk can source part of the old history while writing the
     * same array.  Advance the tail in order so the reference layout has no
     * read-after-write race and needs no temporary GB10 packing buffer.
     */
    for (size_t tail = 0; tail < history_tokens; tail++) {
        const size_t source = tokens + tail;
        history[tail * channels + channel] = source < history_tokens
            ? history[source * channels + channel]
            : input[(source - history_tokens) * channels + channel];
    }
}

/*
 * One block owns one channel.  Threads cover the token dimension, then a
 * block barrier makes the history update occur only after every convolution
 * read has completed.  Thread zero advances the tail in the same order as
 * gdn_conv_history_tail_kernel, so short chunks retain the exact reference
 * semantics without a temporary buffer or a write/read race.
 */
__global__ static void gdn_conv_silu_fused_kernel(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output) {
    const size_t channel = (size_t)blockIdx.x;
    if (channel >= channels) return;
    const size_t history_tokens = kernel_size - 1u;
    for (size_t token = threadIdx.x; token < tokens;
         token += (size_t)blockDim.x) {
        const size_t current = history_tokens + token;
        float sum = 0.0f;
        for (size_t tap = 0; tap < kernel_size; tap++) {
            const size_t distance = kernel_size - 1u - tap;
            const size_t source = current - distance;
            const float sample =
                source < history_tokens
                    ? history[source * channels + channel]
                    : input[(source - history_tokens) * channels + channel];
            sum += gdn_conv_weight(kernel_type, kernel,
                                   tap * channels + channel) * sample;
        }
        output[token * channels + channel] =
            sum / (1.0f + expf(-sum));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (size_t tail = 0; tail < history_tokens; tail++) {
            const size_t source = tokens + tail;
            history[tail * channels + channel] =
                source < history_tokens
                    ? history[source * channels + channel]
                    : input[(source - history_tokens) * channels + channel];
        }
    }
}

__global__ static void gdn_split_qkv_kernel(const float *qkv, size_t tokens,
                                            float *q, float *k, float *v) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_GDN_QKV_CHANNELS;
    if (index >= total) return;
    const size_t token = index / Q38_GDN_QKV_CHANNELS;
    const size_t channel = index % Q38_GDN_QKV_CHANNELS;
    const float *src = qkv + token * Q38_GDN_QKV_CHANNELS;
    if (channel < Q38_GDN_KEY_CHANNELS)
        q[token * Q38_GDN_KEY_CHANNELS + channel] = src[channel];
    else if (channel < 2u * Q38_GDN_KEY_CHANNELS)
        k[token * Q38_GDN_KEY_CHANNELS + channel - Q38_GDN_KEY_CHANNELS] =
            src[channel];
    else
        v[token * Q38_GDN_VALUE_CHANNELS + channel - 2u * Q38_GDN_KEY_CHANNELS] =
            src[channel];
}

__global__ static void gdn_repeat_key_kernel(const float *key, size_t tokens,
                                             float *value) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_GDN_VALUE_CHANNELS;
    if (index >= total) return;
    const size_t token = index / Q38_GDN_VALUE_CHANNELS;
    const size_t value_channel = index % Q38_GDN_VALUE_CHANNELS;
    const size_t value_head = value_channel / Q38_GDN_HEAD_DIM;
    const size_t dimension = value_channel % Q38_GDN_HEAD_DIM;
    const size_t key_head = value_head / 3u;
    value[index] = key[token * Q38_GDN_KEY_CHANNELS +
                       key_head * Q38_GDN_HEAD_DIM + dimension];
}

/*
 * One thread owns one value head so every token's decay, prediction, update,
 * and read are completed in reference order before that thread advances.
 * This intentionally favors a transparent FP32 baseline over throughput.
 */
__global__ static void gdn_recurrence_kernel(
    float *state, size_t tokens, const float *q, const float *k,
    const float *v, const float *decay, const float *beta, float scale,
    float *output) {
    const size_t head = threadIdx.x;
    const size_t head_stride = (size_t)Q38_GDN_HEAD_DIM;
    const size_t state_stride = head_stride * head_stride;
    float delta[Q38_GDN_HEAD_DIM];
    float *matrix = state + head * state_stride;

    for (size_t token = 0; token < tokens; token++) {
        const float decay_head = decay[token * Q38_GDN_VALUE_HEADS + head];
        const float beta_head = beta[token * Q38_GDN_VALUE_HEADS + head];
        const float *q_head =
            q + token * Q38_GDN_VALUE_CHANNELS + head * head_stride;
        const float *k_head =
            k + token * Q38_GDN_VALUE_CHANNELS + head * head_stride;
        const float *v_head =
            v + token * Q38_GDN_VALUE_CHANNELS + head * head_stride;
        float *output_head =
            output + token * Q38_GDN_VALUE_CHANNELS + head * head_stride;

        for (size_t row = 0; row < head_stride; row++) {
            float *state_row = matrix + row * head_stride;
            for (size_t column = 0; column < head_stride; column++)
                state_row[column] *= decay_head;
        }

        for (size_t column = 0; column < head_stride; column++) {
            float prediction = 0.0f;
            for (size_t row = 0; row < head_stride; row++)
                prediction += matrix[row * head_stride + column] * k_head[row];
            delta[column] = (v_head[column] - prediction) * beta_head;
        }

        for (size_t row = 0; row < head_stride; row++) {
            float *state_row = matrix + row * head_stride;
            for (size_t column = 0; column < head_stride; column++)
                state_row[column] += k_head[row] * delta[column];
        }

        for (size_t column = 0; column < head_stride; column++) {
            float value = 0.0f;
            for (size_t row = 0; row < head_stride; row++)
                value += matrix[row * head_stride + column] * q_head[row];
            output_head[column] = scale * value;
        }
    }
}

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

static bool valid_grid(size_t elements) {
    return elements && elements <= (size_t)UINT_MAX * 256u;
}

extern "C" bool q38_cuda_gdn_project(
    uint32_t weight_type, const void *weights, size_t rows, size_t cols,
    const float *input, size_t tokens, float *output, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!weights || !rows || !cols || !input || !tokens || !output ||
        tokens > SIZE_MAX / cols || rows > SIZE_MAX / tokens ||
        !valid_grid(rows * tokens)) {
        set_error(error, error_len, "invalid GDN projection arguments");
        return false;
    }
    if (weight_type == Q38_QUANT_Q2_K) {
        if (cols % Q38_QUANT_QK_K) {
            set_error(error, error_len, "Q2 GDN projection cols are not block aligned");
            return false;
        }
        for (size_t token = 0; token < tokens; token++) {
            if (!q38_cuda_q2_matvec(
                    weights, rows, cols, input + token * cols,
                    output + token * rows, stream, error, error_len))
                return false;
        }
        return true;
    }
    if (weight_type == Q38_GDN_WEIGHT_BF16) {
        for (size_t token = 0; token < tokens; token++) {
            if (!q38_cuda_bf16_matvec(
                    (const uint16_t *)weights, rows, cols,
                    input + token * cols, output + token * rows, stream,
                    error, error_len))
                return false;
        }
        return true;
    }
    if (weight_type != Q38_GDN_WEIGHT_F32 &&
        (weight_type != Q38_GDN_WEIGHT_Q8_0 || cols % 32u)) {
        set_error(error, error_len, "unsupported GDN projection weight type");
        return false;
    }
    gdn_dense_project_kernel<<<(unsigned)(rows * tokens), 256, 0, stream>>>(
        weight_type, weights, rows, cols, input, tokens, output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len, "GDN projection launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_conv(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!kernel || !input || !tokens || !channels || !history || !output ||
        kernel_size < 2u || kernel_type > Q38_GDN_WEIGHT_BF16 ||
        (kernel_type != Q38_GDN_WEIGHT_F32 &&
         kernel_type != Q38_GDN_WEIGHT_BF16) ||
        tokens > SIZE_MAX / channels ||
        tokens > SIZE_MAX - (kernel_size - 1u) ||
        (kernel_size - 1u) > SIZE_MAX / channels ||
        !valid_grid(tokens * channels) ||
        !valid_grid((kernel_size - 1u) * channels)) {
        set_error(error, error_len, "invalid GDN convolution arguments");
        return false;
    }
    gdn_conv_kernel<<<(unsigned)((tokens * channels + 255u) / 256u), 256, 0,
                      stream>>>(kernel_type, kernel, input, tokens, channels,
                                kernel_size, history, output);
    gdn_conv_history_tail_kernel<<<(unsigned)((channels + 255u) / 256u), 256,
                                   0, stream>>>(
        input, tokens, channels, kernel_size, history);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len, "GDN convolution launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_conv_update(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len) {
    return q38_cuda_gdn_conv(kernel_type, kernel, input, tokens, channels,
                             kernel_size, history, output, stream, error,
                             error_len);
}

extern "C" bool q38_cuda_gdn_conv_silu(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len) {
    if (!q38_cuda_gdn_conv(kernel_type, kernel, input, tokens, channels,
                           kernel_size, history, output, stream, error,
                           error_len))
        return false;
    return q38_cuda_silu(output, output, tokens * channels, stream, error,
                         error_len);
}

extern "C" bool q38_cuda_gdn_conv_silu_fused(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!kernel || !input || !tokens || !channels || !history || !output ||
        kernel_size < 2u || kernel_type > Q38_GDN_WEIGHT_BF16 ||
        (kernel_type != Q38_GDN_WEIGHT_F32 &&
         kernel_type != Q38_GDN_WEIGHT_BF16) ||
        tokens > SIZE_MAX / channels ||
        tokens > SIZE_MAX - (kernel_size - 1u) ||
        (kernel_size - 1u) > SIZE_MAX / channels ||
        !valid_grid(channels) ||
        !valid_grid((kernel_size - 1u) * channels)) {
        set_error(error, error_len, "invalid fused GDN convolution arguments");
        return false;
    }
    const unsigned threads =
        (unsigned)(tokens < 256u ? tokens : 256u);
    gdn_conv_silu_fused_kernel<<<(unsigned)channels, threads, 0, stream>>>(
        kernel_type, kernel, input, tokens, channels, kernel_size, history,
        output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len,
                     "fused GDN convolution launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_split_qkv(
    const float *qkv, size_t tokens, float *q, float *k, float *v,
    cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!qkv || !tokens || !q || !k || !v ||
        tokens > SIZE_MAX / Q38_GDN_QKV_CHANNELS ||
        !valid_grid(tokens * Q38_GDN_QKV_CHANNELS)) {
        set_error(error, error_len, "invalid GDN QKV split arguments");
        return false;
    }
    gdn_split_qkv_kernel<<<
        (unsigned)((tokens * Q38_GDN_QKV_CHANNELS + 255u) / 256u), 256, 0,
        stream>>>(qkv, tokens, q, k, v);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len, "GDN QKV split launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_repeat_key_heads(
    const float *key, size_t tokens, float *value, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!key || !tokens || !value ||
        tokens > SIZE_MAX / Q38_GDN_VALUE_CHANNELS ||
        !valid_grid(tokens * Q38_GDN_VALUE_CHANNELS)) {
        set_error(error, error_len, "invalid GDN head-repeat arguments");
        return false;
    }
    gdn_repeat_key_kernel<<<
        (unsigned)((tokens * Q38_GDN_VALUE_CHANNELS + 255u) / 256u), 256, 0,
        stream>>>(key, tokens, value);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len, "GDN head-repeat launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_recurrence(
    float *state, size_t tokens, const float *q, const float *k,
    const float *v, const float *decay, const float *beta, float scale,
    float *output, cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!state || !tokens || !q || !k || !v || !decay || !beta || !output ||
        tokens > SIZE_MAX / Q38_GDN_VALUE_CHANNELS ||
        tokens > SIZE_MAX / Q38_GDN_VALUE_HEADS) {
        set_error(error, error_len, "invalid GDN recurrence arguments");
        return false;
    }
    gdn_recurrence_kernel<<<1, Q38_GDN_VALUE_HEADS, 0, stream>>>(
        state, tokens, q, k, v, decay, beta, scale, output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len, "GDN recurrence launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_recurrence_reset(float *state,
                                               cudaStream_t stream,
                                               char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!state) {
        set_error(error, error_len, "invalid GDN recurrence reset state");
        return false;
    }
    cudaError_t status = cudaMemsetAsync(
        state, 0,
        (size_t)Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM *
            sizeof(float),
        stream);
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len, "GDN recurrence reset failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}
