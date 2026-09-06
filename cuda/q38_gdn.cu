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

__device__ static float gdn_bf16_to_float(uint16_t bits) {
    return __uint_as_float((uint32_t)bits << 16);
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
    float *output, bool channel_major) {
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
            const size_t weight_index = channel_major
                ? channel * kernel_size + tap
                : tap * channels + channel;
            sum += gdn_conv_weight(kernel_type, kernel, weight_index) * sample;
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

__device__ static float gdn_scalar_weight(uint32_t type, const void *weights,
                                          size_t index) {
    if (type == Q38_GDN_WEIGHT_BF16)
        return gdn_bf16_to_float(((const uint16_t *)weights)[index]);
    return ((const float *)weights)[index];
}

__global__ static void gdn_prepare_recurrence_kernel(
    const float *conv, const float *a, const float *b,
    uint32_t scalar_weight_type, const void *a_log, const void *dt_bias,
    size_t tokens, float *q, float *k, float *v, float *decay,
    float *beta) {
    const size_t head = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_heads = tokens * Q38_GDN_VALUE_HEADS;
    if (head >= total_heads) return;
    const size_t token = head / Q38_GDN_VALUE_HEADS;
    const size_t value_head = head % Q38_GDN_VALUE_HEADS;
    const size_t key_head = value_head / 3u;
    const float *conv_token = conv + token * Q38_GDN_QKV_CHANNELS;
    float *q_head = q + head * Q38_GDN_HEAD_DIM;
    float *k_head = k + head * Q38_GDN_HEAD_DIM;
    float *v_head = v + head * Q38_GDN_HEAD_DIM;
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (size_t d = 0; d < Q38_GDN_HEAD_DIM; ++d) {
        const float q_value =
            conv_token[key_head * Q38_GDN_HEAD_DIM + d];
        const float k_value =
            conv_token[Q38_GDN_KEY_CHANNELS +
                       key_head * Q38_GDN_HEAD_DIM + d];
        const float v_value =
            conv_token[2u * Q38_GDN_KEY_CHANNELS +
                       value_head * Q38_GDN_HEAD_DIM + d];
        q_head[d] = q_value;
        k_head[d] = k_value;
        v_head[d] = v_value;
        q_norm += q_value * q_value;
        k_norm += k_value * k_value;
    }
    const float q_scale = rsqrtf(q_norm + 1.0e-6f);
    const float k_scale = rsqrtf(k_norm + 1.0e-6f);
    for (size_t d = 0; d < Q38_GDN_HEAD_DIM; ++d) {
        q_head[d] *= q_scale;
        k_head[d] *= k_scale;
    }
    const float av =
        a[token * Q38_GDN_VALUE_HEADS + value_head] +
        gdn_scalar_weight(scalar_weight_type, dt_bias, value_head);
    decay[head] =
        expf(-expf(gdn_scalar_weight(scalar_weight_type, a_log, value_head)) *
             log1pf(expf(av)));
    beta[head] =
        1.0f / (1.0f + expf(-b[token * Q38_GDN_VALUE_HEADS + value_head]));
}

__global__ static void gdn_post_recurrence_kernel(
    const float *recurrent, const float *z, uint32_t norm_weight_type,
    const void *norm, size_t tokens, float *output) {
    const size_t head = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_heads = tokens * Q38_GDN_VALUE_HEADS;
    if (head >= total_heads) return;
    const float *recurrent_head = recurrent + head * Q38_GDN_HEAD_DIM;
    float *output_head = output + head * Q38_GDN_HEAD_DIM;
    double sum = 0.0;
    for (size_t d = 0; d < Q38_GDN_HEAD_DIM; ++d)
        sum += (double)recurrent_head[d] * recurrent_head[d];
    const float scale =
        1.0f / sqrtf((float)(sum / Q38_GDN_HEAD_DIM) + 1.0e-6f);
    const float *z_head = z + head * Q38_GDN_HEAD_DIM;
    for (size_t d = 0; d < Q38_GDN_HEAD_DIM; ++d) {
        const float weight =
            gdn_scalar_weight(norm_weight_type, norm, d);
        output_head[d] = recurrent_head[d] * scale * weight /
                         (1.0f + expf(-z_head[d]));
    }
}

/*
 * Single-token island for the non-projection GDN stages.  One block owns one
 * value head, so the recurrence ordering and FP32 state update remain the
 * same as gdn_recurrence_kernel while the projection-sized intermediates stay
 * in shared memory.
 */
__global__ static void gdn_fused_recurrent_kernel(
    const float *qkv, const float *z, const float *a, const float *b,
    uint32_t conv_weight_type, const void *conv_kernel,
    uint32_t scalar_weight_type, const void *a_log, const void *dt_bias,
    uint32_t norm_weight_type, const void *norm, float *state,
    const float *history, float *output) {
    const size_t head = blockIdx.x;
    if (head >= Q38_GDN_VALUE_HEADS) return;
    constexpr size_t dim = Q38_GDN_HEAD_DIM;
    constexpr size_t state_stride = dim * dim;
    constexpr size_t history_tokens = Q38_GDN_CONV_KERNEL - 1u;
    const size_t key_head = head / 3u;
    const size_t key_offset = key_head * dim;
    const size_t value_offset = 2u * Q38_GDN_KEY_CHANNELS + head * dim;
    __shared__ float q_values[dim];
    __shared__ float k_values[dim];
    __shared__ float v_values[dim];
    __shared__ float delta[dim];
    __shared__ float recurrent_values[dim];
    __shared__ float scalars[2];
    const unsigned lane = threadIdx.x;

    if (lane < dim) {
        const size_t d = lane;
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float v_sum = 0.0f;
        for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; ++tap) {
            const size_t source =
                history_tokens - (Q38_GDN_CONV_KERNEL - 1u - tap);
            const float qkv_sample = source < history_tokens
                ? history[source * Q38_GDN_QKV_CHANNELS + key_offset + d]
                : qkv[key_offset + d];
            const float k_sample = source < history_tokens
                ? history[source * Q38_GDN_QKV_CHANNELS +
                          Q38_GDN_KEY_CHANNELS + key_offset + d]
                : qkv[Q38_GDN_KEY_CHANNELS + key_offset + d];
            const float v_sample = source < history_tokens
                ? history[source * Q38_GDN_QKV_CHANNELS + value_offset + d]
                : qkv[value_offset + d];
            q_sum += gdn_conv_weight(
                conv_weight_type, conv_kernel,
                (key_offset + d) * Q38_GDN_CONV_KERNEL + tap) * qkv_sample;
            k_sum += gdn_conv_weight(
                conv_weight_type, conv_kernel,
                (Q38_GDN_KEY_CHANNELS + key_offset + d) *
                    Q38_GDN_CONV_KERNEL + tap) * k_sample;
            v_sum += gdn_conv_weight(
                conv_weight_type, conv_kernel,
                (value_offset + d) * Q38_GDN_CONV_KERNEL + tap) * v_sample;
        }
        q_values[d] = q_sum / (1.0f + expf(-q_sum));
        k_values[d] = k_sum / (1.0f + expf(-k_sum));
        v_values[d] = v_sum / (1.0f + expf(-v_sum));
    }
    __syncthreads();

    if (lane == 0) {
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            q_norm += q_values[d] * q_values[d];
            k_norm += k_values[d] * k_values[d];
        }
        const float q_scale = rsqrtf(q_norm + 1.0e-6f);
        const float k_scale = rsqrtf(k_norm + 1.0e-6f);
        for (size_t d = 0; d < dim; ++d) {
            q_values[d] *= q_scale;
            k_values[d] *= k_scale;
        }
        const float av = a[head] +
            gdn_scalar_weight(scalar_weight_type, dt_bias, head);
        scalars[0] =
            expf(-expf(gdn_scalar_weight(scalar_weight_type, a_log, head)) *
                 log1pf(expf(av)));
        scalars[1] = 1.0f / (1.0f + expf(-b[head]));
    }
    __syncthreads();

    float *matrix = state + head * state_stride;
    for (size_t index = lane; index < state_stride; index += blockDim.x)
        matrix[index] *= scalars[0];
    __syncthreads();

    if (lane < dim) {
        float prediction = 0.0f;
        for (size_t row = 0; row < dim; ++row)
            prediction += matrix[row * dim + lane] * k_values[row];
        delta[lane] = (v_values[lane] - prediction) * scalars[1];
    }
    __syncthreads();

    for (size_t index = lane; index < state_stride; index += blockDim.x) {
        const size_t row = index / dim;
        const size_t column = index % dim;
        matrix[index] += k_values[row] * delta[column];
    }
    __syncthreads();

    if (lane < dim) {
        float value = 0.0f;
        for (size_t row = 0; row < dim; ++row)
            value += matrix[row * dim + lane] * q_values[row];
        recurrent_values[lane] = value / sqrtf((float)dim);
    }
    __syncthreads();

    if (lane == 0) {
        double sum = 0.0;
        for (size_t d = 0; d < dim; ++d)
            sum += (double)recurrent_values[d] * recurrent_values[d];
        scalars[0] =
            1.0f / sqrtf((float)(sum / dim) + 1.0e-6f);
    }
    __syncthreads();

    if (lane < dim) {
        const float norm_weight =
            gdn_scalar_weight(norm_weight_type, norm, lane);
        const float gate = 1.0f / (1.0f + expf(-z[head * dim + lane]));
        output[head * dim + lane] =
            recurrent_values[lane] * scalars[0] * norm_weight * gate;
    }
}

/*
 * One block owns one value head.  The matrix scale and rank-1 update are
 * element-parallel, while each prediction/output dot product keeps the
 * reference row order in one thread.  Barriers preserve token sequencing.
 */
__global__ static void gdn_recurrence_kernel(
    float *state, size_t tokens, const float *q, const float *k,
    const float *v, const float *decay, const float *beta, float scale,
    float *output) {
    const size_t head = blockIdx.x;
    const size_t head_stride = (size_t)Q38_GDN_HEAD_DIM;
    const size_t state_stride = head_stride * head_stride;
    __shared__ float delta[Q38_GDN_HEAD_DIM];
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

        for (size_t index = threadIdx.x; index < state_stride;
             index += blockDim.x)
            matrix[index] *= decay_head;
        __syncthreads();

        if (threadIdx.x < head_stride) {
            const size_t column = threadIdx.x;
            float prediction = 0.0f;
            for (size_t row = 0; row < head_stride; row++)
                prediction += matrix[row * head_stride + column] * k_head[row];
            delta[column] = (v_head[column] - prediction) * beta_head;
        }
        __syncthreads();

        for (size_t index = threadIdx.x; index < state_stride;
             index += blockDim.x) {
            const size_t row = index / head_stride;
            const size_t column = index % head_stride;
            matrix[index] += k_head[row] * delta[column];
        }
        __syncthreads();

        if (threadIdx.x < head_stride) {
            const size_t column = threadIdx.x;
            float value = 0.0f;
            for (size_t row = 0; row < head_stride; row++)
                value += matrix[row * head_stride + column] * q_head[row];
            output_head[column] = scale * value;
        }
        __syncthreads();
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
        output, false);
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

extern "C" bool q38_cuda_gdn_conv_silu_fused_channel_major(
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
        set_error(error, error_len,
                  "invalid channel-major fused GDN convolution arguments");
        return false;
    }
    const unsigned threads =
        (unsigned)(tokens < 256u ? tokens : 256u);
    gdn_conv_silu_fused_kernel<<<(unsigned)channels, threads, 0, stream>>>(
        kernel_type, kernel, input, tokens, channels, kernel_size, history,
        output, true);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len,
                     "channel-major fused GDN convolution launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_history_update(
    const float *input, size_t tokens, size_t channels, size_t kernel_size,
    float *history, cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!input || !tokens || !channels || !history || kernel_size < 2u ||
        tokens > SIZE_MAX / channels ||
        tokens > SIZE_MAX - (kernel_size - 1u) ||
        (kernel_size - 1u) > SIZE_MAX / channels ||
        !valid_grid(channels) ||
        !valid_grid((kernel_size - 1u) * channels)) {
        set_error(error, error_len, "invalid GDN history update arguments");
        return false;
    }
    gdn_conv_history_tail_kernel<<<
        (unsigned)((channels + 255u) / 256u), 256, 0, stream>>>(
        input, tokens, channels, kernel_size, history);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len,
                     "GDN history update launch failed: %s",
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

extern "C" bool q38_cuda_gdn_prepare_recurrence(
    const float *conv, const float *a, const float *b,
    uint32_t scalar_weight_type, const void *a_log, const void *dt_bias,
    size_t tokens, float *q, float *k, float *v, float *decay, float *beta,
    cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!conv || !a || !b || !a_log || !dt_bias || !tokens || !q || !k ||
        !v || !decay || !beta ||
        (scalar_weight_type != Q38_GDN_WEIGHT_F32 &&
         scalar_weight_type != Q38_GDN_WEIGHT_BF16) ||
        tokens > SIZE_MAX / Q38_GDN_VALUE_HEADS) {
        set_error(error, error_len,
                  "invalid GDN recurrence preparation arguments");
        return false;
    }
    const size_t total = tokens * Q38_GDN_VALUE_HEADS;
    gdn_prepare_recurrence_kernel<<<
        (unsigned)((total + 255u) / 256u), 256, 0, stream>>>(
        conv, a, b, scalar_weight_type, a_log, dt_bias, tokens, q, k, v,
        decay, beta);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len,
                     "GDN recurrence preparation launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_post_recurrence(
    const float *recurrent, const float *z, uint32_t norm_weight_type,
    const void *norm, size_t tokens, float *output, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!recurrent || !z || !norm || !tokens || !output ||
        (norm_weight_type != Q38_GDN_WEIGHT_F32 &&
         norm_weight_type != Q38_GDN_WEIGHT_BF16) ||
        tokens > SIZE_MAX / Q38_GDN_VALUE_HEADS) {
        set_error(error, error_len, "invalid GDN post-recurrence arguments");
        return false;
    }
    const size_t total = tokens * Q38_GDN_VALUE_HEADS;
    gdn_post_recurrence_kernel<<<
        (unsigned)((total + 255u) / 256u), 256, 0, stream>>>(
        recurrent, z, norm_weight_type, norm, tokens, output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len,
                     "GDN post-recurrence launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_gdn_fused_recurrent(
    const float *qkv, const float *z, const float *a, const float *b,
    uint32_t conv_weight_type, const void *conv_kernel,
    uint32_t scalar_weight_type, const void *a_log, const void *dt_bias,
    uint32_t norm_weight_type, const void *norm, float *state,
    const float *history, float *output, cudaStream_t stream, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!qkv || !z || !a || !b || !conv_kernel || !a_log || !dt_bias ||
        !norm || !state || !history || !output ||
        (conv_weight_type != Q38_GDN_WEIGHT_F32 &&
         conv_weight_type != Q38_GDN_WEIGHT_BF16) ||
        (scalar_weight_type != Q38_GDN_WEIGHT_F32 &&
         scalar_weight_type != Q38_GDN_WEIGHT_BF16) ||
        (norm_weight_type != Q38_GDN_WEIGHT_F32 &&
         norm_weight_type != Q38_GDN_WEIGHT_BF16)) {
        set_error(error, error_len, "invalid fused GDN recurrence arguments");
        return false;
    }
    gdn_fused_recurrent_kernel<<<Q38_GDN_VALUE_HEADS, 256, 0, stream>>>(
        qkv, z, a, b, conv_weight_type, conv_kernel, scalar_weight_type,
        a_log, dt_bias, norm_weight_type, norm, state, history, output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len)
            snprintf(error, error_len,
                     "fused GDN recurrence launch failed: %s",
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
    gdn_recurrence_kernel<<<Q38_GDN_VALUE_HEADS, 256, 0, stream>>>(
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
