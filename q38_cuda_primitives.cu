#include "q38_cuda_primitives.h"

#include <stdio.h>

__device__ static float half_to_float_device(uint16_t bits) {
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

__device__ static float q2_value(const q38_q2_k_block *block, unsigned element) {
    const unsigned half = element / 128;
    const unsigned within = element % 128;
    const unsigned group = within / 16;
    const unsigned l = within % 16;
    const unsigned shift = (group / 2) * 2;
    const unsigned scale_index = half * 8 + group;
    const uint8_t scale = block->scales[scale_index];
    const unsigned qindex = half * 32 + (group % 2) * 16 + l;
    const unsigned quant = (block->qs[qindex] >> shift) & 3u;
    return half_to_float_device(block->d) * (scale & 0xfu) * quant -
           half_to_float_device(block->dmin) * (scale >> 4);
}

__device__ static void q4_scale_min(unsigned index, const uint8_t *scales,
                                    uint8_t *scale, uint8_t *minimum) {
    if (index < 4) {
        *scale = scales[index] & 63u;
        *minimum = scales[index + 4] & 63u;
    } else {
        *scale = (scales[index + 4] & 0xfu) |
                 ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) |
                   ((scales[index] >> 6) << 4);
    }
}

__device__ static float q4_value(const q38_q4_k_block *block, unsigned element) {
    const unsigned group = element / 32;
    const unsigned l = element % 32;
    uint8_t scale, minimum;
    q4_scale_min(group, block->scales, &scale, &minimum);
    const uint8_t packed = block->qs[(group / 2) * 32 + l];
    const unsigned quant = group % 2 ? packed >> 4 : packed & 0xfu;
    return half_to_float_device(block->d) * scale * quant -
           half_to_float_device(block->dmin) * minimum;
}

__global__ static void dequant_kernel(uint32_t type, const void *blocks,
                                      size_t block_count, float *out) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = block_count * Q38_QUANT_QK_K;
    if (index >= elements) return;
    const size_t block = index / Q38_QUANT_QK_K;
    const unsigned element = (unsigned)(index % Q38_QUANT_QK_K);
    if (type == Q38_QUANT_Q2_K)
        out[index] = q2_value((const q38_q2_k_block *)blocks + block, element);
    else
        out[index] = q4_value((const q38_q4_k_block *)blocks + block, element);
}

__global__ static void rms_norm_kernel(const float *input, const float *weight,
                                       float *output, size_t elements,
                                       float epsilon) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float sum = 0.0f;
    for (size_t i = 0; i < elements; i++) sum += input[i] * input[i];
    output[index] = input[index] * rsqrtf(sum / (float)elements + epsilon) *
                    weight[index];
}

__global__ static void silu_kernel(const float *input, float *output,
                                   size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index] / (1.0f + expf(-input[index]));
}

__global__ static void q2_matvec_kernel(const q38_q2_k_block *weights,
                                        size_t rows, size_t cols,
                                        const float *input, float *output) {
    constexpr unsigned warp_count = 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const size_t row = (size_t)blockIdx.x;
    if (row >= rows) return;
    const size_t blocks_per_row = cols / Q38_QUANT_QK_K;
    __shared__ double warp_sums[warp_count];
    double sum = 0.0;
    /* Each warp owns complete Q2_K blocks, keeping the quantized block decode
     * and dot product cooperative while allowing all blocks in a row to be
     * processed by the same block. */
    for (size_t block_index = warp; block_index < blocks_per_row;
         block_index += warp_count) {
        const q38_q2_k_block *block =
            weights + row * blocks_per_row + block_index;
        for (unsigned element = lane; element < Q38_QUANT_QK_K; element += 32)
            sum += (double)q2_value(block, element) *
                   (double)input[block_index * Q38_QUANT_QK_K + element];
    }
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0;
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) output[row] = sum;
    }
}

__device__ static float bf16_to_float_device(uint16_t bits) {
    return __uint_as_float((uint32_t)bits << 16);
}

__global__ static void bf16_matvec_kernel(const uint16_t *weights, size_t rows,
                                          size_t cols, const float *input,
                                          float *output) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (size_t col = 0; col < cols; col++)
        sum += bf16_to_float_device(weights[row * cols + col]) * input[col];
    output[row] = sum;
}

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

extern "C" bool q38_cuda_dequantize_row(uint32_t type, const void *blocks,
                                        size_t block_count, float *out,
                                        cudaStream_t stream, char *error,
                                        size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if ((type != Q38_QUANT_Q2_K && type != Q38_QUANT_Q4_K) ||
        !blocks || !out || !block_count ||
        block_count > SIZE_MAX / Q38_QUANT_QK_K) {
        set_error(error, error_len, "invalid CUDA dequantization arguments");
        return false;
    }
    const size_t elements = block_count * Q38_QUANT_QK_K;
    const unsigned grid = (unsigned)((elements + 255) / 256);
    dequant_kernel<<<grid, 256, 0, stream>>>(type, blocks, block_count, out);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len) snprintf(error, error_len, "CUDA launch failed: %s",
                                         cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_rms_norm(const float *input, const float *weight,
                                   float *output, size_t elements,
                                   float epsilon, cudaStream_t stream,
                                   char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!input || !weight || !output || !elements) {
        set_error(error, error_len, "invalid CUDA RMSNorm arguments");
        return false;
    }
    rms_norm_kernel<<<(unsigned)((elements + 255) / 256), 256, 0, stream>>>(
        input, weight, output, elements, epsilon);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len) snprintf(error, error_len, "CUDA RMSNorm launch failed: %s",
                                         cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_silu(const float *input, float *output, size_t elements,
                              cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!input || !output || !elements) {
        set_error(error, error_len, "invalid CUDA SiLU arguments");
        return false;
    }
    silu_kernel<<<(unsigned)((elements + 255) / 256), 256, 0, stream>>>(
        input, output, elements);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len) snprintf(error, error_len, "CUDA SiLU launch failed: %s",
                                         cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_q2_matvec(const void *weights, size_t rows,
                                    size_t cols, const float *input,
                                    float *output, cudaStream_t stream,
                                    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!weights || !rows || !cols || cols % Q38_QUANT_QK_K ||
        !input || !output) {
        set_error(error, error_len, "invalid CUDA Q2 matvec arguments");
        return false;
    }
    q2_matvec_kernel<<<(unsigned)rows, 256, 0, stream>>>(
        (const q38_q2_k_block *)weights, rows, cols, input, output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len) snprintf(error, error_len, "CUDA Q2 matvec launch failed: %s",
                                         cudaGetErrorString(status));
        return false;
    }
    return true;
}

extern "C" bool q38_cuda_bf16_matvec(const uint16_t *weights, size_t rows,
                                      size_t cols, const float *input,
                                      float *output, cudaStream_t stream,
                                      char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!weights || !rows || !cols || !input || !output) {
        set_error(error, error_len, "invalid CUDA BF16 matvec arguments");
        return false;
    }
    bf16_matvec_kernel<<<(unsigned)((rows + 255) / 256), 256, 0, stream>>>(
        weights, rows, cols, input, output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len) snprintf(error, error_len, "CUDA BF16 matvec launch failed: %s",
                                         cudaGetErrorString(status));
        return false;
    }
    return true;
}
