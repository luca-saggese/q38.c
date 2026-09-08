#include "q38_moe_cuda.h"
#include "q38_moe_ref.h"
#include "q38_quant.h"
#include "q38_topk_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

__device__ __forceinline__ static float nvfp4_decode_fp4(uint32_t nibble) {
    constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[nibble & 0xFu];
}

__device__ __forceinline__ static float nvfp4_decode_e4m3(uint32_t value) {
    const int sign = (value & 0x80u) ? -1 : 1;
    const int exponent = (value >> 3) & 0xFu;
    const int mantissa = value & 0x7u;
    float decoded;
    if (exponent == 0)
        decoded = (static_cast<float>(mantissa) / 8.0f) * exp2f(-6.0f);
    else if (exponent == 0xFu)
        decoded = (1.0f + static_cast<float>(mantissa) / 8.0f) * 256.0f;
    else
        decoded = (1.0f + static_cast<float>(mantissa) / 8.0f) *
                  exp2f(static_cast<float>(exponent - 7));
    return static_cast<float>(sign) * decoded;
}

__device__ __forceinline__ static int nvfp4_round_even(float value) {
    const float lower = floorf(value);
    const float fraction = value - lower;
    if (fraction < 0.5f) return static_cast<int>(lower);
    if (fraction > 0.5f) return static_cast<int>(lower + 1.0f);
    const int integer = static_cast<int>(lower);
    return (integer & 1) == 0 ? integer : integer + 1;
}

__device__ __forceinline__ static uint32_t nvfp4_encode_e4m3(float value) {
    const uint32_t sign = value < 0.0f ? 0x80u : 0u;
    const float magnitude = fabsf(value);
    if (magnitude == 0.0f) return sign;
    if (!isfinite(magnitude) || magnitude >= 448.0f) return sign | 0x7Eu;
    if (magnitude < exp2f(-6.0f)) {
        int mantissa = nvfp4_round_even(magnitude * exp2f(9.0f));
        if (mantissa >= 8) mantissa = 8;
        return sign | static_cast<uint32_t>(mantissa);
    }
    const int exponent = static_cast<int>(floorf(log2f(magnitude)));
    int exponent_field = exponent + 7;
    int mantissa = nvfp4_round_even(
        (magnitude / exp2f(static_cast<float>(exponent)) - 1.0f) * 8.0f);
    if (mantissa >= 8) {
        ++exponent_field;
        mantissa = 0;
    }
    if (exponent_field >= 15) return sign | 0x7Eu;
    return sign | (static_cast<uint32_t>(exponent_field) << 3) |
           static_cast<uint32_t>(mantissa & 7);
}

__device__ __forceinline__ static uint32_t nvfp4_encode_fp4(float value) {
    const float magnitude = fabsf(value);
    uint32_t code;
    if (magnitude <= 0.25f) code = 0;
    else if (magnitude < 0.75f) code = 1;
    else if (magnitude <= 1.25f) code = 2;
    else if (magnitude < 1.75f) code = 3;
    else if (magnitude <= 2.5f) code = 4;
    else if (magnitude < 3.5f) code = 5;
    else if (magnitude <= 5.0f) code = 6;
    else code = 7;
    return code | (value < 0.0f ? 0x8u : 0u);
}

__global__ static void nvfp4_quantize_kernel(
    const float *input, uint8_t *packed, uint8_t *scales,
    const float *input_scale, const uint16_t *expert_ids) {
    __shared__ float values[16];
    __shared__ float block_scale;
    const unsigned lane = threadIdx.x;
    const unsigned start = blockIdx.x * 16u;
    if (lane < 16u) values[lane] = input[start + lane];
    __syncthreads();
    if (lane == 0) {
        float amax = 0.0f;
        for (unsigned i = 0; i < 16u; ++i)
            amax = fmaxf(amax, fabsf(values[i]));
        const float scale = input_scale[expert_ids[0]];
        const uint8_t encoded =
            static_cast<uint8_t>(nvfp4_encode_e4m3(amax / (6.0f * scale)));
        scales[blockIdx.x] = encoded;
        block_scale = nvfp4_decode_e4m3(encoded) * scale;
        if (block_scale < 1.0e-5f) block_scale = 1.0f;
    }
    __syncthreads();
    if (lane < 8u) {
        const uint32_t lo = nvfp4_encode_fp4(values[lane * 2] / block_scale);
        const uint32_t hi =
            nvfp4_encode_fp4(values[lane * 2 + 1] / block_scale);
        packed[blockIdx.x * 8u + lane] =
            static_cast<uint8_t>(lo | (hi << 4));
    }
}

__global__ static void nvfp4_grouped_gate_up_kernel(
    const uint8_t *gate_weight, const uint8_t *gate_scale,
    const float *gate_scale_2, const float *gate_input_scale,
    const uint8_t *up_weight, const uint8_t *up_scale,
    const float *up_scale_2, const float *up_input_scale,
    const uint8_t *activation, const uint8_t *activation_scale,
    const uint16_t *expert_ids, float *mid, size_t experts) {
    extern __shared__ uint8_t staged[];
    uint8_t *staged_activation = staged;
    uint8_t *staged_scale = staged + 1280u;
    for (unsigned i = threadIdx.x; i < 1280u; i += blockDim.x)
        staged_activation[i] = activation[i];
    for (unsigned i = threadIdx.x; i < 160u; i += blockDim.x)
        staged_scale[i] = activation_scale[i];
    __syncthreads();
    const unsigned warp = threadIdx.x / 32u;
    const unsigned lane = threadIdx.x & 31u;
    const size_t global_row = (size_t)blockIdx.x * 4u + warp;
    const size_t total_rows = experts * 640u;
    if (global_row >= total_rows) return;
    const size_t selected = global_row / 640u;
    const uint32_t expert = expert_ids[selected];
    const uint32_t row = global_row % 640u;
    const size_t weight_stride = 640u * 1280u;
    const size_t scale_stride = 640u * 160u;
    float gate = 0.0f, up = 0.0f;
    for (uint32_t k = lane; k < 2560u; k += 32u) {
        const uint8_t a = staged_activation[k / 2u];
        const uint32_t an = (k & 1u) ? a >> 4 : a & 0xFu;
        const float activation_value =
            nvfp4_decode_fp4(an) *
            nvfp4_decode_e4m3(staged_scale[k / 16u]) *
            gate_input_scale[expert];
        const uint8_t gw = gate_weight[expert * weight_stride +
                                        row * 1280u + k / 2u];
        const uint8_t uw = up_weight[expert * weight_stride +
                                      row * 1280u + k / 2u];
        const uint8_t ws = gate_scale[expert * scale_stride +
                                      row * 160u + k / 16u];
        const uint8_t us = up_scale[expert * scale_stride +
                                    row * 160u + k / 16u];
        gate += nvfp4_decode_fp4((k & 1u) ? gw >> 4 : gw & 0xFu) *
                nvfp4_decode_e4m3(ws) * gate_scale_2[expert] *
                activation_value;
        up += nvfp4_decode_fp4((k & 1u) ? uw >> 4 : uw & 0xFu) *
              nvfp4_decode_e4m3(us) * up_scale_2[expert] *
              activation_value;
    }
    for (unsigned offset = 16; offset; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        mid[selected * 640u + row] = gate / (1.0f + expf(-gate)) * up;
}

__global__ static void nvfp4_grouped_quantize_kernel(
    const float *input, uint8_t *packed, uint8_t *scales,
    const float *input_scale, const uint16_t *expert_ids, size_t experts) {
    __shared__ float values[16];
    __shared__ float block_scale;
    const unsigned lane = threadIdx.x;
    const unsigned expert = blockIdx.x;
    const unsigned block = blockIdx.y;
    if (expert >= experts) return;
    if (lane < 16u)
        values[lane] = input[expert * 640u + block * 16u + lane];
    __syncthreads();
    if (lane == 0) {
        float amax = 0.0f;
        for (unsigned i = 0; i < 16u; ++i)
            amax = fmaxf(amax, fabsf(values[i]));
        const float scale = input_scale[expert_ids[expert]];
        const uint8_t encoded =
            static_cast<uint8_t>(nvfp4_encode_e4m3(amax / (6.0f * scale)));
        scales[expert * 40u + block] = encoded;
        block_scale = nvfp4_decode_e4m3(encoded) * scale;
        if (block_scale < 1.0e-5f) block_scale = 1.0f;
    }
    __syncthreads();
    if (lane < 8u) {
        const uint32_t lo = nvfp4_encode_fp4(values[lane * 2] / block_scale);
        const uint32_t hi =
            nvfp4_encode_fp4(values[lane * 2 + 1] / block_scale);
        packed[expert * 320u + block * 8u + lane] =
            static_cast<uint8_t>(lo | (hi << 4));
    }
}

__global__ static void nvfp4_grouped_down_kernel(
    const uint8_t *weight, const uint8_t *scale, const float *scale_2,
    const float *input_scale, const uint8_t *activation,
    const uint8_t *activation_scale, const uint16_t *expert_ids,
    const float *route_weights, float *output, size_t experts) {
    extern __shared__ uint8_t staged[];
    uint8_t *staged_activation = staged;
    uint8_t *staged_scale = staged + experts * 320u;
    for (size_t i = threadIdx.x; i < experts * 320u; i += blockDim.x)
        staged_activation[i] = activation[i];
    for (size_t i = threadIdx.x; i < experts * 40u; i += blockDim.x)
        staged_scale[i] = activation_scale[i];
    __syncthreads();
    const unsigned warp = threadIdx.x / 32u;
    const unsigned lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 4u + warp;
    if (row >= 2560u) return;
    float weighted_sum = 0.0f;
    const size_t weight_stride = 2560u * 320u;
    const size_t scale_stride = 2560u * 40u;
    for (unsigned selected = 0; selected < experts; ++selected) {
        const uint32_t expert = expert_ids[selected];
        float sum = 0.0f;
        for (uint32_t k = lane; k < 640u; k += 32u) {
            const uint8_t packed =
                weight[expert * weight_stride + row * 320u + k / 2u];
            const uint8_t act =
                staged_activation[selected * 320u + k / 2u];
            const uint32_t wn = (k & 1u) ? packed >> 4 : packed & 0xFu;
            const uint32_t an = (k & 1u) ? act >> 4 : act & 0xFu;
            const float weight_value =
                nvfp4_decode_fp4(wn) *
                nvfp4_decode_e4m3(scale[expert * scale_stride +
                                         row * 40u + k / 16u]) *
                scale_2[expert];
            const float activation_value =
                nvfp4_decode_fp4(an) *
                nvfp4_decode_e4m3(staged_scale[selected * 40u + k / 16u]) *
                input_scale[expert];
            sum += weight_value * activation_value;
        }
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            weighted_sum = __fadd_rn(
                weighted_sum, __fmul_rn(route_weights[selected], sum));
    }
    if (lane == 0) output[row] = weighted_sum;
}

__global__ static void route_weights_kernel(
    const float *logits, const uint32_t *indices, size_t tokens,
    uint16_t *expert_ids, float *weights) {
    const size_t token = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    const float *row = logits + token * Q38_MOE_EXPERTS;
    float max_value = -INFINITY;
    for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
        const float value = __bfloat162float(__float2bfloat16_rn(row[e]));
        max_value = fmaxf(max_value, value);
    }
    float selected_sum = 0.0f;
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        const uint32_t expert = indices[token * Q38_MOE_TOP_K + k];
        expert_ids[token * Q38_MOE_TOP_K + k] = (uint16_t)expert;
        const float value =
            __bfloat162float(__float2bfloat16_rn(row[expert]));
        selected_sum += expf(value - max_value);
    }
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        const uint32_t expert = indices[token * Q38_MOE_TOP_K + k];
        const float value =
            __bfloat162float(__float2bfloat16_rn(row[expert]));
        const float normalized = expf(value - max_value) / selected_sum;
        weights[token * Q38_MOE_TOP_K + k] =
            __bfloat162float(__float2bfloat16_rn(normalized));
    }
}

__device__ static bool route_ascending_before(
    float left_score, uint16_t left_id, float right_score,
    uint16_t right_id) {
    return left_score < right_score ||
           (left_score == right_score && left_id > right_id);
}

__global__ static void q38_moe_route_top10_decode_kernel(
    const float *logits, uint32_t *indices, uint16_t *expert_ids,
    float *weights) {
    __shared__ float scores[Q38_MOE_EXPERTS];
    __shared__ uint16_t ids[Q38_MOE_EXPERTS];
    __shared__ float selected_exp[Q38_MOE_TOP_K];
    __shared__ float normalizer;
    const unsigned thread = threadIdx.x;

    for (unsigned offset = 0; offset < 2; ++offset) {
        const unsigned expert = thread + offset * 256u;
        const float effective =
            __bfloat162float(__float2bfloat16_rn(logits[expert]));
        scores[expert] = effective;
        ids[expert] = (uint16_t)expert;
    }
    __syncthreads();

    for (unsigned length = 2; length <= Q38_MOE_EXPERTS; length <<= 1) {
        for (unsigned stride = length >> 1; stride; stride >>= 1) {
            for (unsigned offset = 0; offset < 2; ++offset) {
                const unsigned index = thread + offset * 256u;
                const unsigned partner = index ^ stride;
                if (partner > index) {
                    const bool ascending = (index & length) == 0;
                    const bool before = route_ascending_before(
                        scores[index], ids[index], scores[partner],
                        ids[partner]);
                    const bool swap = ascending ? !before : before;
                    if (swap) {
                        const float score = scores[index];
                        const uint16_t id = ids[index];
                        scores[index] = scores[partner];
                        ids[index] = ids[partner];
                        scores[partner] = score;
                        ids[partner] = id;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (thread < Q38_MOE_TOP_K)
        selected_exp[thread] =
            expf(scores[Q38_MOE_EXPERTS - 1u - thread] -
                 scores[Q38_MOE_EXPERTS - 1u]);
    __syncthreads();
    if (thread == 0) {
        float sum = 0.0f;
        for (unsigned k = 0; k < Q38_MOE_TOP_K; ++k)
            sum += selected_exp[k];
        normalizer = sum;
    }
    __syncthreads();
    if (thread < Q38_MOE_TOP_K) {
        const uint16_t id = ids[Q38_MOE_EXPERTS - 1u - thread];
        const float normalized = selected_exp[thread] / normalizer;
        indices[thread] = id;
        expert_ids[thread] = id;
        weights[thread] =
            __bfloat162float(__float2bfloat16_rn(normalized));
    }
}

__global__ static void router_kernel(const float *hidden, size_t tokens,
                                     const float *router, float *logits) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_MOE_EXPERTS;
    if (i >= total) return;
    const size_t token = i / Q38_MOE_EXPERTS;
    const size_t expert = i % Q38_MOE_EXPERTS;
    float value = 0.0f;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        value += hidden[token * Q38_MOE_HIDDEN + d] *
                 router[expert * Q38_MOE_HIDDEN + d];
    logits[i] = value;
}

extern "C" bool q38_moe_cuda_router(const float *device_hidden,
                                    size_t token_count,
                                    const float *device_router,
                                    float *device_logits, cudaStream_t stream,
                                    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!device_hidden || !token_count || !device_router || !device_logits ||
        token_count > SIZE_MAX / Q38_MOE_EXPERTS)
        return fail(error, error_len, "invalid CUDA MoE router arguments");
    const size_t total = token_count * Q38_MOE_EXPERTS;
    router_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        device_hidden, token_count, device_router, device_logits);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_route_weights(
    const float *device_logits, size_t token_count, uint32_t *device_indices,
    uint16_t *device_expert_ids, float *device_weights, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!device_logits || !token_count || !device_indices ||
        !device_expert_ids || !device_weights)
        return fail(error, error_len, "invalid CUDA route weight arguments");
    if (token_count == 1) {
        q38_moe_route_top10_decode_kernel<<<1, 256, 0, stream>>>(
            device_logits, device_indices, device_expert_ids, device_weights);
        const cudaError_t status = cudaGetLastError();
        return status == cudaSuccess ||
               fail(error, error_len, cudaGetErrorString(status));
    }
    if (!q38_topk_cuda(device_logits, token_count, Q38_MOE_EXPERTS,
                       Q38_MOE_TOP_K, device_indices, stream, error,
                       error_len))
        return false;
    route_weights_kernel<<<(unsigned)((token_count + 255) / 256), 256, 0,
                            stream>>>(
        device_logits, device_indices, token_count, device_expert_ids,
        device_weights);
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ||
           fail(error, error_len, cudaGetErrorString(status));
}

extern "C" bool q38_moe_cuda_route(
    const float *device_hidden, size_t token_count, const float *device_router,
    float *device_logits, q38_moe_route10 *host_routes, cudaStream_t stream,
    char *error, size_t error_len) {
    if (!host_routes || !q38_moe_cuda_router(device_hidden, token_count,
                                             device_router, device_logits,
                                             stream, error, error_len))
        return false;
    const size_t count = token_count * Q38_MOE_EXPERTS;
    float *logits = (float *)malloc(count * sizeof(float));
    if (!logits) return fail(error, error_len, "CUDA MoE route host allocation failed");
    cudaError_t status = cudaMemcpyAsync(logits, device_logits,
                                         count * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        free(logits);
        return fail(error, error_len, cudaGetErrorString(status));
    }
    for (size_t t = 0; t < token_count; ++t) {
        /* Reuse the scalar tie/normalization policy without routing a second
         * projection: logits are converted to a probability-equivalent
         * one-hot hidden input only by this local selection loop. */
        float max_logit = -INFINITY, sum = 0.0f;
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e)
            max_logit = fmaxf(max_logit, logits[t * Q38_MOE_EXPERTS + e]);
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
            logits[t * Q38_MOE_EXPERTS + e] =
                expf(logits[t * Q38_MOE_EXPERTS + e] - max_logit);
            sum += logits[t * Q38_MOE_EXPERTS + e];
        }
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e)
            logits[t * Q38_MOE_EXPERTS + e] /= sum;
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
            size_t best = Q38_MOE_EXPERTS;
            for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
                bool used = false;
                for (size_t j = 0; j < k; ++j) used |= host_routes[t].expert[j] == e;
                if (!used && (best == Q38_MOE_EXPERTS ||
                    logits[t * Q38_MOE_EXPERTS + e] >
                        logits[t * Q38_MOE_EXPERTS + best] ||
                    (logits[t * Q38_MOE_EXPERTS + e] ==
                     logits[t * Q38_MOE_EXPERTS + best] && e < best)))
                    best = e;
            }
            host_routes[t].expert[k] = (uint16_t)best;
            host_routes[t].weight[k] = logits[t * Q38_MOE_EXPERTS + best];
        }
        float selected = 0.0f;
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) selected += host_routes[t].weight[k];
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) host_routes[t].weight[k] /= selected;
    }
    free(logits);
    return true;
}

extern "C" bool q38_moe_cuda_nvfp4_grouped_indexed(
    const uint8_t *gate_weight, const uint8_t *gate_scale,
    const float *gate_scale_2, const float *gate_input_scale,
    const uint8_t *up_weight, const uint8_t *up_scale,
    const float *up_scale_2, const float *up_input_scale,
    const uint8_t *down_weight, const uint8_t *down_scale,
    const float *down_scale_2, const float *down_input_scale,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    float *device_output, float *device_mid,
    uint8_t *device_activation, uint8_t *device_activation_scale,
    uint8_t *device_down_activation,
    uint8_t *device_down_activation_scale, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!gate_weight || !gate_scale || !gate_scale_2 ||
        !gate_input_scale || !up_weight || !up_scale || !up_scale_2 ||
        !up_input_scale || !down_weight || !down_scale || !down_scale_2 ||
        !down_input_scale || !device_hidden || !device_expert_ids ||
        !device_route_weights || !device_output || !device_mid ||
        !device_activation || !device_activation_scale ||
        !device_down_activation || !device_down_activation_scale ||
        expert_count == 0 || expert_count > Q38_MOE_TOP_K)
        return fail(error, error_len, "invalid CUDA NVFP4 grouped arguments");
    nvfp4_quantize_kernel<<<160u, 32u, 0, stream>>>(
        device_hidden, device_activation, device_activation_scale,
        gate_input_scale, device_expert_ids);
    nvfp4_grouped_gate_up_kernel<<<
        (unsigned)((expert_count * 640u + 3u) / 4u), 128u,
        1280u + 160u, stream>>>(
        gate_weight, gate_scale, gate_scale_2, gate_input_scale,
        up_weight, up_scale, up_scale_2, up_input_scale,
        device_activation, device_activation_scale, device_expert_ids,
        device_mid, expert_count);
    dim3 quant_grid((unsigned)expert_count, 40u, 1u);
    nvfp4_grouped_quantize_kernel<<<quant_grid, 32u, 0, stream>>>(
        device_mid, device_down_activation, device_down_activation_scale,
        down_input_scale, device_expert_ids, expert_count);
    nvfp4_grouped_down_kernel<<<
        (unsigned)((2560u + 3u) / 4u), 128u,
        (unsigned)(expert_count * (320u + 40u)), stream>>>(
        down_weight, down_scale, down_scale_2, down_input_scale,
        device_down_activation, device_down_activation_scale,
        device_expert_ids, device_route_weights, device_output,
        expert_count);
    const cudaError_t status = cudaGetLastError();
    return status == cudaSuccess ||
           fail(error, error_len, cudaGetErrorString(status));
}

__device__ static float q2_value(const q38_q2_k_block *blocks,
                                 size_t row, size_t column,
                                 size_t blocks_per_row) {
    const size_t element = column % 256;
    const q38_q2_k_block *b =
        blocks + row * blocks_per_row + column / 256;
    const size_t half = element / 128;
    const size_t within = element % 128;
    const size_t group = within / 16;
    const size_t l = within % 16;
    const unsigned shift = (unsigned)((group / 2) * 2);
    const uint8_t scale = b->scales[half * 8 + group];
    const size_t qindex = half * 32 + (group & 1) * 16 + l;
    const float d = __half2float(*reinterpret_cast<const __half *>(&b->d));
    const float m = __half2float(*reinterpret_cast<const __half *>(&b->dmin));
    return d * (scale & 0xf) * ((b->qs[qindex] >> shift) & 3) -
           m * (scale >> 4);
}

__global__ static void q2_gate_up_kernel(const q38_q2_k_block *weights,
                                         const float *hidden, float *mid) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q38_MOE_INTERMEDIATE) return;
    float g = 0.0f, u = 0.0f;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
        g += q2_value(weights, i, d, 10) * hidden[d];
        u += q2_value(weights, Q38_MOE_INTERMEDIATE + i, d, 10) *
             hidden[d];
    }
    mid[i] = (g / (1.0f + expf(-g))) * u;
}

__device__ static float q2_dot_row_warp(
    const q38_q2_k_block *weights, const float *hidden, size_t row,
    unsigned lane) {
    float sum = 0.0f;
    for (unsigned block = 0; block < 10; ++block) {
        const q38_q2_k_block *q = weights + row * 10u + block;
        for (unsigned element = lane; element < 256; element += 32u) {
            const unsigned half = element >> 7;
            const unsigned within = element & 127u;
            const unsigned group = within >> 4;
            const unsigned l = within & 15u;
            const unsigned shift = (group >> 1) << 1;
            const uint8_t scale = q->scales[half * 8u + group];
            const unsigned qindex = half * 32u + (group & 1u) * 16u + l;
            const float d = __half2float(
                *reinterpret_cast<const __half *>(&q->d));
            const float m = __half2float(
                *reinterpret_cast<const __half *>(&q->dmin));
            const float value =
                d * (scale & 0xfu) * ((q->qs[qindex] >> shift) & 3u) -
                m * (scale >> 4);
            sum += value * hidden[block * 256u + element];
        }
    }
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}

__global__ static void q2_gate_up_candidate_kernel(
    const q38_q2_k_block *weights, const float *hidden, float *mid) {
    const unsigned lane = threadIdx.x & 31u;
    const size_t row = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= Q38_MOE_INTERMEDIATE) return;
    const float gate = q2_dot_row_warp(weights, hidden, row, lane);
    const float up = q2_dot_row_warp(
        weights, hidden, Q38_MOE_INTERMEDIATE + row, lane);
    if (lane == 0)
        mid[row] = (gate / (1.0f + expf(-gate))) * up;
}

__global__ static void q2_down_kernel(const q38_q2_k_block *weights,
                                      const float *mid, float *output) {
    size_t d = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Q38_MOE_HIDDEN) return;
    float value = 0.0f;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
        value += q2_value(weights, i, d, 10) * mid[i];
    output[d] = value;
}

__global__ static void q2_grouped_gate_up_kernel(
    const q38_q2_k_block *weights, const float *hidden, size_t expert_count,
    const uint16_t *expert_ids, size_t expert_blocks, float *mid) {
    const unsigned lane = threadIdx.x & 31u;
    const size_t warp =
        ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const size_t total_rows = expert_count * Q38_MOE_INTERMEDIATE;
    if (warp >= total_rows) return;
    const size_t expert_slot = warp / Q38_MOE_INTERMEDIATE;
    const size_t row = warp % Q38_MOE_INTERMEDIATE;
    const size_t expert = expert_ids ? expert_ids[expert_slot] : expert_slot;
    const q38_q2_k_block *expert_weights =
        weights + expert * expert_blocks;
    const float gate = q2_dot_row_warp(expert_weights, hidden, row, lane);
    const float up = q2_dot_row_warp(
        expert_weights, hidden, Q38_MOE_INTERMEDIATE + row, lane);
    if (lane == 0)
        mid[expert_slot * Q38_MOE_INTERMEDIATE + row] =
            (gate / (1.0f + expf(-gate))) * up;
}

__global__ static void q2_grouped_down_weighted_kernel(
    const q38_q2_k_block *weights, const float *mid,
    const float *route_weights, size_t expert_count,
    const uint16_t *expert_ids, size_t expert_blocks, float *output) {
    const size_t expert_slot = blockIdx.x;
    const size_t d = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (expert_slot >= expert_count) return;
    if (d >= Q38_MOE_HIDDEN) return;
    const size_t expert =
        expert_ids ? expert_ids[expert_slot] : expert_slot;
    const q38_q2_k_block *expert_weights =
        weights + expert * expert_blocks;
    const float *expert_mid = mid + expert_slot * Q38_MOE_INTERMEDIATE;
    float value = 0.0f;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
        value += q2_value(expert_weights, i, d, 10) * expert_mid[i];
    atomicAdd(output + d, route_weights[expert_slot] * value);
}

__global__ static void q2_grouped_down_private_kernel(
    const q38_q2_k_block *weights, const float *mid,
    size_t expert_count, const uint16_t *expert_ids, size_t expert_blocks,
    float *expert_outputs) {
    const size_t expert_slot = blockIdx.x;
    const size_t d = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (expert_slot >= expert_count) return;
    if (d >= Q38_MOE_HIDDEN) return;
    const size_t expert =
        expert_ids ? expert_ids[expert_slot] : expert_slot;
    const q38_q2_k_block *expert_weights =
        weights + expert * expert_blocks;
    const float *expert_mid = mid + expert_slot * Q38_MOE_INTERMEDIATE;
    float value = 0.0f;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
        value += q2_value(expert_weights, i, d, 10) * expert_mid[i];
    expert_outputs[expert_slot * Q38_MOE_HIDDEN + d] = value;
}

__global__ static void q2_grouped_deterministic_reduce_kernel(
    const float *expert_outputs, const float *route_weights,
    size_t expert_count, float *output) {
    const size_t d = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Q38_MOE_HIDDEN) return;
    float value = 0.0f;
    for (size_t expert_slot = 0; expert_slot < expert_count; ++expert_slot)
        value = __fadd_rn(
            value, __fmul_rn(
                       route_weights[expert_slot],
                       expert_outputs[expert_slot * Q38_MOE_HIDDEN + d]));
    output[d] = value;
}

extern "C" bool q38_moe_cuda_q2_gate_up_candidate(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, unsigned threads_per_block, cudaStream_t stream,
    char *error, size_t error_len);

__device__ static void q4_scale_min(const q38_q4_k_block *block,
                                    unsigned index, uint8_t *scale,
                                    uint8_t *minimum) {
    if (index < 4) {
        *scale = block->scales[index] & 63u;
        *minimum = block->scales[index + 4] & 63u;
    } else {
        *scale = (block->scales[index + 4] & 0xfu) |
                 ((block->scales[index - 4] >> 6) << 4);
        *minimum = (block->scales[index + 4] >> 4) |
                   ((block->scales[index] >> 6) << 4);
    }
}

__device__ static float q4_value(const q38_q4_k_block *blocks,
                                 size_t row, size_t column,
                                 size_t blocks_per_row) {
    const size_t element = column % 256;
    const q38_q4_k_block *block =
        blocks + row * blocks_per_row + column / 256;
    const unsigned group = (unsigned)(element / 64);
    const unsigned within = (unsigned)(element % 64);
    const unsigned scale_index = group * 2u + (within >= 32u);
    uint8_t scale, minimum;
    q4_scale_min(block, scale_index, &scale, &minimum);
    const unsigned qindex = within % 32u;
    const uint8_t packed = block->qs[group * 32u + qindex];
    const unsigned quant = (within >= 32u) ? (packed >> 4) : (packed & 0xfu);
    const float d = __half2float(*reinterpret_cast<const __half *>(&block->d));
    const float m = __half2float(*reinterpret_cast<const __half *>(
        &block->dmin));
    return d * scale * quant - m * minimum;
}

__global__ static void q4_gate_up_kernel(const q38_q4_k_block *weights,
                                         const float *hidden, float *mid) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q38_MOE_INTERMEDIATE) return;
    float gate = 0.0f, up = 0.0f;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
        gate += q4_value(weights, i, d, 10) * hidden[d];
        up += q4_value(weights, Q38_MOE_INTERMEDIATE + i, d, 10) *
              hidden[d];
    }
    mid[i] = (gate / (1.0f + expf(-gate))) * up;
}

__global__ static void q4_down_kernel(const q38_q4_k_block *weights,
                                      const float *mid, float *output) {
    const size_t d = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Q38_MOE_HIDDEN) return;
    float value = 0.0f;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
        value += q4_value(weights, i, d, 10) * mid[i];
    output[d] = value;
}

extern "C" bool q38_moe_cuda_expert_q4_workspace(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, float *device_mid,
    cudaStream_t stream, char *error, size_t error_len) {
    if (!device_gate_up || !device_down || !device_hidden || !device_output ||
        !device_mid)
        return fail(error, error_len, "invalid CUDA Q4 expert arguments");
    q4_gate_up_kernel<<<3, 256, 0, stream>>>(
        (const q38_q4_k_block *)device_gate_up, device_hidden, device_mid);
    q4_down_kernel<<<10, 256, 0, stream>>>(
        (const q38_q4_k_block *)device_down, device_mid, device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_q4_gate_up(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, cudaStream_t stream, char *error, size_t error_len) {
    if (!device_gate_up || !device_hidden || !device_mid)
        return fail(error, error_len, "invalid CUDA Q4 gate/up arguments");
    q4_gate_up_kernel<<<3, 256, 0, stream>>>(
        (const q38_q4_k_block *)device_gate_up, device_hidden, device_mid);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_q4_down(
    const void *device_down, const float *device_mid,
    float *device_output, cudaStream_t stream, char *error, size_t error_len) {
    if (!device_down || !device_mid || !device_output)
        return fail(error, error_len, "invalid CUDA Q4 down arguments");
    q4_down_kernel<<<10, 256, 0, stream>>>(
        (const q38_q4_k_block *)device_down, device_mid, device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_expert_q4(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, cudaStream_t stream,
    char *error, size_t error_len) {
    float *mid = nullptr;
    cudaError_t status = cudaMalloc(&mid, Q38_MOE_INTERMEDIATE * sizeof(float));
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    const bool ok = q38_moe_cuda_expert_q4_workspace(
        device_gate_up, device_down, device_hidden, device_output, mid, stream,
        error, error_len);
    if (ok) status = cudaStreamSynchronize(stream);
    cudaFree(mid);
    if (ok && status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return ok;
}

extern "C" bool q38_moe_cuda_expert_q2_workspace(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, float *device_mid,
    cudaStream_t stream, char *error, size_t error_len) {
    if (!device_gate_up || !device_down || !device_hidden || !device_output ||
        !device_mid)
        return fail(error, error_len, "invalid CUDA Q2 expert arguments");
    q2_gate_up_kernel<<<3, 256, 0, stream>>>(
        (const q38_q2_k_block *)device_gate_up, device_hidden, device_mid);
    q2_down_kernel<<<10, 256, 0, stream>>>(
        (const q38_q2_k_block *)device_down, device_mid, device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_expert_q2(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, cudaStream_t stream,
    char *error, size_t error_len) {
    float *mid = nullptr;
    cudaError_t status = cudaMalloc(&mid, Q38_MOE_INTERMEDIATE * sizeof(float));
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    const bool ok = q38_moe_cuda_expert_q2_workspace(
        device_gate_up, device_down, device_hidden, device_output, mid, stream,
        error, error_len);
    if (ok) status = cudaStreamSynchronize(stream);
    cudaFree(mid);
    if (ok && status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return ok;
}

extern "C" bool q38_moe_cuda_q2_gate_up(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, cudaStream_t stream, char *error, size_t error_len) {
    if (!device_gate_up || !device_hidden || !device_mid)
        return fail(error, error_len, "invalid CUDA Q2 gate/up arguments");
    return q38_moe_cuda_q2_gate_up_candidate(
        device_gate_up, device_hidden, device_mid, 128, stream, error,
        error_len);
}

extern "C" bool q38_moe_cuda_q2_gate_up_candidate(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, unsigned threads_per_block, cudaStream_t stream,
    char *error, size_t error_len) {
    if (!device_gate_up || !device_hidden || !device_mid ||
        (threads_per_block != 128 && threads_per_block != 256 &&
         threads_per_block != 512))
        return fail(error, error_len, "invalid Q2 candidate geometry");
    const unsigned warps_per_block = threads_per_block / 32u;
    const unsigned blocks =
        (Q38_MOE_INTERMEDIATE + warps_per_block - 1u) / warps_per_block;
    q2_gate_up_candidate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        (const q38_q2_k_block *)device_gate_up, device_hidden, device_mid);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_q2_down(
    const void *device_down, const float *device_mid,
    float *device_output, cudaStream_t stream, char *error, size_t error_len) {
    if (!device_down || !device_mid || !device_output)
        return fail(error, error_len, "invalid CUDA Q2 down arguments");
    q2_down_kernel<<<10, 256, 0, stream>>>(
        (const q38_q2_k_block *)device_down, device_mid, device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

__global__ static void q2_weighted_accum_kernel(
    float *accum, const float *expert, float weight) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Q38_MOE_HIDDEN) return;
    accum[d] = __fadd_rn(accum[d], __fmul_rn(weight, expert[d]));
}

extern "C" bool q38_moe_cuda_accumulate_weighted(
    float *device_accum, const float *device_expert, float weight,
    cudaStream_t stream, char *error, size_t error_len) {
    if (!device_accum || !device_expert)
        return fail(error, error_len,
                    "invalid CUDA MoE accumulation arguments");
    q2_weighted_accum_kernel<<<10, 256, 0, stream>>>(
        device_accum, device_expert, weight);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_q2_grouped_indexed(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    size_t gate_expert_blocks, size_t down_expert_blocks,
    float *device_output, float *device_mid, cudaStream_t stream,
    char *error, size_t error_len);

extern "C" bool q38_moe_cuda_q2_grouped(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const float *device_route_weights,
    size_t expert_count, float *device_output, float *device_mid,
    cudaStream_t stream, char *error, size_t error_len) {
    if (!device_gate_up || !device_down || !device_hidden ||
        !device_route_weights || !expert_count || expert_count > Q38_MOE_TOP_K ||
        !device_output || !device_mid)
        return fail(error, error_len, "invalid grouped Q2 MoE arguments");
    const size_t gate_expert_blocks =
        2u * Q38_MOE_INTERMEDIATE * 10u;
    const size_t down_expert_blocks = Q38_MOE_INTERMEDIATE * 10u;
    return q38_moe_cuda_q2_grouped_indexed(
        device_gate_up, device_down, device_hidden, nullptr,
        device_route_weights, expert_count, gate_expert_blocks,
        down_expert_blocks, device_output, device_mid, stream, error,
        error_len);
}

extern "C" bool q38_moe_cuda_q2_grouped_indexed(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    size_t gate_expert_blocks, size_t down_expert_blocks,
    float *device_output, float *device_mid, cudaStream_t stream,
    char *error, size_t error_len) {
    if (!device_gate_up || !device_down || !device_hidden ||
        !device_route_weights || !expert_count || expert_count > Q38_MOE_TOP_K ||
        !gate_expert_blocks || !down_expert_blocks || !device_output ||
        !device_mid)
        return fail(error, error_len, "invalid indexed grouped Q2 arguments");
    const unsigned threads = 128u;
    const size_t total_rows = expert_count * Q38_MOE_INTERMEDIATE;
    const unsigned gate_blocks = (unsigned)((total_rows +
                                             (threads / 32u) - 1u) /
                                            (threads / 32u));
    q2_grouped_gate_up_kernel<<<gate_blocks, threads, 0, stream>>>(
        (const q38_q2_k_block *)device_gate_up, device_hidden, expert_count,
        device_expert_ids, gate_expert_blocks, device_mid);
    if (cudaMemsetAsync(device_output, 0, Q38_MOE_HIDDEN * sizeof(float),
                        stream) != cudaSuccess)
        return fail(error, error_len, "grouped Q2 output clear failed");
    q2_grouped_down_weighted_kernel<<<dim3((unsigned)expert_count, 10, 1),
                                      256, 0, stream>>>(
        (const q38_q2_k_block *)device_down, device_mid, device_route_weights,
        expert_count, device_expert_ids, down_expert_blocks, device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

extern "C" bool q38_moe_cuda_q2_grouped_indexed_deterministic(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    size_t gate_expert_blocks, size_t down_expert_blocks,
    float *device_output, float *device_mid, float *device_expert_outputs,
    cudaStream_t stream, char *error, size_t error_len) {
    if (!device_gate_up || !device_down || !device_hidden ||
        !device_route_weights || !expert_count ||
        expert_count > Q38_MOE_TOP_K || !gate_expert_blocks ||
        !down_expert_blocks || !device_output || !device_mid ||
        !device_expert_outputs)
        return fail(error, error_len,
                    "invalid deterministic indexed grouped Q2 arguments");
    const unsigned threads = 128u;
    const size_t total_rows = expert_count * Q38_MOE_INTERMEDIATE;
    const unsigned gate_blocks = (unsigned)((total_rows +
                                             (threads / 32u) - 1u) /
                                            (threads / 32u));
    q2_grouped_gate_up_kernel<<<gate_blocks, threads, 0, stream>>>(
        (const q38_q2_k_block *)device_gate_up, device_hidden, expert_count,
        device_expert_ids, gate_expert_blocks, device_mid);
    q2_grouped_down_private_kernel<<<dim3((unsigned)expert_count, 10, 1),
                                    256, 0, stream>>>(
        (const q38_q2_k_block *)device_down, device_mid, expert_count,
        device_expert_ids, down_expert_blocks,
        device_expert_outputs);
    q2_grouped_deterministic_reduce_kernel<<<10, 256, 0, stream>>>(
        device_expert_outputs, device_route_weights, expert_count,
        device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

__global__ static void shared_kernel(const float *hidden, size_t tokens,
                                     const float *gate_proj, const float *up_proj,
                                     const float *down_proj,
                                     const float *gate_weight, float *output) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * Q38_MOE_HIDDEN) return;
    const size_t token = index / Q38_MOE_HIDDEN;
    const size_t d = index % Q38_MOE_HIDDEN;
    const float *x = hidden + token * Q38_MOE_HIDDEN;
    float shared_gate = 0.0f;
    for (size_t j = 0; j < Q38_MOE_HIDDEN; ++j)
        shared_gate += gate_weight[j] * x[j];
    shared_gate = 1.0f / (1.0f + expf(-shared_gate));
    float value = 0.0f;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        float g = 0.0f, u = 0.0f;
        for (size_t j = 0; j < Q38_MOE_HIDDEN; ++j) {
            g += gate_proj[i * Q38_MOE_HIDDEN + j] * x[j];
            u += up_proj[i * Q38_MOE_HIDDEN + j] * x[j];
        }
        const float mid = (g / (1.0f + expf(-g))) * u;
        value += down_proj[d * Q38_MOE_INTERMEDIATE + i] * mid;
    }
    output[index] = value * shared_gate;
}

extern "C" bool q38_moe_cuda_shared_f32(
    const float *device_hidden, size_t token_count,
    const float *device_gate_proj, const float *device_up_proj,
    const float *device_down_proj, const float *device_gate_weight,
    float *device_output, cudaStream_t stream, char *error, size_t error_len) {
    if (!device_hidden || !token_count || !device_gate_proj ||
        !device_up_proj || !device_down_proj || !device_gate_weight ||
        !device_output || token_count > SIZE_MAX / Q38_MOE_HIDDEN)
        return fail(error, error_len, "invalid CUDA shared expert arguments");
    const size_t total = token_count * Q38_MOE_HIDDEN;
    shared_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        device_hidden, token_count, device_gate_proj, device_up_proj,
        device_down_proj, device_gate_weight, device_output);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    return true;
}
