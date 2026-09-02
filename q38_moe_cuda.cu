#include "q38_moe_cuda.h"
#include "q38_moe_ref.h"
#include "q38_quant.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_fp16.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
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

__global__ static void q2_down_kernel(const q38_q2_k_block *weights,
                                      const float *mid, float *output) {
    size_t d = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= Q38_MOE_HIDDEN) return;
    float value = 0.0f;
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
        value += q2_value(weights, i, d, 10) * mid[i];
    output[d] = value;
}

extern "C" bool q38_moe_cuda_expert_q2(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, cudaStream_t stream,
    char *error, size_t error_len) {
    if (!device_gate_up || !device_down || !device_hidden || !device_output)
        return fail(error, error_len, "invalid CUDA Q2 expert arguments");
    float *mid = nullptr;
    cudaError_t status = cudaMalloc(&mid, Q38_MOE_INTERMEDIATE * sizeof(float));
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    q2_gate_up_kernel<<<3, 256, 0, stream>>>(
        (const q38_q2_k_block *)device_gate_up, device_hidden, mid);
    q2_down_kernel<<<10, 256, 0, stream>>>(
        (const q38_q2_k_block *)device_down, mid, device_output);
    status = cudaGetLastError();
    cudaFree(mid);
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
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
