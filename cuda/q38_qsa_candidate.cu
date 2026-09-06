#include "q38_qsa_candidate.h"

#include <cuda_runtime.h>

#include <stdint.h>

__device__ static float candidate_bf16_to_float(uint16_t value) {
    return __int_as_float((int)((uint32_t)value << 16));
}

__global__ static void candidate_project_kernel(
    const uint16_t *weights, size_t rows, size_t cols, const float *input,
    size_t tokens, float *output) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const size_t index = (size_t)blockIdx.x * (blockDim.x >> 5) + warp;
    const size_t elements = rows * tokens;
    if (index >= elements) return;
    const size_t token = index / rows;
    const size_t row = index - token * rows;
    float sum = 0.0f;
    for (size_t base = 0; base < cols; base += 32) {
        const size_t col = base + lane;
        const float product =
            col < cols
                ? __fmul_rn(candidate_bf16_to_float(weights[row * cols + col]),
                            input[token * cols + col])
                : 0.0f;
        for (unsigned source = 0; source < 32; ++source) {
            const float value = __shfl_sync(0xffffffffu, product, source);
            if (lane == 0) sum = __fadd_rn(sum, value);
        }
    }
    if (lane == 0) output[index] = sum;
}

extern "C" int q38_qsa_candidate_abi(void) {
    return Q38_QSA_CANDIDATE_ABI_VERSION;
}

extern "C" int q38_qsa_candidate_project(
    const void *wq, const void *wk, const void *wv,
    const float *input, float *q, float *k, float *v,
    size_t token_count, size_t cols, void *stream) {
    if (!wq || !wk || !wv || !input || !q || !k || !v ||
        !token_count || !cols)
        return 1;
    const size_t q_rows = 12288;
    const size_t k_rows = 512;
    const size_t v_rows = 512;
    candidate_project_kernel<<<
        (unsigned)((q_rows * token_count + 7) / 8), 256, 0,
        (cudaStream_t)stream>>>(
        (const uint16_t *)wq, q_rows, cols, input, token_count, q);
    candidate_project_kernel<<<
        (unsigned)((k_rows * token_count + 7) / 8), 256, 0,
        (cudaStream_t)stream>>>(
        (const uint16_t *)wk, k_rows, cols, input, token_count, k);
    candidate_project_kernel<<<
        (unsigned)((v_rows * token_count + 7) / 8), 256, 0,
        (cudaStream_t)stream>>>(
        (const uint16_t *)wv, v_rows, cols, input, token_count, v);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
