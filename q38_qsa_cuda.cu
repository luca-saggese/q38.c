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
    for (size_t col = 0; col < cols; ++col) {
        const float product = __fmul_rn(
            bf16_to_float(weights[row * cols + col]),
            input[token * cols + col]);
        sum = __fadd_rn(sum, product);
    }
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

__global__ static void index_scores_kernel(
    const float *raw_keys, size_t token_count, const float *queries,
    size_t query_count, size_t heads, size_t head_dim, size_t ratio,
    float *scores) {
    const size_t blocks = (token_count + ratio - 1) / ratio;
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= query_count * blocks) return;
    const size_t query = index / blocks;
    const size_t block = index % blocks;
    const size_t begin = block * ratio;
    const size_t end = begin + ratio < token_count ? begin + ratio : token_count;
    float total = 0.0f;
    for (size_t head = 0; head < heads; ++head) {
        float dot = 0.0f;
        float norm = 0.0f;
        float qnorm = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            float pooled = 0.0f;
            for (size_t token = begin; token < end; ++token)
                pooled += raw_keys[token * head_dim + d];
            pooled /= (float)(end - begin);
            dot += queries[(query * heads + head) * head_dim + d] * pooled;
            norm += pooled * pooled;
            const float q = queries[(query * heads + head) * head_dim + d];
            qnorm += q * q;
        }
        dot /= sqrtf(norm / (float)head_dim + 1e-6f);
        dot /= sqrtf(qnorm / (float)head_dim + 1e-6f);
        total += dot > 0.0f ? dot : 0.0f;
    }
    scores[index] = total / sqrtf((float)head_dim);
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

extern "C" bool q38_qsa_cuda_index_scores(
    const float *device_raw_keys, size_t token_count,
    const float *device_queries, size_t query_count, size_t heads,
    size_t head_dim, size_t ratio, float *device_scores, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!device_raw_keys || !token_count || !device_queries || !query_count ||
        !heads || !head_dim || !ratio || !device_scores ||
        ratio > SIZE_MAX - token_count + 1 ||
        heads > SIZE_MAX / head_dim) {
        return fail(error, error_len, "invalid CUDA QSA index arguments");
    }
    const size_t blocks = (token_count + ratio - 1) / ratio;
    if (query_count > SIZE_MAX / blocks)
        return fail(error, error_len, "CUDA QSA index size overflows");
    const size_t elements = query_count * blocks;
    index_scores_kernel<<<(unsigned)((elements + 255) / 256), 256, 0, stream>>>(
        device_raw_keys, token_count, device_queries, query_count, heads,
        head_dim, ratio, device_scores);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    return true;
}

__global__ static void gather_kernel(
    const float *source, size_t kv_count, size_t kv_heads, size_t head_dim,
    const uint32_t *ids, size_t selected_count, float *output) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = selected_count * kv_heads * head_dim;
    if (index >= elements) return;
    const size_t d = index % head_dim;
    const size_t head = (index / head_dim) % kv_heads;
    const size_t selected = index / (kv_heads * head_dim);
    const uint32_t row = ids[selected];
    if ((size_t)row >= kv_count) {
        output[index] = __int_as_float(0x7fc00000);
        return;
    }
    output[index] = source[((size_t)row * kv_heads + head) * head_dim + d];
}

__global__ static void attention_kernel(
    const float *query, size_t query_count, size_t query_heads,
    size_t head_dim, const float *selected_k, const float *selected_v,
    size_t selected_count, size_t kv_heads, float *output) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = query_count * query_heads * head_dim;
    if (index >= elements) return;
    const size_t d = index % head_dim;
    const size_t head = (index / head_dim) % query_heads;
    const size_t token = index / (query_heads * head_dim);
    const size_t group = query_heads / kv_heads;
    const size_t kv_head = head / group;
    const float *q = query + (token * query_heads + head) * head_dim;
    float maximum = -1.0e30f;
    for (size_t row = 0; row < selected_count; ++row) {
        const float *k = selected_k + (row * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (size_t i = 0; i < head_dim; ++i) dot += q[i] * k[i];
        maximum = fmaxf(maximum, dot / sqrtf((float)head_dim));
    }
    float denominator = 0.0f, numerator = 0.0f;
    for (size_t row = 0; row < selected_count; ++row) {
        const float *k = selected_k + (row * kv_heads + kv_head) * head_dim;
        const float *v = selected_v + (row * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (size_t i = 0; i < head_dim; ++i) dot += q[i] * k[i];
        const float weight = expf(dot / sqrtf((float)head_dim) - maximum);
        denominator += weight;
        if (d < head_dim) numerator += weight * v[d];
    }
    output[index] = numerator / denominator;
}

extern "C" bool q38_qsa_cuda_gather_attention(
    const float *device_k, const float *device_v, size_t kv_count,
    size_t kv_heads, size_t head_dim, const uint32_t *device_ids,
    size_t selected_count, float *device_selected_k,
    float *device_selected_v, const float *device_query, size_t query_count,
    size_t query_heads, float *device_output, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!device_k || !device_v || !kv_count || !kv_heads || !head_dim ||
        !device_ids || !selected_count || !device_selected_k ||
        !device_selected_v || !device_query || !query_count || !query_heads ||
        query_heads < kv_heads || query_heads % kv_heads != 0 ||
        kv_heads > SIZE_MAX / head_dim ||
        query_heads > SIZE_MAX / head_dim ||
        selected_count > SIZE_MAX / (kv_heads * head_dim) ||
        query_count > SIZE_MAX / (query_heads * head_dim))
        return fail(error, error_len, "invalid CUDA QSA gather arguments");
    const size_t gather_elements = selected_count * kv_heads * head_dim;
    const size_t attention_elements = query_count * query_heads * head_dim;
    gather_kernel<<<(unsigned)((gather_elements + 255) / 256), 256, 0,
                    stream>>>(device_k, kv_count, kv_heads, head_dim,
                              device_ids, selected_count, device_selected_k);
    gather_kernel<<<(unsigned)((gather_elements + 255) / 256), 256, 0,
                    stream>>>(device_v, kv_count, kv_heads, head_dim,
                              device_ids, selected_count, device_selected_v);
    attention_kernel<<<(unsigned)((attention_elements + 255) / 256), 256, 0,
                       stream>>>(device_query, query_count, query_heads,
                                 head_dim, device_selected_k, device_selected_v,
                                 selected_count, kv_heads, device_output);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return fail(error, error_len, cudaGetErrorString(status));
    return true;
}
