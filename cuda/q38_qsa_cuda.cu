#include "q38_qsa_cuda.h"

#include "q38_gdn.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

static bool fail_cuda(char *error, size_t error_len) {
    return fail(error, error_len, cudaGetErrorString(cudaGetLastError()));
}

__device__ static float bf16_to_float(uint16_t value) {
    return __int_as_float((int)((uint32_t)value << 16));
}

__device__ static float chain_rope_angle(size_t component, size_t rotary,
                                         uint64_t position) {
    float frequency = 1.0f;
    const float base = powf(10000000.0f, -2.0f / (float)rotary);
    for (size_t i = 0; i < component; ++i) frequency *= base;
    return (float)position * frequency;
}

__global__ static void chain_prepare_kernel(
    const float *qfull, const float *raw_k, const float *raw_v,
    const float *index, const uint16_t *q_norm, const uint16_t *k_norm,
    const uint16_t *index_q_norm, size_t position, float *q, float *k,
    float *v, float *index_q, float *raw_index) {
    const size_t head = blockIdx.x;
    const size_t lane = threadIdx.x;
    if (lane != 0) return;
    if (head < 24) {
        float sum = 0.0f;
        for (size_t d = 0; d < 256; ++d) {
            const float value = qfull[head * 512 + d];
            sum += value * value;
        }
        const float scale = rsqrtf(sum / 256.0f + 1e-6f);
        for (size_t p = 0; p < 32; ++p) {
            const float angle = chain_rope_angle(p, 64, position);
            const float c = cosf(angle), s = sinf(angle);
            const size_t a = p, b = 32 + p;
            const float x = qfull[head * 512 + a] * scale *
                            (1.0f + bf16_to_float(q_norm[a]));
            const float y = qfull[head * 512 + b] * scale *
                            (1.0f + bf16_to_float(q_norm[b]));
            q[head * 256 + a] = x * c - y * s;
            q[head * 256 + b] = x * s + y * c;
        }
        for (size_t d = 64; d < 256; ++d)
            q[head * 256 + d] =
                qfull[head * 512 + d] * scale *
                (1.0f + bf16_to_float(q_norm[d]));
        return;
    }
    if (head < 26) {
        const size_t kv_head = head - 24;
        float sum = 0.0f;
        for (size_t d = 0; d < 256; ++d) {
            const float value = raw_k[kv_head * 256 + d];
            sum += value * value;
        }
        const float scale = rsqrtf(sum / 256.0f + 1e-6f);
        for (size_t p = 0; p < 32; ++p) {
            const float angle = chain_rope_angle(p, 64, position);
            const float c = cosf(angle), s = sinf(angle);
            const float x = raw_k[kv_head * 256 + p] * scale *
                            (1.0f + bf16_to_float(k_norm[p]));
            const float y = raw_k[kv_head * 256 + 32 + p] * scale *
                            (1.0f + bf16_to_float(k_norm[32 + p]));
            k[kv_head * 256 + p] = x * c - y * s;
            k[kv_head * 256 + 32 + p] = x * s + y * c;
        }
        for (size_t d = 64; d < 256; ++d)
            k[kv_head * 256 + d] =
                raw_k[kv_head * 256 + d] * scale *
                (1.0f + bf16_to_float(k_norm[d]));
        for (size_t d = 0; d < 256; ++d)
            v[kv_head * 256 + d] = raw_v[kv_head * 256 + d];
        return;
    }
    const size_t index_head = head - 26;
    if (index_head < 4) {
        float sum = 0.0f;
        for (size_t d = 0; d < 128; ++d) {
            const float value = index[index_head * 128 + d];
            sum += value * value;
        }
        const float scale = rsqrtf(sum / 128.0f + 1e-6f);
        for (size_t p = 0; p < 32; ++p) {
            const float angle = chain_rope_angle(p, 64, position);
            const float c = cosf(angle), s = sinf(angle);
            const float x = index[index_head * 128 + p] * scale *
                            (1.0f + bf16_to_float(index_q_norm[p]));
            const float y = index[index_head * 128 + 32 + p] * scale *
                            (1.0f + bf16_to_float(index_q_norm[32 + p]));
            index_q[index_head * 128 + p] = x * c - y * s;
            index_q[index_head * 128 + 32 + p] = x * s + y * c;
        }
        for (size_t d = 64; d < 128; ++d)
            index_q[index_head * 128 + d] =
                index[index_head * 128 + d] * scale *
                (1.0f + bf16_to_float(index_q_norm[d]));
        return;
    }
    if (index_head == 4)
        for (size_t d = 0; d < 128; ++d)
            raw_index[d] = index[4 * 128 + d];
}

__global__ static void chain_append_kernel(
    const float *k, const float *v, const float *raw_index, float *state_k,
    float *state_v, float *state_index, size_t row) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 512) {
        state_k[row * 512 + index] = k[index];
        state_v[row * 512 + index] = v[index];
    } else if (index < 640)
        state_index[row * 128 + index - 512] = raw_index[index - 512];
}

__global__ static void chain_select_kernel(
    const float *state_index, const float *index_q, const uint16_t *index_k_norm,
    size_t visible, uint32_t *selected, size_t selected_capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const size_t ratio = 4, dim = 128, budget = 2048;
    const size_t complete = visible / ratio;
    const size_t groups = complete < budget / ratio ? complete : budget / ratio;
    float scores[512];
    uint32_t ids[512];
    size_t used = 0;
    for (size_t candidate = 0; candidate < complete; ++candidate) {
        const size_t begin = candidate * ratio;
        float total = 0.0f;
        float key[128];
        for (size_t d = 0; d < dim; ++d) {
            float value = 0.0f;
            for (size_t j = 0; j < ratio; ++j)
                value += state_index[(begin + j) * dim + d];
            key[d] = value * 0.25f;
        }
        float norm = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            key[d] *= 1.0f + bf16_to_float(index_k_norm[d]);
            norm += key[d] * key[d];
        }
        const float scale = rsqrtf(norm / dim + 1e-6f);
        for (size_t d = 0; d < dim; ++d) key[d] *= scale;
        for (size_t h = 0; h < 4; ++h) {
            float dot = 0.0f;
            for (size_t d = 0; d < dim; ++d)
                dot += index_q[h * dim + d] * key[d];
            total += dot > 0.0f ? dot : 0.0f;
        }
        const float score = total / sqrtf((float)dim);
        size_t at = used;
        while (at > 0) {
            const size_t previous = ids[at - 1];
            if (scores[at - 1] > score ||
                (scores[at - 1] == score && previous < candidate))
                break;
            --at;
        }
        if (at < groups) {
            if (used < groups) ++used;
            for (size_t j = used; j > at + 1; --j) {
                scores[j - 1] = scores[j - 2];
                ids[j - 1] = ids[j - 2];
            }
            scores[at] = score;
            ids[at] = (uint32_t)candidate;
        }
    }
    size_t out = 0;
    for (size_t rank = 0; rank < groups; ++rank)
        for (size_t j = 0; j < ratio; ++j)
            if (out < selected_capacity)
                selected[out++] = ids[rank] * ratio + (uint32_t)j;
    for (size_t i = complete * ratio; i < visible; ++i)
        if (out < selected_capacity) selected[out++] = (uint32_t)i;
}

__global__ static void chain_gate_kernel(const float *qfull, float *attention) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 6144) return;
    const size_t head = index / 256, d = index % 256;
    const float gate = 1.0f /
        (1.0f + expf(-qfull[head * 512 + 256 + d]));
    attention[index] *= gate;
}

extern "C" bool q38_qsa_cuda_chain_reserve(
    q38_qsa_cuda_chain_state *state, size_t capacity, cudaStream_t stream,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!state || !capacity || capacity <= state->capacity) return true;
    if (capacity > SIZE_MAX / (512u * sizeof(float)) ||
        capacity > SIZE_MAX / (128u * sizeof(float)))
        return fail(error, error_len, "QSA chain state size overflows");
    float *main_k = nullptr, *main_v = nullptr, *index_k = nullptr;
    if (cudaMalloc((void **)&main_k, capacity * 512 * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc((void **)&main_v, capacity * 512 * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc((void **)&index_k, capacity * 128 * sizeof(float)) !=
            cudaSuccess) {
        cudaFree(main_k); cudaFree(main_v); cudaFree(index_k);
        return fail(error, error_len, "QSA chain state allocation failed");
    }
    if (state->count) {
        if (cudaMemcpyAsync(main_k, state->main_k,
                            state->count * 512 * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(main_v, state->main_v,
                            state->count * 512 * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(index_k, state->index_k,
                            state->count * 128 * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            cudaFree(main_k); cudaFree(main_v); cudaFree(index_k);
            return fail(error, error_len, "QSA chain state growth copy failed");
        }
    }
    cudaFree(state->main_k); cudaFree(state->main_v); cudaFree(state->index_k);
    state->main_k = main_k;
    state->main_v = main_v;
    state->index_k = index_k;
    state->capacity = capacity;
    return true;
}

extern "C" void q38_qsa_cuda_chain_release(q38_qsa_cuda_chain_state *state) {
    if (!state) return;
    cudaFree(state->main_k); cudaFree(state->main_v); cudaFree(state->index_k);
    *state = {};
}

extern "C" void q38_qsa_cuda_chain_reset(q38_qsa_cuda_chain_state *state) {
    if (!state) return;
    state->count = 0;
    state->position = 0;
}

extern "C" bool q38_qsa_cuda_chain_decode(
    const uint16_t *q_proj, const uint16_t *k_proj, const uint16_t *v_proj,
    const uint16_t *index_qk_proj, const uint16_t *o_proj,
    const uint16_t *q_norm, const uint16_t *k_norm,
    const uint16_t *index_q_norm, const uint16_t *index_k_norm,
    const float *device_input, float *device_output, uint64_t position,
    q38_qsa_cuda_chain_state *state, q38_qsa_cuda_chain_workspace *workspace,
    cudaStream_t stream, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q_proj || !k_proj || !v_proj || !index_qk_proj || !o_proj ||
        !q_norm || !k_norm || !index_q_norm || !index_k_norm ||
        !device_input || !device_output || !state || !workspace ||
        !state->main_k || !state->main_v || !state->index_k ||
        !workspace->qfull || !workspace->q || !workspace->k ||
        !workspace->v || !workspace->index || !workspace->index_q ||
        !workspace->raw_index || !workspace->attention ||
        !workspace->selected_k || !workspace->selected_v ||
        !workspace->selected || !workspace->selected_capacity ||
        state->count >= state->capacity || state->position != position)
        return fail(error, error_len, "invalid QSA chain decode state");
    if (!q38_qsa_cuda_project_device(
            q_proj, 12288, k_proj, 512, v_proj, 512, 2560, device_input, 1,
            workspace->qfull, workspace->k, workspace->v, stream, error,
            error_len) ||
        !q38_cuda_bf16_matvec_device(
            index_qk_proj, 640, 2560, device_input, workspace->index, stream,
            error, error_len))
        return false;
    chain_prepare_kernel<<<30, 1, 0, stream>>>(
        workspace->qfull, workspace->k, workspace->v, workspace->index,
        q_norm, k_norm, index_q_norm, (size_t)position, workspace->q,
        workspace->k, workspace->v, workspace->index_q, workspace->raw_index);
    chain_append_kernel<<<(640 + 255) / 256, 256, 0, stream>>>(
        workspace->k, workspace->v, workspace->raw_index, state->main_k,
        state->main_v, state->index_k, state->count);
    const size_t visible = state->count + 1;
    chain_select_kernel<<<1, 1, 0, stream>>>(
        state->index_k, workspace->index_q, index_k_norm, visible,
        workspace->selected, workspace->selected_capacity);
    const size_t selected_count =
        (visible / 4 < 512 ? visible / 4 : 512) * 4 + visible % 4;
    if (!q38_qsa_cuda_gather_attention(
            state->main_k, state->main_v, visible, 2, 256, workspace->selected,
            selected_count, workspace->selected_k, workspace->selected_v,
            workspace->q, 1, 24, workspace->attention, stream, error,
            error_len))
        return false;
    chain_gate_kernel<<<(6144 + 255) / 256, 256, 0, stream>>>(
        workspace->qfull, workspace->attention);
    if (!q38_cuda_bf16_matvec_device(
            o_proj, 2560, 6144, workspace->attention, device_output, stream,
            error, error_len))
        return false;
    if (cudaGetLastError() != cudaSuccess) return fail_cuda(error, error_len);
    state->count = visible;
    state->position = position + 1;
    return true;
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
    if (!q38_qsa_cuda_project_device(
            q_proj, q_rows, k_proj, k_rows, v_proj, v_rows, cols,
            device_input, token_count, device_q, device_k, device_v, stream,
            error, error_len))
        return false;
    return true;
}

extern "C" bool q38_qsa_cuda_project_device(
    const uint16_t *q_proj, size_t q_rows, const uint16_t *k_proj,
    size_t k_rows, const uint16_t *v_proj, size_t v_rows, size_t cols,
    const float *device_input, size_t token_count, float *device_q,
    float *device_k, float *device_v, cudaStream_t stream, char *error,
    size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!q_proj || !k_proj || !v_proj || !cols || !device_input ||
        !token_count || !device_q || !device_k || !device_v) {
        return fail(error, error_len, "invalid CUDA QSA device projection arguments");
    }
    if (q_rows > SIZE_MAX / token_count || k_rows > SIZE_MAX / token_count ||
        v_rows > SIZE_MAX / token_count)
        return fail(error, error_len, "CUDA QSA device projection size overflows");
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
