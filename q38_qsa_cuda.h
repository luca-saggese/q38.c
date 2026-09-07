#ifndef Q38_QSA_CUDA_H
#define Q38_QSA_CUDA_H

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *main_k;
    float *main_v;
    float *index_k;
    size_t capacity;
    size_t count;
    uint64_t position;
} q38_qsa_cuda_chain_state;

typedef struct {
    float *qfull;
    float *q;
    float *k;
    float *v;
    float *index;
    float *index_q;
    float *raw_index;
    float *attention;
    float *selected_k;
    float *selected_v;
    uint32_t *selected;
    size_t selected_capacity;
} q38_qsa_cuda_chain_workspace;

bool q38_qsa_cuda_chain_reserve(q38_qsa_cuda_chain_state *state,
                                size_t capacity, cudaStream_t stream,
                                char *error, size_t error_len);
void q38_qsa_cuda_chain_release(q38_qsa_cuda_chain_state *state);
void q38_qsa_cuda_chain_reset(q38_qsa_cuda_chain_state *state);

/*
 * Single-token QSA chain.  All weights, activations, cache rows, and
 * workspaces are device-resident.  The function only enqueues work; the
 * caller owns the boundary input upload/output download and synchronization.
 */
bool q38_qsa_cuda_chain_decode(
    const uint16_t *q_proj, const uint16_t *k_proj, const uint16_t *v_proj,
    const uint16_t *index_qk_proj, const uint16_t *o_proj,
    const uint16_t *q_norm, const uint16_t *k_norm,
    const uint16_t *index_q_norm, const uint16_t *index_k_norm,
    const float *device_input, float *device_output, uint64_t position,
    q38_qsa_cuda_chain_state *state, q38_qsa_cuda_chain_workspace *workspace,
    cudaStream_t stream, char *error, size_t error_len);

bool q38_qsa_cuda_project_main(const uint16_t *q_proj, size_t q_rows,
                               const uint16_t *k_proj, size_t k_rows,
                               const uint16_t *v_proj, size_t v_rows,
                               size_t cols, const float *device_input,
                               size_t token_count, float *device_q,
                               float *device_k, float *device_v,
                               cudaStream_t stream, char *error,
                               size_t error_len);

bool q38_qsa_cuda_project_device(
    const uint16_t *q_proj, size_t q_rows, const uint16_t *k_proj,
    size_t k_rows, const uint16_t *v_proj, size_t v_rows, size_t cols,
    const float *device_input, size_t token_count, float *device_q,
    float *device_k, float *device_v, cudaStream_t stream, char *error,
    size_t error_len);

bool q38_qsa_cuda_apply_rope(float *device_tensor, size_t token_count,
                             size_t head_count, size_t head_dim,
                             size_t rotary_dims, int64_t position,
                             const uint32_t sections[4], cudaStream_t stream,
                             char *error, size_t error_len);

bool q38_qsa_cuda_index_scores(const float *device_raw_keys,
                               size_t token_count,
                               const float *device_queries,
                               size_t query_count, size_t heads,
                               size_t head_dim, size_t ratio,
                               float *device_scores, cudaStream_t stream,
                               char *error, size_t error_len);

bool q38_qsa_cuda_gather_attention(
    const float *device_k, const float *device_v, size_t kv_count,
    size_t kv_heads, size_t head_dim, const uint32_t *device_ids,
    size_t selected_count, float *device_selected_k,
    float *device_selected_v, const float *device_query, size_t query_count,
    size_t query_heads, float *device_output, cudaStream_t stream,
    char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
