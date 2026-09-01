#ifndef Q38_QSA_CUDA_H
#define Q38_QSA_CUDA_H

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool q38_qsa_cuda_project_main(const uint16_t *q_proj, size_t q_rows,
                               const uint16_t *k_proj, size_t k_rows,
                               const uint16_t *v_proj, size_t v_rows,
                               size_t cols, const float *device_input,
                               size_t token_count, float *device_q,
                               float *device_k, float *device_v,
                               cudaStream_t stream, char *error,
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
