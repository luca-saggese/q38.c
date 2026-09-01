#ifndef Q38_QSA_REF_H
#define Q38_QSA_REF_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool q38_qsa_index_scores_ref(const float *raw_keys, size_t token_count,
                              const float *queries, size_t query_count,
                              size_t heads, size_t head_dim, size_t ratio,
                              float *scores, char *error, size_t error_len);

bool q38_qsa_expand_block_scores_ref(const float *block_scores,
                                     size_t block_count, size_t token_count,
                                     size_t ratio, float *token_scores,
                                     char *error, size_t error_len);

bool q38_qsa_selected_width_ref(size_t kv_count, size_t top_k, size_t ratio,
                                size_t *width, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
