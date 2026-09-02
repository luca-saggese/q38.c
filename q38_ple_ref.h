#ifndef Q38_PLE_REF_H
#define Q38_PLE_REF_H

#include "q38_session.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_PLE_MAX_NGRAM 3u
#define Q38_PLE_MAX_HEADS 16u

typedef struct {
    uint32_t ngram_size;
    uint32_t heads_per_ngram;
    uint64_t multipliers[Q38_PLE_MAX_NGRAM];
    uint32_t head_offsets[Q38_PLE_MAX_HEADS];
    uint32_t head_vocab_sizes[Q38_PLE_MAX_HEADS];
} q38_ple_hash_config;

bool q38_ple_hash_config_validate(const q38_ple_hash_config *config,
                                  char *error, size_t error_len);

bool q38_ple_ngram_ids_ref(const q38_ple_hash_config *config,
                           const q38_ngram_history *history,
                           uint32_t current_token, uint32_t eos_token,
                           uint32_t *ids, size_t ids_count,
                           char *error, size_t error_len);

bool q38_ple_decode_row_ref(uint32_t qtype, const void *blocks,
                            uint32_t row_width, float *out,
                            size_t out_elements, char *error,
                            size_t error_len);

typedef struct {
    size_t hidden;
    size_t streams;
    size_t heads;
    size_t row_width;
    size_t kernel;
    size_t dilation;
    float eps;
} q38_ple_forward_config;

void q38_ple_grouped_norm_inplace(float *values, const float *weight,
                                   size_t tokens, size_t streams,
                                   size_t hidden, float eps);

/*
 * Reference PLE injection. All matrices are output-by-input, rows are
 * token-major, and `history` contains the normalized convolution tail from
 * the preceding call. The history is updated in place.
 */
bool q38_ple_forward_ref(const q38_ple_forward_config *config,
                         const float *hidden, size_t token_count,
                         const float *embedding, const float *key_proj,
                         const float *value_proj, const float *norm_key,
                         const float *norm_query, const float *norm_conv,
                         const float *conv, float *history, float *contribution,
                         float *after, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
