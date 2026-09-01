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

#ifdef __cplusplus
}
#endif

#endif
