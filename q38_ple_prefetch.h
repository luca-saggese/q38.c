#ifndef Q38_PLE_PREFETCH_H
#define Q38_PLE_PREFETCH_H

#include "q38_ple.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enabled;
    size_t max_rows;
} q38_ple_prefetch_config;

typedef struct {
    size_t requested_rows;
    size_t prefetched_rows;
    uint64_t bytes_advised;
    bool supported;
} q38_ple_prefetch_stats;

/*
 * Issue advisory page-cache hints for the next quantized rows. This never
 * allocates or copies table data. The default configuration is disabled
 * because the loader has no CUDA-visible staging queue yet.
 */
bool q38_ple_prefetch_rows(const q38_ple_store *store,
                           const uint64_t *rows, size_t row_count,
                           const q38_ple_prefetch_config *config,
                           q38_ple_prefetch_stats *stats,
                           char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
