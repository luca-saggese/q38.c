#ifndef Q38_PLE_CACHE_H
#define Q38_PLE_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t lookup_count;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t insert_count;
    uint64_t eviction_count;
} q38_ple_cache_stats;

typedef struct {
    uint8_t *storage;
    uint64_t *keys;
    uint8_t *valid;
    size_t slot_count;
    size_t row_bytes;
    size_t next_victim;
    size_t budget_bytes;
    q38_ple_cache_stats stats;
} q38_ple_cache;

bool q38_ple_cache_init(q38_ple_cache *cache, size_t budget_bytes,
                        size_t row_bytes, char *error, size_t error_len);
void q38_ple_cache_destroy(q38_ple_cache *cache);
void q38_ple_cache_reset(q38_ple_cache *cache);

bool q38_ple_cache_lookup(q38_ple_cache *cache, uint64_t row,
                          void *row_data, char *error, size_t error_len);
bool q38_ple_cache_insert(q38_ple_cache *cache, uint64_t row,
                          const void *row_data, char *error, size_t error_len);

size_t q38_ple_cache_resident_bytes(const q38_ple_cache *cache);
const q38_ple_cache_stats *q38_ple_cache_get_stats(
    const q38_ple_cache *cache);

#ifdef __cplusplus
}
#endif

#endif
