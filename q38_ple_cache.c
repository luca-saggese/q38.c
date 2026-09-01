#include "q38_ple_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

static size_t slot_offset(const q38_ple_cache *cache, size_t slot) {
    return slot * cache->row_bytes;
}

bool q38_ple_cache_init(q38_ple_cache *cache, size_t budget_bytes,
                        size_t row_bytes, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!cache) return fail(error, error_len, "PLE cache is null");
    if (budget_bytes == 0 || row_bytes == 0) {
        return fail(error, error_len, "PLE cache geometry is zero");
    }
    const size_t slot_count = budget_bytes / row_bytes;
    if (slot_count == 0) {
        return fail(error, error_len, "PLE cache budget is smaller than one row");
    }
    if (slot_count > SIZE_MAX / row_bytes) {
        return fail(error, error_len, "PLE cache storage size overflows");
    }

    memset(cache, 0, sizeof(*cache));
    cache->storage = (uint8_t *)malloc(slot_count * row_bytes);
    cache->keys = (uint64_t *)malloc(slot_count * sizeof(*cache->keys));
    cache->valid = (uint8_t *)calloc(slot_count, sizeof(*cache->valid));
    if (!cache->storage || !cache->keys || !cache->valid) {
        q38_ple_cache_destroy(cache);
        return fail(error, error_len, "PLE cache allocation failed");
    }
    cache->slot_count = slot_count;
    cache->row_bytes = row_bytes;
    cache->budget_bytes = slot_count * row_bytes;
    return true;
}

void q38_ple_cache_destroy(q38_ple_cache *cache) {
    if (!cache) return;
    free(cache->storage);
    free(cache->keys);
    free(cache->valid);
    memset(cache, 0, sizeof(*cache));
}

void q38_ple_cache_reset(q38_ple_cache *cache) {
    if (!cache || !cache->valid) return;
    memset(cache->valid, 0, cache->slot_count * sizeof(*cache->valid));
    cache->next_victim = 0;
    memset(&cache->stats, 0, sizeof(cache->stats));
}

bool q38_ple_cache_lookup(q38_ple_cache *cache, uint64_t row,
                          void *row_data, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!cache || !cache->storage || !cache->keys || !cache->valid ||
        !row_data || cache->slot_count == 0 || cache->row_bytes == 0) {
        return fail(error, error_len, "invalid PLE cache lookup arguments");
    }
    ++cache->stats.lookup_count;
    for (size_t slot = 0; slot < cache->slot_count; ++slot) {
        if (cache->valid[slot] && cache->keys[slot] == row) {
            memcpy(row_data, cache->storage + slot_offset(cache, slot),
                   cache->row_bytes);
            ++cache->stats.cache_hits;
            return true;
        }
    }
    ++cache->stats.cache_misses;
    return false;
}

bool q38_ple_cache_insert(q38_ple_cache *cache, uint64_t row,
                          const void *row_data, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!cache || !cache->storage || !cache->keys || !cache->valid ||
        !row_data || cache->slot_count == 0 || cache->row_bytes == 0) {
        return fail(error, error_len, "invalid PLE cache insert arguments");
    }

    size_t victim = cache->slot_count;
    for (size_t slot = 0; slot < cache->slot_count; ++slot) {
        if (cache->valid[slot] && cache->keys[slot] == row) {
            victim = slot;
            break;
        }
    }
    if (victim == cache->slot_count) {
        for (size_t step = 0; step < cache->slot_count; ++step) {
            const size_t slot =
                (cache->next_victim + step) % cache->slot_count;
            if (!cache->valid[slot]) {
                victim = slot;
                break;
            }
        }
        if (victim == cache->slot_count) {
            victim = cache->next_victim;
            cache->next_victim = (cache->next_victim + 1) % cache->slot_count;
            ++cache->stats.eviction_count;
        }
        ++cache->stats.insert_count;
    }
    cache->keys[victim] = row;
    cache->valid[victim] = 1;
    memcpy(cache->storage + slot_offset(cache, victim), row_data,
           cache->row_bytes);
    return true;
}

size_t q38_ple_cache_resident_bytes(const q38_ple_cache *cache) {
    return cache ? cache->budget_bytes : 0;
}

const q38_ple_cache_stats *q38_ple_cache_get_stats(
    const q38_ple_cache *cache) {
    return cache ? &cache->stats : NULL;
}
