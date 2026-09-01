#include "q38_gguf.h"
#include "q38_memory.h"
#include "q38_ple.h"
#include "q38_ple_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *data;
    size_t size;
    size_t capacity;
} builder;

static void put_u32(unsigned char *p, uint32_t value) {
    memcpy(p, &value, sizeof(value));
}

static void put_u64(unsigned char *p, uint64_t value) {
    memcpy(p, &value, sizeof(value));
}

static void reserve(builder *b, size_t extra) {
    if (extra > SIZE_MAX - b->size) exit(2);
    size_t needed = b->size + extra;
    if (needed <= b->capacity) return;
    size_t capacity = b->capacity ? b->capacity : 256;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) exit(2);
        capacity *= 2;
    }
    b->data = (unsigned char *)realloc(b->data, capacity);
    if (!b->data) exit(2);
    b->capacity = capacity;
}

static void bytes(builder *b, const void *data, size_t size) {
    reserve(b, size);
    memcpy(b->data + b->size, data, size);
    b->size += size;
}

static void u32(builder *b, uint32_t value) {
    unsigned char data[4];
    put_u32(data, value);
    bytes(b, data, sizeof(data));
}

static void u64(builder *b, uint64_t value) {
    unsigned char data[8];
    put_u64(data, value);
    bytes(b, data, sizeof(data));
}

static void string(builder *b, const char *value) {
    size_t length = strlen(value);
    u64(b, length);
    bytes(b, value, length);
}

static void align32(builder *b) {
    unsigned char zero[32] = {0};
    size_t padding = (32 - (b->size % 32)) % 32;
    bytes(b, zero, padding);
}

static void build_fixture(builder *b) {
    static const char *prefix =
        "ple.ple_embedding.ngram_embedding.shard_";
    enum { SHARDS = 3, ROWS_PER_SHARD = 2, ROW_BYTES = 170 };
    u32(b, Q38_GGUF_MAGIC);
    u32(b, 3);
    u64(b, SHARDS);
    u64(b, 0);
    for (unsigned shard = 0; shard < SHARDS; ++shard) {
        char name[128];
        snprintf(name, sizeof(name), "%s%u.weight", prefix, shard);
        string(b, name);
        u32(b, 2);                 /* ndim */
        u64(b, ROWS_PER_SHARD);   /* rows */
        u64(b, 160);               /* row width */
        u32(b, 8);                 /* GGML_TYPE_Q8_0 */
        u64(b, shard * ROWS_PER_SHARD * ROW_BYTES);
    }
    align32(b);
    for (unsigned shard = 0; shard < SHARDS; ++shard) {
        for (unsigned row = 0; row < ROWS_PER_SHARD; ++row) {
            unsigned char value = (unsigned char)(0x20 + shard * 8 + row);
            unsigned char row_data[ROW_BYTES];
            memset(row_data, value, sizeof(row_data));
            bytes(b, row_data, sizeof(row_data));
        }
    }
}

static int check_row(const unsigned char *row, size_t size,
                     unsigned char value) {
    for (size_t i = 0; i < size; ++i) {
        if (row[i] != value) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *path = "artifacts/m4/ple_loader_fixture.gguf";
    const char *report = argc > 1 ? argv[1] : NULL;
    builder fixture = {0};
    build_fixture(&fixture);
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(fixture.data, 1, fixture.size, file) != fixture.size) {
        fprintf(stderr, "cannot write GGUF fixture\n");
        if (file) fclose(file);
        free(fixture.data);
        return 1;
    }
    fclose(file);

    char error[256];
    q38_gguf *model = q38_gguf_open(path, error, sizeof(error));
    if (!model) {
        fprintf(stderr, "GGUF open failed: %s\n", error);
        remove(path);
        free(fixture.data);
        return 1;
    }
    q38_ple_store store;
    if (!q38_ple_store_bind_gguf(
            model, "ple.ple_embedding.ngram_embedding.shard_", 3, 160,
            &store, error, sizeof(error)) ||
        store.rows != 6 || store.row_width != 160 || store.row_bytes != 170 ||
        store.qtype != 8 || store.shard_count != 3) {
        fprintf(stderr, "GGUF PLE bind failed: %s\n", error);
        q38_gguf_close(model);
        remove(path);
        free(fixture.data);
        return 1;
    }

    q38_memory_tracker tracker;
    q38_memory_tracker_init(&tracker);
    q38_memory_snapshot before, after;
    q38_memory_capture(&tracker, "ple_mmap_bound", model->size,
                       model->size, 0, &before);
    const uint64_t full_table_bytes = store.rows * store.row_bytes;
    const uint64_t bounded_cache_bytes = store.row_bytes * 2;
    if (bounded_cache_bytes >= full_table_bytes) {
        fprintf(stderr, "residency fixture does not exercise a bounded cache\n");
        q38_gguf_close(model);
        remove(path);
        free(fixture.data);
        return 1;
    }
    q38_ple_cache cache;
    if (!q38_ple_cache_init(&cache, (size_t)bounded_cache_bytes,
                            (size_t)store.row_bytes, error, sizeof(error))) {
        fprintf(stderr, "bounded PLE cache init failed: %s\n", error);
        q38_gguf_close(model);
        remove(path);
        free(fixture.data);
        return 1;
    }

    uint64_t ids[] = {5, 0, 3, 5};
    unsigned char output[4][192];
    memset(output, 0xa5, sizeof(output));
    if (!q38_ple_store_read_rows(&store, ids, 4, output, sizeof(output[0]),
                                 error, sizeof(error))) {
        fprintf(stderr, "PLE row batch read failed: %s\n", error);
        q38_ple_cache_destroy(&cache);
        q38_gguf_close(model);
        remove(path);
        free(fixture.data);
        return 1;
    }
    unsigned char cached[170];
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
        if (!q38_ple_cache_lookup(&cache, ids[i], cached, error,
                                  sizeof(error))) {
            if (!q38_ple_store_read_row(&store, ids[i], cached, sizeof(cached),
                                        error, sizeof(error)) ||
                !q38_ple_cache_insert(&cache, ids[i], cached, error,
                                      sizeof(error))) {
                fprintf(stderr, "bounded PLE cache fill failed: %s\n", error);
                q38_ple_cache_destroy(&cache);
                q38_gguf_close(model);
                remove(path);
                free(fixture.data);
                return 1;
            }
        }
    }
    q38_memory_capture(&tracker, "ple_rows_read", model->size, model->size,
                       q38_ple_cache_resident_bytes(&cache), &after);
    if (!check_row(output[0], 170, 0x31) ||
        !check_row(output[1], 170, 0x20) ||
        !check_row(output[2], 170, 0x29) ||
        !check_row(output[3], 170, 0x31) ||
        output[0][191] != 0xa5 || output[1][191] != 0xa5) {
        fprintf(stderr, "PLE rows were not read in logical ID order\n");
        q38_ple_cache_destroy(&cache);
        q38_gguf_close(model);
        remove(path);
        free(fixture.data);
        return 1;
    }
    unsigned char single[170];
    if (!q38_ple_store_read_row(&store, 4, single, sizeof(single),
                                error, sizeof(error)) ||
        !check_row(single, sizeof(single), 0x30) ||
        q38_ple_store_read_row(&store, 6, single, sizeof(single), error,
                               sizeof(error))) {
        fprintf(stderr, "PLE single-row range handling failed: %s\n", error);
        q38_ple_cache_destroy(&cache);
        q38_gguf_close(model);
        remove(path);
        free(fixture.data);
        return 1;
    }
    if (q38_ple_cache_resident_bytes(&cache) > (size_t)bounded_cache_bytes) {
        fprintf(stderr, "bounded PLE cache exceeded configured budget\n");
        q38_ple_cache_destroy(&cache);
        remove(path);
        free(fixture.data);
        return 1;
    }

    const q38_ple_cache_stats *cache_stats = q38_ple_cache_get_stats(&cache);
    q38_gguf_close(model);
    if (report) {
        FILE *output_file = fopen(report, "w");
        if (!output_file) {
            q38_ple_cache_destroy(&cache);
            remove(path);
            free(fixture.data);
            return 1;
        }
        fprintf(output_file,
                "{\"gate\":\"M4-C10\",\"backend\":\"GGUF mmap\","
                "\"model_file_bytes\":%llu,\"model_mapped_bytes\":%llu,"
                "\"full_table_bytes\":%llu,\"bounded_cache_bytes\":%llu,"
                "\"rows_requested\":4,\"row_bytes\":%llu,"
                "\"cache_hits\":%llu,\"cache_misses\":%llu,"
                "\"cache_resident_bytes\":%zu,"
                "\"before_rss_bytes\":%llu,\"after_rss_bytes\":%llu,"
                "\"cuda_staging\":\"not measured; full forward runtime absent\","
                "\"status\":\"pass\"}\n",
                (unsigned long long)before.model_file_bytes,
                (unsigned long long)before.model_mapped_bytes,
                (unsigned long long)full_table_bytes,
                (unsigned long long)bounded_cache_bytes,
                (unsigned long long)store.row_bytes,
                (unsigned long long)cache_stats->cache_hits,
                (unsigned long long)cache_stats->cache_misses,
                q38_ple_cache_resident_bytes(&cache),
                (unsigned long long)before.rss_bytes,
                (unsigned long long)after.rss_bytes);
        fclose(output_file);
    }
    q38_ple_cache_destroy(&cache);
    remove(path);
    free(fixture.data);
    puts("test_m4_ple_loader: GGUF shard binding and file-backed row reads passed");
    return 0;
}
