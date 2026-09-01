#ifndef Q38_PLE_H
#define Q38_PLE_H

#include "q38_gguf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_PLE_MAX_SHARDS 128u

typedef struct {
    const q38_tensor *tensor;
    const q38_gguf *model;
    const q38_tensor *shard[Q38_PLE_MAX_SHARDS];
    uint64_t shard_first_row[Q38_PLE_MAX_SHARDS];
    uint64_t shard_rows[Q38_PLE_MAX_SHARDS];
    uint32_t shard_count;
    uint32_t qtype;
    uint64_t rows;
    uint32_t row_width;
    uint64_t row_bytes;
    uint64_t file_offset;
} q38_ple_store;

bool q38_ple_store_bind(const q38_tensor *tensor, uint32_t expected_row_width,
                        q38_ple_store *store, char *error, size_t error_len);

bool q38_ple_store_row_range(const q38_ple_store *store, uint64_t row,
                             uint64_t *offset, char *error, size_t error_len);

/* Bind the verified GGUF PLE shard layout used by the local Qwen checkpoint.
 * Shards are named prefix + decimal index + ".weight" and store rows as
 * [row, row_width]. The table remains in the GGUF mapping; only descriptors
 * are retained here. */
bool q38_ple_store_bind_gguf(const q38_gguf *model, const char *shard_prefix,
                             uint32_t shard_count, uint32_t row_width,
                             q38_ple_store *store, char *error,
                             size_t error_len);

/* Copy one or more quantized rows into caller-owned storage. IDs are logical
 * row IDs and output order is preserved. No table-sized allocation occurs. */
bool q38_ple_store_read_row(const q38_ple_store *store, uint64_t row,
                            void *row_data, size_t row_data_bytes,
                            char *error, size_t error_len);
bool q38_ple_store_read_rows(const q38_ple_store *store, const uint64_t *rows,
                             size_t row_count, void *row_data,
                             size_t row_stride, char *error,
                             size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
