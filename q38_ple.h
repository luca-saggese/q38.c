#ifndef Q38_PLE_H
#define Q38_PLE_H

#include "q38_gguf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const q38_tensor *tensor;
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

#ifdef __cplusplus
}
#endif

#endif
