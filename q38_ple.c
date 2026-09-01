#include "q38_ple.h"

#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_ple_store_bind(const q38_tensor *tensor, uint32_t expected_row_width,
                        q38_ple_store *store, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!tensor || !store) return fail(error, error_len, "PLE tensor/store is null");
    if (tensor->ndim != 2 || tensor->dim[0] != expected_row_width ||
        tensor->dim[1] == 0 || tensor->bytes == 0) {
        return fail(error, error_len, "PLE table geometry is invalid");
    }
    memset(store, 0, sizeof(*store));
    store->tensor = tensor;
    store->qtype = tensor->type;
    store->rows = tensor->dim[1];
    store->row_width = expected_row_width;
    store->row_bytes = tensor->bytes % store->rows == 0
        ? tensor->bytes / store->rows : 0;
    store->file_offset = tensor->abs_offset;
    return true;
}

bool q38_ple_store_row_range(const q38_ple_store *store, uint64_t row,
                             uint64_t *offset, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!store || !store->tensor || !offset) {
        return fail(error, error_len, "PLE store/offset is null");
    }
    if (row >= store->rows) return fail(error, error_len, "PLE row is out of range");
    if (store->row_bytes == 0) {
        return fail(error, error_len, "PLE row is not byte-addressable");
    }
    if (row > UINT64_MAX / store->row_bytes ||
        store->file_offset > UINT64_MAX - row * store->row_bytes) {
        return fail(error, error_len, "PLE row offset overflows");
    }
    *offset = store->file_offset + row * store->row_bytes;
    return true;
}
