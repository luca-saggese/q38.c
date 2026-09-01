#include "q38_ple.h"

#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

static bool tensor_name_is_shard(const q38_tensor *tensor,
                                 const char *prefix, uint32_t index) {
    char suffix[32];
    int suffix_len = snprintf(suffix, sizeof(suffix), "%u.weight", index);
    if (suffix_len < 0 || (size_t)suffix_len >= sizeof(suffix)) return false;
    size_t prefix_len = strlen(prefix);
    size_t total = prefix_len + (size_t)suffix_len;
    return tensor->name.len == total &&
           memcmp(tensor->name.ptr, prefix, prefix_len) == 0 &&
           memcmp(tensor->name.ptr + prefix_len, suffix, (size_t)suffix_len) == 0;
}

static const q38_tensor *find_shard(const q38_gguf *model,
                                    const char *prefix, uint32_t index) {
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        if (tensor_name_is_shard(&model->tensors[i], prefix, index))
            return &model->tensors[i];
    }
    return NULL;
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

bool q38_ple_store_bind_gguf(const q38_gguf *model, const char *shard_prefix,
                             uint32_t shard_count, uint32_t row_width,
                             q38_ple_store *store, char *error,
                             size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!model || !model->map || !shard_prefix || !shard_prefix[0] ||
        !store || shard_count == 0 || shard_count > Q38_PLE_MAX_SHARDS ||
        row_width == 0) {
        return fail(error, error_len, "invalid GGUF PLE binding arguments");
    }

    q38_ple_store bound;
    memset(&bound, 0, sizeof(bound));
    bound.model = model;
    bound.row_width = row_width;
    bound.shard_count = shard_count;
    for (uint32_t i = 0; i < shard_count; ++i) {
        const q38_tensor *tensor = find_shard(model, shard_prefix, i);
        if (!tensor) {
            char message[256];
            snprintf(message, sizeof(message), "missing PLE shard %u", i);
            return fail(error, error_len, message);
        }
        if (tensor->ndim != 2 || tensor->dim[1] != row_width ||
            tensor->dim[0] == 0 || tensor->bytes == 0 ||
            !q38_gguf_type_nbytes(tensor->type, row_width,
                                   &bound.row_bytes)) {
            char message[256];
            snprintf(message, sizeof(message), "invalid PLE shard geometry %u", i);
            return fail(error, error_len, message);
        }
        if (tensor->dim[0] > UINT64_MAX / bound.row_bytes ||
            tensor->dim[0] * bound.row_bytes != tensor->bytes) {
            char message[256];
            snprintf(message, sizeof(message), "PLE shard byte size mismatch %u", i);
            return fail(error, error_len, message);
        }
        if (i > 0 && tensor->type != bound.qtype) {
            return fail(error, error_len, "PLE shard quantization types differ");
        }
        bound.shard[i] = tensor;
        bound.shard_rows[i] = tensor->dim[0];
        bound.shard_first_row[i] = bound.rows;
        if (bound.rows > UINT64_MAX - tensor->dim[0])
            return fail(error, error_len, "PLE row count overflows");
        bound.rows += tensor->dim[0];
        bound.qtype = tensor->type;
    }
    if (bound.rows == 0 || bound.row_bytes == 0) {
        return fail(error, error_len, "PLE GGUF table is empty");
    }
    *store = bound;
    return true;
}

bool q38_ple_store_row_range(const q38_ple_store *store, uint64_t row,
                             uint64_t *offset, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!store || !offset ||
        (!store->tensor && store->shard_count == 0)) {
        return fail(error, error_len, "PLE store/offset is null");
    }
    if (row >= store->rows) return fail(error, error_len, "PLE row is out of range");
    if (store->row_bytes == 0) {
        return fail(error, error_len, "PLE row is not byte-addressable");
    }
    if (store->shard_count != 0) {
        for (uint32_t i = 0; i < store->shard_count; ++i) {
            const uint64_t first = store->shard_first_row[i];
            if (row < first || row - first >= store->shard_rows[i]) continue;
            const q38_tensor *shard = store->shard[i];
            uint64_t local = row - first;
            if (local > UINT64_MAX / store->row_bytes ||
                shard->abs_offset > UINT64_MAX - local * store->row_bytes) {
                return fail(error, error_len, "PLE row offset overflows");
            }
            *offset = shard->abs_offset + local * store->row_bytes;
            return true;
        }
        return fail(error, error_len, "PLE row has no shard");
    }
    if (row > UINT64_MAX / store->row_bytes ||
        store->file_offset > UINT64_MAX - row * store->row_bytes) {
        return fail(error, error_len, "PLE row offset overflows");
    }
    *offset = store->file_offset + row * store->row_bytes;
    return true;
}

bool q38_ple_store_read_row(const q38_ple_store *store, uint64_t row,
                            void *row_data, size_t row_data_bytes,
                            char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!store || !store->model || !store->model->map || !row_data ||
        row_data_bytes < store->row_bytes) {
        return fail(error, error_len, "invalid PLE row read arguments");
    }
    uint64_t offset;
    if (!q38_ple_store_row_range(store, row, &offset, error, error_len))
        return false;
    if (offset > store->model->size ||
        store->row_bytes > store->model->size - offset) {
        return fail(error, error_len, "PLE row exceeds mapped GGUF");
    }
    memcpy(row_data, store->model->map + offset, (size_t)store->row_bytes);
    return true;
}

bool q38_ple_store_read_rows(const q38_ple_store *store, const uint64_t *rows,
                             size_t row_count, void *row_data,
                             size_t row_stride, char *error,
                             size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!store || !rows || row_count == 0 || !row_data ||
        row_stride < store->row_bytes) {
        return fail(error, error_len, "invalid PLE row batch arguments");
    }
    if (row_count > SIZE_MAX / row_stride) {
        return fail(error, error_len, "PLE row batch size overflows");
    }
    for (size_t i = 0; i < row_count; ++i) {
        if (!q38_ple_store_read_row(
                store, rows[i], (uint8_t *)row_data + i * row_stride,
                (size_t)store->row_bytes, error, error_len))
            return false;
    }
    return true;
}
