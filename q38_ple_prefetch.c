#include "q38_ple_prefetch.h"

#include <errno.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_ple_prefetch_rows(const q38_ple_store *store,
                           const uint64_t *rows, size_t row_count,
                           const q38_ple_prefetch_config *config,
                           q38_ple_prefetch_stats *stats,
                           char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (stats) {
        stats->requested_rows = 0;
        stats->prefetched_rows = 0;
        stats->bytes_advised = 0;
        stats->supported = false;
    }
    if (!store || !rows || row_count == 0 || !config) {
        return fail(error, error_len, "invalid PLE prefetch arguments");
    }
    if (config->max_rows != 0 && row_count > config->max_rows) {
        return fail(error, error_len, "PLE prefetch row count exceeds limit");
    }
    if (stats) stats->requested_rows = row_count;
    if (!config->enabled) return true;
    if (!store->model || !store->model->map || store->model->size == 0 ||
        store->row_bytes == 0) {
        return fail(error, error_len, "PLE prefetch requires a mapped store");
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return fail(error, error_len, "invalid system page size");
    for (size_t i = 0; i < row_count; ++i) {
        uint64_t offset = 0;
        if (!q38_ple_store_row_range(store, rows[i], &offset, error, error_len))
            return false;
        if (offset > store->model->size ||
            store->row_bytes > store->model->size - offset) {
            return fail(error, error_len, "PLE prefetch row exceeds mapping");
        }
        const uint64_t page = offset - offset % (uint64_t)page_size;
        const uint64_t end = offset + store->row_bytes;
        const uint64_t advised_end =
            (end + (uint64_t)page_size - 1) /
            (uint64_t)page_size * (uint64_t)page_size;
        const uint64_t length = advised_end - page;
        if (page > (uint64_t)store->model->size ||
            length > (uint64_t)store->model->size - page) {
            return fail(error, error_len, "PLE prefetch page exceeds mapping");
        }
        if (madvise((void *)(store->model->map + page), (size_t)length,
                    MADV_WILLNEED) != 0) {
            if (errno == ENOSYS || errno == EINVAL) {
                return fail(error, error_len, "mapped-row prefetch is unsupported");
            }
            return fail(error, error_len, "mapped-row prefetch failed");
        }
        if (stats) {
            ++stats->prefetched_rows;
            stats->bytes_advised += length;
            stats->supported = true;
        }
    }
    return true;
}
