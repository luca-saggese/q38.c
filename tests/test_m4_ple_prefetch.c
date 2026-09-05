#include "q38_ple_prefetch.h"

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const size_t size = page * 2;
    uint8_t *mapping = mmap(NULL, size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        return 1;
    }
    q38_gguf model = {.fd = -1, .map = mapping, .size = size};
    q38_tensor tensor = {.abs_offset = 0, .bytes = size};
    q38_ple_store store = {
        .tensor = &tensor,
        .model = &model,
        .rows = 2,
        .row_width = 1,
        .row_bytes = page,
        .file_offset = 0,
    };
    uint64_t rows[] = {0, 1};
    q38_ple_prefetch_stats stats;
    q38_ple_prefetch_config disabled = {.enabled = false, .max_rows = 2};
    char error[128];
    if (!q38_ple_prefetch_rows(&store, rows, 2, &disabled, &stats,
                               error, sizeof(error)) ||
        stats.requested_rows != 2 || stats.prefetched_rows != 0 ||
        stats.bytes_advised != 0 || stats.supported) {
        fprintf(stderr, "disabled prefetch changed state: %s\n", error);
        munmap(mapping, size);
        return 1;
    }
    q38_ple_prefetch_config enabled = {.enabled = true, .max_rows = 2};
    if (!q38_ple_prefetch_rows(&store, rows, 2, &enabled, &stats,
                               error, sizeof(error)) ||
        stats.requested_rows != 2 ||
        (stats.prefetched_rows != 2 && !strstr(error, "unsupported"))) {
        fprintf(stderr, "enabled prefetch failed: %s\n", error);
        munmap(mapping, size);
        return 1;
    }
    q38_ple_scheduler *scheduler =
        q38_ple_scheduler_create(&store, error, sizeof(error));
    if (!scheduler) {
        fprintf(stderr, "scheduler creation failed: %s\n", error);
        munmap(mapping, size);
        return 1;
    }
    uint64_t duplicate_rows[] = {1, 0, 1};
    if (!q38_ple_scheduler_submit(scheduler, duplicate_rows, 3, error,
                                  sizeof(error)) ||
        !q38_ple_scheduler_wait(scheduler, error, sizeof(error))) {
        fprintf(stderr, "scheduler execution failed: %s\n", error);
        q38_ple_scheduler_destroy(scheduler);
        munmap(mapping, size);
        return 1;
    }
    q38_ple_scheduler_stats scheduler_stats;
    if (!q38_ple_scheduler_get_stats(scheduler, &scheduler_stats) ||
        scheduler_stats.logical_accesses != 3 ||
        scheduler_stats.unique_rows != 2 ||
        scheduler_stats.unique_physical_blocks != 1 ||
        scheduler_stats.logical_bytes != 3 * page ||
        scheduler_stats.file_read_ops != 0 ||
        scheduler_stats.physical_bytes != 0) {
        fprintf(stderr, "scheduler telemetry/coalescing mismatch\n");
        q38_ple_scheduler_destroy(scheduler);
        munmap(mapping, size);
        return 1;
    }
    q38_ple_scheduler_destroy(scheduler);
    munmap(mapping, size);
    puts("test_m4_ple_prefetch: advisory and async scheduler prefetch passed");
    return 0;
}
