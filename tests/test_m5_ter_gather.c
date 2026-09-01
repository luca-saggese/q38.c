#include "q38_ple.h"

#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed(const struct timespec *a, const struct timespec *b) {
    return (double)(b->tv_sec - a->tv_sec) * 1e6 +
           (double)(b->tv_nsec - a->tv_nsec) / 1e3;
}

int main(int argc, char **argv) {
    const size_t row_bytes = 160 * 2;
    const size_t rows = 8192;
    uint8_t *payload = malloc(rows * row_bytes);
    uint8_t *direct = malloc(4096 * row_bytes);
    uint8_t *parallel = malloc(4096 * row_bytes);
    uint64_t *ids = malloc(4096 * sizeof(*ids));
    if (!payload || !direct || !parallel || !ids) return 1;
    for (size_t i = 0; i < rows * row_bytes; ++i) payload[i] = (uint8_t)i;
    q38_gguf model = {.map = payload, .size = rows * row_bytes};
    q38_tensor descriptor = {0};
    q38_ple_store store = {.tensor = &descriptor, .model = &model,
                           .rows = rows, .row_bytes = row_bytes,
                           .file_offset = 0};
    char error[128];
    FILE *out = argc > 1 ? fopen(argv[1], "w") : NULL;
    if (!out) return 2;
    fprintf(out, "{\"cases\":[");
    const size_t cases[] = {1,2,4,8,16,32,64,128,256,384,512,768,1024,2048,4096};
    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); ++c) {
        size_t count = cases[c];
        for (size_t i = 0; i < count; ++i) ids[i] = (i * 17) % rows;
        if (count > 8) for (size_t i = 0; i < count / 4; ++i) ids[i] = ids[0];
        struct timespec a, b;
        q38_ple_gather_stats ds, ps;
        clock_gettime(CLOCK_MONOTONIC, &a);
        if (!q38_ple_store_read_rows_mode(&store, ids, count, direct, row_bytes,
                                          Q38_PLE_GATHER_DIRECT, &ds, error,
                                          sizeof(error))) return 1;
        clock_gettime(CLOCK_MONOTONIC, &b);
        double direct_us = elapsed(&a, &b);
        clock_gettime(CLOCK_MONOTONIC, &a);
        if (!q38_ple_store_read_rows_mode(&store, ids, count, parallel, row_bytes,
                                          Q38_PLE_GATHER_PARALLEL, &ps, error,
                                          sizeof(error))) return 1;
        clock_gettime(CLOCK_MONOTONIC, &b);
        double parallel_us = elapsed(&a, &b);
        if (memcmp(direct, parallel, count * row_bytes) != 0) return 1;
        fprintf(out, "%s{\"unique_rows\":%zu,\"direct_us\":%.3f,\"parallel_us\":%.3f,\"bytes\":%zu,\"deduplicated\":%" PRIu64 "}",
                c ? "," : "", count, direct_us, parallel_us, count * row_bytes,
                ps.deduplicated_rows);
    }
    fprintf(out, "],\"status\":\"pass\",\"threshold_policy\":\"measured; default 512 only when no tuning is supplied\"}\n");
    fclose(out);
    free(payload); free(direct); free(parallel); free(ids);
    puts("test_m5_ter_gather: direct/parallel row bytes are equivalent");
    return 0;
}
