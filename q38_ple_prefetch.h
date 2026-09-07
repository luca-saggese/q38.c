#ifndef Q38_PLE_PREFETCH_H
#define Q38_PLE_PREFETCH_H

#include "q38_ple.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enabled;
    size_t max_rows;
} q38_ple_prefetch_config;

typedef struct {
    size_t requested_rows;
    size_t prefetched_rows;
    uint64_t bytes_advised;
    bool supported;
} q38_ple_prefetch_stats;

typedef struct q38_ple_scheduler q38_ple_scheduler;

typedef enum {
    Q38_PLE_T0_TOKEN_FORWARD_BEGIN = 0,
    Q38_PLE_T2_LAYER0_BEGIN,
    Q38_PLE_T3_LAYER0_END,
    Q38_PLE_T4_LAYER1_BEGIN,
    Q38_PLE_T5_LAYER1_END,
    Q38_PLE_T11_TOKEN_FORWARD_END
} q38_ple_timeline_boundary;

typedef struct {
    uint64_t logical_accesses;
    uint64_t unique_rows;
    uint64_t unique_physical_blocks;
    uint64_t file_read_ops;
    uint64_t madvise_calls;
    uint64_t logical_bytes;
    uint64_t physical_bytes;
    uint64_t cache_hits;
    uint64_t cache_misses;
    double start_ms;
    double ready_ms;
    double consume_ms;
    double elapsed_ms;
    double overlap_ms;
    double wait_ms;
    double request_build_ms;
    double history_ngram_ms;
    double index_lookup_ms;
    double async_submit_ms;
    double file_io_ms;
    double worker_cpu_ms;
    double result_publish_ms;
    double decode_dequant_ms;
    double accumulation_ms;
    double injection_ms;
    double wait_at_injection_ms;
    uint64_t file_read_min_bytes;
    uint64_t file_read_max_bytes;
    bool file_reads_sequential;
    uint64_t request_id;
    uint64_t token_position;
    uint64_t submit_position;
    uint64_t injection_position;
    double t0_token_forward_begin_ms;
    double t1_ple_request_submit_ms;
    double t2_layer0_begin_ms;
    double t3_layer0_end_ms;
    double t4_layer1_begin_ms;
    double t5_layer1_end_ms;
    double t6_ple_injection_arrival_ms;
    double t7_ple_wait_begin_ms;
    double t8_ple_wait_end_ms;
    double t9_ple_injection_begin_ms;
    double t10_ple_injection_end_ms;
    double t11_token_forward_end_ms;
} q38_ple_scheduler_stats;

/*
 * Issue advisory page-cache hints for the next quantized rows. This never
 * allocates or copies table data. The default configuration is disabled
 * because the loader has no CUDA-visible staging queue yet.
 */
bool q38_ple_prefetch_rows(const q38_ple_store *store,
                           const uint64_t *rows, size_t row_count,
                           const q38_ple_prefetch_config *config,
                           q38_ple_prefetch_stats *stats,
                           char *error, size_t error_len);

/*
 * The scheduler retains only row IDs and physical ranges.  It never retains
 * decoded rows or a table-sized mirror.  A worker uses the existing mmap and
 * file descriptor to warm the page cache before the consumer reaches PLE.
 */
q38_ple_scheduler *q38_ple_scheduler_create(
    const q38_ple_store *store, char *error, size_t error_len);
void q38_ple_scheduler_destroy(q38_ple_scheduler *scheduler);
void q38_ple_scheduler_reset(q38_ple_scheduler *scheduler);
bool q38_ple_scheduler_submit(q38_ple_scheduler *scheduler,
                              const uint64_t *rows, size_t row_count,
                              char *error, size_t error_len);
bool q38_ple_scheduler_wait(q38_ple_scheduler *scheduler,
                            char *error, size_t error_len);
bool q38_ple_scheduler_get_stats(
    const q38_ple_scheduler *scheduler, q38_ple_scheduler_stats *stats);
bool q38_ple_scheduler_record_request_timing(
    q38_ple_scheduler *scheduler, double request_build_ms,
    double history_ngram_ms, double index_lookup_ms, double async_submit_ms);
bool q38_ple_scheduler_record_injection_timing(
    q38_ple_scheduler *scheduler, double decode_dequant_ms,
    double accumulation_ms, double injection_ms);
bool q38_ple_scheduler_record_injection_begin(
    q38_ple_scheduler *scheduler);
bool q38_ple_scheduler_record_timeline(
    q38_ple_scheduler *scheduler, q38_ple_timeline_boundary boundary,
    uint64_t request_id, uint64_t token_position);

#ifdef __cplusplus
}
#endif

#endif
