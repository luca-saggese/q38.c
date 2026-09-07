#include "q38_ple_prefetch.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

typedef struct {
    uint64_t offset;
    uint64_t bytes;
} q38_ple_block;

struct q38_ple_scheduler {
    q38_ple_store store;
    pthread_mutex_t mutex;
    pthread_cond_t work;
    pthread_cond_t done;
    pthread_t worker;
    bool worker_started;
    bool stop;
    bool pending;
    bool ready;
    uint64_t *rows;
    size_t row_count;
    q38_ple_block *blocks;
    size_t block_count;
    q38_ple_scheduler_stats stats;
    double request_build_ms;
    double history_ngram_ms;
    double index_lookup_ms;
    double async_submit_ms;
    double decode_dequant_ms;
    double accumulation_ms;
    double injection_ms;
    uint64_t next_request_id;
};

static double scheduler_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static double scheduler_thread_cpu_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static int compare_block(const void *left, const void *right) {
    const q38_ple_block *a = (const q38_ple_block *)left;
    const q38_ple_block *b = (const q38_ple_block *)right;
    return a->offset < b->offset ? -1 : a->offset > b->offset ? 1 : 0;
}

static void scheduler_free_job(q38_ple_scheduler *scheduler) {
    free(scheduler->rows);
    free(scheduler->blocks);
    scheduler->rows = NULL;
    scheduler->blocks = NULL;
    scheduler->row_count = 0;
    scheduler->block_count = 0;
}

static bool range_pages(const q38_ple_scheduler *scheduler,
                        uint64_t offset, uint64_t bytes,
                        uint64_t *page, size_t *length, size_t *pages) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || offset > scheduler->store.model->size ||
        bytes > scheduler->store.model->size - offset)
        return false;
    const uint64_t psize = (uint64_t)page_size;
    const uint64_t first = offset - offset % psize;
    const uint64_t end = offset + bytes;
    const uint64_t rounded = (end + psize - 1) / psize * psize;
    if (rounded < first || rounded - first > SIZE_MAX) return false;
    *page = first;
    *length = (size_t)(rounded - first);
    *pages = *length / (size_t)page_size;
    return true;
}

static bool block_resident(const q38_ple_scheduler *scheduler,
                           const q38_ple_block *block) {
    if (!scheduler->store.model || !scheduler->store.model->map) return false;
    uint64_t page = 0;
    size_t length = 0, pages = 0;
    if (!range_pages(scheduler, block->offset, block->bytes, &page,
                     &length, &pages) || pages == 0)
        return false;
    unsigned char *resident = (unsigned char *)calloc(pages, 1);
    if (!resident) return false;
    const int result = mincore((void *)(scheduler->store.model->map + page),
                               length, resident);
    bool hit = result == 0;
    if (hit)
        for (size_t i = 0; i < pages; ++i)
            if (!(resident[i] & 1u)) {
                hit = false;
                break;
            }
    free(resident);
    return hit;
}

static void warm_block(q38_ple_scheduler *scheduler,
                       const q38_ple_block *block,
                       q38_ple_scheduler_stats *stats) {
    if (!scheduler->store.model || !scheduler->store.model->map) return;
    uint64_t page = 0;
    size_t length = 0, pages = 0;
    if (!range_pages(scheduler, block->offset, block->bytes, &page,
                     &length, &pages))
        return;
    if (block_resident(scheduler, block)) {
        ++stats->cache_hits;
        return;
    }
    ++stats->cache_misses;
    (void)madvise((void *)(scheduler->store.model->map + page), length,
                  MADV_WILLNEED);

    /*
     * pread is intentionally chunked into a small transient buffer.  This
     * forces file-backed pages in kernels which ignore MADV_WILLNEED without
     * creating a resident PLE copy.
     */
    const size_t chunk_size = 64u * 1024u;
    uint8_t *buffer = (uint8_t *)malloc(chunk_size);
    bool read_complete = false;
    const double io_started = scheduler_now_ms();
    if (buffer && scheduler->store.model->fd >= 0) {
        uint64_t done = 0;
        while (done < block->bytes) {
            const size_t want = (size_t)((block->bytes - done) > chunk_size
                ? chunk_size : block->bytes - done);
            const ssize_t got = pread(scheduler->store.model->fd, buffer, want,
                                      (off_t)(block->offset + done));
            if (got <= 0) break;
            ++stats->file_read_ops;
            stats->physical_bytes += (uint64_t)got;
            if (stats->file_read_min_bytes == 0 ||
                (uint64_t)got < stats->file_read_min_bytes)
                stats->file_read_min_bytes = (uint64_t)got;
            if ((uint64_t)got > stats->file_read_max_bytes)
                stats->file_read_max_bytes = (uint64_t)got;
            done += (uint64_t)got;
            if ((size_t)got != want) break;
        }
        read_complete = done == block->bytes;
    }
    stats->file_io_ms += scheduler_now_ms() - io_started;
    if (!read_complete) {
        volatile uint8_t touch = 0;
        for (size_t i = 0; i < pages; ++i)
            touch ^= scheduler->store.model->map[page +
                         i * (size_t)sysconf(_SC_PAGESIZE)];
        (void)touch;
    }
    free(buffer);
}

static void *scheduler_worker(void *arg) {
    q38_ple_scheduler *scheduler = (q38_ple_scheduler *)arg;
    pthread_mutex_lock(&scheduler->mutex);
    for (;;) {
        while (!scheduler->pending && !scheduler->stop)
            pthread_cond_wait(&scheduler->work, &scheduler->mutex);
        if (scheduler->stop) break;
        scheduler->ready = false;
        q38_ple_scheduler_stats stats = scheduler->stats;
        stats.request_build_ms = scheduler->request_build_ms;
        stats.history_ngram_ms = scheduler->history_ngram_ms;
        stats.index_lookup_ms = scheduler->index_lookup_ms;
        stats.async_submit_ms = scheduler->async_submit_ms;
        stats.decode_dequant_ms = scheduler->decode_dequant_ms;
        stats.accumulation_ms = scheduler->accumulation_ms;
        stats.injection_ms = scheduler->injection_ms;
        stats.start_ms = scheduler_now_ms();
        const double cpu_start = scheduler_thread_cpu_ms();
        q38_ple_block *blocks = scheduler->blocks;
        const size_t block_count = scheduler->block_count;
        pthread_mutex_unlock(&scheduler->mutex);

        for (size_t i = 0; i < block_count; ++i)
            warm_block(scheduler, &blocks[i], &stats);
        stats.ready_ms = scheduler_now_ms();
        stats.t6_ple_injection_arrival_ms = stats.ready_ms;
        stats.elapsed_ms = stats.ready_ms - stats.start_ms;
        stats.worker_cpu_ms = scheduler_thread_cpu_ms() - cpu_start;
        stats.result_publish_ms = 0.0;

        pthread_mutex_lock(&scheduler->mutex);
        const double publish_start = scheduler_now_ms();
        scheduler->stats = stats;
        scheduler->pending = false;
        scheduler->ready = true;
        pthread_cond_broadcast(&scheduler->done);
        scheduler->stats.result_publish_ms = scheduler_now_ms() -
                                             publish_start;
    }
    pthread_mutex_unlock(&scheduler->mutex);
    return NULL;
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

q38_ple_scheduler *q38_ple_scheduler_create(
    const q38_ple_store *store, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!store || !store->model || !store->model->map ||
        !store->row_bytes || !store->rows)
        return NULL;
    q38_ple_scheduler *scheduler =
        (q38_ple_scheduler *)calloc(1, sizeof(*scheduler));
    if (!scheduler) return NULL;
    scheduler->store = *store;
    if (pthread_mutex_init(&scheduler->mutex, NULL) != 0) {
        free(scheduler);
        return NULL;
    }
    if (pthread_cond_init(&scheduler->work, NULL) != 0) {
        pthread_mutex_destroy(&scheduler->mutex);
        free(scheduler);
        return NULL;
    }
    if (pthread_cond_init(&scheduler->done, NULL) != 0) {
        pthread_cond_destroy(&scheduler->work);
        pthread_mutex_destroy(&scheduler->mutex);
        free(scheduler);
        return NULL;
    }
    if (pthread_create(&scheduler->worker, NULL, scheduler_worker, scheduler) != 0) {
        pthread_cond_destroy(&scheduler->done);
        pthread_cond_destroy(&scheduler->work);
        pthread_mutex_destroy(&scheduler->mutex);
        free(scheduler);
        return NULL;
    }
    scheduler->worker_started = true;
    return scheduler;
}

void q38_ple_scheduler_destroy(q38_ple_scheduler *scheduler) {
    if (!scheduler) return;
    if (scheduler->worker_started) {
        pthread_mutex_lock(&scheduler->mutex);
        scheduler->stop = true;
        pthread_cond_signal(&scheduler->work);
        pthread_mutex_unlock(&scheduler->mutex);
        pthread_join(scheduler->worker, NULL);
    }
    scheduler_free_job(scheduler);
    pthread_cond_destroy(&scheduler->done);
    pthread_cond_destroy(&scheduler->work);
    pthread_mutex_destroy(&scheduler->mutex);
    free(scheduler);
}

void q38_ple_scheduler_reset(q38_ple_scheduler *scheduler) {
    if (!scheduler) return;
    pthread_mutex_lock(&scheduler->mutex);
    /*
     * Forward state reset is serialized with execution.  A completed job is
     * consumed here; an in-flight job is allowed to finish before clearing it.
     */
    while (scheduler->pending || (!scheduler->ready && scheduler->rows))
        pthread_cond_wait(&scheduler->done, &scheduler->mutex);
    scheduler_free_job(scheduler);
    scheduler->pending = false;
    scheduler->ready = false;
    memset(&scheduler->stats, 0, sizeof(scheduler->stats));
    pthread_mutex_unlock(&scheduler->mutex);
}

bool q38_ple_scheduler_submit(q38_ple_scheduler *scheduler,
                              const uint64_t *rows, size_t row_count,
                              char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!scheduler || !rows || row_count == 0 ||
        row_count > SIZE_MAX / sizeof(*rows))
        return fail(error, error_len, "invalid PLE scheduler submission");
    uint64_t *unique_rows =
        (uint64_t *)malloc(row_count * sizeof(*unique_rows));
    q38_ple_block *blocks =
        (q38_ple_block *)malloc(row_count * sizeof(*blocks));
    if (!unique_rows || !blocks) {
        free(unique_rows);
        free(blocks);
        return fail(error, error_len, "PLE scheduler metadata allocation failed");
    }
    memcpy(unique_rows, rows, row_count * sizeof(*unique_rows));
    qsort(unique_rows, row_count, sizeof(*unique_rows), compare_u64);
    size_t unique_count = 0;
    for (size_t i = 0; i < row_count; ++i) {
        if (unique_count == 0 || unique_rows[i] != unique_rows[unique_count - 1])
            unique_rows[unique_count++] = unique_rows[i];
    }
    size_t block_count = 0;
    for (size_t i = 0; i < unique_count; ++i) {
        uint64_t offset = 0;
        if (!q38_ple_store_row_range(scheduler ? &scheduler->store : NULL,
                                     unique_rows[i], &offset, error, error_len)) {
            free(unique_rows);
            free(blocks);
            return false;
        }
        blocks[block_count++] = (q38_ple_block){offset, scheduler->store.row_bytes};
    }
    qsort(blocks, block_count, sizeof(*blocks), compare_block);
    size_t coalesced = 0;
    for (size_t i = 0; i < block_count; ++i) {
        q38_ple_block block = blocks[i];
        if (coalesced && blocks[coalesced - 1].offset +
                         blocks[coalesced - 1].bytes == block.offset) {
            blocks[coalesced - 1].bytes += block.bytes;
        } else {
            blocks[coalesced++] = block;
        }
    }

    pthread_mutex_lock(&scheduler->mutex);
    while (scheduler->pending)
        pthread_cond_wait(&scheduler->done, &scheduler->mutex);
    scheduler_free_job(scheduler);
    scheduler->ready = false;
    scheduler->rows = unique_rows;
    scheduler->row_count = unique_count;
    scheduler->blocks = blocks;
    scheduler->block_count = coalesced;
    const double t0 = scheduler->stats.t0_token_forward_begin_ms;
    const uint64_t token_position = scheduler->stats.token_position;
    memset(&scheduler->stats, 0, sizeof(scheduler->stats));
    scheduler->stats.t0_token_forward_begin_ms = t0;
    scheduler->stats.token_position = token_position;
    scheduler->stats.logical_accesses = row_count;
    scheduler->stats.unique_rows = unique_count;
    scheduler->stats.unique_physical_blocks = coalesced;
    if (scheduler->store.row_bytes != 0 &&
        row_count > UINT64_MAX / scheduler->store.row_bytes) {
        scheduler_free_job(scheduler);
        pthread_mutex_unlock(&scheduler->mutex);
        return fail(error, error_len, "PLE scheduler byte count overflows");
    }
    scheduler->stats.logical_bytes = (uint64_t)row_count *
                                     scheduler->store.row_bytes;
    scheduler->stats.file_reads_sequential = true;
    scheduler->stats.request_id = ++scheduler->next_request_id;
    scheduler->stats.t1_ple_request_submit_ms = scheduler_now_ms();
    scheduler->pending = true;
    scheduler->ready = false;
    pthread_cond_signal(&scheduler->work);
    pthread_mutex_unlock(&scheduler->mutex);
    return true;
}

bool q38_ple_scheduler_wait(q38_ple_scheduler *scheduler,
                            char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!scheduler) return true;
    pthread_mutex_lock(&scheduler->mutex);
    if (!scheduler->rows && !scheduler->pending) {
        pthread_mutex_unlock(&scheduler->mutex);
        return true;
    }
    const double consume_ms = scheduler_now_ms();
    scheduler->stats.t7_ple_wait_begin_ms = consume_ms;
    while (!scheduler->ready)
        pthread_cond_wait(&scheduler->done, &scheduler->mutex);
    scheduler->stats.consume_ms = consume_ms;
    scheduler->stats.wait_ms =
        scheduler->stats.ready_ms > consume_ms
        ? scheduler->stats.ready_ms - consume_ms : 0.0;
    scheduler->stats.t8_ple_wait_end_ms = scheduler_now_ms();
    scheduler->stats.wait_at_injection_ms =
        scheduler->stats.t8_ple_wait_end_ms -
        scheduler->stats.t7_ple_wait_begin_ms;
    scheduler->stats.wait_ms = scheduler->stats.wait_at_injection_ms;
    scheduler->stats.overlap_ms =
        scheduler->stats.elapsed_ms > scheduler->stats.wait_ms
        ? scheduler->stats.elapsed_ms - scheduler->stats.wait_ms : 0.0;
    scheduler_free_job(scheduler);
    scheduler->ready = false;
    pthread_cond_broadcast(&scheduler->done);
    pthread_mutex_unlock(&scheduler->mutex);
    return true;
}

bool q38_ple_scheduler_record_request_timing(
    q38_ple_scheduler *scheduler, double request_build_ms,
    double history_ngram_ms, double index_lookup_ms, double async_submit_ms) {
    if (!scheduler) return false;
    pthread_mutex_lock(&scheduler->mutex);
    scheduler->request_build_ms = request_build_ms;
    scheduler->history_ngram_ms = history_ngram_ms;
    scheduler->index_lookup_ms = index_lookup_ms;
    scheduler->async_submit_ms = async_submit_ms;
    scheduler->stats.request_build_ms = request_build_ms;
    scheduler->stats.history_ngram_ms = history_ngram_ms;
    scheduler->stats.index_lookup_ms = index_lookup_ms;
    scheduler->stats.async_submit_ms = async_submit_ms;
    pthread_mutex_unlock(&scheduler->mutex);
    return true;
}

bool q38_ple_scheduler_record_injection_timing(
    q38_ple_scheduler *scheduler, double decode_dequant_ms,
    double accumulation_ms, double injection_ms) {
    if (!scheduler) return false;
    pthread_mutex_lock(&scheduler->mutex);
    scheduler->decode_dequant_ms = decode_dequant_ms;
    scheduler->accumulation_ms = accumulation_ms;
    scheduler->injection_ms = injection_ms;
    scheduler->stats.decode_dequant_ms = decode_dequant_ms;
    scheduler->stats.accumulation_ms = accumulation_ms;
    scheduler->stats.injection_ms = injection_ms;
    scheduler->stats.t10_ple_injection_end_ms = scheduler_now_ms();
    pthread_mutex_unlock(&scheduler->mutex);
    return true;
}

bool q38_ple_scheduler_record_injection_begin(
    q38_ple_scheduler *scheduler) {
    if (!scheduler) return false;
    pthread_mutex_lock(&scheduler->mutex);
    scheduler->stats.t9_ple_injection_begin_ms = scheduler_now_ms();
    pthread_mutex_unlock(&scheduler->mutex);
    return true;
}

bool q38_ple_scheduler_record_timeline(
    q38_ple_scheduler *scheduler, q38_ple_timeline_boundary boundary,
    uint64_t request_id, uint64_t token_position) {
    if (!scheduler || boundary < Q38_PLE_T0_TOKEN_FORWARD_BEGIN ||
        boundary > Q38_PLE_T11_TOKEN_FORWARD_END)
        return false;
    pthread_mutex_lock(&scheduler->mutex);
    if (request_id != 0) scheduler->stats.request_id = request_id;
    scheduler->stats.token_position = token_position;
    scheduler->stats.submit_position = token_position;
    scheduler->stats.injection_position = token_position;
    const double timestamp = scheduler_now_ms();
    switch (boundary) {
    case Q38_PLE_T0_TOKEN_FORWARD_BEGIN:
        scheduler->stats.t0_token_forward_begin_ms = timestamp;
        break;
    case Q38_PLE_T2_LAYER0_BEGIN:
        scheduler->stats.t2_layer0_begin_ms = timestamp;
        break;
    case Q38_PLE_T3_LAYER0_END:
        scheduler->stats.t3_layer0_end_ms = timestamp;
        break;
    case Q38_PLE_T4_LAYER1_BEGIN:
        scheduler->stats.t4_layer1_begin_ms = timestamp;
        break;
    case Q38_PLE_T5_LAYER1_END:
        scheduler->stats.t5_layer1_end_ms = timestamp;
        break;
    case Q38_PLE_T11_TOKEN_FORWARD_END:
        scheduler->stats.t11_token_forward_end_ms = timestamp;
        break;
    default:
        break;
    }
    pthread_mutex_unlock(&scheduler->mutex);
    return true;
}

bool q38_ple_scheduler_get_stats(
    const q38_ple_scheduler *scheduler, q38_ple_scheduler_stats *stats) {
    if (!scheduler || !stats) return false;
    q38_ple_scheduler *mutable_scheduler = (q38_ple_scheduler *)scheduler;
    pthread_mutex_lock(&mutable_scheduler->mutex);
    *stats = mutable_scheduler->stats;
    pthread_mutex_unlock(&mutable_scheduler->mutex);
    return true;
}
