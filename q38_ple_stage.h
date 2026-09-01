#ifndef Q38_PLE_STAGE_H
#define Q38_PLE_STAGE_H

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    void *host_ptr;
    size_t capacity;
    size_t used;
    bool in_flight;
    cudaEvent_t ready;
} q38_ple_stage_buffer;

typedef struct {
    q38_ple_stage_buffer *buffers;
    uint32_t count;
    size_t bytes_per_buffer;
    uint32_t next;
    uint64_t high_watermark;
    uint64_t wait_count;
    uint64_t wait_us;
    uint64_t h2d_bytes;
    uint64_t h2d_us;
} q38_ple_stage_pool;

bool q38_ple_stage_pool_init(q38_ple_stage_pool *pool, uint32_t count,
                             size_t bytes_per_buffer, char *error,
                             size_t error_len);
void q38_ple_stage_pool_destroy(q38_ple_stage_pool *pool);
bool q38_ple_stage_pool_acquire(q38_ple_stage_pool *pool, uint32_t *index,
                                char *error, size_t error_len);
void *q38_ple_stage_pool_data(q38_ple_stage_pool *pool, uint32_t index);
bool q38_ple_stage_pool_submit(q38_ple_stage_pool *pool, uint32_t index,
                               void *device_dst, size_t bytes,
                               cudaStream_t stream, char *error,
                               size_t error_len);

#endif
