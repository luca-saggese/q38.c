#include "q38_ple_stage.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static uint64_t now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000u + (uint64_t)tv.tv_usec;
}

bool q38_ple_stage_pool_init(q38_ple_stage_pool *pool, uint32_t count,
                             size_t bytes_per_buffer, char *error,
                             size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!pool || !count || !bytes_per_buffer)
        return fail(error, error_len, "invalid staging pool geometry");
    memset(pool, 0, sizeof(*pool));
    pool->buffers = (q38_ple_stage_buffer *)calloc(count, sizeof(*pool->buffers));
    if (!pool->buffers) return fail(error, error_len, "staging pool allocation failed");
    pool->count = count;
    pool->bytes_per_buffer = bytes_per_buffer;
    for (uint32_t i = 0; i < count; ++i) {
        cudaError_t s = cudaHostAlloc(&pool->buffers[i].host_ptr,
                                      bytes_per_buffer, cudaHostAllocDefault);
        if (s != cudaSuccess) {
            q38_ple_stage_pool_destroy(pool);
            return fail(error, error_len, cudaGetErrorString(s));
        }
        s = cudaEventCreateWithFlags(&pool->buffers[i].ready,
                                     cudaEventDisableTiming);
        if (s != cudaSuccess) {
            q38_ple_stage_pool_destroy(pool);
            return fail(error, error_len, cudaGetErrorString(s));
        }
    }
    return true;
}

void q38_ple_stage_pool_destroy(q38_ple_stage_pool *pool) {
    if (!pool) return;
    for (uint32_t i = 0; i < pool->count; ++i) {
        if (pool->buffers[i].ready) cudaEventDestroy(pool->buffers[i].ready);
        if (pool->buffers[i].host_ptr) cudaFreeHost(pool->buffers[i].host_ptr);
    }
    free(pool->buffers);
    memset(pool, 0, sizeof(*pool));
}

bool q38_ple_stage_pool_acquire(q38_ple_stage_pool *pool, uint32_t *index,
                                char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!pool || !pool->buffers || !index)
        return fail(error, error_len, "invalid staging pool acquire");
    const uint64_t start = now_us();
    for (uint32_t n = 0; n < pool->count; ++n) {
        uint32_t i = (pool->next + n) % pool->count;
        q38_ple_stage_buffer *b = &pool->buffers[i];
        if (b->in_flight) {
            cudaError_t s = cudaEventQuery(b->ready);
            if (s == cudaErrorNotReady) continue;
            if (s != cudaSuccess) return fail(error, error_len, cudaGetErrorString(s));
            b->in_flight = false;
        }
        *index = i;
        pool->next = (i + 1) % pool->count;
        pool->wait_us += now_us() - start;
        return true;
    }
    uint32_t i = pool->next % pool->count;
    q38_ple_stage_buffer *b = &pool->buffers[i];
    pool->wait_count++;
    cudaError_t s = cudaEventSynchronize(b->ready);
    if (s != cudaSuccess) return fail(error, error_len, cudaGetErrorString(s));
    b->in_flight = false;
    *index = i;
    pool->next = (i + 1) % pool->count;
    pool->wait_us += now_us() - start;
    return true;
}

void *q38_ple_stage_pool_data(q38_ple_stage_pool *pool, uint32_t index) {
    return pool && pool->buffers && index < pool->count
        ? pool->buffers[index].host_ptr : NULL;
}

bool q38_ple_stage_pool_submit(q38_ple_stage_pool *pool, uint32_t index,
                               void *device_dst, size_t bytes,
                               cudaStream_t stream, char *error,
                               size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!pool || !pool->buffers || index >= pool->count || !device_dst ||
        !bytes || bytes > pool->bytes_per_buffer)
        return fail(error, error_len, "invalid staging pool submit");
    q38_ple_stage_buffer *b = &pool->buffers[index];
    const uint64_t start = now_us();
    cudaError_t s = cudaMemcpyAsync(device_dst, b->host_ptr, bytes,
                                    cudaMemcpyHostToDevice, stream);
    if (s != cudaSuccess) return fail(error, error_len, cudaGetErrorString(s));
    s = cudaEventRecord(b->ready, stream);
    if (s != cudaSuccess) return fail(error, error_len, cudaGetErrorString(s));
    b->used = bytes;
    b->in_flight = true;
    if (bytes > pool->high_watermark) pool->high_watermark = bytes;
    pool->h2d_bytes += bytes;
    pool->h2d_us += now_us() - start;
    return true;
}
