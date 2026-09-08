#include "q38_nvfp4_residency.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t Q38_NVFP4_STAGE_BYTES = 128u * 1024u * 1024u;

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static cudaError_t release_staging(
    void *stage[2], cudaEvent_t events[2], cudaStream_t stream) {
    cudaError_t first = cudaSuccess;
    for (uint32_t i = 0; i < 2; ++i) {
        if (events[i]) {
            cudaError_t status = cudaEventDestroy(events[i]);
            if (first == cudaSuccess && status != cudaSuccess)
                first = status;
            events[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < 2; ++i) {
        if (stage[i]) {
            cudaError_t status = cudaFreeHost(stage[i]);
            if (first == cudaSuccess && status != cudaSuccess)
                first = status;
            stage[i] = NULL;
        }
    }
    if (stream) {
        cudaError_t status = cudaStreamDestroy(stream);
        if (first == cudaSuccess && status != cudaSuccess)
            first = status;
    }
    return first;
}

static bool fail_cuda(char *error, size_t error_len, const char *operation,
                      cudaError_t status) {
    if (error && error_len) {
        snprintf(error, error_len, "%s failed: %s", operation,
                 cudaGetErrorString(status));
    }
    return false;
}

void q38_nvfp4_cuda_residency_init(q38_nvfp4_cuda_residency *residency) {
    if (residency) memset(residency, 0, sizeof(*residency));
}

void q38_nvfp4_cuda_residency_destroy(
    q38_nvfp4_cuda_residency *residency) {
    if (!residency) return;
    for (uint32_t i = 0; i < Q38_NVFP4_REGION_COUNT; ++i) {
        if (residency->device_regions[i]) {
            cudaFree(residency->device_regions[i]);
            residency->device_regions[i] = NULL;
        }
    }
    q38_nvfp4_cuda_residency_init(residency);
}

bool q38_nvfp4_cuda_residency_load(
    const q38_nvfp4_pack *pack,
    q38_nvfp4_cuda_residency *residency,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!pack || !residency)
        return fail(error, error_len, "invalid NVFP4 residency arguments");
    if (q38_nvfp4_pack_is_source_backed(pack))
        return fail(error, error_len,
                    "CUDA residency requires a materialized NVFP4 pack");

    q38_nvfp4_cuda_residency_destroy(residency);
    void *stage[2] = {NULL, NULL};
    cudaEvent_t reuse_events[2] = {NULL, NULL};
    bool slot_in_flight[2] = {false, false};
    cudaStream_t transfer_stream = NULL;
    cudaError_t status = cudaStreamCreateWithFlags(
        &transfer_stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        fail_cuda(error, error_len, "cudaStreamCreate NVFP4 residency", status);
        goto fail_load;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        status = cudaMallocHost(&stage[i], Q38_NVFP4_STAGE_BYTES);
        if (status != cudaSuccess) {
            fail_cuda(error, error_len, "cudaMallocHost NVFP4 staging", status);
            goto fail_load;
        }
        status = cudaEventCreateWithFlags(
            &reuse_events[i], cudaEventDisableTiming);
        if (status != cudaSuccess) {
            fail_cuda(error, error_len, "cudaEventCreate NVFP4 staging", status);
            goto fail_load;
        }
    }
    for (uint32_t region = 0; region < Q38_NVFP4_REGION_COUNT; ++region) {
        q38_nvfp4_region_view view;
        if (!q38_nvfp4_pack_get_region_view(
                pack, region, &view, error, error_len))
            goto fail_load;
        if (!view.data || view.bytes == 0) {
            fail(error, error_len, "NVFP4 region is empty");
            goto fail_load;
        }
        if (view.bytes > SIZE_MAX) {
            fail(error, error_len, "NVFP4 region size overflows host size");
            goto fail_load;
        }
        status = cudaMalloc(
            &residency->device_regions[region], (size_t)view.bytes);
        if (status != cudaSuccess) {
            fail_cuda(error, error_len, "cudaMalloc NVFP4 region", status);
            goto fail_load;
        }
        residency->allocated_bytes += view.bytes;
        residency->allocation_count++;
        size_t offset = 0;
        while (offset < (size_t)view.bytes) {
            const uint32_t slot = (uint32_t)((offset / Q38_NVFP4_STAGE_BYTES) & 1u);
            if (slot_in_flight[slot]) {
                status = cudaEventSynchronize(reuse_events[slot]);
                if (status != cudaSuccess) {
                    fail_cuda(error, error_len,
                              "cudaEventSynchronize NVFP4 staging", status);
                    goto fail_load;
                }
                slot_in_flight[slot] = false;
            }
            const size_t remaining = (size_t)view.bytes - offset;
            const size_t bytes = remaining < Q38_NVFP4_STAGE_BYTES
                ? remaining : Q38_NVFP4_STAGE_BYTES;
            memcpy(stage[slot], (const uint8_t *)view.data + offset, bytes);
            status = cudaMemcpyAsync(
                (uint8_t *)residency->device_regions[region] + offset,
                stage[slot], bytes, cudaMemcpyHostToDevice, transfer_stream);
            if (status != cudaSuccess) {
                fail_cuda(error, error_len,
                          "cudaMemcpyAsync NVFP4 region", status);
                goto fail_load;
            }
            status = cudaEventRecord(reuse_events[slot], transfer_stream);
            if (status != cudaSuccess) {
                fail_cuda(error, error_len,
                          "cudaEventRecord NVFP4 staging", status);
                goto fail_load;
            }
            slot_in_flight[slot] = true;
            offset += bytes;
        }
        residency->copied_bytes += view.bytes;
    }
    status = cudaStreamSynchronize(transfer_stream);
    if (status != cudaSuccess) {
        fail_cuda(error, error_len,
                  "cudaStreamSynchronize NVFP4 residency", status);
        goto fail_load;
    }
    status = release_staging(stage, reuse_events, transfer_stream);
    transfer_stream = NULL;
    if (status != cudaSuccess) {
        fail_cuda(error, error_len, "NVFP4 staging cleanup", status);
        goto fail_load;
    }
    return true;

fail_load:
    if (transfer_stream) {
        cudaError_t sync_status = cudaStreamSynchronize(transfer_stream);
        if (status == cudaSuccess && sync_status != cudaSuccess)
            status = sync_status;
    }
    {
        cudaError_t cleanup_status =
            release_staging(stage, reuse_events, transfer_stream);
        if (status == cudaSuccess && cleanup_status != cudaSuccess)
            status = cleanup_status;
    }
    if (status != cudaSuccess && error && error_len && !error[0])
        fail_cuda(error, error_len, "NVFP4 residency cleanup", status);
    q38_nvfp4_cuda_residency_destroy(residency);
    return false;
}
