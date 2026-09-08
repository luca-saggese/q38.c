#include "q38_nvfp4_residency.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
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
    for (uint32_t region = 0; region < Q38_NVFP4_REGION_COUNT; ++region) {
        q38_nvfp4_region_view view;
        if (!q38_nvfp4_pack_get_region_view(
                pack, region, &view, error, error_len))
            goto fail_load;
        if (!view.data || view.bytes == 0)
            return fail(error, error_len, "NVFP4 region is empty");
        cudaError_t status = cudaMalloc(
            &residency->device_regions[region], (size_t)view.bytes);
        if (status != cudaSuccess) {
            fail_cuda(error, error_len, "cudaMalloc NVFP4 region", status);
            goto fail_load;
        }
        residency->allocated_bytes += view.bytes;
        residency->allocation_count++;
        status = cudaMemcpy(
            residency->device_regions[region], view.data, (size_t)view.bytes,
            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            fail_cuda(error, error_len, "cudaMemcpy NVFP4 region", status);
            goto fail_load;
        }
        residency->copied_bytes += view.bytes;
    }
    return true;

fail_load:
    q38_nvfp4_cuda_residency_destroy(residency);
    return false;
}
