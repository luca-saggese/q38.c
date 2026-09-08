#ifndef Q38_NVFP4_RESIDENCY_H
#define Q38_NVFP4_RESIDENCY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q38_nvfp4_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *device_regions[Q38_NVFP4_REGION_COUNT];
    uint64_t allocated_bytes;
    uint64_t copied_bytes;
    uint32_t allocation_count;
} q38_nvfp4_cuda_residency;

void q38_nvfp4_cuda_residency_init(q38_nvfp4_cuda_residency *residency);
bool q38_nvfp4_cuda_residency_load(
    const q38_nvfp4_pack *pack,
    q38_nvfp4_cuda_residency *residency,
    char *error, size_t error_len);
void q38_nvfp4_cuda_residency_destroy(
    q38_nvfp4_cuda_residency *residency);

#ifdef __cplusplus
}
#endif

#endif
