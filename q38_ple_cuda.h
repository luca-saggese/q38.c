#ifndef Q38_PLE_CUDA_H
#define Q38_PLE_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#include "q38_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decode rows selected by host-side PLE IDs. The table, IDs, and output are
 * device allocations; output is contiguous as [id_count][row_width]. */
bool q38_ple_cuda_lookup_rows(uint32_t qtype, const void *device_table,
                              uint64_t table_rows, uint32_t row_width,
                              const uint32_t *device_ids, size_t id_count,
                              float *device_rows, cudaStream_t stream,
                              char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
