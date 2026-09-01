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

typedef struct {
    size_t input_count;
    size_t unique_count;
} q38_ple_cuda_lookup_stats;

/* Decode rows selected by host-side PLE IDs. The table, IDs, and output are
 * device allocations; output is contiguous as [id_count][row_width]. */
bool q38_ple_cuda_lookup_rows(uint32_t qtype, const void *device_table,
                              uint64_t table_rows, uint32_t row_width,
                              const uint32_t *device_ids, size_t id_count,
                              float *device_rows, cudaStream_t stream,
                              char *error, size_t error_len);

/* Decode each distinct host-side ID once, then expand rows back into the
 * original order on the device. Duplicate IDs therefore change neither the
 * output order nor the accumulation order of a caller. */
bool q38_ple_cuda_lookup_rows_dedup(
    uint32_t qtype, const void *device_table, uint64_t table_rows,
    uint32_t row_width, const uint32_t *host_ids, size_t id_count,
    float *device_rows, cudaStream_t stream, q38_ple_cuda_lookup_stats *stats,
    char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
