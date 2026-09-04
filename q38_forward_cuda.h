#ifndef Q38_FORWARD_CUDA_H
#define Q38_FORWARD_CUDA_H

#include <stdbool.h>
#include <stddef.h>

#include "q38_forward.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_forward_cuda_context q38_forward_cuda_context;
typedef void (*q38_forward_cuda_allocation_observer)(size_t bytes, void *user);
typedef struct q38_forward_cuda_telemetry {
    const char *subsystem;
    uint32_t layer;
    const char *logical_stage;
    const char *tensor_name;
    uint32_t qtype;
    size_t rows;
    size_t cols;
    size_t bytes;
    bool resident_hit;
    bool resident_miss;
    size_t upload_bytes;
    size_t weight_bytes;
    size_t activation_read_bytes;
    size_t activation_write_bytes;
    size_t d2h_bytes;
    float upload_ms;
    float kernel_ms;
    float backend_overhead_ms;
    uint64_t allocation_count;
    uint64_t sync_count;
    uint64_t host_syncs;
} q38_forward_cuda_telemetry;
typedef void (*q38_forward_cuda_telemetry_observer)(
    const q38_forward_cuda_telemetry *telemetry, void *user);
typedef void (*q38_forward_cuda_residency_progress_observer)(
    const char *group, const q38_tensor *tensor, size_t cumulative_bytes,
    size_t free_bytes, size_t total_bytes, void *user);
typedef struct {
    size_t matrix_upload_bytes;
    uint64_t resident_hits;
    uint64_t resident_misses;
    uint64_t cuda_allocations;
    bool lm_head_resident;
    const void *lm_head_device_pointer;
    bool all_non_ple_resident;
    size_t persistent_resident_bytes;
    uint64_t persistent_resident_tensors;
    uint64_t persistent_hits;
    uint64_t persistent_misses;
    size_t persistent_expected_bytes;
    uint64_t persistent_expected_tensors;
    uint64_t persistent_duplicate_tensors;
    uint64_t persistent_ple_tensors;
    uint64_t persistent_ple_entries;
    bool persistent_coverage_ok;
    const char *persistent_failure;
    size_t persistent_loaded_bytes;
    uint64_t persistent_loaded_tensors;
    bool exec_strict;
    uint64_t resident_lookup_in_decode;
    uint64_t gguf_name_lookup_in_decode;
    uint64_t non_ple_residency_miss;
    size_t non_ple_upload_bytes_per_token;
    float gpu_argmax_kernel_ms;
} q38_forward_cuda_residency_stats;

q38_forward_cuda_context *q38_forward_cuda_context_create(char *error,
                                                           size_t error_len);
void q38_forward_cuda_context_destroy(q38_forward_cuda_context *context);
void *q38_forward_cuda_stream(q38_forward_cuda_context *context);
void q38_forward_cuda_set_allocation_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_allocation_observer observer, void *user);
void q38_forward_cuda_set_telemetry_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_telemetry_observer observer, void *user);
void q38_forward_cuda_set_stage_context(q38_forward_cuda_context *context,
                                        uint32_t layer,
                                        const char *logical_stage);
bool q38_forward_cuda_enable_all_non_ple_residency(
    q38_forward_cuda_context *context, const q38_gguf *model,
    char *error, size_t error_len);
void q38_forward_cuda_set_residency_progress_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_residency_progress_observer observer, void *user);
bool q38_forward_cuda_prepare_lm_head(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_tensor *tensor, char *error, size_t error_len);
void q38_forward_cuda_get_residency_stats(
    const q38_forward_cuda_context *context,
    q38_forward_cuda_residency_stats *stats);

bool q38_forward_cuda_matvec_backend(
    const q38_gguf *model, const q38_tensor *tensor, size_t row,
    const float *input, size_t cols, float *output, void *user, char *error,
    size_t error_len);

bool q38_forward_cuda_matrix_backend(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t rows, size_t cols, float *output, void *user, char *error,
    size_t error_len);

/* Reduce the most recent device-side matrix result without downloading it. */
bool q38_forward_cuda_greedy_argmax(q38_forward_cuda_context *context,
                                    uint32_t *token, char *error,
                                    size_t error_len);

bool q38_forward_cuda_expert_backend(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, size_t expert, const float *input, float *output,
    void *user, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
