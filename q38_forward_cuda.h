#ifndef Q38_FORWARD_CUDA_H
#define Q38_FORWARD_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q38_diagnostics.h"
#include "q38_forward.h"
#include "q38_qsa_candidate.h"
#include "q38_directional_steering.h"
#include "q38_nvfp4_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_forward_cuda_context q38_forward_cuda_context;
typedef void (*q38_forward_cuda_allocation_observer)(size_t bytes, void *user);

typedef enum {
    Q38_CUDA_SYNC_MOE_ROUTED_D2H = 0,
    Q38_CUDA_SYNC_MOE_GROUPED_D2H,
    Q38_CUDA_SYNC_GDN_OUTPUT,
    Q38_CUDA_SYNC_GDN_TRACE_STATE,
    Q38_CUDA_SYNC_GR_READ,
    Q38_CUDA_SYNC_GR_WRITE,
    Q38_CUDA_SYNC_QSA_QKV,
    Q38_CUDA_SYNC_MATVEC_D2H,
    Q38_CUDA_SYNC_MATRIX_D2H,
    Q38_CUDA_SYNC_MATRIX_BATCH_D2H,
    Q38_CUDA_SYNC_ARGMAX,
    Q38_CUDA_SYNC_PLE_STAGE_WAIT,
    Q38_CUDA_SYNC_RESIDENCY_INIT,
    Q38_CUDA_SYNC_STEERING_INIT,
    Q38_CUDA_SYNC_LM_HEAD_RESIDENCY_INIT,
    Q38_CUDA_SYNC_REASON_COUNT
} q38_forward_cuda_sync_reason;

typedef struct {
    uint64_t real_cuda_sync_count;
    double host_blocked_on_cuda_ms;
    double telemetry_callback_wall_ms;
    uint64_t reason_count[Q38_CUDA_SYNC_REASON_COUNT];
    double reason_ms[Q38_CUDA_SYNC_REASON_COUNT];
    double reason_max_ms[Q38_CUDA_SYNC_REASON_COUNT];
} q38_forward_cuda_sync_stats;

typedef struct q38_forward_cuda_telemetry {
    const char *subsystem;
    uint32_t layer;
    const char *logical_stage;
    const char *operation;
    const char *fallback_path;
    const char *tensor_name;
    uint32_t tensor_id;
    uint32_t qtype;
    size_t rows;
    size_t cols;
    size_t bytes;
    bool resident_hit;
    bool resident_miss;
    bool non_ple_residency_miss;
    bool ple_file_backed_access;
    size_t upload_bytes;
    size_t ple_file_bytes;
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
    float callback_wall_ms;
    float host_wait_ms;
} q38_forward_cuda_telemetry;
typedef void (*q38_forward_cuda_telemetry_observer)(
    const q38_forward_cuda_telemetry *telemetry, void *user);
typedef void (*q38_forward_cuda_residency_progress_observer)(
    const char *group, const q38_tensor *tensor, size_t cumulative_bytes,
    size_t free_bytes, size_t total_bytes, void *user);
typedef struct {
    uint64_t source_offset;
    size_t bytes;
    double source_copy_ms;
    double h2d_enqueue_ms;
} q38_residency_span_timing;
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
    uint64_t persistent_pointer_fingerprint;
    uint64_t cuda_context_identity;
    uint64_t cuda_stream_identity;
    uint64_t workspace_pointer_fingerprint;
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
    uint64_t residency_planned_spans;
    uint64_t residency_transfer_calls;
    uint64_t residency_device_copies;
    uint64_t residency_final_syncs;
    size_t residency_stage_bytes;
    size_t residency_planned_bytes;
    size_t residency_staged_bytes;
    size_t residency_h2d_bytes;
    double residency_plan_ms;
    double residency_device_alloc_ms;
    double residency_source_copy_ms;
    double residency_h2d_enqueue_ms;
    double residency_d2d_enqueue_ms;
    double residency_final_wait_ms;
    uint64_t residency_allocations;
    size_t residency_allocated_bytes;
    uint64_t residency_mincore_pages_before;
    uint64_t residency_mincore_pages_after;
    long residency_minor_faults_before;
    long residency_minor_faults_after;
    long residency_major_faults_before;
    long residency_major_faults_after;
    const q38_residency_span_timing *residency_span_timings;
    size_t residency_span_timing_count;
    bool exec_strict;
    uint64_t resident_lookup_in_decode;
    uint64_t gguf_name_lookup_in_decode;
    uint64_t non_ple_residency_miss;
    size_t non_ple_upload_bytes_per_token;
    uint64_t ple_file_backed_accesses;
    size_t ple_file_bytes;
    float gpu_argmax_kernel_ms;
    uint64_t routed_layers_executed;
    uint64_t selected_experts_total;
    uint64_t q2_gate_up_fast_calls;
    uint64_t q2_gate_up_legacy_calls;
    uint64_t q2_gate_up_fallback_calls;
    uint64_t q2_down_calls;
    double q2_gate_up_fast_total_kernel_ms;
    double q2_gate_up_legacy_total_kernel_ms;
    double q2_down_total_kernel_ms;
    double q2_weighted_reduce_total_kernel_ms;
    double expert_backend_total_wall_ms;
    uint64_t expert_host_sync_count;
    uint64_t expert_kernel_launches;
    uint64_t expert_H2D_bytes;
    uint64_t expert_D2H_bytes;
    uint64_t qsa_chain_calls;
    uint64_t qsa_chain_kernel_launches;
    uint64_t qsa_chain_syncs;
    uint64_t qsa_chain_h2d_bytes;
    uint64_t qsa_chain_d2h_bytes;
    uint64_t qsa_chain_internal_h2d_bytes;
    uint64_t qsa_chain_internal_d2h_bytes;
    uint64_t expert_fast_calls_by_layer[Q38_MODEL_LAYERS];
    uint64_t expert_legacy_calls_by_layer[Q38_MODEL_LAYERS];
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
bool q38_forward_cuda_load_directional_steering(
    q38_forward_cuda_context *context,
    const q38_directional_steering *steering, char *error, size_t error_len);
void q38_forward_cuda_set_directional_steering_scales(
    q38_forward_cuda_context *context, float ffn_scale, float attn_scale);
bool q38_forward_cuda_apply_directional_steering(
    q38_forward_cuda_context *context, float *device_values,
    uint32_t layer, size_t width, size_t rows, float scale,
    char *error, size_t error_len);
bool q38_forward_cuda_enable_all_non_ple_residency(
    q38_forward_cuda_context *context, const q38_gguf *model,
    char *error, size_t error_len);
bool q38_forward_cuda_enable_nvfp4_residency(
    q38_forward_cuda_context *context, const q38_nvfp4_pack *pack,
    const q38_gguf *model, char *error, size_t error_len);
void q38_forward_cuda_set_residency_progress_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_residency_progress_observer observer, void *user);
bool q38_forward_cuda_prepare_lm_head(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_tensor *tensor, char *error, size_t error_len);
bool q38_forward_cuda_prepare_nvfp4_lm_head(
    q38_forward_cuda_context *context, const q38_nvfp4_pack *pack,
    const q38_tensor *tensor, char *error, size_t error_len);
void q38_forward_cuda_get_residency_stats(
    const q38_forward_cuda_context *context,
    q38_forward_cuda_residency_stats *stats);
void q38_forward_cuda_get_sync_stats(
    const q38_forward_cuda_context *context,
    q38_forward_cuda_sync_stats *stats);
void q38_forward_cuda_reset_sync_stats(q38_forward_cuda_context *context);
const char *q38_forward_cuda_sync_reason_name(
    q38_forward_cuda_sync_reason reason);
void q38_forward_cuda_set_qsa_candidate(
    q38_forward_cuda_context *context, q38_qsa_candidate_fn candidate);
void q38_forward_cuda_record_route(
    q38_forward_cuda_context *context, uint32_t layer, size_t selected_count);
void q38_forward_cuda_get_expert_layer_calls(
    const q38_forward_cuda_context *context, uint32_t layer,
    uint64_t *fast_calls, uint64_t *legacy_calls);
void q38_forward_cuda_reset_gdn_state(q38_forward_cuda_context *context);
bool q38_forward_cuda_load_gdn_state(
    const q38_forward_state *state, void *user, char *error, size_t error_len);
bool q38_forward_cuda_sync_gdn_state(
    q38_forward_state *state, void *user, char *error, size_t error_len);
bool q38_forward_cuda_load_qsa_state(
    const q38_forward_state *state, void *user, char *error, size_t error_len);

bool q38_forward_cuda_matvec_backend(
    const q38_gguf *model, const q38_tensor *tensor, size_t row,
    const float *input, size_t cols, float *output, void *user, char *error,
    size_t error_len);

bool q38_forward_cuda_matrix_backend(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t rows, size_t cols, float *output, void *user, char *error,
    size_t error_len);
bool q38_forward_cuda_matrix_batch_backend(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t token_count, size_t rows, size_t cols, float *output, void *user,
    char *error, size_t error_len);

bool q38_forward_cuda_gr_read_backend(
    const q38_gguf *model, const q38_gr_weights *weights,
    const float *residual, size_t token_count, float *input, float *normed,
    void *user, char *error, size_t error_len);

bool q38_forward_cuda_gr_write_backend(
    const q38_gguf *model, const q38_gr_weights *weights,
    const float *residual, float *normed, const float *block,
    size_t token_count, float *updated, void *user, char *error,
    size_t error_len);

bool q38_forward_cuda_qsa_qkv_backend(
    const q38_gguf *model, const q38_tensor *q_proj,
    const q38_tensor *k_proj, const q38_tensor *v_proj,
    const float *host_input, size_t token_count, float *host_q,
    float *host_k, float *host_v, q38_forward_qsa_timing *timing,
    void *user, char *error, size_t error_len);

bool q38_forward_cuda_qsa_chain_backend(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_qsa_state *state, const float *host_input, size_t token_count,
    uint32_t layer_number, float *host_output, q38_forward_qsa_timing *timing,
    void *user, char *error, size_t error_len);

/* Reduce the most recent device-side matrix result without downloading it. */
bool q38_forward_cuda_greedy_argmax(q38_forward_cuda_context *context,
                                    uint32_t *token, char *error,
                                    size_t error_len);

bool q38_forward_cuda_expert_backend(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, size_t expert, const float *input, float *output,
    void *user, char *error, size_t error_len);
bool q38_forward_cuda_moe_layer_q2_backend(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, const q38_moe_route10 *route,
    const float *host_input, float *host_output, void *user, char *error,
    size_t error_len);

bool q38_forward_cuda_gdn_layer_backend(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_forward_state *state, const float *input, size_t token_count,
    uint32_t layer_number, float *output, void *user, char *error,
    size_t error_len);

/* Device-only layer primitives.  These enqueue work on the context stream;
 * ownership of the boundary transfer and final wait remains with the caller. */
bool q38_forward_cuda_gr_read_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_gr_weights *weights, const float *device_residual,
    float *device_input, float *device_normed, char *error, size_t error_len);
bool q38_forward_cuda_gr_write_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_gr_weights *weights, const float *device_residual,
    float *device_normed, const float *device_block, float *device_updated,
    char *error, size_t error_len);
bool q38_forward_cuda_gdn_layer_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_layer_weights *layer, q38_forward_state *state,
    const float *device_input, uint32_t layer_number, float *device_output,
    char *error, size_t error_len);
bool q38_forward_cuda_qsa_chain_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_layer_weights *layer, q38_qsa_state *state,
    const float *device_input, uint32_t layer_number, float *device_output,
    char *error, size_t error_len);

typedef bool (*q38_forward_cuda_decoder_layer_chain_fn)(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_forward_state *state, uint32_t layer_number, const float *host_input,
    float *host_output, void *user, char *error, size_t error_len);
bool q38_forward_cuda_decoder_layer_chain_backend(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_forward_state *state, uint32_t layer_number, const float *host_input,
    float *host_output, void *user, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
