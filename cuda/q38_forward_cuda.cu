#include "q38_forward_cuda.h"

#include "q38_cuda_primitives.h"
#include "q38_gdn.h"
#include "q38_moe_cuda.h"
#include "q38_qsa_cuda.h"
#include "q38_topk_cuda.h"
#include "q38_gr_ref.h"
#include "q38_diagnostics.h"
#include "q38_residency_plan.h"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>

namespace cg = cooperative_groups;

extern "C" void q38_profile_nvtx_push(const char *name);
extern "C" void q38_profile_nvtx_pop(void);

#if Q38_DIAG_ENABLED
static cudaError_t q38_diag_memcpy_async(void *dst, const void *src,
                                         size_t bytes, cudaMemcpyKind kind,
                                         cudaStream_t stream) {
    const char *name = kind == cudaMemcpyHostToDevice ? "H2D" :
                       kind == cudaMemcpyDeviceToHost ? "D2H" :
                       kind == cudaMemcpyDeviceToDevice ? "D2D" : "MEMCPY";
    q38_profile_nvtx_push(name);
    cudaError_t status = ::cudaMemcpyAsync(dst, src, bytes, kind, stream);
    q38_profile_nvtx_pop();
    return status;
}

static cudaError_t q38_diag_stream_synchronize(cudaStream_t stream) {
    q38_profile_nvtx_push("HOST_WAIT_STREAM");
    cudaError_t status = ::cudaStreamSynchronize(stream);
    q38_profile_nvtx_pop();
    return status;
}

static cudaError_t q38_diag_event_synchronize(cudaEvent_t event) {
    q38_profile_nvtx_push("HOST_WAIT_EVENT");
    cudaError_t status = ::cudaEventSynchronize(event);
    q38_profile_nvtx_pop();
    return status;
}

#define cudaMemcpyAsync q38_diag_memcpy_async
#define cudaStreamSynchronize q38_diag_stream_synchronize
#define cudaEventSynchronize q38_diag_event_synchronize
#endif

#if Q38_DIAGNOSTICS
#define Q38_CUDA_DIAG_ONLY(statement) do { statement; } while (0)
#else
#define Q38_CUDA_DIAG_ONLY(statement) do { } while (0)
#endif

#define Q38_CUDA_DIAG_COLLECT(context) \
    (Q38_DIAG_ENABLED && (context) && (context)->telemetry_observer)

struct persistent_tensor {
    const void *host;
    void *device;
    size_t bytes;
};

typedef enum {
    Q38_STORAGE_RESIDENT,
    Q38_STORAGE_FILE_BACKED_PLE,
} q38_storage_class;

typedef struct {
    const void *ptr;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t qtype;
    uint32_t tensor_id;
    q38_storage_class storage;
    uint64_t gguf_offset;
    const char *name;
} q38_exec_tensor;

struct q38_forward_cuda_context {
    void *device_weights;
    size_t device_weights_bytes;
    float *device_input;
    size_t device_input_elements;
    float *device_output;
    size_t device_output_bytes;
    size_t device_output_elements;
    float *device_hidden_a;
    size_t device_hidden_a_bytes;
    float *device_hidden_b;
    size_t device_hidden_b_bytes;
    uint32_t *device_argmax;
    size_t device_argmax_bytes;
    void *device_aux;
    size_t device_aux_bytes;
    float *device_moe_mid;
    size_t device_moe_mid_bytes;
    float *device_moe_grouped_mid;
    size_t device_moe_grouped_mid_bytes;
    uint16_t *device_moe_route_ids;
    size_t device_moe_route_ids_bytes;
    uint32_t *device_moe_route_indices;
    size_t device_moe_route_indices_bytes;
    float *device_moe_route_weights;
    size_t device_moe_route_weights_bytes;
    float *device_moe_accum;
    size_t device_moe_accum_bytes;
    float *device_moe_expert_outputs;
    size_t device_moe_expert_outputs_bytes;
    float *device_moe_logits;
    size_t device_moe_logits_bytes;
    float *device_moe_shared_gate;
    size_t device_moe_shared_gate_bytes;
    float *device_moe_shared_up;
    size_t device_moe_shared_up_bytes;
    float *device_moe_shared_output;
    size_t device_moe_shared_output_bytes;
    float *device_moe_shared_weight;
    size_t device_moe_shared_weight_bytes;
    float *device_gr_residual;
    size_t device_gr_residual_bytes;
    float *device_gr_norm;
    size_t device_gr_norm_bytes;
    float *device_gr_down;
    size_t device_gr_down_bytes;
    float *device_gr_bottleneck;
    size_t device_gr_bottleneck_bytes;
    float *device_gr_up;
    size_t device_gr_up_bytes;
    float *device_gr_input;
    size_t device_gr_input_bytes;
    float *device_gr_block;
    size_t device_gr_block_bytes;
    float *device_gr_inject;
    size_t device_gr_inject_bytes;
    float *device_gr_updated;
    size_t device_gr_updated_bytes;
    float *device_qsa_input;
    size_t device_qsa_input_bytes;
    float *device_qsa_output;
    size_t device_qsa_output_bytes;
    float *host_qsa_output;
    size_t host_qsa_output_bytes;
    q38_qsa_cuda_chain_state qsa_chain_state[Q38_MODEL_LAYERS];
    q38_qsa_cuda_chain_workspace qsa_chain_workspace;
    bool qsa_chain_workspace_ready;
    float *device_gdn_input;
    size_t device_gdn_input_bytes;
    float *device_gdn_qkv;
    size_t device_gdn_qkv_bytes;
    float *device_gdn_z;
    size_t device_gdn_z_bytes;
    float *device_gdn_a;
    size_t device_gdn_a_bytes;
    float *device_gdn_b;
    size_t device_gdn_b_bytes;
    float *device_gdn_conv;
    size_t device_gdn_conv_bytes;
    float *device_gdn_q;
    size_t device_gdn_q_bytes;
    float *device_gdn_k;
    size_t device_gdn_k_bytes;
    float *device_gdn_v;
    size_t device_gdn_v_bytes;
    float *device_gdn_decay;
    size_t device_gdn_decay_bytes;
    float *device_gdn_beta;
    size_t device_gdn_beta_bytes;
    float *device_gdn_recurrent;
    size_t device_gdn_recurrent_bytes;
    float *device_gdn_gated;
    size_t device_gdn_gated_bytes;
    float *device_gdn_state;
    size_t device_gdn_state_bytes;
    float *device_gdn_history;
    size_t device_gdn_history_bytes;
    bool device_gdn_state_initialized;
    uint64_t gdn_c3_calls;
    uint64_t gdn_c3_launches;
    uint64_t gdn_c3_syncs;
    uint64_t gdn_c3_h2d_bytes;
    uint64_t gdn_c3_d2h_bytes;
    float *device_steering;
    size_t device_steering_bytes;
    float directional_steering_ffn_scale;
    float directional_steering_attn_scale;
    void *lm_head_device_weights;
    size_t lm_head_device_weights_bytes;
    const void *lm_head_host_data;
    bool lm_head_resident;
    bool lm_head_uses_persistent;
    size_t matrix_upload_bytes;
    uint64_t resident_hits;
    uint64_t resident_misses;
    uint64_t cuda_allocations;
    uint64_t cuda_synchronizations;
#if Q38_DIAGNOSTICS
    q38_forward_cuda_sync_stats sync_stats;
    double telemetry_wait_baseline_ms;
#endif
    cudaStream_t stream;
    q38_qsa_candidate_fn qsa_candidate;
    q38_forward_cuda_allocation_observer allocation_observer;
    void *allocation_observer_user;
    q38_forward_cuda_telemetry_observer telemetry_observer;
    void *telemetry_observer_user;
    uint32_t current_layer;
    const char *current_stage;
    persistent_tensor *persistent;
    size_t persistent_count;
    size_t persistent_bytes;
    bool all_non_ple_resident;
    uint64_t persistent_hits;
    uint64_t persistent_misses;
    size_t persistent_expected_bytes;
    uint64_t persistent_expected_tensors;
    uint64_t persistent_duplicate_tensors;
    uint64_t persistent_ple_tensors;
    uint64_t persistent_ple_entries;
    bool persistent_coverage_ok;
    char persistent_failure[256];
    size_t persistent_loaded_bytes;
    uint64_t persistent_loaded_tensors;
    bool exec_strict;
    void *residency_stage;
    size_t residency_stage_bytes;
    void *residency_transfer;
    size_t residency_transfer_bytes;
    void *residency_stage_buffers[2];
    void *residency_transfer_buffers[2];
    cudaEvent_t residency_reuse_events[2];
    bool residency_reuse_events_ready[2];
    uint64_t residency_transfer_calls;
    uint64_t residency_device_copies;
    uint64_t residency_final_syncs;
    uint64_t residency_planned_spans;
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
    q38_residency_span_timing *residency_span_timings;
    size_t residency_span_timing_count;
    const q38_gguf *exec_model;
    q38_exec_tensor *exec_tensors;
    size_t exec_tensor_count;
    uint64_t resident_lookup_in_decode;
    uint64_t gguf_name_lookup_in_decode;
    uint64_t non_ple_residency_miss;
    size_t non_ple_upload_bytes_per_token;
    uint64_t ple_file_backed_accesses;
    size_t ple_file_bytes;
    float gpu_argmax_kernel_ms;
    q38_forward_cuda_residency_progress_observer progress_observer;
    void *progress_observer_user;
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
};

__global__ static void q38_directional_steering_kernel(
        float *values, const float *directions, uint32_t layer,
        uint32_t width, uint32_t rows, float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || !width || !scale) return;
    float *value = values + (uint64_t)row * width;
    const float *direction = directions + (uint64_t)layer * width;
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        dot += value[i] * direction[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float coefficient = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x)
        value[i] -= coefficient * direction[i];
}

__device__ static float gr_bf16_value(uint16_t bits) {
    return __uint_as_float((uint32_t)bits << 16);
}

__device__ static float gr_silu(float value) {
    value /= 4.0f;
    return value / (1.0f + expf(-value));
}

__global__ static void gr_fused_normalize_down_kernel(
    const float *residual, const uint16_t *gamma, const uint16_t *weights,
    float *norm_sums, float *normalized, float *down) {
    cg::grid_group grid = cg::this_grid();
    __shared__ double partial[128];
    if (blockIdx.x < Q38_GR_BRANCHES) {
        const unsigned branch = blockIdx.x;
        double sum = 0.0;
        for (unsigned channel = threadIdx.x; channel < Q38_GR_HIDDEN;
             channel += blockDim.x) {
            const float value =
                residual[branch * Q38_GR_HIDDEN + channel];
            sum += (double)value * (double)value;
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned stride = 64; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            norm_sums[branch] =
                (float)(partial[0] / (double)Q38_GR_HIDDEN) + 1e-6f;
    }
    grid.sync();

    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < Q38_GR_BRANCHES * Q38_GR_HIDDEN;
         index += (size_t)gridDim.x * blockDim.x) {
        const unsigned branch = (unsigned)(index / Q38_GR_HIDDEN);
        normalized[index] =
            residual[index] * rsqrtf(norm_sums[branch]) *
            (1.0f + gr_bf16_value(gamma[index]));
    }
    grid.sync();

    __shared__ float warp_sums[4];
    for (size_t row = blockIdx.x; row < Q38_GR_RANK; row += gridDim.x) {
        const unsigned lane = threadIdx.x & 31u;
        const unsigned warp = threadIdx.x >> 5;
        float sum = 0.0f;
        for (size_t col = threadIdx.x;
             col < Q38_GR_BRANCHES * Q38_GR_HIDDEN;
             col += blockDim.x)
            sum += gr_bf16_value(weights[row * Q38_GR_BRANCHES *
                                             Q38_GR_HIDDEN + col]) *
                   normalized[col];
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < 4 ? warp_sums[lane] : 0.0f;
            for (unsigned offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0) down[row] = sum;
        }
        __syncthreads();
    }
}

__global__ static void gr_fused_lowrank_up_kernel(
    const uint16_t *weights, const float *down, float *bottleneck,
    float *up) {
    cg::grid_group grid = cg::this_grid();
    for (size_t rank = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         rank < Q38_GR_RANK; rank += (size_t)gridDim.x * blockDim.x)
        bottleneck[rank] = gr_silu(down[rank]);
    grid.sync();

    __shared__ float warp_sums[4];
    for (size_t row = blockIdx.x;
         row < Q38_GR_BRANCHES * Q38_GR_HIDDEN; row += gridDim.x) {
        const unsigned lane = threadIdx.x & 31u;
        const unsigned warp = threadIdx.x >> 5;
        float sum = 0.0f;
        for (size_t col = threadIdx.x; col < Q38_GR_RANK;
             col += blockDim.x)
            sum += gr_bf16_value(
                       weights[row * Q38_GR_RANK + col]) *
                   bottleneck[col];
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < 4 ? warp_sums[lane] : 0.0f;
            for (unsigned offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (lane == 0) up[row] = sum;
        }
        __syncthreads();
    }
}

__global__ static void gr_fused_branch_read_kernel(
    const float *normalized, const float *up, float *input) {
    const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= Q38_GR_HIDDEN) return;
    float value = 0.0f;
    for (unsigned branch = 0; branch < Q38_GR_BRANCHES; ++branch) {
        const size_t index = branch * Q38_GR_HIDDEN + channel;
        value += (1.0f / (1.0f + expf(-up[index]))) * normalized[index];
    }
    input[channel] = value / (float)Q38_GR_BRANCHES;
}

__global__ static void gr_normalize_kernel(
    const float *residual, const uint16_t *gamma, float *norm_sums,
    float *normalized) {
    const unsigned branch = blockIdx.x;
    if (branch >= Q38_GR_BRANCHES) return;
    __shared__ double partial[256];
    double sum = 0.0;
    for (unsigned channel = threadIdx.x; channel < Q38_GR_HIDDEN;
         channel += blockDim.x) {
        const float value = residual[branch * Q38_GR_HIDDEN + channel];
        sum += (double)value * (double)value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        norm_sums[branch] =
            (float)(partial[0] / (double)Q38_GR_HIDDEN) + 1e-6f;
    __syncthreads();
    for (unsigned channel = threadIdx.x; channel < Q38_GR_HIDDEN;
         channel += blockDim.x) {
        const size_t index = branch * Q38_GR_HIDDEN + channel;
        normalized[index] =
            residual[index] * rsqrtf(norm_sums[branch]) *
            (1.0f + gr_bf16_value(gamma[index]));
    }
}

__global__ static void gr_fused_writeback_kernel(
    const float *residual, const float *block, const float *inject,
    float *updated) {
    const size_t index =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Q38_GR_BRANCHES * Q38_GR_HIDDEN) return;
    const unsigned branch = index / Q38_GR_HIDDEN;
    const float scale = 2.0f /
        (1.0f + expf(-inject[branch] / 4.0f));
    updated[index] = residual[index] +
                     scale * block[index % Q38_GR_HIDDEN];
}

__global__ static void moe_shared_silu_mul_kernel(
    const float *gate, const float *up, float *output, size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = gate[index];
    output[index] = value / (1.0f + expf(-value)) * up[index];
}

__global__ static void moe_shared_add_kernel(
    const float *routed, const float *shared, const float *gate,
    float *output, size_t elements) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float scale = 1.0f / (1.0f + expf(-gate[0]));
    output[index] = routed[index] + scale * shared[index];
}

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static bool is_lm_head_tensor(const q38_tensor *tensor) {
    return tensor && tensor->name.len == 14 &&
           memcmp(tensor->name.ptr, "lm_head.weight", 14) == 0;
}


static bool tensor_shape(const q38_tensor *tensor, size_t *rows, size_t *cols) {
    if (!tensor || !rows || !cols || !tensor->ndim || tensor->ndim > 3)
        return false;
    size_t r = 1;
    for (uint32_t i = 0; i + 1 < tensor->ndim; ++i) {
        if (!tensor->dim[i] || r > SIZE_MAX / (size_t)tensor->dim[i])
            return false;
        r *= (size_t)tensor->dim[i];
    }
    if (!tensor->dim[tensor->ndim - 1] ||
        tensor->dim[tensor->ndim - 1] > SIZE_MAX)
        return false;
    *rows = r;
    *cols = (size_t)tensor->dim[tensor->ndim - 1];
    return true;
}

static bool ensure_buffer(void **buffer, size_t *capacity, size_t bytes,
                          q38_forward_cuda_allocation_observer observer,
                          void *observer_user, uint64_t *allocation_count);

static const char *subsystem_for_stage(const char *stage) {
    if (stage && strstr(stage, "qsa")) return "qsa";
    if (stage && (strstr(stage, "moe") || strstr(stage, "expert") ||
                  strstr(stage, "router"))) return "moe";
    if (stage && strstr(stage, "ple")) return "ple";
    if (stage && strstr(stage, "lm_head")) return "lm_head";
    if (stage && strstr(stage, "gdn")) return "gdn";
    if (stage && strstr(stage, "gr_")) return "gr";
    return "unknown";
}

static bool is_gr_projection_stage(const char *stage) {
    return stage &&
           (!strcmp(stage, "gr_read_down") ||
            !strcmp(stage, "gr_read_up") ||
            !strcmp(stage, "gr_write_inject"));
}

static bool is_gdn_projection_stage(const char *stage) {
    return stage &&
           (!strcmp(stage, "gdn_qkv_projection") ||
            !strcmp(stage, "gdn_z_projection") ||
            !strcmp(stage, "gdn_a_projection") ||
            !strcmp(stage, "gdn_b_projection") ||
            !strcmp(stage, "gdn_output_projection"));
}

static void copy_tensor_name(const q38_tensor *tensor, char *out,
                             size_t out_len) {
    if (!out || !out_len) return;
    size_t n = tensor && tensor->name.len < out_len - 1
        ? (size_t)tensor->name.len : out_len - 1;
    if (tensor && tensor->name.ptr && n) memcpy(out, tensor->name.ptr, n);
    out[n] = '\0';
}

static uint32_t tensor_id_for(const q38_forward_cuda_context *context,
                              const q38_gguf *model,
                              const q38_tensor *tensor) {
    if (!context || !model || !tensor || model != context->exec_model ||
        !context->exec_tensors || tensor < model->tensors ||
        tensor >= model->tensors + model->n_tensors)
        return UINT32_MAX;
    return context->exec_tensors[tensor - model->tensors].tensor_id;
}

static bool is_ple_embedding_table(const q38_tensor *tensor) {
    if (!tensor || !tensor->name.ptr) return false;
    const char *name = tensor->name.ptr;
    const size_t len = (size_t)tensor->name.len;
    return len >= 41 &&
           memmem(name, len,
                  ".ple.ple_embedding.ngram_embedding.shard_", 41) != NULL;
}

static bool residency_plan_is_ple(const q38_tensor *tensor, void *user) {
    (void)user;
    return is_ple_embedding_table(tensor);
}

#if Q38_DIAGNOSTICS
static double residency_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static uint64_t residency_mincore_pages(const q38_gguf *model) {
    if (!model || !model->map || !model->size) return 0;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t address = (uintptr_t)model->map;
    const uintptr_t base = address & ~(uintptr_t)(page - 1);
    const size_t offset = (size_t)(address - base);
    const size_t pages = (offset + model->size + page - 1) / page;
    unsigned char *vec = (unsigned char *)calloc(pages, 1);
    if (!vec) return 0;
    const uint64_t result =
        mincore((void *)base, pages * page, vec) == 0
            ? [&]() {
                uint64_t resident = 0;
                for (size_t i = 0; i < pages; ++i)
                    resident += (vec[i] & 1u) != 0;
                return resident;
            }()
            : 0;
    free(vec);
    return result;
}
#endif

static const char *residency_group(const q38_tensor *tensor) {
    if (!tensor || !tensor->name.ptr) return "unknown";
    const char *name = tensor->name.ptr;
    size_t len = (size_t)tensor->name.len;
    if (memmem(name, len, "embed_tokens", 12)) return "embedding";
    if (memmem(name, len, "hyper_connection", 17)) return "GR";
    if (memmem(name, len, "linear_attn", 11)) return "GDN";
    if (memmem(name, len, "self_attn", 9) || memmem(name, len, "indexer", 7))
        return "QSA";
    if (memmem(name, len, ".mlp.gate.weight", 16)) return "router";
    if (memmem(name, len, "shared_expert", 13)) return "shared_experts";
    if (memmem(name, len, "experts.", 8)) return "routed_experts_Q2";
    if (memmem(name, len, "lm_head.weight", 14)) return "LM-head";
    return "unknown";
}

static persistent_tensor *persistent_find(q38_forward_cuda_context *context,
                                          const void *host) {
    if (!context || !host) return NULL;
    for (size_t i = 0; i < context->persistent_count; ++i)
        if (context->persistent[i].host == host)
            return &context->persistent[i];
    return NULL;
}

static q38_exec_tensor *exec_tensor_for(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_tensor *tensor) {
    if (!context || !model || model != context->exec_model || !tensor ||
        !context->exec_tensors || tensor < model->tensors ||
        tensor >= model->tensors + model->n_tensors)
        return NULL;
    return &context->exec_tensors[tensor - model->tensors];
}

static bool exec_tensor_is_resident(const q38_exec_tensor *exec,
                                    const q38_tensor *tensor) {
    return exec && exec->storage == Q38_STORAGE_RESIDENT && exec->ptr &&
           exec->bytes == tensor->bytes;
}

#if Q38_DIAGNOSTICS
static double host_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static double cuda_sync_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void record_cuda_sync(q38_forward_cuda_context *context,
                             q38_forward_cuda_sync_reason reason,
                             double elapsed_ms) {
    if (!context || reason >= Q38_CUDA_SYNC_REASON_COUNT ||
        elapsed_ms < 0.0)
        return;
    context->sync_stats.real_cuda_sync_count++;
    context->sync_stats.host_blocked_on_cuda_ms += elapsed_ms;
    context->sync_stats.reason_count[reason]++;
    context->sync_stats.reason_ms[reason] += elapsed_ms;
    if (elapsed_ms > context->sync_stats.reason_max_ms[reason])
        context->sync_stats.reason_max_ms[reason] = elapsed_ms;
}

#define Q38_CUDA_SYNC_CALL(context, reason, call) \
    ([&]() { \
        const double q38_sync_started = cuda_sync_now_ms(); \
        const cudaError_t q38_sync_status = (call); \
        record_cuda_sync((context), (reason), \
                         cuda_sync_now_ms() - q38_sync_started); \
        return q38_sync_status; \
    }())
#else
#define host_now_ms() 0.0
#define Q38_CUDA_SYNC_CALL(context, reason, call) (call)
#endif

#if Q38_DIAGNOSTICS
static float event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    return cudaEventElapsedTime(&ms, start, stop) == cudaSuccess ? ms : 0.0f;
}
#else
#define event_elapsed(...) 0.0f
#endif

#if Q38_DIAGNOSTICS
static bool telemetry_events_create(bool collect, cudaEvent_t *upload_start,
                                    cudaEvent_t *upload_stop,
                                    cudaEvent_t *kernel_start,
                                    cudaEvent_t *kernel_stop) {
    if (!collect) return true;
    if (cudaEventCreate(upload_start) == cudaSuccess &&
        cudaEventCreate(upload_stop) == cudaSuccess &&
        cudaEventCreate(kernel_start) == cudaSuccess &&
        cudaEventCreate(kernel_stop) == cudaSuccess)
        return true;
    if (*upload_start) cudaEventDestroy(*upload_start);
    if (*upload_stop) cudaEventDestroy(*upload_stop);
    if (*kernel_start) cudaEventDestroy(*kernel_start);
    if (*kernel_stop) cudaEventDestroy(*kernel_stop);
    *upload_start = NULL;
    *upload_stop = NULL;
    *kernel_start = NULL;
    *kernel_stop = NULL;
    return false;
}

static void telemetry_events_destroy(cudaEvent_t upload_start,
                                     cudaEvent_t upload_stop,
                                     cudaEvent_t kernel_start,
                                     cudaEvent_t kernel_stop) {
    if (upload_start) cudaEventDestroy(upload_start);
    if (upload_stop) cudaEventDestroy(upload_stop);
    if (kernel_start) cudaEventDestroy(kernel_start);
    if (kernel_stop) cudaEventDestroy(kernel_stop);
}

static bool telemetry_event_record(bool collect, cudaEvent_t event,
                                   cudaStream_t stream) {
    return !collect || cudaEventRecord(event, stream) == cudaSuccess;
}
#else
#define telemetry_events_create(...) true
#define telemetry_events_destroy(...) do { } while (0)
#define telemetry_event_record(...) true
#endif

#if Q38_DIAGNOSTICS
static void emit_telemetry(q38_forward_cuda_context *context,
                           const q38_gguf *model, const q38_tensor *tensor,
                           size_t rows, size_t cols,
                           size_t bytes, bool hit, bool miss,
                           size_t upload_bytes, float upload_ms,
                           float kernel_ms, double wall_ms,
                           uint64_t allocations, uint64_t syncs,
                           const char *operation,
                           const char *fallback_path) {
    if (!context) return;
    const bool ple = is_ple_embedding_table(tensor);
    if (ple && (miss || upload_bytes)) {
        Q38_CUDA_DIAG_ONLY(++context->ple_file_backed_accesses);
        Q38_CUDA_DIAG_ONLY(context->ple_file_bytes += upload_bytes);
    }
    if (!context->telemetry_observer) return;
    char name[128];
    copy_tensor_name(tensor, name, sizeof(name));
    const double total_wait_ms = context->sync_stats.host_blocked_on_cuda_ms;
    const double callback_wait_ms =
        fmax(0.0, total_wait_ms - context->telemetry_wait_baseline_ms);
    context->telemetry_wait_baseline_ms = total_wait_ms;
    q38_forward_cuda_telemetry record = {
        subsystem_for_stage(context->current_stage), context->current_layer,
        context->current_stage ? context->current_stage : "backend",
        operation ? operation : "unknown",
        fallback_path ? fallback_path : "none",
        name, tensor_id_for(context, model, tensor),
        tensor ? tensor->type : 0, rows, cols, bytes,
        ple ? false : hit, ple ? false : miss, ple ? false : miss,
        ple && (miss || upload_bytes), ple ? 0 : upload_bytes,
        ple ? upload_bytes : 0,
        bytes,
        (kernel_ms > 0.0f || wall_ms > 0.0) ? cols * sizeof(float) : 0,
        (kernel_ms > 0.0f || wall_ms > 0.0) ? rows * sizeof(float) : 0,
        (kernel_ms > 0.0f || wall_ms > 0.0) ? rows * sizeof(float) : 0,
        upload_ms, kernel_ms,
        (float)fmax(0.0, wall_ms - (double)upload_ms - (double)kernel_ms),
        allocations, syncs, syncs,
        (float)wall_ms, (float)callback_wait_ms
    };
#if Q38_DIAGNOSTICS
    const double callback_started = cuda_sync_now_ms();
#endif
    context->telemetry_observer(&record, context->telemetry_observer_user);
#if Q38_DIAGNOSTICS
    context->sync_stats.telemetry_callback_wall_ms +=
        cuda_sync_now_ms() - callback_started;
#endif
}
#else
#define emit_telemetry(...) do { } while (0)
#endif

extern "C" bool q38_forward_cuda_expert_backend(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, size_t expert, const float *input, float *output,
    void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t gate_rows, gate_cols, down_rows, down_cols;
    if (!context || !model || !gate_up || !down || !input || !output ||
        !tensor_shape(gate_up, &gate_rows, &gate_cols) ||
        !tensor_shape(down, &down_rows, &down_cols) ||
        (gate_up->type != 10 && gate_up->type != 12) ||
        down->type != gate_up->type ||
        gate_cols != 2560 || down_cols != 2560 ||
        gate_rows % 1280 != 0 || down_rows % 640 != 0 ||
        expert >= gate_rows / 1280 || expert >= down_rows / 640)
        return fail(error, error_len, "unsupported CUDA routed expert geometry");
    const size_t gate_row_bytes = (size_t)(gate_up->bytes / gate_rows);
    const size_t down_row_bytes = (size_t)(down->bytes / down_rows);
    const size_t gate_bytes = 1280u * gate_row_bytes;
    const size_t down_bytes = 640u * down_row_bytes;
    q38_exec_tensor *gate_exec = exec_tensor_for(context, model, gate_up);
    q38_exec_tensor *down_exec = exec_tensor_for(context, model, down);
    const bool use_persistent =
        context->all_non_ple_resident &&
        exec_tensor_is_resident(gate_exec, gate_up) &&
        exec_tensor_is_resident(down_exec, down);
    if (context->exec_strict && !use_persistent)
        return fail(error, error_len,
                    "Q38_EXEC_STRICT: routed expert tensor is not resident");
    const void *gate_data = use_persistent ? NULL :
        q38_gguf_tensor_data(model, gate_up);
    const void *down_data = use_persistent ? NULL :
        q38_gguf_tensor_data(model, down);
    if (!use_persistent) Q38_CUDA_DIAG_ONLY(context->gguf_name_lookup_in_decode += 2);
    if (!use_persistent && (!gate_data || !down_data))
        return fail(error, error_len, "invalid CUDA routed expert payload");
    if (context->all_non_ple_resident) {
        if (use_persistent) Q38_CUDA_DIAG_ONLY(++context->persistent_hits);
        else {
            Q38_CUDA_DIAG_ONLY(++context->persistent_misses);
            Q38_CUDA_DIAG_ONLY(++context->non_ple_residency_miss);
        }
    }
    const bool collect_telemetry = Q38_CUDA_DIAG_COLLECT(context);
    const double host_started = collect_telemetry ? host_now_ms() : 0.0;
    cudaEvent_t upload_start = NULL, upload_stop = NULL;
    cudaEvent_t kernel_start = NULL, kernel_stop = NULL;
    cudaEvent_t gate_start = NULL, gate_stop = NULL;
    cudaEvent_t down_start = NULL, down_stop = NULL;
    if (!telemetry_events_create(collect_telemetry, &upload_start,
                                 &upload_stop, &kernel_start, &kernel_stop))
        return fail(error, error_len, "CUDA telemetry event allocation failed");
    if (collect_telemetry && gate_up->type == Q38_QUANT_Q2_K &&
        (cudaEventCreate(&gate_start) != cudaSuccess ||
         cudaEventCreate(&gate_stop) != cudaSuccess ||
         cudaEventCreate(&down_start) != cudaSuccess ||
         cudaEventCreate(&down_stop) != cudaSuccess))
        return fail(error, error_len, "CUDA expert event allocation failed");
    const uint64_t allocation_before = context->cuda_allocations;
    if ((!use_persistent &&
        !ensure_buffer(&context->device_weights, &context->device_weights_bytes,
                       gate_bytes, context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) ||
        (!use_persistent &&
        !ensure_buffer(&context->device_aux, &context->device_aux_bytes,
                       down_bytes, context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements, 2560u * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes, 2560u * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_mid,
                       &context->device_moe_mid_bytes,
                       Q38_MOE_INTERMEDIATE * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "CUDA routed expert allocation failed");
    const unsigned char *gate_src = gate_data
        ? (const unsigned char *)gate_data + expert * 1280u * gate_row_bytes
        : NULL;
    const unsigned char *down_src = down_data
        ? (const unsigned char *)down_data + expert * 640u * down_row_bytes
        : NULL;
    void *gate_storage = use_persistent
        ? (unsigned char *)gate_exec->ptr + expert * 1280u * gate_row_bytes
        : context->device_weights;
    void *down_storage = use_persistent
        ? (unsigned char *)down_exec->ptr + expert * 640u * down_row_bytes
        : context->device_aux;
    if (!telemetry_event_record(collect_telemetry, upload_start, context->stream) ||
        (!use_persistent &&
         (cudaMemcpyAsync(gate_storage, gate_src, gate_bytes,
                          cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
          cudaMemcpyAsync(down_storage, down_src, down_bytes,
                          cudaMemcpyHostToDevice, context->stream) != cudaSuccess)) ||
        !telemetry_event_record(collect_telemetry, upload_stop, context->stream) ||
        cudaMemcpyAsync(context->device_input, input, 2560u * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA routed expert upload failed");
    if (!use_persistent)
        Q38_CUDA_DIAG_ONLY(context->non_ple_upload_bytes_per_token += gate_bytes + down_bytes);
    if (!telemetry_event_record(collect_telemetry, kernel_start, context->stream))
        return fail(error, error_len, "CUDA routed expert execution failed");
    bool launched;
    if (gate_up->type == Q38_QUANT_Q2_K) {
        if (!telemetry_event_record(collect_telemetry, gate_start, context->stream) ||
            !q38_moe_cuda_q2_gate_up(
                gate_storage, context->device_input, context->device_moe_mid,
                context->stream, error, error_len) ||
            !telemetry_event_record(collect_telemetry, gate_stop, context->stream) ||
            !telemetry_event_record(collect_telemetry, down_start, context->stream) ||
            !q38_moe_cuda_q2_down(
                down_storage, context->device_moe_mid, context->device_output,
                context->stream, error, error_len) ||
            !telemetry_event_record(collect_telemetry, down_stop, context->stream)) {
            Q38_CUDA_DIAG_ONLY(++context->q2_gate_up_fallback_calls);
            launched = false;
        } else {
            launched = true;
        }
    } else {
        launched = q38_moe_cuda_expert_q4_workspace(
            gate_storage, down_storage, context->device_input,
            context->device_output, context->device_moe_mid, context->stream,
            error, error_len);
    }
    if (!launched)
        return false;
    if (!telemetry_event_record(collect_telemetry, kernel_stop, context->stream) ||
        cudaMemcpyAsync(output, context->device_output, 2560u * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_MOE_ROUTED_D2H,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "CUDA routed expert execution failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    const double expert_wall_ms = collect_telemetry ? host_now_ms() - host_started : 0.0;
    Q38_CUDA_DIAG_ONLY(context->expert_backend_total_wall_ms += expert_wall_ms);
    Q38_CUDA_DIAG_ONLY(++context->expert_host_sync_count);
    Q38_CUDA_DIAG_ONLY(context->expert_H2D_bytes += 2560u * sizeof(float));
    Q38_CUDA_DIAG_ONLY(context->expert_D2H_bytes += 2560u * sizeof(float));
    if (gate_up->type == Q38_QUANT_Q2_K) {
        Q38_CUDA_DIAG_ONLY(++context->q2_gate_up_fast_calls);
        Q38_CUDA_DIAG_ONLY(++context->q2_down_calls);
        if (context->current_layer < Q38_MODEL_LAYERS)
            Q38_CUDA_DIAG_ONLY(++context->expert_fast_calls_by_layer[context->current_layer]);
        Q38_CUDA_DIAG_ONLY(if (collect_telemetry)
            context->q2_gate_up_fast_total_kernel_ms +=
                event_elapsed(gate_start, gate_stop));
        Q38_CUDA_DIAG_ONLY(if (collect_telemetry)
            context->q2_down_total_kernel_ms +=
                event_elapsed(down_start, down_stop));
    }
    emit_telemetry(context, model, gate_up, gate_rows, gate_cols, gate_bytes,

                   use_persistent, !use_persistent, use_persistent ? 0 : gate_bytes,
                   collect_telemetry ? event_elapsed(upload_start, upload_stop) : 0.0f,
                   collect_telemetry ? event_elapsed(kernel_start, kernel_stop) : 0.0f,
                   expert_wall_ms,
               context->cuda_allocations - allocation_before, 1,
               "routed_expert", use_persistent
                   ? "resident_exec_tensor" : "gguf_host_upload");
    emit_telemetry(context, model, down, down_rows, down_cols, down_bytes,
               use_persistent, !use_persistent, use_persistent ? 0 : down_bytes, 0.0f, 0.0f, 0.0,
               0, 0, "routed_expert", use_persistent
                   ? "resident_exec_tensor" : "gguf_host_upload");
    telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                             kernel_stop);
    if (gate_start) cudaEventDestroy(gate_start);
    if (gate_stop) cudaEventDestroy(gate_stop);
    if (down_start) cudaEventDestroy(down_start);
    if (down_stop) cudaEventDestroy(down_stop);
    return true;
}

extern "C" bool q38_forward_cuda_moe_layer_q2_backend(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, const q38_moe_route10 *route,
    const float *host_input, float *host_output, void *user, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t gate_rows, gate_cols, down_rows, down_cols;
    if (!context || !model || !gate_up || !down || !route ||
        !host_input || !host_output ||
        gate_up->type != Q38_QUANT_Q2_K ||
        down->type != Q38_QUANT_Q2_K ||
        !tensor_shape(gate_up, &gate_rows, &gate_cols) ||
        !tensor_shape(down, &down_rows, &down_cols) ||
        gate_cols != 2560 || down_cols != 2560 ||
        gate_rows % 1280 != 0 || down_rows % 640 != 0)
        return fail(error, error_len, "unsupported CUDA Q2 MoE layer geometry");

    q38_exec_tensor *gate_exec = exec_tensor_for(context, model, gate_up);
    q38_exec_tensor *down_exec = exec_tensor_for(context, model, down);
    if (!exec_tensor_is_resident(gate_exec, gate_up) ||
        !exec_tensor_is_resident(down_exec, down))
        return fail(error, error_len,
                    "Q38 Q2 MoE layer requires resident expert weights");

    if (!ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       Q38_MOE_HIDDEN * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes,
                       Q38_MOE_HIDDEN * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_mid,
                       &context->device_moe_mid_bytes,
                       Q38_MOE_INTERMEDIATE * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_grouped_mid,
                       &context->device_moe_grouped_mid_bytes,
                       Q38_MOE_TOP_K * Q38_MOE_INTERMEDIATE * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_route_ids,
                       &context->device_moe_route_ids_bytes,
                       Q38_MOE_TOP_K * sizeof(uint16_t),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_route_weights,
                       &context->device_moe_route_weights_bytes,
                       Q38_MOE_TOP_K * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_accum,
                       &context->device_moe_accum_bytes,
                       Q38_MOE_HIDDEN * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_moe_expert_outputs,
                       &context->device_moe_expert_outputs_bytes,
                       Q38_MOE_TOP_K * Q38_MOE_HIDDEN * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "CUDA Q2 MoE layer allocation failed");

    const double started = host_now_ms();
    uint16_t route_ids[Q38_MOE_TOP_K];
    for (unsigned k = 0; k < Q38_MOE_TOP_K; ++k) {
        const unsigned e = route->expert[k];
        if (e >= gate_rows / 1280 || e >= down_rows / 640)
            return fail(error, error_len, "invalid Q2 routed expert ID");
        route_ids[k] = (uint16_t)e;
    }
    if (cudaMemcpyAsync(context->device_input, host_input,
                        Q38_MOE_HIDDEN * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_moe_route_ids, route_ids,
                        sizeof(route_ids), cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_moe_route_weights, route->weight,
                        Q38_MOE_TOP_K * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        !q38_moe_cuda_q2_grouped_indexed_deterministic(
            gate_exec->ptr, down_exec->ptr, context->device_input,
            context->device_moe_route_ids, context->device_moe_route_weights,
            Q38_MOE_TOP_K, 1280u * 10u, 640u * 10u,
            context->device_moe_accum, context->device_moe_grouped_mid,
            context->device_moe_expert_outputs,
            context->stream, error, error_len) ||
        cudaMemcpyAsync(host_output, context->device_moe_accum,
                        Q38_MOE_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_MOE_GROUPED_D2H,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "CUDA grouped Q2 MoE layer execution failed");

    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    Q38_CUDA_DIAG_ONLY(++context->expert_host_sync_count);
    Q38_CUDA_DIAG_ONLY(context->expert_backend_total_wall_ms += host_now_ms() - started);
    Q38_CUDA_DIAG_ONLY(context->expert_H2D_bytes +=
        Q38_MOE_HIDDEN * sizeof(float) +
        Q38_MOE_TOP_K * (sizeof(uint16_t) + sizeof(float)));
    Q38_CUDA_DIAG_ONLY(context->expert_D2H_bytes += Q38_MOE_HIDDEN * sizeof(float));
    Q38_CUDA_DIAG_ONLY(context->q2_gate_up_fast_calls += Q38_MOE_TOP_K);
    Q38_CUDA_DIAG_ONLY(context->q2_down_calls += Q38_MOE_TOP_K);
    Q38_CUDA_DIAG_ONLY(context->expert_kernel_launches += 3);
    Q38_CUDA_DIAG_ONLY(context->persistent_hits += 2);
    if (context->current_layer < Q38_MODEL_LAYERS)
        context->expert_fast_calls_by_layer[context->current_layer] +=
            Q38_MOE_TOP_K;
    return true;
}

static bool ensure_buffer(void **buffer, size_t *capacity, size_t bytes,
                          q38_forward_cuda_allocation_observer observer,
                          void *observer_user, uint64_t *allocation_count) {
    if (*buffer && *capacity >= bytes) return true;
    if (*buffer) cudaFree(*buffer);
    *buffer = NULL;
    *capacity = 0;
    if (cudaMalloc(buffer, bytes) != cudaSuccess) return false;
    Q38_CUDA_DIAG_ONLY(if (observer) observer(bytes, observer_user));
    Q38_CUDA_DIAG_ONLY(if (allocation_count) ++*allocation_count);
    *capacity = bytes;
    return true;
}

extern "C" q38_forward_cuda_context *
q38_forward_cuda_context_create(char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)calloc(1, sizeof(*context));
    if (!context) {
        fail(error, error_len, "CUDA forward context allocation failed");
        return NULL;
    }
    context->exec_strict = getenv("Q38_EXEC_STRICT") != NULL;
    if (cudaStreamCreate(&context->stream) != cudaSuccess) {
        free(context);
        fail(error, error_len, "CUDA forward stream creation failed");
        return NULL;
    }
    return context;
}

extern "C" void
q38_forward_cuda_reset_gdn_state(q38_forward_cuda_context *context) {
    if (!context || !context->device_gdn_state ||
        !context->device_gdn_history)
        return;
    const size_t state_bytes =
        (size_t)Q38_GDN_LAYER_COUNT * Q38_GDN_VALUE_HEADS *
        Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM * sizeof(float);
    const size_t history_bytes =
        (size_t)Q38_GDN_LAYER_COUNT * (Q38_GDN_CONV_KERNEL - 1u) *
        Q38_GDN_CONV_CHANNELS * sizeof(float);
    if (cudaMemsetAsync(context->device_gdn_state, 0, state_bytes,
                        context->stream) == cudaSuccess &&
        cudaMemsetAsync(context->device_gdn_history, 0, history_bytes,
                        context->stream) == cudaSuccess)
        context->device_gdn_state_initialized = true;
}

extern "C" bool q38_forward_cuda_load_gdn_state(
    const q38_forward_state *state, void *user, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    if (!state || !context || !state->storage.recurrent_state ||
        !state->storage.conv_history)
        return fail(error, error_len, "invalid GDN state upload arguments");
    const size_t state_bytes = (size_t)state->storage.layout.recurrent.bytes;
    const size_t history_bytes =
        (size_t)state->storage.layout.conv_history.bytes;
    if (!ensure_buffer((void **)&context->device_gdn_state,
                       &context->device_gdn_state_bytes, state_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_gdn_history,
                       &context->device_gdn_history_bytes, history_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "GDN state upload storage allocation failed");
    if (cudaMemcpyAsync(context->device_gdn_state,
                        state->storage.recurrent_state, state_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_gdn_history,
                        state->storage.conv_history, history_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "GDN state upload failed");
    context->device_gdn_state_initialized = true;
    return true;
}

extern "C" bool q38_forward_cuda_sync_gdn_state(
    q38_forward_state *state, void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    if (!state || !context)
        return fail(error, error_len, "invalid GDN state sync arguments");
    if (!context->device_gdn_state_initialized)
        return true;
    if (!context->device_gdn_state || !context->device_gdn_history ||
        !state->storage.recurrent_state || !state->storage.conv_history)
        return fail(error, error_len, "GDN state sync storage is unavailable");
    const size_t state_bytes =
        (size_t)state->storage.layout.recurrent.bytes;
    const size_t history_bytes =
        (size_t)state->storage.layout.conv_history.bytes;
    if (state_bytes > context->device_gdn_state_bytes ||
        history_bytes > context->device_gdn_history_bytes)
        return fail(error, error_len, "GDN state sync layout exceeds device storage");
    if (cudaMemcpyAsync(state->storage.recurrent_state,
                        context->device_gdn_state, state_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(state->storage.conv_history,
                        context->device_gdn_history, history_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_GDN_TRACE_STATE,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "GDN state sync transfer failed");
    return true;
}

extern "C" bool q38_forward_cuda_load_qsa_state(
    const q38_forward_state *state, void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    if (!state || !context)
        return fail(error, error_len, "invalid QSA state upload arguments");
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        const q38_qsa_state *host = &state->qsa[layer];
        q38_qsa_cuda_chain_state *device = &context->qsa_chain_state[layer];
        if (!host->position) {
            q38_qsa_cuda_chain_reset(device);
            continue;
        }
        if (host->main_k.count != host->position ||
            host->main_v.count != host->position ||
            host->index_k.count != host->position ||
            !q38_qsa_cuda_chain_reserve(
                device, host->position, context->stream, error, error_len) ||
            cudaMemcpyAsync(
                device->main_k, host->main_k.data,
                host->main_k.count * host->main_k.row_bytes,
                cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
            cudaMemcpyAsync(
                device->main_v, host->main_v.data,
                host->main_v.count * host->main_v.row_bytes,
                cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
            cudaMemcpyAsync(
                device->index_k, host->index_k.data,
                host->index_k.count * host->index_k.row_bytes,
                cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
            return fail(error, error_len, "QSA state upload failed");
        device->count = host->position;
        device->position = host->position;
    }
    return true;
}

extern "C" bool q38_forward_cuda_enable_all_non_ple_residency(
    q38_forward_cuda_context *context, const q38_gguf *model,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !model)
        return fail(error, error_len, "invalid all-non-PLE residency arguments");
    if (context->all_non_ple_resident) return true;
#if Q38_DIAGNOSTICS
    struct rusage usage_before = {};
    getrusage(RUSAGE_SELF, &usage_before);
    context->residency_mincore_pages_before = residency_mincore_pages(model);
    const double plan_started = residency_now_ms();
#endif
    q38_residency_plan plan;
    q38_residency_plan_init(&plan);
    if (!q38_residency_plan_build(
            model, residency_plan_is_ple, NULL, 64u * 1024u,
            256u * 1024u * 1024u, &plan, error, error_len))
        return false;
#if Q38_DIAGNOSTICS
    context->residency_plan_ms = residency_now_ms() - plan_started;
    context->residency_planned_bytes = (size_t)plan.resident_bytes;
    context->residency_span_timing_count = plan.span_count;
    context->residency_span_timings = (q38_residency_span_timing *)calloc(
        plan.span_count ? plan.span_count : 1,
        sizeof(*context->residency_span_timings));
    if (!context->residency_span_timings) {
        q38_residency_plan_destroy(&plan);
        return fail(error, error_len, "residency timing allocation failed");
    }
#endif
    const size_t count = plan.entry_count;
    if (plan.resident_bytes > SIZE_MAX) {
        q38_residency_plan_destroy(&plan);
        return fail(error, error_len, "all-non-PLE residency size overflow");
    }
    const size_t total = (size_t)plan.resident_bytes;
    context->persistent_expected_tensors = count;
    context->persistent_expected_bytes = total;
    context->persistent_ple_tensors = plan.excluded_ple_tensors;
    context->residency_planned_spans = plan.span_count;
    /*
     * On unified-memory systems, cudaMemGetInfo() reports immediately free
     * device pages and excludes reclaimable host page cache.  Do not reject
     * residency based on that snapshot; the actual cudaMalloc calls below
     * remain the authoritative allocation gate.
     */
    persistent_tensor *entries =
        (persistent_tensor *)calloc(count ? count : 1, sizeof(*entries));
    if (!entries) return fail(error, error_len, "all-non-PLE residency index allocation failed");
    context->exec_tensors = (q38_exec_tensor *)calloc(
        model->n_tensors ? model->n_tensors : 1, sizeof(*context->exec_tensors));
    if (!context->exec_tensors) {
        free(entries);
        return fail(error, error_len,
                    "execution tensor descriptor allocation failed");
    }
    context->exec_model = model;
    context->exec_tensor_count = (size_t)model->n_tensors;
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        const q38_tensor *tensor = &model->tensors[i];
        q38_exec_tensor *exec = &context->exec_tensors[i];
        size_t exec_rows = 0, exec_cols = 0;
        (void)tensor_shape(tensor, &exec_rows, &exec_cols);
        exec->bytes = tensor->bytes;
        exec->rows = (uint32_t)exec_rows;
        exec->cols = (uint32_t)exec_cols;
        exec->qtype = tensor->type;
        exec->tensor_id = (uint32_t)i;
        exec->gguf_offset = tensor->abs_offset;
        exec->name = tensor->name.ptr;
        exec->storage = is_ple_embedding_table(tensor)
            ? Q38_STORAGE_FILE_BACKED_PLE : Q38_STORAGE_RESIDENT;
    }
    size_t at = 0;
    size_t loaded_bytes = 0;
#if Q38_DIAGNOSTICS
    const double allocation_started = residency_now_ms();
#endif
    for (size_t p = 0; p < plan.entry_count; ++p) {
        const q38_residency_plan_entry *planned = &plan.entries[p];
        const uint32_t tensor_index = planned->tensor_index;
        const q38_tensor *tensor = &model->tensors[tensor_index];
        q38_exec_tensor *exec = &context->exec_tensors[tensor_index];
        const void *host = q38_gguf_tensor_data(model, tensor);
        bool duplicate = false;
        for (size_t j = 0; j < at; ++j)
            if (host && entries[j].host == host) duplicate = true;
        if (duplicate) {
            Q38_CUDA_DIAG_ONLY(++context->persistent_duplicate_tensors);
            for (size_t j = 0; j < at; ++j) cudaFree(entries[j].device);
            free(entries);
            q38_residency_plan_destroy(&plan);
            return fail(error, error_len,
                        "duplicate tensor in all-non-PLE residency set");
        }
        if (!host || cudaMalloc(&entries[at].device, (size_t)tensor->bytes) !=
                         cudaSuccess) {
            cudaFree(entries[at].device);
            context->persistent = entries;
            context->persistent_count = at;
            context->persistent_bytes = loaded_bytes;
            context->persistent_loaded_bytes = loaded_bytes;
            context->persistent_loaded_tensors = at;
            context->all_non_ple_resident = false;
            char name[128];
            copy_tensor_name(tensor, name, sizeof(name));
            if (error && error_len)
                snprintf(error, error_len,
                         "all-non-PLE residency upload failed at %s/%s: %s",
                         residency_group(tensor), name,
                         cudaGetErrorString(cudaGetLastError()));
            const char *detail = error && error_len ? error :
                "all-non-PLE residency upload failed";
            snprintf(context->persistent_failure,
                     sizeof(context->persistent_failure), "%s", detail);
            q38_residency_plan_destroy(&plan);
            return false;
        }
        context->residency_allocations++;
        context->residency_allocated_bytes += (size_t)tensor->bytes;
        entries[at].host = host;
        entries[at].bytes = (size_t)tensor->bytes;
        exec->ptr = entries[at].device;
        ++at;
        loaded_bytes += (size_t)tensor->bytes;
    }
    size_t largest_span = 0;
    for (size_t i = 0; i < plan.span_count; ++i)
        if (plan.spans[i].bytes > largest_span)
            largest_span = (size_t)plan.spans[i].bytes;
    for (size_t slot = 0; slot < 2 && largest_span; ++slot) {
        if (cudaMallocHost(&context->residency_stage_buffers[slot],
                           largest_span) != cudaSuccess ||
            cudaMalloc(&context->residency_transfer_buffers[slot],
                       largest_span) != cudaSuccess) {
            for (size_t cleanup = 0; cleanup <= slot; ++cleanup) {
                if (context->residency_reuse_events_ready[cleanup])
                    cudaEventDestroy(context->residency_reuse_events[cleanup]);
                cudaFree(context->residency_transfer_buffers[cleanup]);
                cudaFreeHost(context->residency_stage_buffers[cleanup]);
            }
            context->residency_stage_buffers[0] = NULL;
            context->residency_stage_buffers[1] = NULL;
            context->residency_transfer_buffers[0] = NULL;
            context->residency_transfer_buffers[1] = NULL;
            context->residency_stage = NULL;
            context->residency_transfer = NULL;
            context->persistent = entries;
            context->persistent_count = at;
            context->persistent_bytes = loaded_bytes;
            context->persistent_loaded_bytes = loaded_bytes;
            context->persistent_loaded_tensors = at;
            context->all_non_ple_resident = false;
            snprintf(context->persistent_failure,
                     sizeof(context->persistent_failure),
                     "coalesced residency staging allocation failed");
            q38_residency_plan_destroy(&plan);
            return fail(error, error_len, context->persistent_failure);
        }
        if (cudaEventCreateWithFlags(&context->residency_reuse_events[slot],
                                     cudaEventDisableTiming) != cudaSuccess) {
            for (size_t cleanup = 0; cleanup <= slot; ++cleanup) {
                if (context->residency_reuse_events_ready[cleanup])
                    cudaEventDestroy(context->residency_reuse_events[cleanup]);
                cudaFree(context->residency_transfer_buffers[cleanup]);
                cudaFreeHost(context->residency_stage_buffers[cleanup]);
            }
            context->residency_stage_buffers[0] = NULL;
            context->residency_stage_buffers[1] = NULL;
            context->residency_transfer_buffers[0] = NULL;
            context->residency_transfer_buffers[1] = NULL;
            context->residency_stage = NULL;
            context->residency_transfer = NULL;
            context->persistent = entries;
            context->persistent_count = at;
            context->persistent_bytes = loaded_bytes;
            context->persistent_loaded_bytes = loaded_bytes;
            context->persistent_loaded_tensors = at;
            context->all_non_ple_resident = false;
            snprintf(context->persistent_failure,
                     sizeof(context->persistent_failure),
                     "coalesced residency staging allocation failed");
            q38_residency_plan_destroy(&plan);
            return fail(error, error_len, context->persistent_failure);
        }
        context->residency_reuse_events_ready[slot] = true;
    }
    if (largest_span) {
        context->residency_stage = context->residency_stage_buffers[0];
        context->residency_transfer = context->residency_transfer_buffers[0];
    }
    if (largest_span) {
        context->residency_allocations += 4;
        context->residency_allocated_bytes += largest_span * 4;
    }
#if Q38_DIAGNOSTICS
    context->residency_device_alloc_ms =
        residency_now_ms() - allocation_started;
#endif
    context->residency_stage_bytes = largest_span;
    context->residency_transfer_bytes = largest_span;
    context->persistent = entries;
    context->persistent_count = at;
    for (size_t s = 0; s < plan.span_count; ++s) {
        const q38_residency_plan_span *span = &plan.spans[s];
        const size_t slot = s % 2;
        if (s >= 2 &&
            cudaEventSynchronize(context->residency_reuse_events[slot]) !=
                cudaSuccess) {
            q38_residency_plan_destroy(&plan);
            return fail(error, error_len,
                        "coalesced residency staging reuse wait failed");
        }
        void *stage = context->residency_stage_buffers[slot];
        void *transfer = context->residency_transfer_buffers[slot];
        const double source_started =
#if Q38_DIAGNOSTICS
            residency_now_ms();
#else
            0.0;
#endif
        memcpy(stage,
               model->map + span->file_offset, (size_t)span->bytes);
#if Q38_DIAGNOSTICS
        const double source_ms = residency_now_ms() - source_started;
        context->residency_source_copy_ms += source_ms;
        context->residency_span_timings[s].source_offset = span->file_offset;
        context->residency_span_timings[s].bytes = (size_t)span->bytes;
        context->residency_span_timings[s].source_copy_ms = source_ms;
#endif
        context->residency_staged_bytes += (size_t)span->bytes;
        if (cudaMemcpyAsync(transfer, stage, (size_t)span->bytes,
                            cudaMemcpyHostToDevice, context->stream) !=
            cudaSuccess) {
            q38_residency_plan_destroy(&plan);
            return fail(error, error_len,
                        "coalesced residency H2D upload failed");
        }
#if Q38_DIAGNOSTICS
        context->residency_span_timings[s].h2d_enqueue_ms =
            residency_now_ms() - source_started - source_ms;
        context->residency_h2d_enqueue_ms +=
            context->residency_span_timings[s].h2d_enqueue_ms;
#endif
        context->residency_h2d_bytes += (size_t)span->bytes;
        context->residency_transfer_calls++;
        const double d2d_started =
#if Q38_DIAGNOSTICS
            residency_now_ms();
#else
            0.0;
#endif
        for (size_t j = 0; j < span->entry_count; ++j) {
            const q38_residency_plan_entry *planned =
                &plan.entries[span->first_entry + j];
            const size_t entry_index = span->first_entry + j;
            const uint64_t relative =
                planned->file_offset - span->file_offset;
            if (cudaMemcpyAsync(
                    entries[entry_index].device,
                    (const char *)transfer + relative,
                    (size_t)planned->bytes, cudaMemcpyDeviceToDevice,
                    context->stream) != cudaSuccess) {
                q38_residency_plan_destroy(&plan);
                return fail(error, error_len,
                            "coalesced residency tensor copy failed");
            }
            context->residency_device_copies++;
        }
        if (cudaEventRecord(context->residency_reuse_events[slot],
                            context->stream) != cudaSuccess) {
            q38_residency_plan_destroy(&plan);
            return fail(error, error_len,
                        "coalesced residency staging event failed");
        }
#if Q38_DIAGNOSTICS
        context->residency_d2d_enqueue_ms +=
            residency_now_ms() - d2d_started;
#endif
    }
    const double final_wait_started =
#if Q38_DIAGNOSTICS
        residency_now_ms();
#else
        0.0;
#endif
    if (Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_RESIDENCY_INIT,
                           cudaStreamSynchronize(context->stream)) !=
        cudaSuccess) {
        q38_residency_plan_destroy(&plan);
        return fail(error, error_len,
                    "coalesced residency synchronization failed");
    }
#if Q38_DIAGNOSTICS
    context->residency_final_wait_ms =
        residency_now_ms() - final_wait_started;
    struct rusage usage_after = {};
    getrusage(RUSAGE_SELF, &usage_after);
    context->residency_minor_faults_before = usage_before.ru_minflt;
    context->residency_minor_faults_after = usage_after.ru_minflt;
    context->residency_major_faults_before = usage_before.ru_majflt;
    context->residency_major_faults_after = usage_after.ru_majflt;
    context->residency_mincore_pages_after = residency_mincore_pages(model);
#endif
    context->residency_final_syncs++;
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    size_t progress_free = 0, progress_total = 0;
    (void)cudaMemGetInfo(&progress_free, &progress_total);
    if (context->progress_observer)
        context->progress_observer("coalesced", NULL, loaded_bytes,
                                   progress_free, progress_total,
                                   context->progress_observer_user);
    q38_residency_plan_destroy(&plan);
    context->persistent_bytes = loaded_bytes;
    context->persistent_loaded_bytes = loaded_bytes;
    context->persistent_loaded_tensors = at;
    context->all_non_ple_resident = true;
    context->persistent_coverage_ok =
        at == context->persistent_expected_tensors &&
        total == context->persistent_expected_bytes &&
        context->persistent_ple_entries == 0;
    if (!context->persistent_coverage_ok && context->exec_strict)
        return fail(error, error_len,
                    "Q38_EXEC_STRICT: incomplete non-PLE execution descriptors");
    return true;
}

extern "C" void
q38_forward_cuda_context_destroy(q38_forward_cuda_context *context) {
    if (!context) return;
    cudaFree(context->device_weights);
    cudaFree(context->device_input);
    cudaFree(context->device_output);
    cudaFree(context->device_hidden_a);
    cudaFree(context->device_hidden_b);
    cudaFree(context->device_argmax);
    cudaFree(context->device_aux);
    cudaFree(context->device_moe_mid);
    cudaFree(context->device_moe_grouped_mid);
    cudaFree(context->device_moe_route_ids);
    cudaFree(context->device_moe_route_indices);
    cudaFree(context->device_moe_route_weights);
    cudaFree(context->device_moe_accum);
    cudaFree(context->device_moe_expert_outputs);
    cudaFree(context->device_moe_logits);
    cudaFree(context->device_moe_shared_gate);
    cudaFree(context->device_moe_shared_up);
    cudaFree(context->device_moe_shared_output);
    cudaFree(context->device_moe_shared_weight);
    cudaFree(context->device_gr_residual);
    cudaFree(context->device_gr_norm);
    cudaFree(context->device_gr_down);
    cudaFree(context->device_gr_bottleneck);
    cudaFree(context->device_gr_up);
    cudaFree(context->device_gr_input);
    cudaFree(context->device_gr_block);
    cudaFree(context->device_gr_inject);
    cudaFree(context->device_gr_updated);
    cudaFree(context->device_qsa_input);
    cudaFree(context->device_qsa_output);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer)
        q38_qsa_cuda_chain_release(&context->qsa_chain_state[layer]);
    cudaFree(context->qsa_chain_workspace.qfull);
    cudaFree(context->qsa_chain_workspace.q);
    cudaFree(context->qsa_chain_workspace.k);
    cudaFree(context->qsa_chain_workspace.v);
    cudaFree(context->qsa_chain_workspace.index);
    cudaFree(context->qsa_chain_workspace.index_q);
    cudaFree(context->qsa_chain_workspace.raw_index);
    cudaFree(context->qsa_chain_workspace.attention);
    cudaFree(context->qsa_chain_workspace.selected_k);
    cudaFree(context->qsa_chain_workspace.selected_v);
    cudaFree(context->qsa_chain_workspace.selected);
    cudaFree(context->device_gdn_input);
    cudaFree(context->device_gdn_qkv);
    cudaFree(context->device_gdn_z);
    cudaFree(context->device_gdn_a);
    cudaFree(context->device_gdn_b);
    cudaFree(context->device_gdn_conv);
    cudaFree(context->device_gdn_q);
    cudaFree(context->device_gdn_k);
    cudaFree(context->device_gdn_v);
    cudaFree(context->device_gdn_decay);
    cudaFree(context->device_gdn_beta);
    cudaFree(context->device_gdn_recurrent);
    cudaFree(context->device_gdn_gated);
    cudaFree(context->device_gdn_state);
    cudaFree(context->device_gdn_history);
    cudaFree(context->device_steering);
    for (size_t slot = 0; slot < 2; ++slot) {
        if (context->residency_reuse_events_ready[slot])
            cudaEventDestroy(context->residency_reuse_events[slot]);
        cudaFree(context->residency_transfer_buffers[slot]);
        cudaFreeHost(context->residency_stage_buffers[slot]);
    }
    free(context->residency_span_timings);
    free(context->host_qsa_output);
    if (!context->lm_head_uses_persistent)
        cudaFree(context->lm_head_device_weights);
    for (size_t i = 0; i < context->persistent_count; ++i)
        cudaFree(context->persistent[i].device);
    free(context->persistent);
    free(context->exec_tensors);
    if (context->stream) cudaStreamDestroy(context->stream);
    free(context);
}

extern "C" bool q38_forward_cuda_load_directional_steering(
    q38_forward_cuda_context *context,
    const q38_directional_steering *steering, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !steering || !steering->directions ||
        steering->layers != Q38_DIRECTIONAL_STEERING_LAYERS ||
        steering->hidden_size != Q38_DIRECTIONAL_STEERING_HIDDEN)
        return fail(error, error_len, "invalid CUDA Q38 steering state");
    if (!context->device_steering &&
        cudaMalloc((void **)&context->device_steering,
                   Q38_DIRECTIONAL_STEERING_BYTES) != cudaSuccess)
        return fail(error, error_len, "Q38 steering CUDA allocation failed");
    context->device_steering_bytes = Q38_DIRECTIONAL_STEERING_BYTES;
    if (cudaMemcpyAsync(context->device_steering, steering->directions,
                        Q38_DIRECTIONAL_STEERING_BYTES,
                        cudaMemcpyHostToDevice, context->stream) !=
            cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_STEERING_INIT,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "Q38 steering CUDA upload failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    return true;
}

extern "C" void q38_forward_cuda_set_directional_steering_scales(
    q38_forward_cuda_context *context, float ffn_scale, float attn_scale) {
    if (!context) return;
    context->directional_steering_ffn_scale = ffn_scale;
    context->directional_steering_attn_scale = attn_scale;
}

static bool apply_directional_steering_device(
    q38_forward_cuda_context *context, float *values, uint32_t layer,
    size_t width, size_t rows, float scale, char *error, size_t error_len) {
    if (!context || !context->device_steering || !values || !rows ||
        !scale)
        return true;
    if (width != Q38_DIRECTIONAL_STEERING_HIDDEN ||
        layer >= Q38_DIRECTIONAL_STEERING_LAYERS ||
        rows > UINT32_MAX)
        return fail(error, error_len, "invalid CUDA Q38 steering geometry");
    uint32_t threads = 256u;
    while (threads > width && threads > 1u) threads >>= 1;
    q38_directional_steering_kernel<<<(unsigned)rows, threads, 0,
                                      context->stream>>>(
        values, context->device_steering, layer, (uint32_t)width,
        (uint32_t)rows, scale);
    return cudaGetLastError() == cudaSuccess ||
           fail(error, error_len, "Q38 steering kernel launch failed");
}

extern "C" bool q38_forward_cuda_apply_directional_steering(
    q38_forward_cuda_context *context, float *device_values, uint32_t layer,
    size_t width, size_t rows, float scale, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    return apply_directional_steering_device(
        context, device_values, layer, width, rows, scale, error, error_len);
}

extern "C" bool q38_forward_cuda_prepare_lm_head(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_tensor *tensor, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !model || !tensor || !is_lm_head_tensor(tensor) ||
        !tensor->bytes)
        return fail(error, error_len, "invalid LM-head residency tensor");
    const void *data = q38_gguf_tensor_data(model, tensor);
    if (!data)
        return fail(error, error_len, "invalid LM-head residency payload");
    if (context->lm_head_resident &&
        context->lm_head_host_data == data &&
        context->lm_head_device_weights_bytes == tensor->bytes)
        return true;
    const void *persistent = q38_gguf_tensor_data(model, tensor);
    persistent_tensor *entry = persistent_find(context, persistent);
    if (entry) {
        context->lm_head_device_weights = entry->device;
        context->lm_head_device_weights_bytes = entry->bytes;
        context->lm_head_host_data = persistent;
        context->lm_head_resident = true;
        context->lm_head_uses_persistent = true;
        return true;
    }
    if (context->lm_head_device_weights &&
        !context->lm_head_uses_persistent &&
        context->lm_head_device_weights_bytes < tensor->bytes) {
        cudaFree(context->lm_head_device_weights);
        context->lm_head_device_weights = NULL;
        context->lm_head_device_weights_bytes = 0;
    }

    if (!context->lm_head_device_weights &&
        cudaMalloc(&context->lm_head_device_weights, tensor->bytes) !=
            cudaSuccess)
        return fail(error, error_len, "LM-head residency allocation failed");
    if (cudaMemcpyAsync(context->lm_head_device_weights, data, tensor->bytes,
                        cudaMemcpyHostToDevice, context->stream) !=
            cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_LM_HEAD_RESIDENCY_INIT,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "LM-head residency upload failed");
    context->lm_head_device_weights_bytes = tensor->bytes;
    context->lm_head_host_data = data;
    context->lm_head_resident = true;
    context->lm_head_uses_persistent = false;
    return true;
}

extern "C" void q38_forward_cuda_get_residency_stats(
    const q38_forward_cuda_context *context,
    q38_forward_cuda_residency_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    if (!context) return;
    stats->matrix_upload_bytes = context->matrix_upload_bytes;
    stats->resident_hits = context->resident_hits;
    stats->resident_misses = context->resident_misses;
    stats->cuda_allocations = context->cuda_allocations;
    stats->lm_head_resident = context->lm_head_resident;
    stats->lm_head_device_pointer = context->lm_head_device_weights;
    stats->all_non_ple_resident = context->all_non_ple_resident;
    stats->persistent_resident_bytes = context->persistent_bytes;
    stats->persistent_resident_tensors = context->persistent_count;
    uint64_t pointer_hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < context->persistent_count; ++i) {
        const uintptr_t pointer = (uintptr_t)context->persistent[i].device;
        for (size_t byte = 0; byte < sizeof(pointer); ++byte)
            pointer_hash = (pointer_hash ^
                            (uint8_t)(pointer >> (byte * 8))) *
                           UINT64_C(1099511628211);
    }
    stats->persistent_pointer_fingerprint = pointer_hash;
    stats->cuda_context_identity = (uint64_t)(uintptr_t)context;
    stats->cuda_stream_identity = (uint64_t)(uintptr_t)context->stream;
    uint64_t workspace_hash = UINT64_C(1469598103934665603);
    const uintptr_t workspace_pointers[] = {
        (uintptr_t)context->device_input,
        (uintptr_t)context->device_output,
        (uintptr_t)context->device_moe_mid,
        (uintptr_t)context->device_moe_grouped_mid,
        (uintptr_t)context->device_moe_route_ids,
        (uintptr_t)context->device_moe_route_indices,
        (uintptr_t)context->device_moe_route_weights,
        (uintptr_t)context->device_moe_accum,
        (uintptr_t)context->device_moe_expert_outputs,
        (uintptr_t)context->device_qsa_input,
        (uintptr_t)context->device_qsa_output,
        (uintptr_t)context->device_hidden_a,
        (uintptr_t)context->device_hidden_b,
        (uintptr_t)context->host_qsa_output,
    };
    for (size_t i = 0; i < sizeof(workspace_pointers) /
                           sizeof(workspace_pointers[0]); ++i) {
        for (size_t byte = 0; byte < sizeof(workspace_pointers[i]); ++byte)
            workspace_hash =
                (workspace_hash ^
                 (uint8_t)(workspace_pointers[i] >> (byte * 8))) *
                UINT64_C(1099511628211);
    }
    stats->workspace_pointer_fingerprint = workspace_hash;
    stats->persistent_hits = context->persistent_hits;
    stats->persistent_misses = context->persistent_misses;
    stats->persistent_expected_bytes = context->persistent_expected_bytes;
    stats->persistent_expected_tensors = context->persistent_expected_tensors;
    stats->persistent_duplicate_tensors = context->persistent_duplicate_tensors;
    stats->persistent_ple_tensors = context->persistent_ple_tensors;
    stats->persistent_ple_entries = context->persistent_ple_entries;
    stats->persistent_coverage_ok = context->persistent_coverage_ok;
    stats->persistent_failure = context->persistent_failure[0]
        ? context->persistent_failure : NULL;
    stats->persistent_loaded_bytes = context->persistent_loaded_bytes;
    stats->persistent_loaded_tensors = context->persistent_loaded_tensors;
    stats->residency_planned_spans = context->residency_planned_spans;
    stats->residency_transfer_calls = context->residency_transfer_calls;
    stats->residency_device_copies = context->residency_device_copies;
    stats->residency_final_syncs = context->residency_final_syncs;
    stats->residency_stage_bytes = context->residency_stage_bytes;
    stats->residency_planned_bytes = context->residency_planned_bytes;
    stats->residency_staged_bytes = context->residency_staged_bytes;
    stats->residency_h2d_bytes = context->residency_h2d_bytes;
    stats->residency_plan_ms = context->residency_plan_ms;
    stats->residency_device_alloc_ms = context->residency_device_alloc_ms;
    stats->residency_source_copy_ms = context->residency_source_copy_ms;
    stats->residency_h2d_enqueue_ms = context->residency_h2d_enqueue_ms;
    stats->residency_d2d_enqueue_ms = context->residency_d2d_enqueue_ms;
    stats->residency_final_wait_ms = context->residency_final_wait_ms;
    stats->residency_allocations = context->residency_allocations;
    stats->residency_allocated_bytes = context->residency_allocated_bytes;
    stats->residency_mincore_pages_before =
        context->residency_mincore_pages_before;
    stats->residency_mincore_pages_after =
        context->residency_mincore_pages_after;
    stats->residency_minor_faults_before =
        context->residency_minor_faults_before;
    stats->residency_minor_faults_after =
        context->residency_minor_faults_after;
    stats->residency_major_faults_before =
        context->residency_major_faults_before;
    stats->residency_major_faults_after =
        context->residency_major_faults_after;
    stats->residency_span_timings = context->residency_span_timings;
    stats->residency_span_timing_count =
        context->residency_span_timing_count;
    stats->exec_strict = context->exec_strict;
    stats->resident_lookup_in_decode = context->resident_lookup_in_decode;
    stats->gguf_name_lookup_in_decode = context->gguf_name_lookup_in_decode;
    stats->non_ple_residency_miss = context->non_ple_residency_miss;
    stats->non_ple_upload_bytes_per_token =
        context->non_ple_upload_bytes_per_token;
    stats->ple_file_backed_accesses = context->ple_file_backed_accesses;
    stats->ple_file_bytes = context->ple_file_bytes;
    stats->gpu_argmax_kernel_ms = context->gpu_argmax_kernel_ms;
    stats->routed_layers_executed = context->routed_layers_executed;
    stats->selected_experts_total = context->selected_experts_total;
    stats->q2_gate_up_fast_calls = context->q2_gate_up_fast_calls;
    stats->q2_gate_up_legacy_calls = context->q2_gate_up_legacy_calls;
    stats->q2_gate_up_fallback_calls = context->q2_gate_up_fallback_calls;
    stats->q2_down_calls = context->q2_down_calls;
    stats->q2_gate_up_fast_total_kernel_ms =
        context->q2_gate_up_fast_total_kernel_ms;
    stats->q2_gate_up_legacy_total_kernel_ms =
        context->q2_gate_up_legacy_total_kernel_ms;
    stats->q2_down_total_kernel_ms = context->q2_down_total_kernel_ms;
    stats->q2_weighted_reduce_total_kernel_ms =
        context->q2_weighted_reduce_total_kernel_ms;
    stats->expert_backend_total_wall_ms =
        context->expert_backend_total_wall_ms;
    stats->expert_host_sync_count = context->expert_host_sync_count;
    stats->expert_kernel_launches = context->expert_kernel_launches;
    stats->expert_H2D_bytes = context->expert_H2D_bytes;
    stats->expert_D2H_bytes = context->expert_D2H_bytes;
    stats->qsa_chain_calls = context->qsa_chain_calls;
    stats->qsa_chain_kernel_launches = context->qsa_chain_kernel_launches;
    stats->qsa_chain_syncs = context->qsa_chain_syncs;
    stats->qsa_chain_h2d_bytes = context->qsa_chain_h2d_bytes;
    stats->qsa_chain_d2h_bytes = context->qsa_chain_d2h_bytes;
    stats->qsa_chain_internal_h2d_bytes =
        context->qsa_chain_internal_h2d_bytes;
    stats->qsa_chain_internal_d2h_bytes =
        context->qsa_chain_internal_d2h_bytes;
    memcpy(stats->expert_fast_calls_by_layer,
           context->expert_fast_calls_by_layer,
           sizeof(stats->expert_fast_calls_by_layer));
    memcpy(stats->expert_legacy_calls_by_layer,
           context->expert_legacy_calls_by_layer,
           sizeof(stats->expert_legacy_calls_by_layer));
}

extern "C" void q38_forward_cuda_get_sync_stats(
    const q38_forward_cuda_context *context,
    q38_forward_cuda_sync_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
#if Q38_DIAGNOSTICS
    if (context) *stats = context->sync_stats;
#else
    (void)context;
#endif
}

extern "C" void q38_forward_cuda_reset_sync_stats(
    q38_forward_cuda_context *context) {
#if Q38_DIAGNOSTICS
    if (context) {
        memset(&context->sync_stats, 0, sizeof(context->sync_stats));
        context->telemetry_wait_baseline_ms = 0.0;
    }
#else
    (void)context;
#endif
}

extern "C" const char *q38_forward_cuda_sync_reason_name(
    q38_forward_cuda_sync_reason reason) {
    static const char *const names[Q38_CUDA_SYNC_REASON_COUNT] = {
        "MOE_ROUTED_D2H",
        "MOE_GROUPED_D2H",
        "GDN_OUTPUT",
        "GDN_TRACE_STATE",
        "GR_READ",
        "GR_WRITE",
        "QSA_QKV",
        "MATVEC_D2H",
        "MATRIX_D2H",
        "MATRIX_BATCH_D2H",
        "ARGMAX",
        "PLE_STAGE_WAIT",
        "RESIDENCY_INIT",
        "STEERING_INIT",
        "LM_HEAD_RESIDENCY_INIT",
    };
    return reason < Q38_CUDA_SYNC_REASON_COUNT ? names[reason] : "UNKNOWN";
}

extern "C" void q38_forward_cuda_set_qsa_candidate(
    q38_forward_cuda_context *context, q38_qsa_candidate_fn candidate) {
    if (context) context->qsa_candidate = candidate;
}

extern "C" void q38_forward_cuda_record_route(
    q38_forward_cuda_context *context, uint32_t layer, size_t selected_count) {
    if (!context || layer >= Q38_MODEL_LAYERS) return;
    Q38_CUDA_DIAG_ONLY(++context->routed_layers_executed);
    Q38_CUDA_DIAG_ONLY(context->selected_experts_total += selected_count);
}

extern "C" void q38_forward_cuda_get_expert_layer_calls(
    const q38_forward_cuda_context *context, uint32_t layer,
    uint64_t *fast_calls, uint64_t *legacy_calls) {
    if (fast_calls) *fast_calls = 0;
    if (legacy_calls) *legacy_calls = 0;
    if (!context || layer >= Q38_MODEL_LAYERS) return;
    if (fast_calls) *fast_calls = context->expert_fast_calls_by_layer[layer];
    if (legacy_calls)
        *legacy_calls = context->expert_legacy_calls_by_layer[layer];
}

extern "C" void *
q38_forward_cuda_stream(q38_forward_cuda_context *context) {
    return context ? (void *)context->stream : NULL;
}

extern "C" void q38_forward_cuda_set_allocation_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_allocation_observer observer, void *user) {
    if (!context) return;
    context->allocation_observer = observer;
    context->allocation_observer_user = user;
}

extern "C" void q38_forward_cuda_set_telemetry_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_telemetry_observer observer, void *user) {
    if (!context) return;
    context->telemetry_observer = observer;
    context->telemetry_observer_user = user;
}

extern "C" void q38_forward_cuda_set_residency_progress_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_residency_progress_observer observer, void *user) {
    if (!context) return;
    context->progress_observer = observer;
    context->progress_observer_user = user;
}

extern "C" void q38_forward_cuda_set_stage_context(
    q38_forward_cuda_context *context, uint32_t layer,
    const char *logical_stage) {
    if (!context) return;
    context->current_layer = layer;
    context->current_stage = logical_stage;
}

extern "C" bool q38_forward_cuda_matvec_backend(
    const q38_gguf *model, const q38_tensor *tensor, size_t row,
    const float *input, size_t cols, float *output, void *user, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t rows, actual_cols;
    if (!context || !model || !tensor || !input || !output ||
        !tensor_shape(tensor, &rows, &actual_cols) || row >= rows ||
        actual_cols != cols || tensor->bytes % rows != 0)
        return fail(error, error_len, "invalid CUDA forward matvec geometry");
    const size_t row_bytes = (size_t)(tensor->bytes / rows);
    if (!row_bytes || row > SIZE_MAX / row_bytes)
        return fail(error, error_len, "invalid CUDA forward tensor payload");
    q38_exec_tensor *exec = exec_tensor_for(context, model, tensor);
    const bool use_persistent = context->all_non_ple_resident &&
        exec_tensor_is_resident(exec, tensor);
    if (context->exec_strict && !use_persistent &&
        !is_ple_embedding_table(tensor))
        return fail(error, error_len,
                    "Q38_EXEC_STRICT: matvec tensor is not resident");
    const void *data = use_persistent ? NULL :
        q38_gguf_tensor_data(model, tensor);
    if (!use_persistent && !data)
        return fail(error, error_len, "invalid CUDA forward tensor payload");
    const void *row_data = data
        ? (const unsigned char *)data + row * row_bytes : NULL;

    size_t weight_bytes = row_bytes;
    if (tensor->type != 0 && tensor->type != 8 && tensor->type != 10 &&
        tensor->type != 30)
        return fail(error, error_len, "unsupported CUDA forward matvec type");
    if (cols > SIZE_MAX / sizeof(float))
        return fail(error, error_len, "CUDA forward matvec size overflow");
    if ((!use_persistent &&
         !ensure_buffer(&context->device_weights,
                        &context->device_weights_bytes, weight_bytes,
                        context->allocation_observer,
                        context->allocation_observer_user,
                        &context->cuda_allocations)) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       cols * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes, sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "CUDA forward matvec allocation failed");
    if (use_persistent) {
        Q38_CUDA_DIAG_ONLY(++context->persistent_hits);
        const void *weight_storage =
            (const unsigned char *)exec->ptr + row * row_bytes;
        bool launched = false;
        if (tensor->type == 30)
            launched = q38_cuda_bf16_matvec(
                (const uint16_t *)weight_storage, 1, cols,
                context->device_input, context->device_output, context->stream,
                error, error_len);
        else if (tensor->type == 10)
            launched = q38_cuda_q2_matvec(
                (void *)weight_storage, 1, cols, context->device_input,
                context->device_output, context->stream, error, error_len);
        else
            launched = q38_cuda_gdn_project(
                tensor->type == 8 ? Q38_GDN_WEIGHT_Q8_0 : Q38_GDN_WEIGHT_F32,
                (void *)weight_storage, 1, cols, context->device_input, 1,
                context->device_output, context->stream, error, error_len);
        if (!launched ||
            cudaMemcpyAsync(output, context->device_output, sizeof(float),
                            cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
            Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_MATVEC_D2H,
                               cudaStreamSynchronize(context->stream)) !=
                cudaSuccess)
            return fail(error, error_len, "CUDA resident matvec execution failed");
        Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
        emit_telemetry(context, model, tensor, 1, cols, weight_bytes, true,
                       false, 0, 0.0f, 0.0f, 0.0, 0, 1,
                       "matvec", "resident_exec_tensor");
        return true;
    }
    if (context->all_non_ple_resident) {
        Q38_CUDA_DIAG_ONLY(++context->persistent_misses);
        if (!is_ple_embedding_table(tensor))
            Q38_CUDA_DIAG_ONLY(++context->non_ple_residency_miss);
    }

    if (cudaMemcpyAsync(context->device_weights, row_data, weight_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_input, input, cols * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA forward matvec upload failed");
    if (!is_ple_embedding_table(tensor))
        Q38_CUDA_DIAG_ONLY(context->non_ple_upload_bytes_per_token += weight_bytes);
    if (!is_ple_embedding_table(tensor))
        Q38_CUDA_DIAG_ONLY(++context->gguf_name_lookup_in_decode);

    bool launched = false;
    if (tensor->type == 30) {
        launched = q38_cuda_bf16_matvec(
            (const uint16_t *)context->device_weights, 1, cols,
            context->device_input, context->device_output, context->stream,
            error, error_len);
    } else if (tensor->type == 10) {
        launched = q38_cuda_q2_matvec(
            context->device_weights, 1, cols, context->device_input,
            context->device_output, context->stream, error, error_len);
    } else {
        const uint32_t type = tensor->type == 8 ? Q38_GDN_WEIGHT_Q8_0
                                                : Q38_GDN_WEIGHT_F32;
        launched = q38_cuda_gdn_project(
            type, context->device_weights, 1, cols, context->device_input, 1,
            context->device_output, context->stream, error, error_len);
    }
    if (!launched ||
        cudaMemcpyAsync(output, context->device_output, sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_MATVEC_D2H,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return launched ? fail(error, error_len,
                               "CUDA forward matvec download failed")
                        : false;
    emit_telemetry(context, model, tensor, 1, cols, weight_bytes, false,
                   true, weight_bytes, 0.0f, 0.0f, 0.0, 0, 1,
                   "matvec", "gguf_host_upload");
    return true;
}

extern "C" bool q38_forward_cuda_matrix_backend(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t rows, size_t cols, float *output, void *user, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t actual_rows, actual_cols;
    if (!context || !model || !tensor || !input || !output ||
        !tensor_shape(tensor, &actual_rows, &actual_cols) ||
        actual_rows != rows || actual_cols != cols)
        return fail(error, error_len, "invalid CUDA forward matrix geometry");
    if (!tensor->bytes || rows > SIZE_MAX / sizeof(float) ||
        cols > SIZE_MAX / sizeof(float))
        return fail(error, error_len, "invalid CUDA forward matrix payload");
    const bool collect_telemetry = Q38_CUDA_DIAG_COLLECT(context);
    const double host_started = collect_telemetry ? host_now_ms() : 0.0;
    cudaEvent_t upload_start = NULL, upload_stop = NULL;
    cudaEvent_t kernel_start = NULL, kernel_stop = NULL;
    if (!telemetry_events_create(collect_telemetry, &upload_start,
                                 &upload_stop, &kernel_start, &kernel_stop))
        return fail(error, error_len, "CUDA telemetry event allocation failed");
    const uint64_t allocation_before = context->cuda_allocations;
    q38_exec_tensor *exec = exec_tensor_for(context, model, tensor);
    const bool use_exec_resident = context->all_non_ple_resident &&
        exec_tensor_is_resident(exec, tensor);
    const void *data = use_exec_resident ? NULL :
        q38_gguf_tensor_data(model, tensor);
    if (!use_exec_resident) Q38_CUDA_DIAG_ONLY(++context->gguf_name_lookup_in_decode);
    if (context->exec_strict && !use_exec_resident &&
        !is_ple_embedding_table(tensor))
        return fail(error, error_len,
                    "Q38_EXEC_STRICT: matrix tensor is not resident");
    if (!use_exec_resident && !data)
        return fail(error, error_len, "invalid CUDA forward matrix payload");
    const bool use_resident_lm_head =
        is_lm_head_tensor(tensor) &&
        context->lm_head_resident && context->lm_head_host_data == data &&
        context->lm_head_device_weights_bytes == tensor->bytes;
    const bool use_persistent_weight = use_exec_resident;
    if (use_resident_lm_head || use_persistent_weight)
        Q38_CUDA_DIAG_ONLY(++context->resident_hits);
    else {
        Q38_CUDA_DIAG_ONLY(++context->resident_misses);
        if (context->all_non_ple_resident) {
            Q38_CUDA_DIAG_ONLY(++context->persistent_misses);
            if (!is_ple_embedding_table(tensor))
                Q38_CUDA_DIAG_ONLY(++context->non_ple_residency_miss);
        }
    }
    if (use_persistent_weight) Q38_CUDA_DIAG_ONLY(++context->persistent_hits);
    if ((!use_resident_lm_head && !use_persistent_weight &&
         !ensure_buffer(&context->device_weights,
                        &context->device_weights_bytes, (size_t)tensor->bytes,
                        context->allocation_observer,
                        context->allocation_observer_user,
                       &context->cuda_allocations)) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       cols * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes,
                       rows * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "CUDA forward matrix allocation failed");
    if (!telemetry_event_record(collect_telemetry, upload_start, context->stream) ||
        (!use_resident_lm_head && !use_persistent_weight &&
         cudaMemcpyAsync(context->device_weights, data, (size_t)tensor->bytes,
                         cudaMemcpyHostToDevice, context->stream) !=
             cudaSuccess) ||
        !telemetry_event_record(collect_telemetry, upload_stop, context->stream) ||
        cudaMemcpyAsync(context->device_input, input, cols * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA forward matrix upload failed");
    if (!telemetry_event_record(collect_telemetry, kernel_start, context->stream))
        return fail(error, error_len, "CUDA matrix timing failed");
    if (!use_resident_lm_head && !use_persistent_weight)
        Q38_CUDA_DIAG_ONLY(context->matrix_upload_bytes += tensor->bytes);
    if (!use_resident_lm_head && !use_persistent_weight &&
        !is_ple_embedding_table(tensor))
        Q38_CUDA_DIAG_ONLY(context->non_ple_upload_bytes_per_token += tensor->bytes);
    bool launched = false;
    void *weight_storage = use_resident_lm_head
                               ? context->lm_head_device_weights
                               : use_persistent_weight ? (void *)exec->ptr
                               : context->device_weights;
    if (tensor->type == 30) {
        launched = q38_cuda_bf16_matvec(
            (const uint16_t *)weight_storage, rows, cols,
            context->device_input, context->device_output, context->stream,
            error, error_len);
    } else if (tensor->type == 10) {
        launched = q38_cuda_q2_matvec(
            weight_storage, rows, cols, context->device_input,
            context->device_output, context->stream, error, error_len);
    } else if (tensor->type == 0 || tensor->type == 8) {
        const uint32_t type = tensor->type == 8 ? Q38_GDN_WEIGHT_Q8_0
                                                : Q38_GDN_WEIGHT_F32;
        launched = q38_cuda_gdn_project(
            type, weight_storage, rows, cols, context->device_input, 1,
            context->device_output, context->stream, error, error_len);
    } else {
        return fail(error, error_len, "unsupported CUDA forward matrix type");
    }
    if (!launched) {
        telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                 kernel_stop);
        return false;
    }
    if (context->current_stage &&
        (!strcmp(context->current_stage, "qsa_output_projection") ||
         !strcmp(context->current_stage, "gdn_output_projection")) &&
        !apply_directional_steering_device(
            context, context->device_output, context->current_layer, rows, 1,
            context->directional_steering_attn_scale, error, error_len)) {
        telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                 kernel_stop);
        return false;
    }
    if (!telemetry_event_record(collect_telemetry, kernel_stop, context->stream))
        return fail(error, error_len, "CUDA kernel timing failed");
    if (cudaMemcpyAsync(output, context->device_output, rows * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_MATRIX_D2H,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "CUDA forward matrix download failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    context->device_output_elements = rows;
    float upload_ms = collect_telemetry ? event_elapsed(upload_start, upload_stop) : 0.0f;
    float kernel_ms = collect_telemetry ? event_elapsed(kernel_start, kernel_stop) : 0.0f;
    emit_telemetry(context, model, tensor, rows, cols, (size_t)tensor->bytes,

                   use_resident_lm_head || use_persistent_weight,
                   !(use_resident_lm_head || use_persistent_weight),
                   (use_resident_lm_head || use_persistent_weight) ? 0 :
                       (size_t)tensor->bytes,
                   upload_ms, kernel_ms,
                   collect_telemetry ? host_now_ms() - host_started : 0.0,
                   context->cuda_allocations - allocation_before, 1,
                   "matrix", use_resident_lm_head || use_persistent_weight
                       ? "resident_exec_tensor" : "gguf_host_upload");
    telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                             kernel_stop);
    return true;
}

extern "C" bool q38_forward_cuda_matrix_batch_backend(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t token_count, size_t rows, size_t cols, float *output, void *user,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t actual_rows, actual_cols;
    if (!context || !model || !tensor || !input || !output || !token_count ||
        !tensor_shape(tensor, &actual_rows, &actual_cols) ||
        actual_rows != rows || actual_cols != cols ||
        (tensor->type != 0 && tensor->type != 8 &&
         tensor->type != 10 && tensor->type != 30))
        return fail(error, error_len,
                    "invalid CUDA batched matrix geometry");
    if (token_count > SIZE_MAX / cols ||
        token_count * cols > SIZE_MAX / sizeof(float) ||
        token_count > SIZE_MAX / rows ||
        token_count * rows > SIZE_MAX / sizeof(float))
        return fail(error, error_len, "CUDA batched matrix size overflow");
    q38_exec_tensor *exec = exec_tensor_for(context, model, tensor);
    const bool resident = context->all_non_ple_resident &&
        exec_tensor_is_resident(exec, tensor);
    if (!resident)
        return false;
    const size_t input_bytes = token_count * cols * sizeof(float);
    const size_t output_bytes = token_count * rows * sizeof(float);
    if (!ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements, input_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes, output_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len,
                    "CUDA batched matrix workspace allocation failed");
    cudaEvent_t upload_start = NULL, upload_stop = NULL;
    cudaEvent_t kernel_start = NULL, kernel_stop = NULL;
    const bool collect_telemetry = Q38_CUDA_DIAG_COLLECT(context);
    if (!telemetry_events_create(collect_telemetry, &upload_start,
                                 &upload_stop, &kernel_start, &kernel_stop)) {
        return fail(error, error_len,
                    "CUDA batched matrix telemetry event allocation failed");
    }
    const double started = collect_telemetry ? host_now_ms() : 0.0;
    if (!telemetry_event_record(collect_telemetry, upload_start, context->stream) ||
        cudaMemcpyAsync(context->device_input, input, input_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        !telemetry_event_record(collect_telemetry, upload_stop, context->stream) ||
        !telemetry_event_record(collect_telemetry, kernel_start, context->stream)) {
        telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                 kernel_stop);
        return fail(error, error_len, "CUDA batched matrix upload failed");
    }
    const bool gr_bf16_candidate =
        token_count == 1 && tensor->type == 30 &&
        is_gr_projection_stage(context->current_stage);
    const unsigned gr_threads =
        gr_bf16_candidate && context->current_stage &&
                !strcmp(context->current_stage, "gr_read_up")
            ? 128u
            : 256u;
    const bool gdn_bf16_candidate =
        token_count == 1 && tensor->type == 30 &&
        is_gdn_projection_stage(context->current_stage);
    const bool qsa_bf16_candidate =
        token_count == 1 && tensor->type == 30 &&
        context->current_stage &&
        !strcmp(context->current_stage, "qsa_output_projection");
    const bool launched = gr_bf16_candidate
        ? q38_cuda_bf16_matvec_configured(
              (const uint16_t *)exec->ptr, rows, cols,
              context->device_input, context->device_output, gr_threads,
              context->stream, error, error_len)
        : gdn_bf16_candidate
        ? q38_cuda_gdn_project(
              Q38_GDN_WEIGHT_BF16, exec->ptr, rows, cols,
              context->device_input, token_count, context->device_output,
              context->stream, error, error_len)
        : qsa_bf16_candidate
        ? q38_cuda_bf16_matvec_configured(
              (const uint16_t *)exec->ptr, rows, cols,
              context->device_input, context->device_output, 256u,
              context->stream, error, error_len)
        : q38_cuda_matrix_batch_generic(
              tensor->type, exec->ptr, context->device_input, token_count,
              rows, cols, context->device_output, context->stream, error,
              error_len);
    if (!launched) {
        telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                 kernel_stop);
        return false;
    }
    if (context->current_stage &&
        (!strcmp(context->current_stage, "qsa_output_projection") ||
         !strcmp(context->current_stage, "gdn_output_projection")) &&
        !apply_directional_steering_device(
            context, context->device_output, context->current_layer, rows,
            token_count, context->directional_steering_attn_scale, error,
            error_len)) {
        telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                 kernel_stop);
        return false;
    }
    if (!telemetry_event_record(collect_telemetry, kernel_stop, context->stream)) {
        telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                 kernel_stop);
        return fail(error, error_len, "CUDA batched matrix timing failed");
    }
    if (cudaMemcpyAsync(output, context->device_output, output_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_MATRIX_BATCH_D2H,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        {
            telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                                     kernel_stop);
            return fail(error, error_len,
                        "CUDA batched matrix execution failed");
        }
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    context->device_output_elements = token_count * rows;
    Q38_CUDA_DIAG_ONLY(++context->persistent_hits);
    const float upload_ms =
        collect_telemetry ? event_elapsed(upload_start, upload_stop) : 0.0f;
    const float kernel_ms =
        collect_telemetry ? event_elapsed(kernel_start, kernel_stop) : 0.0f;
    emit_telemetry(context, model, tensor, token_count * rows, cols,
                   (size_t)tensor->bytes, true, false, 0, upload_ms, kernel_ms,
                   collect_telemetry ? host_now_ms() - started : 0.0, 0, 1,
                   "matrix_batch",
                   gdn_bf16_candidate ? "resident_gdn_projection"
                                     : "resident_exec_tensor");
    telemetry_events_destroy(upload_start, upload_stop, kernel_start,
                             kernel_stop);
    return true;
}

extern "C" bool q38_forward_cuda_gdn_layer_backend(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_forward_state *state, const float *input, size_t token_count,
    uint32_t layer_number, float *output, void *user, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    if (!context || !model || !layer || !state || !input || !output ||
        token_count != 1)
        return false;
    const q38_tensor *tensors[] = {
        layer->gdn.in_proj_qkv, layer->gdn.in_proj_z,
        layer->gdn.in_proj_a, layer->gdn.in_proj_b, layer->gdn.conv1d,
        layer->gdn.A_log, layer->gdn.dt_bias, layer->gdn.norm,
        layer->gdn.out_proj,
    };
    for (const q38_tensor *tensor : tensors) {
        if (!tensor || tensor->type != Q38_GDN_WEIGHT_BF16)
            return fail(error, error_len,
                        "GDN-C3 requires resident BF16 GDN tensors");
    }
    const int slot = q38_gdn_slot_for_layer(
        &state->storage.layout, layer_number);
    if (slot < 0 || (uint32_t)slot >= Q38_GDN_LAYER_COUNT)
        return fail(error, error_len, "invalid GDN-C3 layer slot");
    q38_exec_tensor *exec[9] = {};
    for (size_t i = 0; i < 9; ++i) {
        exec[i] = exec_tensor_for(context, model, tensors[i]);
        if (!exec[i] || !exec_tensor_is_resident(exec[i], tensors[i]))
            return fail(error, error_len,
                        "GDN-C3 requires resident execution tensors");
    }
    const size_t state_slot_elements =
        (size_t)Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;
    const size_t history_slot_elements =
        (size_t)(Q38_GDN_CONV_KERNEL - 1u) * Q38_GDN_CONV_CHANNELS;
    const size_t state_bytes =
        (size_t)Q38_GDN_LAYER_COUNT * state_slot_elements * sizeof(float);
    const size_t history_bytes =
        (size_t)Q38_GDN_LAYER_COUNT * history_slot_elements * sizeof(float);
    const auto ensure = [&](void **buffer, size_t *capacity, size_t bytes) {
        return ensure_buffer(buffer, capacity, bytes,
                             context->allocation_observer,
                             context->allocation_observer_user,
                             &context->cuda_allocations);
    };
    if (!ensure((void **)&context->device_gdn_input,
                &context->device_gdn_input_bytes,
                Q38_GDN_INPUT_DIM * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_qkv,
                &context->device_gdn_qkv_bytes,
                Q38_GDN_QKV_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_z,
                &context->device_gdn_z_bytes,
                Q38_GDN_Z_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_a,
                &context->device_gdn_a_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_b,
                &context->device_gdn_b_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_conv,
                &context->device_gdn_conv_bytes,
                Q38_GDN_QKV_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_q,
                &context->device_gdn_q_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_k,
                &context->device_gdn_k_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_v,
                &context->device_gdn_v_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_decay,
                &context->device_gdn_decay_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_beta,
                &context->device_gdn_beta_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_recurrent,
                &context->device_gdn_recurrent_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_gated,
                &context->device_gdn_gated_bytes,
                Q38_GDN_Z_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_output,
                &context->device_output_bytes,
                Q38_GR_HIDDEN * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_state,
                &context->device_gdn_state_bytes, state_bytes) ||
        !ensure((void **)&context->device_gdn_history,
                &context->device_gdn_history_bytes, history_bytes))
        return fail(error, error_len, "GDN-C3 workspace allocation failed");
    if (!context->device_gdn_state_initialized) {
        if (cudaMemsetAsync(context->device_gdn_state, 0, state_bytes,
                            context->stream) != cudaSuccess ||
            cudaMemsetAsync(context->device_gdn_history, 0, history_bytes,
                            context->stream) != cudaSuccess)
            return fail(error, error_len, "GDN-C3 state initialization failed");
        context->device_gdn_state_initialized = true;
    }
    float *device_state =
        context->device_gdn_state + (size_t)slot * state_slot_elements;
    float *device_history =
        context->device_gdn_history + (size_t)slot * history_slot_elements;
    if (cudaMemcpyAsync(
            context->device_gdn_input, input,
            Q38_GDN_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice,
            context->stream) != cudaSuccess)
        return fail(error, error_len, "GDN-C3 input upload failed");
    Q38_CUDA_DIAG_ONLY(context->gdn_c3_h2d_bytes += Q38_GDN_INPUT_DIM * sizeof(float));
    char cuda_error[256] = {};
    const auto project = [&](size_t index, size_t rows, size_t cols,
                             float *destination) {
        if (!q38_cuda_bf16_matvec_device(
                (const uint16_t *)exec[index]->ptr, rows, cols,
                context->device_gdn_input, destination, context->stream,
                cuda_error, sizeof(cuda_error)))
            return false;
        Q38_CUDA_DIAG_ONLY(++context->gdn_c3_launches);
        return true;
    };
    if (!project(0, Q38_GDN_QKV_CHANNELS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_qkv) ||
        !project(1, Q38_GDN_Z_CHANNELS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_z) ||
        !project(2, Q38_GDN_VALUE_HEADS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_a) ||
        !project(3, Q38_GDN_VALUE_HEADS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_b) ||
        !q38_cuda_gdn_fused_recurrent(
            context->device_gdn_qkv, context->device_gdn_z,
            context->device_gdn_a, context->device_gdn_b, Q38_GDN_WEIGHT_BF16,
            exec[4]->ptr, Q38_GDN_WEIGHT_BF16, exec[5]->ptr, exec[6]->ptr,
            Q38_GDN_WEIGHT_BF16, exec[7]->ptr, device_state, device_history,
            context->device_gdn_gated, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_gdn_history_update(
            context->device_gdn_qkv, 1, Q38_GDN_QKV_CHANNELS,
            Q38_GDN_CONV_KERNEL, device_history, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_bf16_matvec_device(
            (const uint16_t *)exec[8]->ptr, Q38_GR_HIDDEN,
            Q38_GDN_Z_CHANNELS, context->device_gdn_gated,
            context->device_output, context->stream, cuda_error,
            sizeof(cuda_error))) {
        return fail(error, error_len,
                    cuda_error[0] ? cuda_error : "GDN-C3 launch failed");
    }
    Q38_CUDA_DIAG_ONLY(context->gdn_c3_launches += 3);
    if (cudaMemcpyAsync(output, context->device_output,
                        Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_GDN_OUTPUT,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "GDN-C3 output transfer failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    Q38_CUDA_DIAG_ONLY(++context->gdn_c3_syncs);
    Q38_CUDA_DIAG_ONLY(++context->gdn_c3_calls);
    Q38_CUDA_DIAG_ONLY(context->gdn_c3_d2h_bytes += Q38_GR_HIDDEN * sizeof(float));
    return true;
}

extern "C" bool q38_forward_cuda_gdn_layer_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_layer_weights *layer, q38_forward_state *state,
    const float *device_input, uint32_t layer_number, float *device_output,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !model || !layer || !state || !device_input ||
        !device_output || layer_number >= Q38_MODEL_LAYERS)
        return false;
    const q38_tensor *tensors[] = {
        layer->gdn.in_proj_qkv, layer->gdn.in_proj_z,
        layer->gdn.in_proj_a, layer->gdn.in_proj_b, layer->gdn.conv1d,
        layer->gdn.A_log, layer->gdn.dt_bias, layer->gdn.norm,
        layer->gdn.out_proj,
    };
    for (const q38_tensor *tensor : tensors)
        if (!tensor || tensor->type != Q38_GDN_WEIGHT_BF16)
            return fail(error, error_len,
                        "GDN device chain requires BF16 tensors");
    const int slot = q38_gdn_slot_for_layer(
        &state->storage.layout, layer_number);
    if (slot < 0 || (uint32_t)slot >= Q38_GDN_LAYER_COUNT)
        return fail(error, error_len, "invalid GDN device chain slot");
    q38_exec_tensor *exec[9] = {};
    for (size_t i = 0; i < 9; ++i) {
        exec[i] = exec_tensor_for(context, model, tensors[i]);
        if (!exec[i] || !exec_tensor_is_resident(exec[i], tensors[i]))
            return fail(error, error_len,
                        "GDN device chain requires resident tensors");
    }
    const size_t state_slot_elements =
        (size_t)Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;
    const size_t history_slot_elements =
        (size_t)(Q38_GDN_CONV_KERNEL - 1u) * Q38_GDN_CONV_CHANNELS;
    const size_t state_bytes =
        (size_t)Q38_GDN_LAYER_COUNT * state_slot_elements * sizeof(float);
    const size_t history_bytes =
        (size_t)Q38_GDN_LAYER_COUNT * history_slot_elements * sizeof(float);
    const auto ensure = [&](void **buffer, size_t *capacity, size_t bytes) {
        return ensure_buffer(buffer, capacity, bytes,
                             context->allocation_observer,
                             context->allocation_observer_user,
                             &context->cuda_allocations);
    };
    if (!ensure((void **)&context->device_gdn_input,
                &context->device_gdn_input_bytes,
                Q38_GDN_INPUT_DIM * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_qkv,
                &context->device_gdn_qkv_bytes,
                Q38_GDN_QKV_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_z,
                &context->device_gdn_z_bytes,
                Q38_GDN_Z_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_a,
                &context->device_gdn_a_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_b,
                &context->device_gdn_b_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_conv,
                &context->device_gdn_conv_bytes,
                Q38_GDN_QKV_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_q,
                &context->device_gdn_q_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_k,
                &context->device_gdn_k_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_v,
                &context->device_gdn_v_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_decay,
                &context->device_gdn_decay_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_beta,
                &context->device_gdn_beta_bytes,
                Q38_GDN_VALUE_HEADS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_recurrent,
                &context->device_gdn_recurrent_bytes,
                Q38_GDN_VALUE_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_gated,
                &context->device_gdn_gated_bytes,
                Q38_GDN_Z_CHANNELS * sizeof(float)) ||
        !ensure((void **)&context->device_gdn_state,
                &context->device_gdn_state_bytes, state_bytes) ||
        !ensure((void **)&context->device_gdn_history,
                &context->device_gdn_history_bytes, history_bytes))
        return fail(error, error_len, "GDN device chain workspace allocation failed");
    if (!context->device_gdn_state_initialized) {
        if (cudaMemsetAsync(context->device_gdn_state, 0, state_bytes,
                            context->stream) != cudaSuccess ||
            cudaMemsetAsync(context->device_gdn_history, 0, history_bytes,
                            context->stream) != cudaSuccess)
            return fail(error, error_len, "GDN device chain state init failed");
        context->device_gdn_state_initialized = true;
    }
    float *device_state =
        context->device_gdn_state + (size_t)slot * state_slot_elements;
    float *device_history =
        context->device_gdn_history + (size_t)slot * history_slot_elements;
    char cuda_error[256] = {};
    const auto project = [&](size_t index, size_t rows, size_t cols,
                             float *destination) {
        return q38_cuda_bf16_matvec_device(
            (const uint16_t *)exec[index]->ptr, rows, cols, device_input,
            destination, context->stream, cuda_error, sizeof(cuda_error));
    };
    if (!project(0, Q38_GDN_QKV_CHANNELS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_qkv) ||
        !project(1, Q38_GDN_Z_CHANNELS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_z) ||
        !project(2, Q38_GDN_VALUE_HEADS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_a) ||
        !project(3, Q38_GDN_VALUE_HEADS, Q38_GDN_INPUT_DIM,
                 context->device_gdn_b) ||
        !q38_cuda_gdn_fused_recurrent(
            context->device_gdn_qkv, context->device_gdn_z,
            context->device_gdn_a, context->device_gdn_b, Q38_GDN_WEIGHT_BF16,
            exec[4]->ptr, Q38_GDN_WEIGHT_BF16, exec[5]->ptr, exec[6]->ptr,
            Q38_GDN_WEIGHT_BF16, exec[7]->ptr, device_state, device_history,
            context->device_gdn_gated, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_gdn_history_update(
            context->device_gdn_qkv, 1, Q38_GDN_QKV_CHANNELS,
            Q38_GDN_CONV_KERNEL, device_history, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_bf16_matvec_device(
            (const uint16_t *)exec[8]->ptr, Q38_GR_HIDDEN,
            Q38_GDN_Z_CHANNELS, context->device_gdn_gated, device_output,
            context->stream, cuda_error, sizeof(cuda_error)))
        return fail(error, error_len,
                    cuda_error[0] ? cuda_error : "GDN device chain launch failed");
    Q38_CUDA_DIAG_ONLY(++context->gdn_c3_calls);
    Q38_CUDA_DIAG_ONLY(context->gdn_c3_launches += 7);
    return true;
}

static unsigned gr_cooperative_grid(const void *kernel, unsigned minimum) {
    int device = 0;
    int multiprocessors = 0;
    int active_blocks = 0;
    int cooperative = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                               device) != cudaSuccess ||
        !cooperative ||
        cudaDeviceGetAttribute(&multiprocessors,
                               cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, kernel, 128, 0) != cudaSuccess ||
        active_blocks <= 0)
        return 0;
    const unsigned maximum =
        (unsigned)multiprocessors * (unsigned)active_blocks;
    return maximum >= minimum ? maximum : 0;
}

static bool ensure_gr_buffers(q38_forward_cuda_context *context) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    return ensure_buffer((void **)&context->device_gr_residual,
                         &context->device_gr_residual_bytes,
                         width * sizeof(float), context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_norm,
                         &context->device_gr_norm_bytes,
                         width * sizeof(float), context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_down,
                         &context->device_gr_down_bytes,
                         Q38_GR_RANK * sizeof(float),
                         context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_bottleneck,
                         &context->device_gr_bottleneck_bytes,
                         Q38_GR_RANK * sizeof(float),
                         context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_up,
                         &context->device_gr_up_bytes,
                         width * sizeof(float), context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_input,
                         &context->device_gr_input_bytes,
                         Q38_GR_HIDDEN * sizeof(float),
                         context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_block,
                         &context->device_gr_block_bytes,
                         Q38_GR_HIDDEN * sizeof(float),
                         context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_inject,
                         &context->device_gr_inject_bytes,
                         Q38_GR_BRANCHES * sizeof(float),
                         context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations) &&
           ensure_buffer((void **)&context->device_gr_updated,
                         &context->device_gr_updated_bytes,
                         width * sizeof(float), context->allocation_observer,
                         context->allocation_observer_user,
                         &context->cuda_allocations);
}

static bool gr_read_device_impl(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_gr_weights *weights, const float *device_residual,
    float *device_input, float *device_normed, char *error, size_t error_len) {
    size_t gamma_rows, gamma_cols, down_rows, down_cols, up_rows, up_cols;
    if (!context || !model || !weights || !device_residual || !device_input ||
        !device_normed ||
        !tensor_shape(weights->hc_norm, &gamma_rows, &gamma_cols) ||
        !tensor_shape(weights->input_mix_weight_down, &down_rows, &down_cols) ||
        !tensor_shape(weights->input_mix_weight_up, &up_rows, &up_cols) ||
        weights->hc_norm->type != 30 ||
        weights->input_mix_weight_down->type != 30 ||
        weights->input_mix_weight_up->type != 30 ||
        gamma_rows != 1 || gamma_cols != Q38_GR_BRANCHES * Q38_GR_HIDDEN ||
        down_rows != Q38_GR_RANK ||
        down_cols != Q38_GR_BRANCHES * Q38_GR_HIDDEN ||
        up_rows != Q38_GR_BRANCHES * Q38_GR_HIDDEN ||
        up_cols != Q38_GR_RANK)
        return false;
    q38_exec_tensor *gamma_exec =
        exec_tensor_for(context, model, weights->hc_norm);
    q38_exec_tensor *down_exec =
        exec_tensor_for(context, model, weights->input_mix_weight_down);
    q38_exec_tensor *up_exec =
        exec_tensor_for(context, model, weights->input_mix_weight_up);
    if (!exec_tensor_is_resident(gamma_exec, weights->hc_norm) ||
        !exec_tensor_is_resident(down_exec, weights->input_mix_weight_down) ||
        !exec_tensor_is_resident(up_exec, weights->input_mix_weight_up) ||
        !ensure_gr_buffers(context))
        return false;
    const unsigned available_grid =
        gr_cooperative_grid((const void *)gr_fused_normalize_down_kernel,
                                          Q38_GR_BRANCHES);
    const unsigned grid = available_grid > Q38_GR_RANK
        ? Q38_GR_RANK : available_grid;
    const unsigned lowrank_grid =
        gr_cooperative_grid((const void *)gr_fused_lowrank_up_kernel, 1);
    if (!grid || !lowrank_grid)
        return fail(error, error_len, "GR-C4 cooperative launch unavailable");
    void *normalize_args[] = {
        (void *)&device_residual, (void *)&gamma_exec->ptr,
        (void *)&down_exec->ptr, &context->device_gr_bottleneck,
        &context->device_gr_norm, &context->device_gr_down,
    };
    if (cudaLaunchCooperativeKernel(
                          (const void *)gr_fused_normalize_down_kernel, dim3(grid),
                          dim3(128), normalize_args, 0, context->stream) != cudaSuccess)
        return fail(error, error_len, "GR-C4 normalize/down launch failed");
    void *up_args[] = {
        (void *)&up_exec->ptr, &context->device_gr_down,
        &context->device_gr_bottleneck, &context->device_gr_up,
    };
    if (cudaLaunchCooperativeKernel(
                          (const void *)gr_fused_lowrank_up_kernel, dim3(lowrank_grid),
                          dim3(128), up_args, 0, context->stream) != cudaSuccess)
        return fail(error, error_len, "GR-C4 low-rank/up launch failed");
    gr_fused_branch_read_kernel<<<
        (Q38_GR_HIDDEN + 255u) / 256u, 256, 0, context->stream>>>(
        context->device_gr_norm, context->device_gr_up, device_input);
    return cudaGetLastError() == cudaSuccess ||
                         fail(error, error_len, "GR-C4 branch read launch failed");
}

static bool gr_write_device_impl(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_gr_weights *weights, const float *device_residual,
    float *device_normed, const float *device_block, float *device_updated,
    char *error, size_t error_len) {
    size_t inject_rows, inject_cols;
    if (!context || !model || !weights || !device_residual ||
        !device_normed || !device_block || !device_updated ||
        !tensor_shape(weights->block_inject_weight, &inject_rows,
                                    &inject_cols) ||
        weights->block_inject_weight->type != 30 ||
        inject_rows != Q38_GR_BRANCHES ||
        inject_cols != Q38_GR_BRANCHES * Q38_GR_HIDDEN)
        return false;
    q38_exec_tensor *inject_exec =
        exec_tensor_for(context, model, weights->block_inject_weight);
    q38_exec_tensor *gamma_exec =
        exec_tensor_for(context, model, weights->hc_norm);
    if (!exec_tensor_is_resident(inject_exec, weights->block_inject_weight) ||
        !exec_tensor_is_resident(gamma_exec, weights->hc_norm) ||
        !ensure_gr_buffers(context))
        return false;
    gr_normalize_kernel<<<Q38_GR_BRANCHES, 256, 0, context->stream>>>(
        device_residual, (const uint16_t *)gamma_exec->ptr,
        context->device_gr_bottleneck, device_normed);
    if (cudaGetLastError() != cudaSuccess ||
        !q38_cuda_bf16_matvec_configured(
                          (const uint16_t *)inject_exec->ptr, Q38_GR_BRANCHES,
                          Q38_GR_BRANCHES * Q38_GR_HIDDEN, device_normed,
                          context->device_gr_inject, 256, context->stream, error,
                          error_len))
        return fail(error, error_len, "GR-C4 inject launch failed");
    gr_fused_writeback_kernel<<<
        (Q38_GR_BRANCHES * Q38_GR_HIDDEN + 255u) / 256u, 256, 0,
        context->stream>>>(
        device_residual, device_block, context->device_gr_inject,
        device_updated);
    return cudaGetLastError() == cudaSuccess ||
                         fail(error, error_len, "GR-C4 writeback launch failed");
}

static bool ensure_qsa_chain_workspace(q38_forward_cuda_context *context,
                                                     char *error, size_t error_len);
static bool qsa_chain_tensor(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_tensor *tensor, size_t rows, size_t cols,
    const uint16_t **pointer, char *error, size_t error_len);

extern "C" bool q38_forward_cuda_gr_read_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_gr_weights *weights, const float *device_residual,
    float *device_input, float *device_normed, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    return gr_read_device_impl(context, model, weights, device_residual,
                                             device_input, device_normed, error, error_len);
}

extern "C" bool q38_forward_cuda_gr_write_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_gr_weights *weights, const float *device_residual,
    float *device_normed, const float *device_block, float *device_updated,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    return gr_write_device_impl(context, model, weights, device_residual,
                                              device_normed, device_block, device_updated,
                                              error, error_len);
}

extern "C" bool q38_forward_cuda_qsa_chain_device(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_layer_weights *layer, q38_qsa_state *host_state,
    const float *device_input, uint32_t layer_number, float *device_output,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !model || !layer || !host_state || !device_input ||
        !device_output || layer_number >= Q38_MODEL_LAYERS)
        return false;
    q38_qsa_cuda_chain_state *chain =
        &context->qsa_chain_state[layer_number];
    if (host_state->position == 0 && chain->position != 0)
        q38_qsa_cuda_chain_reset(chain);
    if (chain->position != host_state->position)
        return fail(error, error_len,
                    "QSA device chain state is not seeded for this position");
    const q38_qsa_weights *weights = &layer->qsa;
    const uint16_t *q_proj, *k_proj, *v_proj, *index_proj, *o_proj;
    const uint16_t *q_norm, *k_norm, *index_q_norm, *index_k_norm;
    if (!qsa_chain_tensor(context, model, weights->q_proj, 12288, 2560,
                          &q_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->k_proj, 512, 2560,
                          &k_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->v_proj, 512, 2560,
                          &v_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->index_qk_proj, 640, 2560,
                          &index_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->o_proj, 2560, 6144,
                          &o_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->q_norm, 1, 256,
                          &q_norm, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->k_norm, 1, 256,
                          &k_norm, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->index_q_norm, 1, 128,
                          &index_q_norm, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->index_k_norm, 1, 128,
                          &index_k_norm, error, error_len) ||
        !ensure_qsa_chain_workspace(context, error, error_len))
        return false;
    const size_t required = chain->count + 1u;
    size_t capacity = chain->capacity ? chain->capacity * 2u : 16u;
    if (capacity < required) capacity = required;
    if (!q38_qsa_cuda_chain_reserve(chain, capacity, context->stream,
                                                  error, error_len))
        return false;
    if (!q38_qsa_cuda_chain_decode(
            q_proj, k_proj, v_proj, index_proj, o_proj, q_norm, k_norm,
            index_q_norm, index_k_norm, device_input, device_output,
            host_state->position, chain, &context->qsa_chain_workspace,
            context->stream, error, error_len))
        return false;
    if (!q38_qsa_state_advance_device(host_state, 1, error, error_len))
        return false;
    ++context->qsa_chain_calls;
    context->qsa_chain_kernel_launches += 9;
    return true;
}

extern "C" bool q38_forward_cuda_decoder_layer_chain_backend(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_forward_state *state, uint32_t layer_number, const float *host_input,
    float *host_output, void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    if (!context || !model || !layer || !state || !host_input || !host_output ||
        layer_number >= Q38_MODEL_LAYERS)
        return fail(error, error_len, "invalid decoder layer chain arguments");
    const auto ensure = [&](void **buffer, size_t *capacity, size_t bytes) {
        return ensure_buffer(buffer, capacity, bytes,
                             context->allocation_observer,
                             context->allocation_observer_user,
                             &context->cuda_allocations);
    };
    if (!ensure((void **)&context->device_hidden_a,
                &context->device_hidden_a_bytes,
                Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float)) ||
        !ensure((void **)&context->device_hidden_b,
                &context->device_hidden_b_bytes,
                Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float)) ||
        !ensure((void **)&context->device_moe_logits,
                &context->device_moe_logits_bytes,
                Q38_MOE_EXPERTS * sizeof(float)) ||
        !ensure((void **)&context->device_moe_route_indices,
                &context->device_moe_route_indices_bytes,
                Q38_MOE_TOP_K * sizeof(uint32_t)) ||
        !ensure((void **)&context->device_moe_route_ids,
                &context->device_moe_route_ids_bytes,
                Q38_MOE_TOP_K * sizeof(uint16_t)) ||
        !ensure((void **)&context->device_moe_route_weights,
                &context->device_moe_route_weights_bytes,
                Q38_MOE_TOP_K * sizeof(float)) ||
        !ensure((void **)&context->device_moe_accum,
                &context->device_moe_accum_bytes,
                Q38_MOE_HIDDEN * sizeof(float)) ||
        !ensure((void **)&context->device_moe_grouped_mid,
                &context->device_moe_grouped_mid_bytes,
                Q38_MOE_TOP_K * Q38_MOE_INTERMEDIATE * sizeof(float)) ||
        !ensure((void **)&context->device_moe_expert_outputs,
                &context->device_moe_expert_outputs_bytes,
                Q38_MOE_TOP_K * Q38_MOE_HIDDEN * sizeof(float)) ||
        !ensure((void **)&context->device_moe_shared_gate,
                &context->device_moe_shared_gate_bytes,
                Q38_MOE_INTERMEDIATE * sizeof(float)) ||
        !ensure((void **)&context->device_moe_shared_up,
                &context->device_moe_shared_up_bytes,
                Q38_MOE_INTERMEDIATE * sizeof(float)) ||
        !ensure((void **)&context->device_moe_shared_output,
                &context->device_moe_shared_output_bytes,
                Q38_MOE_HIDDEN * sizeof(float)) ||
        !ensure((void **)&context->device_moe_shared_weight,
                &context->device_moe_shared_weight_bytes, sizeof(float)))
        return fail(error, error_len, "decoder layer chain workspace allocation failed");
    if (cudaMemcpyAsync(
            context->device_hidden_a, host_input,
            Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
            cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "decoder layer chain input upload failed");

    if (!q38_forward_cuda_gr_read_device(
            context, model, &layer->attn_gr, context->device_hidden_a,
            context->device_gr_input, context->device_gr_norm, error,
            error_len))
        return false;
    if (layer->kind == Q38_LAYER_LINEAR_ATTENTION) {
        if (!q38_forward_cuda_gdn_layer_device(
                context, model, layer, state, context->device_gr_input,
                layer_number, context->device_gr_block, error, error_len))
            return false;
    } else {
        if (!q38_forward_cuda_qsa_chain_device(
                context, model, layer, &state->qsa[layer_number],
                context->device_gr_input, layer_number,
                context->device_gr_block, error, error_len))
            return false;
    }
    if (!q38_forward_cuda_gr_write_device(
            context, model, &layer->attn_gr, context->device_hidden_a,
            context->device_gr_norm, context->device_gr_block,
            context->device_hidden_b, error, error_len))
        return false;
    if (!q38_forward_cuda_gr_read_device(
            context, model, &layer->mlp_gr, context->device_hidden_b,
            context->device_gr_input, context->device_gr_norm, error,
            error_len))
        return false;

    const q38_tensor *router = layer->router;
    const q38_tensor *gate_up = layer->experts.bank_count
        ? layer->experts.bank[0].gate_up : NULL;
    const q38_tensor *down = layer->experts.bank_count
        ? layer->experts.bank[0].down : NULL;
    const q38_tensor *shared_gate_proj = layer->shared_gate_proj;
    const q38_tensor *shared_up_proj = layer->shared_up_proj;
    const q38_tensor *shared_down_proj = layer->shared_down_proj;
    const q38_tensor *shared_gate = layer->shared_expert_gate;
    const q38_tensor *moe_tensors[] = {
        router, gate_up, down, shared_gate_proj, shared_up_proj,
        shared_down_proj, shared_gate,
    };
    for (const q38_tensor *tensor : moe_tensors)
        if (!tensor)
            return fail(error, error_len,
                        "decoder layer chain MoE tensor set is incomplete");
    q38_exec_tensor *router_exec = exec_tensor_for(context, model, router);
    q38_exec_tensor *gate_exec = exec_tensor_for(context, model, gate_up);
    q38_exec_tensor *down_exec = exec_tensor_for(context, model, down);
    q38_exec_tensor *shared_gate_exec =
        exec_tensor_for(context, model, shared_gate_proj);
    q38_exec_tensor *shared_up_exec =
        exec_tensor_for(context, model, shared_up_proj);
    q38_exec_tensor *shared_down_exec =
        exec_tensor_for(context, model, shared_down_proj);
    q38_exec_tensor *shared_weight_exec =
        exec_tensor_for(context, model, shared_gate);
    if (router->type != Q38_GDN_WEIGHT_BF16 ||
        gate_up->type != Q38_QUANT_Q2_K || down->type != Q38_QUANT_Q2_K ||
        shared_gate_proj->type != Q38_GDN_WEIGHT_BF16 ||
        shared_up_proj->type != Q38_GDN_WEIGHT_BF16 ||
        shared_down_proj->type != Q38_GDN_WEIGHT_BF16 ||
        shared_gate->type != Q38_GDN_WEIGHT_BF16 ||
        !exec_tensor_is_resident(router_exec, router) ||
        !exec_tensor_is_resident(gate_exec, gate_up) ||
        !exec_tensor_is_resident(down_exec, down) ||
        !exec_tensor_is_resident(shared_gate_exec, shared_gate_proj) ||
        !exec_tensor_is_resident(shared_up_exec, shared_up_proj) ||
        !exec_tensor_is_resident(shared_down_exec, shared_down_proj) ||
        !exec_tensor_is_resident(shared_weight_exec, shared_gate))
        return fail(error, error_len,
                    "decoder layer chain requires resident MoE tensors");
    char cuda_error[256] = {};
    if (!q38_cuda_bf16_matvec_device(
            (const uint16_t *)router_exec->ptr, Q38_MOE_EXPERTS,
            Q38_MOE_HIDDEN, context->device_gr_input,
            context->device_moe_logits, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_moe_cuda_route_weights(
            context->device_moe_logits, 1,
            context->device_moe_route_indices, context->device_moe_route_ids,
            context->device_moe_route_weights, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_moe_cuda_q2_grouped_indexed_deterministic(
            gate_exec->ptr, down_exec->ptr, context->device_gr_input,
            context->device_moe_route_ids, context->device_moe_route_weights,
            Q38_MOE_TOP_K, 1280u * 10u, 640u * 10u,
            context->device_moe_accum, context->device_moe_grouped_mid,
            context->device_moe_expert_outputs, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_bf16_matvec_device(
            (const uint16_t *)shared_gate_exec->ptr, Q38_MOE_INTERMEDIATE,
            Q38_MOE_HIDDEN, context->device_gr_input,
            context->device_moe_shared_gate, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_bf16_matvec_device(
            (const uint16_t *)shared_up_exec->ptr, Q38_MOE_INTERMEDIATE,
            Q38_MOE_HIDDEN, context->device_gr_input,
            context->device_moe_shared_up, context->stream, cuda_error,
            sizeof(cuda_error)))
        return fail(error, error_len,
                    cuda_error[0] ? cuda_error : "decoder MoE launch failed");
    moe_shared_silu_mul_kernel<<<
        (Q38_MOE_INTERMEDIATE + 255u) / 256u, 256, 0, context->stream>>>(
        context->device_moe_shared_gate, context->device_moe_shared_up,
        context->device_moe_mid, Q38_MOE_INTERMEDIATE);
    if (cudaGetLastError() != cudaSuccess ||
        !q38_cuda_bf16_matvec_device(
            (const uint16_t *)shared_down_exec->ptr, Q38_MOE_HIDDEN,
            Q38_MOE_INTERMEDIATE, context->device_moe_mid,
            context->device_moe_shared_output, context->stream, cuda_error,
            sizeof(cuda_error)) ||
        !q38_cuda_bf16_matvec_device(
            (const uint16_t *)shared_weight_exec->ptr, 1, Q38_MOE_HIDDEN,
            context->device_gr_input, context->device_moe_shared_weight,
            context->stream, cuda_error, sizeof(cuda_error)))
        return fail(error, error_len,
                    cuda_error[0] ? cuda_error : "decoder shared MoE launch failed");
    moe_shared_add_kernel<<<
        (Q38_MOE_HIDDEN + 255u) / 256u, 256, 0, context->stream>>>(
        context->device_moe_accum, context->device_moe_shared_output,
        context->device_moe_shared_weight, context->device_gr_block,
        Q38_MOE_HIDDEN);
    if (cudaGetLastError() != cudaSuccess)
        return fail(error, error_len, "decoder MoE reduction launch failed");
    Q38_CUDA_DIAG_ONLY(++context->routed_layers_executed);
    Q38_CUDA_DIAG_ONLY(context->selected_experts_total += Q38_MOE_TOP_K);
    Q38_CUDA_DIAG_ONLY(context->q2_gate_up_fast_calls += Q38_MOE_TOP_K);
    Q38_CUDA_DIAG_ONLY(context->q2_down_calls += Q38_MOE_TOP_K);
    Q38_CUDA_DIAG_ONLY(context->expert_kernel_launches += 5);
    if (!apply_directional_steering_device(
            context, context->device_gr_block, layer_number,
            Q38_GR_HIDDEN, 1, context->directional_steering_ffn_scale, error,
            error_len) ||
        !q38_forward_cuda_gr_write_device(
            context, model, &layer->mlp_gr, context->device_hidden_b,
            context->device_gr_norm, context->device_gr_block,
            context->device_hidden_a, error, error_len))
        return false;
    if (cudaMemcpyAsync(
            host_output, context->device_hidden_a,
            Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
            cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_GR_WRITE,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "decoder layer chain output download failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    return true;
}

extern "C" bool q38_forward_cuda_gr_read_backend(
    const q38_gguf *model, const q38_gr_weights *weights,
    const float *residual, size_t token_count, float *input, float *normed,
    void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t gamma_rows, gamma_cols, down_rows, down_cols, up_rows, up_cols;
    if (!context || !model || !weights || !residual || !input || !normed ||
        token_count != 1)
        return false;
    if (!tensor_shape(weights->hc_norm, &gamma_rows, &gamma_cols) ||
        !tensor_shape(weights->input_mix_weight_down, &down_rows, &down_cols) ||
        !tensor_shape(weights->input_mix_weight_up, &up_rows, &up_cols) ||
        weights->hc_norm->type != 30 ||
        weights->input_mix_weight_down->type != 30 ||
        weights->input_mix_weight_up->type != 30 ||
        gamma_rows != 1 || gamma_cols != Q38_GR_BRANCHES * Q38_GR_HIDDEN ||
        down_rows != Q38_GR_RANK ||
        down_cols != Q38_GR_BRANCHES * Q38_GR_HIDDEN ||
        up_rows != Q38_GR_BRANCHES * Q38_GR_HIDDEN ||
        up_cols != Q38_GR_RANK)
        return false;

    q38_exec_tensor *gamma_exec =
        exec_tensor_for(context, model, weights->hc_norm);
    q38_exec_tensor *down_exec =
        exec_tensor_for(context, model, weights->input_mix_weight_down);
    q38_exec_tensor *up_exec =
        exec_tensor_for(context, model, weights->input_mix_weight_up);
    if (!exec_tensor_is_resident(gamma_exec, weights->hc_norm) ||
        !exec_tensor_is_resident(down_exec, weights->input_mix_weight_down) ||
        !exec_tensor_is_resident(up_exec, weights->input_mix_weight_up))
        return false;
    if (!ensure_gr_buffers(context))
        return fail(error, error_len, "GR-C4 read workspace allocation failed");

    const unsigned available_grid =
        gr_cooperative_grid((const void *)gr_fused_normalize_down_kernel,
                            Q38_GR_BRANCHES);
    const unsigned grid = available_grid > Q38_GR_RANK
        ? Q38_GR_RANK : available_grid;
    const unsigned lowrank_grid =
        gr_cooperative_grid((const void *)gr_fused_lowrank_up_kernel, 1);
    if (!grid || !lowrank_grid) return false;
    if (cudaMemcpyAsync(context->device_gr_residual, residual,
                        Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "GR-C4 read upload failed");

    void *normalize_args[] = {
        &context->device_gr_residual,
        (void *)&gamma_exec->ptr,
        (void *)&down_exec->ptr,
        &context->device_gr_bottleneck,
        &context->device_gr_norm,
        &context->device_gr_down,
    };
    if (cudaLaunchCooperativeKernel(
            (const void *)gr_fused_normalize_down_kernel, dim3(grid),
            dim3(128), normalize_args, 0, context->stream) != cudaSuccess)
        return fail(error, error_len, "GR-C4 normalize/down launch failed");

    void *up_args[] = {
        (void *)&up_exec->ptr,
        &context->device_gr_down,
        &context->device_gr_bottleneck,
        &context->device_gr_up,
    };
    if (cudaLaunchCooperativeKernel(
            (const void *)gr_fused_lowrank_up_kernel, dim3(lowrank_grid),
            dim3(128), up_args, 0, context->stream) != cudaSuccess)
        return fail(error, error_len, "GR-C4 low-rank/up launch failed");

    gr_fused_branch_read_kernel<<<
        (Q38_GR_HIDDEN + 255u) / 256u, 256, 0, context->stream>>>(
            context->device_gr_norm, context->device_gr_up,
            context->device_gr_input);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(normed, context->device_gr_norm,
                        Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(input, context->device_gr_input,
                        Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_GR_READ,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "GR-C4 read completion failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    return true;
}

extern "C" bool q38_forward_cuda_gr_write_backend(
    const q38_gguf *model, const q38_gr_weights *weights,
    const float *residual, float *normed, const float *block,
    size_t token_count, float *updated, void *user, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t inject_rows, inject_cols;
    if (!context || !model || !weights || !residual || !normed || !block ||
        !updated || token_count != 1 ||
        !tensor_shape(weights->block_inject_weight, &inject_rows,
                      &inject_cols) ||
        weights->block_inject_weight->type != 30 ||
        inject_rows != Q38_GR_BRANCHES ||
        inject_cols != Q38_GR_BRANCHES * Q38_GR_HIDDEN)
        return false;
    q38_exec_tensor *inject_exec =
        exec_tensor_for(context, model, weights->block_inject_weight);
    if (!exec_tensor_is_resident(inject_exec, weights->block_inject_weight) ||
        !ensure_gr_buffers(context))
        return false;

    const uint16_t *gamma = NULL;
    q38_exec_tensor *gamma_exec =
        exec_tensor_for(context, model, weights->hc_norm);
    if (!exec_tensor_is_resident(gamma_exec, weights->hc_norm))
        return false;
    if (!ensure_gr_buffers(context))
        return fail(error, error_len, "GR-C4 write workspace allocation failed");
    gamma = (const uint16_t *)gamma_exec->ptr;
    if (cudaMemcpyAsync(context->device_gr_residual, residual,
                        Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_gr_block, block,
                        Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "GR-C4 write upload failed");

    gr_normalize_kernel<<<Q38_GR_BRANCHES, 256, 0, context->stream>>>(
        context->device_gr_residual, gamma,
        context->device_gr_bottleneck, context->device_gr_norm);
    if (cudaGetLastError() != cudaSuccess ||
        !q38_cuda_bf16_matvec_configured(
            (const uint16_t *)inject_exec->ptr, Q38_GR_BRANCHES,
            Q38_GR_BRANCHES * Q38_GR_HIDDEN, context->device_gr_norm,
            context->device_gr_inject, 256, context->stream, error,
            error_len))
        return fail(error, error_len, "GR-C4 inject launch failed");
    gr_fused_writeback_kernel<<<
        (Q38_GR_BRANCHES * Q38_GR_HIDDEN + 255u) / 256u, 256, 0,
        context->stream>>>(
        context->device_gr_residual, context->device_gr_block,
        context->device_gr_inject, context->device_gr_updated);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(normed, context->device_gr_norm,
                        Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(updated, context->device_gr_updated,
                        Q38_GR_BRANCHES * Q38_GR_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_GR_WRITE,
                           cudaStreamSynchronize(context->stream)) !=
            cudaSuccess)
        return fail(error, error_len, "GR-C4 write completion failed");
    Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    return true;
}

static bool ensure_qsa_chain_workspace(q38_forward_cuda_context *context,
                                       char *error, size_t error_len) {
    if (context->qsa_chain_workspace_ready) return true;
    q38_qsa_cuda_chain_workspace *workspace = &context->qsa_chain_workspace;
    size_t workspace_bytes = 0;
    const size_t selected_capacity = 2051u;
    const size_t selected_kv_elements = selected_capacity * 2u * 256u;
    if (!ensure_buffer((void **)&workspace->qfull, &workspace_bytes,
                       12288u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->q, &workspace_bytes,
                       6144u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->k, &workspace_bytes,
                       512u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->v, &workspace_bytes,
                       512u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->index, &workspace_bytes,
                       640u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->index_q, &workspace_bytes,
                       512u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->raw_index, &workspace_bytes,
                       128u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->attention, &workspace_bytes,
                       6144u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->selected_k, &workspace_bytes,
                       selected_kv_elements * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->selected_v, &workspace_bytes,
                       selected_kv_elements * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace_bytes = 0;
    if (!ensure_buffer((void **)&workspace->selected, &workspace_bytes,
                       selected_capacity * sizeof(uint32_t),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations)) {
        return fail(error, error_len, "QSA chain workspace allocation failed");
    }
    workspace->selected_capacity = selected_capacity;
    context->qsa_chain_workspace_ready = true;
    return true;
}

static bool qsa_chain_tensor(
    q38_forward_cuda_context *context, const q38_gguf *model,
    const q38_tensor *tensor, size_t rows, size_t cols,
    const uint16_t **pointer, char *error, size_t error_len) {
    size_t actual_rows = 0, actual_cols = 0;
    q38_exec_tensor *exec = exec_tensor_for(context, model, tensor);
    if (!tensor || tensor->type != 30 || !tensor_shape(tensor, &actual_rows,
                                                         &actual_cols) ||
        actual_rows != rows || actual_cols != cols ||
        !exec_tensor_is_resident(exec, tensor))
        return fail(error, error_len, "QSA chain requires resident BF16 tensors");
    *pointer = (const uint16_t *)exec->ptr;
    return true;
}

extern "C" bool q38_forward_cuda_qsa_chain_backend(
    const q38_gguf *model, const q38_layer_weights *layer,
    q38_qsa_state *host_state, const float *host_input, size_t token_count,
    uint32_t layer_number, float *host_output, q38_forward_qsa_timing *timing,
    void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    if (!context || !model || !layer || !host_state || !host_input ||
        token_count != 1 || !host_output || !timing ||
        layer_number >= Q38_MODEL_LAYERS)
        return false;
    q38_qsa_cuda_chain_state *chain =
        &context->qsa_chain_state[layer_number];
    if (host_state->position != chain->position &&
        chain->position != 0 && host_state->position != 0)
        return false;
    const q38_qsa_weights *weights = &layer->qsa;
    const uint16_t *q_proj, *k_proj, *v_proj, *index_proj, *o_proj;
    const uint16_t *q_norm, *k_norm, *index_q_norm, *index_k_norm;
    if (!qsa_chain_tensor(context, model, weights->q_proj, 12288, 2560,
                          &q_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->k_proj, 512, 2560,
                          &k_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->v_proj, 512, 2560,
                          &v_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->index_qk_proj, 640, 2560,
                          &index_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->o_proj, 2560, 6144,
                          &o_proj, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->q_norm, 1, 256,
                          &q_norm, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->k_norm, 1, 256,
                          &k_norm, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->index_q_norm, 1, 128,
                          &index_q_norm, error, error_len) ||
        !qsa_chain_tensor(context, model, weights->index_k_norm, 1, 128,
                          &index_k_norm, error, error_len))
        return false;
    if (!ensure_qsa_chain_workspace(context, error, error_len))
        return false;
    if (host_state->position == 0 && chain->position != 0)
        q38_qsa_cuda_chain_reset(chain);
    if (chain->position == 0 && host_state->position != 0) {
        if (host_state->main_k.count != host_state->position ||
            host_state->main_v.count != host_state->position ||
            host_state->index_k.count != host_state->position)
            return false;
        if (!q38_qsa_cuda_chain_reserve(
                chain, host_state->main_k.count, context->stream, error,
                error_len) ||
            cudaMemcpyAsync(
                chain->main_k, host_state->main_k.data,
                host_state->main_k.count * host_state->main_k.row_bytes,
                cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
            cudaMemcpyAsync(
                chain->main_v, host_state->main_v.data,
                host_state->main_v.count * host_state->main_v.row_bytes,
                cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
            cudaMemcpyAsync(
                chain->index_k, host_state->index_k.data,
                host_state->index_k.count * host_state->index_k.row_bytes,
                cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
            return fail(error, error_len, "QSA chain state seed failed");
        chain->count = host_state->position;
        chain->position = host_state->position;
    }
    const size_t required = chain->count + 1;
    size_t reserve_capacity = chain->capacity ? chain->capacity * 2u : 16u;
    if (reserve_capacity < required) reserve_capacity = required;
    if (!q38_qsa_cuda_chain_reserve(chain, reserve_capacity, context->stream,
                                    error,
                                    error_len))
        return false;
    if (!context->device_input ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       2560u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes,
                       2560u * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "QSA chain boundary workspace allocation failed");
    if (cudaMemcpyAsync(context->device_input, host_input,
                        2560u * sizeof(float), cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess)
        return fail(error, error_len, "QSA chain input upload failed");
    const double started = host_now_ms();
    if (!q38_qsa_cuda_chain_decode(
            q_proj, k_proj, v_proj, index_proj, o_proj, q_norm, k_norm,
            index_q_norm, index_k_norm, context->device_input,
            context->device_output, host_state->position, chain,
            &context->qsa_chain_workspace, context->stream, error, error_len))
        return false;
    if (cudaMemcpyAsync(host_output, context->device_output,
                        2560u * sizeof(float), cudaMemcpyDeviceToHost,
                        context->stream) != cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return fail(error, error_len, "QSA chain output download failed");
    if (!q38_qsa_state_advance_device(host_state, 1, error, error_len))
        return false;
    timing->qkv_backend_used = true;
    timing->output_projection_backend_used = true;
    timing->qkv_projection_ms = host_now_ms() - started;
    timing->q_projection_ms = timing->qkv_projection_ms;
    timing->k_projection_ms = 0.0;
    timing->v_projection_ms = 0.0;
    timing->qkv_projection_ms = timing->qkv_projection_ms;
    timing->kernel_launches = 9;
    timing->host_syncs = 1;
    timing->h2d_bytes = 2560u * sizeof(float);
    timing->d2h_bytes = 2560u * sizeof(float);
    timing->total_ms = timing->qkv_projection_ms;
    ++context->qsa_chain_calls;
    context->qsa_chain_kernel_launches += timing->kernel_launches;
    ++context->qsa_chain_syncs;
    context->qsa_chain_h2d_bytes += timing->h2d_bytes;
    context->qsa_chain_d2h_bytes += timing->d2h_bytes;
    return true;
}

extern "C" bool q38_forward_cuda_qsa_qkv_backend(
    const q38_gguf *model, const q38_tensor *q_proj,
    const q38_tensor *k_proj, const q38_tensor *v_proj,
    const float *host_input, size_t token_count, float *host_q,
    float *host_k, float *host_v, q38_forward_qsa_timing *timing,
    void *user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    q38_forward_cuda_context *context =
        (q38_forward_cuda_context *)user;
    size_t q_rows, q_cols, k_rows, k_cols, v_rows, v_cols;
    if (!context || !model || !q_proj || !k_proj || !v_proj ||
        !host_input || !token_count || !host_q || !host_k || !host_v ||
        !timing || !tensor_shape(q_proj, &q_rows, &q_cols) ||
        !tensor_shape(k_proj, &k_rows, &k_cols) ||
        !tensor_shape(v_proj, &v_rows, &v_cols) ||
        q_rows != 12288 || k_rows != 512 || v_rows != 512 ||
        q_cols != 2560 || k_cols != 2560 || v_cols != 2560 ||
        q_proj->type != 30 || k_proj->type != 30 || v_proj->type != 30)
        return fail(error, error_len, "invalid resident QSA QKV geometry");
    if (token_count > SIZE_MAX / q_rows ||
        token_count > SIZE_MAX / k_rows ||
        token_count > SIZE_MAX / v_rows ||
        token_count * q_rows > SIZE_MAX - token_count * k_rows ||
        token_count * (q_rows + k_rows) >
            SIZE_MAX - token_count * v_rows)
        return fail(error, error_len, "QSA QKV size overflow");

    q38_exec_tensor *q_exec = exec_tensor_for(context, model, q_proj);
    q38_exec_tensor *k_exec = exec_tensor_for(context, model, k_proj);
    q38_exec_tensor *v_exec = exec_tensor_for(context, model, v_proj);
    if (!exec_tensor_is_resident(q_exec, q_proj) ||
        !exec_tensor_is_resident(k_exec, k_proj) ||
        !exec_tensor_is_resident(v_exec, v_proj))
        return fail(error, error_len,
                    "QSA QKV tensor is not resident on the CUDA path");

    const size_t q_elements = token_count * q_rows;
    const size_t k_elements = token_count * k_rows;
    const size_t v_elements = token_count * v_rows;
    const size_t output_elements = q_elements + k_elements + v_elements;
    const size_t input_bytes = token_count * q_cols * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const uint64_t allocations_before =
        Q38_DIAG_ENABLED ? context->cuda_allocations : 0;
    const double started = Q38_DIAG_ENABLED ? host_now_ms() : 0.0;
    if (!ensure_buffer((void **)&context->device_qsa_input,
                       &context->device_qsa_input_bytes, input_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations) ||
        !ensure_buffer((void **)&context->device_qsa_output,
                       &context->device_qsa_output_bytes, output_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "QSA QKV CUDA workspace allocation failed");
    if (context->host_qsa_output_bytes < output_bytes) {
        float *grown = (float *)realloc(context->host_qsa_output, output_bytes);
        if (!grown)
            return fail(error, error_len,
                        "QSA QKV host staging allocation failed");
        context->host_qsa_output = grown;
        context->host_qsa_output_bytes = output_bytes;
        Q38_CUDA_DIAG_ONLY(++timing->allocations);
    }
    float *device_q = context->device_qsa_output;
    float *device_k = device_q + q_elements;
    float *device_v = device_k + k_elements;
    bool projection_ok = cudaMemcpyAsync(
                            context->device_qsa_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, context->stream) ==
                        cudaSuccess;
    if (projection_ok) {
        projection_ok = context->qsa_candidate
            ? context->qsa_candidate(
                  q_exec->ptr, k_exec->ptr, v_exec->ptr,
                  context->device_qsa_input, device_q, device_k, device_v,
                  token_count, q_cols, (void *)context->stream) == 0
            : q38_qsa_cuda_project_main(
                  (const uint16_t *)q_exec->ptr, q_rows,
                  (const uint16_t *)k_exec->ptr, k_rows,
                  (const uint16_t *)v_exec->ptr, v_rows, q_cols,
                  context->device_qsa_input, token_count, device_q, device_k,
                  device_v, context->stream, error, error_len);
    }
    if (projection_ok)
        projection_ok = cudaMemcpyAsync(
                            context->host_qsa_output, context->device_qsa_output,
                            output_bytes, cudaMemcpyDeviceToHost,
                            context->stream) == cudaSuccess;
    if (projection_ok)
        projection_ok =
            Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_QSA_QKV,
                               cudaStreamSynchronize(context->stream)) ==
            cudaSuccess;
    if (!projection_ok) {
        if (error && error_len && error[0] == '\0')
            snprintf(error, error_len, "QSA QKV CUDA execution failed: %s",
                     cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    memcpy(host_q, context->host_qsa_output, q_elements * sizeof(float));
    memcpy(host_k, context->host_qsa_output + q_elements,
           k_elements * sizeof(float));
    memcpy(host_v, context->host_qsa_output + q_elements + k_elements,
           v_elements * sizeof(float));
    Q38_CUDA_DIAG_ONLY(timing->qkv_projection_ms += host_now_ms() - started);
    Q38_CUDA_DIAG_ONLY(timing->allocations +=
                       context->cuda_allocations - allocations_before);
    Q38_CUDA_DIAG_ONLY(timing->kernel_launches += 3);
    Q38_CUDA_DIAG_ONLY(timing->host_syncs++);
    Q38_CUDA_DIAG_ONLY(timing->h2d_bytes += input_bytes);
    Q38_CUDA_DIAG_ONLY(timing->d2h_bytes += output_bytes);
    Q38_CUDA_DIAG_ONLY(++context->persistent_hits);
    return true;
}

extern "C" bool q38_forward_cuda_greedy_argmax(
    q38_forward_cuda_context *context, uint32_t *token, char *error,
    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !token || !context->device_output ||
        context->device_output_elements == 0)
        return fail(error, error_len, "CUDA greedy argmax result is unavailable");
    if (!ensure_buffer((void **)&context->device_argmax,
                       &context->device_argmax_bytes,
                       sizeof(*context->device_argmax),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "CUDA greedy argmax allocation failed");
    bool launched = false;
#if Q38_DIAGNOSTICS
    cudaEvent_t start = NULL, stop = NULL;
    if (cudaEventCreate(&start) == cudaSuccess &&
        cudaEventCreate(&stop) == cudaSuccess) {
        cudaEventRecord(start, context->stream);
        launched = q38_argmax_cuda(
            context->device_output, 1, context->device_output_elements,
            context->device_argmax, context->stream, error, error_len);
        cudaEventRecord(stop, context->stream);
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_ARGMAX,
                           cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        if (cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess)
            Q38_CUDA_DIAG_ONLY(context->gpu_argmax_kernel_ms = elapsed);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
#else
    launched = q38_argmax_cuda(
        context->device_output, 1, context->device_output_elements,
        context->device_argmax, context->stream, error, error_len);
#endif
    bool ok = launched &&
        cudaMemcpyAsync(token, context->device_argmax, sizeof(*token),
                        cudaMemcpyDeviceToHost, context->stream) == cudaSuccess &&
        Q38_CUDA_SYNC_CALL(context, Q38_CUDA_SYNC_ARGMAX,
                           cudaStreamSynchronize(context->stream)) ==
            cudaSuccess;
    if (!ok && (!error || !error_len || !error[0]))
        fail(error, error_len, "CUDA greedy argmax execution failed");
    if (ok) Q38_CUDA_DIAG_ONLY(++context->cuda_synchronizations);
    return ok;
}
