#include "q38_forward_cuda.h"

#include "q38_cuda_primitives.h"
#include "q38_gdn.h"
#include "q38_moe_cuda.h"
#include "q38_qsa_cuda.h"
#include "q38_topk_cuda.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
    uint32_t *device_argmax;
    size_t device_argmax_bytes;
    void *device_aux;
    size_t device_aux_bytes;
    float *device_moe_mid;
    size_t device_moe_mid_bytes;
    float *device_moe_accum;
    size_t device_moe_accum_bytes;
    float *device_qsa_input;
    size_t device_qsa_input_bytes;
    float *device_qsa_output;
    size_t device_qsa_output_bytes;
    float *host_qsa_output;
    size_t host_qsa_output_bytes;
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
    uint64_t expert_fast_calls_by_layer[Q38_MODEL_LAYERS];
    uint64_t expert_legacy_calls_by_layer[Q38_MODEL_LAYERS];
};

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
    return "unknown";
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

static bool is_ple_tensor(const q38_tensor *tensor) {
    if (!tensor || !tensor->name.ptr) return false;
    const char *name = tensor->name.ptr;
    const size_t len = (size_t)tensor->name.len;
    return (len >= 4 && memmem(name, len, ".ple", 4)) ||
           (len >= 11 && memmem(name, len, "ngram_heads", 11)) ||
           (len >= 17 && memmem(name, len, "layer_multipliers", 17));
}

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

static double host_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static float event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    return cudaEventElapsedTime(&ms, start, stop) == cudaSuccess ? ms : 0.0f;
}

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
    const bool ple = is_ple_tensor(tensor);
    if (ple && (miss || upload_bytes)) {
        ++context->ple_file_backed_accesses;
        context->ple_file_bytes += upload_bytes;
    }
    if (!context->telemetry_observer) return;
    char name[128];
    copy_tensor_name(tensor, name, sizeof(name));
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
        allocations, syncs, syncs
    };
    context->telemetry_observer(&record, context->telemetry_observer_user);
}

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
    if (!use_persistent) context->gguf_name_lookup_in_decode += 2;
    if (!use_persistent && (!gate_data || !down_data))
        return fail(error, error_len, "invalid CUDA routed expert payload");
    if (context->all_non_ple_resident) {
        if (use_persistent) ++context->persistent_hits;
        else {
            ++context->persistent_misses;
            ++context->non_ple_residency_miss;
        }
    }
    const double host_started = host_now_ms();
    cudaEvent_t upload_start = NULL, upload_stop = NULL;
    cudaEvent_t kernel_start = NULL, kernel_stop = NULL;
    cudaEvent_t gate_start = NULL, gate_stop = NULL;
    cudaEvent_t down_start = NULL, down_stop = NULL;
    if (cudaEventCreate(&upload_start) != cudaSuccess ||
        cudaEventCreate(&upload_stop) != cudaSuccess ||
        cudaEventCreate(&kernel_start) != cudaSuccess ||
        cudaEventCreate(&kernel_stop) != cudaSuccess)
        return fail(error, error_len, "CUDA telemetry event allocation failed");
    if (gate_up->type == Q38_QUANT_Q2_K &&
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
    if (cudaEventRecord(upload_start, context->stream) != cudaSuccess ||
        (!use_persistent &&
         (cudaMemcpyAsync(gate_storage, gate_src, gate_bytes,
                          cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
          cudaMemcpyAsync(down_storage, down_src, down_bytes,
                          cudaMemcpyHostToDevice, context->stream) != cudaSuccess)) ||
        cudaEventRecord(upload_stop, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_input, input, 2560u * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA routed expert upload failed");
    if (!use_persistent)
        context->non_ple_upload_bytes_per_token += gate_bytes + down_bytes;
    if (cudaEventRecord(kernel_start, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA routed expert execution failed");
    bool launched;
    if (gate_up->type == Q38_QUANT_Q2_K) {
        if (cudaEventRecord(gate_start, context->stream) != cudaSuccess ||
            !q38_moe_cuda_q2_gate_up(
                gate_storage, context->device_input, context->device_moe_mid,
                context->stream, error, error_len) ||
            cudaEventRecord(gate_stop, context->stream) != cudaSuccess ||
            cudaEventRecord(down_start, context->stream) != cudaSuccess ||
            !q38_moe_cuda_q2_down(
                down_storage, context->device_moe_mid, context->device_output,
                context->stream, error, error_len) ||
            cudaEventRecord(down_stop, context->stream) != cudaSuccess) {
            ++context->q2_gate_up_fallback_calls;
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
    if (cudaEventRecord(kernel_stop, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(output, context->device_output, 2560u * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA routed expert execution failed");
    ++context->cuda_synchronizations;
    const double expert_wall_ms = host_now_ms() - host_started;
    context->expert_backend_total_wall_ms += expert_wall_ms;
    ++context->expert_host_sync_count;
    context->expert_H2D_bytes += 2560u * sizeof(float);
    context->expert_D2H_bytes += 2560u * sizeof(float);
    if (gate_up->type == Q38_QUANT_Q2_K) {
        ++context->q2_gate_up_fast_calls;
        ++context->q2_down_calls;
        if (context->current_layer < Q38_MODEL_LAYERS)
            ++context->expert_fast_calls_by_layer[context->current_layer];
        context->q2_gate_up_fast_total_kernel_ms +=
            event_elapsed(gate_start, gate_stop);
        context->q2_down_total_kernel_ms +=
            event_elapsed(down_start, down_stop);
    }
    emit_telemetry(context, model, gate_up, gate_rows, gate_cols, gate_bytes,

                   use_persistent, !use_persistent, use_persistent ? 0 : gate_bytes,
                   event_elapsed(upload_start, upload_stop),
                   event_elapsed(kernel_start, kernel_stop),
                   expert_wall_ms,
               context->cuda_allocations - allocation_before, 1,
               "routed_expert", use_persistent
                   ? "resident_exec_tensor" : "gguf_host_upload");
    emit_telemetry(context, model, down, down_rows, down_cols, down_bytes,
               use_persistent, !use_persistent, use_persistent ? 0 : down_bytes, 0.0f, 0.0f, 0.0,
               0, 0, "routed_expert", use_persistent
                   ? "resident_exec_tensor" : "gguf_host_upload");
    cudaEventDestroy(upload_start); cudaEventDestroy(upload_stop);
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_stop);
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

    const size_t gate_row_bytes = (size_t)(gate_up->bytes / gate_rows);
    const size_t down_row_bytes = (size_t)(down->bytes / down_rows);
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
        !ensure_buffer((void **)&context->device_moe_accum,
                       &context->device_moe_accum_bytes,
                       Q38_MOE_HIDDEN * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user,
                       &context->cuda_allocations))
        return fail(error, error_len, "CUDA Q2 MoE layer allocation failed");

    const double started = host_now_ms();
    cudaEvent_t gate_start[Q38_MOE_TOP_K] = {};
    cudaEvent_t gate_stop[Q38_MOE_TOP_K] = {};
    cudaEvent_t down_start[Q38_MOE_TOP_K] = {};
    cudaEvent_t down_stop[Q38_MOE_TOP_K] = {};
    cudaEvent_t reduce_start[Q38_MOE_TOP_K] = {};
    cudaEvent_t reduce_stop[Q38_MOE_TOP_K] = {};
    for (unsigned k = 0; k < Q38_MOE_TOP_K; ++k) {
        if (cudaEventCreate(&gate_start[k]) != cudaSuccess ||
            cudaEventCreate(&gate_stop[k]) != cudaSuccess ||
            cudaEventCreate(&down_start[k]) != cudaSuccess ||
            cudaEventCreate(&down_stop[k]) != cudaSuccess ||
            cudaEventCreate(&reduce_start[k]) != cudaSuccess ||
            cudaEventCreate(&reduce_stop[k]) != cudaSuccess)
            return fail(error, error_len, "CUDA Q2 MoE profiling event allocation failed");
    }
    if (cudaMemcpyAsync(context->device_input, host_input,
                        Q38_MOE_HIDDEN * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemsetAsync(context->device_moe_accum, 0,
                        Q38_MOE_HIDDEN * sizeof(float),
                        context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA Q2 MoE layer upload failed");

    for (unsigned k = 0; k < Q38_MOE_TOP_K; ++k) {
        const unsigned e = route->expert[k];
        if (e >= gate_rows / 1280 || e >= down_rows / 640)
            return fail(error, error_len, "invalid Q2 routed expert ID");
        const void *gate_e =
            (const unsigned char *)gate_exec->ptr +
            (size_t)e * 1280u * gate_row_bytes;
        const void *down_e =
            (const unsigned char *)down_exec->ptr +
            (size_t)e * 640u * down_row_bytes;
        if (cudaEventRecord(gate_start[k], context->stream) != cudaSuccess ||
            !q38_moe_cuda_q2_gate_up(
                gate_e, context->device_input, context->device_moe_mid,
                context->stream, error, error_len) ||
            cudaEventRecord(gate_stop[k], context->stream) != cudaSuccess ||
            cudaEventRecord(down_start[k], context->stream) != cudaSuccess ||
            !q38_moe_cuda_q2_down(
                down_e, context->device_moe_mid, context->device_output,
                context->stream, error, error_len) ||
            cudaEventRecord(down_stop[k], context->stream) != cudaSuccess ||
            cudaEventRecord(reduce_start[k], context->stream) != cudaSuccess ||
            !q38_moe_cuda_accumulate_weighted(
                context->device_moe_accum, context->device_output,
                route->weight[k], context->stream, error, error_len))
            return false;
        if (cudaEventRecord(reduce_stop[k], context->stream) != cudaSuccess)
            return fail(error, error_len, "CUDA Q2 MoE reduction event failed");
    }

    if (cudaMemcpyAsync(host_output, context->device_moe_accum,
                        Q38_MOE_HIDDEN * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA Q2 MoE layer execution failed");

    ++context->cuda_synchronizations;
    ++context->expert_host_sync_count;
    context->expert_backend_total_wall_ms += host_now_ms() - started;
    context->expert_H2D_bytes += Q38_MOE_HIDDEN * sizeof(float);
    context->expert_D2H_bytes += Q38_MOE_HIDDEN * sizeof(float);
    context->q2_gate_up_fast_calls += Q38_MOE_TOP_K;
    context->q2_down_calls += Q38_MOE_TOP_K;
    context->expert_kernel_launches += 3u * Q38_MOE_TOP_K;
    context->persistent_hits += 2;
    if (context->current_layer < Q38_MODEL_LAYERS)
        context->expert_fast_calls_by_layer[context->current_layer] +=
            Q38_MOE_TOP_K;
    for (unsigned k = 0; k < Q38_MOE_TOP_K; ++k) {
        context->q2_gate_up_fast_total_kernel_ms +=
            event_elapsed(gate_start[k], gate_stop[k]);
        context->q2_down_total_kernel_ms +=
            event_elapsed(down_start[k], down_stop[k]);
        context->q2_weighted_reduce_total_kernel_ms +=
            event_elapsed(reduce_start[k], reduce_stop[k]);
        cudaEventDestroy(gate_start[k]);
        cudaEventDestroy(gate_stop[k]);
        cudaEventDestroy(down_start[k]);
        cudaEventDestroy(down_stop[k]);
        cudaEventDestroy(reduce_start[k]);
        cudaEventDestroy(reduce_stop[k]);
    }
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
    if (observer) observer(bytes, observer_user);
    if (allocation_count) ++*allocation_count;
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

extern "C" bool q38_forward_cuda_enable_all_non_ple_residency(
    q38_forward_cuda_context *context, const q38_gguf *model,
    char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!context || !model)
        return fail(error, error_len, "invalid all-non-PLE residency arguments");
    if (context->all_non_ple_resident) return true;
    size_t count = 0, total = 0;
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        const q38_tensor *tensor = &model->tensors[i];
        if (is_ple_tensor(tensor)) {
            ++context->persistent_ple_tensors;
            continue;
        }
        if (!tensor->bytes || tensor->bytes > SIZE_MAX) continue;
        if (count == SIZE_MAX || total > SIZE_MAX - (size_t)tensor->bytes)
            return fail(error, error_len, "all-non-PLE residency size overflow");
        ++count;
        total += (size_t)tensor->bytes;
    }
    context->persistent_expected_tensors = count;
    context->persistent_expected_bytes = total;
    size_t free_bytes = 0, total_bytes = 0;
    const bool forced = getenv("Q38_FORCE_ALL_NON_PLE_RESIDENCY") != NULL;
    if (!forced && cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        free_bytes && total > free_bytes) {
        if (error && error_len)
            snprintf(error, error_len,
                     "insufficient CUDA memory for all-non-PLE residency "
                     "(need=%zu free=%zu total=%zu)", total, free_bytes,
                     total_bytes);
        return false;
    }
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
    size_t at = 0;
    size_t loaded_bytes = 0;
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
        exec->storage = is_ple_tensor(tensor)
            ? Q38_STORAGE_FILE_BACKED_PLE : Q38_STORAGE_RESIDENT;
        if (is_ple_tensor(tensor) || !tensor->bytes) continue;
        const void *host = q38_gguf_tensor_data(model, tensor);
        bool duplicate = false;
        for (size_t j = 0; j < at; ++j)
            if (host && entries[j].host == host) duplicate = true;
        if (duplicate) {
            ++context->persistent_duplicate_tensors;
            for (size_t j = 0; j < at; ++j) cudaFree(entries[j].device);
            free(entries);
            return fail(error, error_len,
                        "duplicate tensor in all-non-PLE residency set");
        }
        if (!host || cudaMalloc(&entries[at].device, (size_t)tensor->bytes) !=
                         cudaSuccess ||
            cudaMemcpyAsync(entries[at].device, host, (size_t)tensor->bytes,
                            cudaMemcpyHostToDevice, context->stream) !=
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
            return false;
        }
        entries[at].host = host;
        entries[at].bytes = (size_t)tensor->bytes;
        exec->ptr = entries[at].device;
        ++at;
        loaded_bytes += (size_t)tensor->bytes;
        if (cudaStreamSynchronize(context->stream) != cudaSuccess) {
            char name[128];
            copy_tensor_name(tensor, name, sizeof(name));
            if (error && error_len)
                snprintf(error, error_len,
                         "all-non-PLE residency copy failed at %s/%s: %s",
                         residency_group(tensor), name,
                         cudaGetErrorString(cudaGetLastError()));
            context->persistent = entries;
            context->persistent_count = at;
            context->persistent_bytes = loaded_bytes;
            context->persistent_loaded_bytes = loaded_bytes;
            context->persistent_loaded_tensors = at;
            context->all_non_ple_resident = false;
            snprintf(context->persistent_failure,
                     sizeof(context->persistent_failure), "%s",
                     error && error_len ? error :
                     "all-non-PLE residency copy failed");
            return false;
        }
        ++context->cuda_synchronizations;
        size_t progress_free = 0, progress_total = 0;
        (void)cudaMemGetInfo(&progress_free, &progress_total);
        if (context->progress_observer)
            context->progress_observer(
                residency_group(tensor), tensor, loaded_bytes,
                progress_free, progress_total,
                context->progress_observer_user);
    }
    if (cudaStreamSynchronize(context->stream) != cudaSuccess) {
        context->persistent = entries;
        context->persistent_count = at;
        context->persistent_bytes = loaded_bytes;
        context->persistent_loaded_bytes = loaded_bytes;
        context->persistent_loaded_tensors = at;
        context->all_non_ple_resident = false;
        snprintf(context->persistent_failure,
                 sizeof(context->persistent_failure),
                 "all-non-PLE residency synchronization failed: %s",
                 cudaGetErrorString(cudaGetLastError()));
        return fail(error, error_len, context->persistent_failure);
    }
    ++context->cuda_synchronizations;
    context->persistent = entries;
    context->persistent_count = at;
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
    cudaFree(context->device_argmax);
    cudaFree(context->device_aux);
    cudaFree(context->device_moe_mid);
    cudaFree(context->device_moe_accum);
    cudaFree(context->device_qsa_input);
    cudaFree(context->device_qsa_output);
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
        cudaStreamSynchronize(context->stream) != cudaSuccess)
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
        (uintptr_t)context->device_moe_accum,
        (uintptr_t)context->device_qsa_input,
        (uintptr_t)context->device_qsa_output,
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
    memcpy(stats->expert_fast_calls_by_layer,
           context->expert_fast_calls_by_layer,
           sizeof(stats->expert_fast_calls_by_layer));
    memcpy(stats->expert_legacy_calls_by_layer,
           context->expert_legacy_calls_by_layer,
           sizeof(stats->expert_legacy_calls_by_layer));
}

extern "C" void q38_forward_cuda_set_qsa_candidate(
    q38_forward_cuda_context *context, q38_qsa_candidate_fn candidate) {
    if (context) context->qsa_candidate = candidate;
}

extern "C" void q38_forward_cuda_record_route(
    q38_forward_cuda_context *context, uint32_t layer, size_t selected_count) {
    if (!context || layer >= Q38_MODEL_LAYERS) return;
    ++context->routed_layers_executed;
    context->selected_experts_total += selected_count;
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
    if (context->exec_strict && !use_persistent && !is_ple_tensor(tensor))
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
        ++context->persistent_hits;
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
            cudaStreamSynchronize(context->stream) != cudaSuccess)
            return fail(error, error_len, "CUDA resident matvec execution failed");
        ++context->cuda_synchronizations;
        emit_telemetry(context, model, tensor, 1, cols, weight_bytes, true,
                       false, 0, 0.0f, 0.0f, 0.0, 0, 1,
                       "matvec", "resident_exec_tensor");
        return true;
    }
    if (context->all_non_ple_resident) {
        ++context->persistent_misses;
        if (!is_ple_tensor(tensor)) ++context->non_ple_residency_miss;
    }

    if (cudaMemcpyAsync(context->device_weights, row_data, weight_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_input, input, cols * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA forward matvec upload failed");
    if (!is_ple_tensor(tensor))
        context->non_ple_upload_bytes_per_token += weight_bytes;
    if (!is_ple_tensor(tensor)) ++context->gguf_name_lookup_in_decode;

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
        cudaStreamSynchronize(context->stream) != cudaSuccess)
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
    const double host_started = host_now_ms();
    cudaEvent_t upload_start = NULL, upload_stop = NULL;
    cudaEvent_t kernel_start = NULL, kernel_stop = NULL;
    if (cudaEventCreate(&upload_start) != cudaSuccess ||
        cudaEventCreate(&upload_stop) != cudaSuccess ||
        cudaEventCreate(&kernel_start) != cudaSuccess ||
        cudaEventCreate(&kernel_stop) != cudaSuccess)
        return fail(error, error_len, "CUDA telemetry event allocation failed");
    const uint64_t allocation_before = context->cuda_allocations;
    q38_exec_tensor *exec = exec_tensor_for(context, model, tensor);
    const bool use_exec_resident = context->all_non_ple_resident &&
        exec_tensor_is_resident(exec, tensor);
    const void *data = use_exec_resident ? NULL :
        q38_gguf_tensor_data(model, tensor);
    if (!use_exec_resident) ++context->gguf_name_lookup_in_decode;
    if (context->exec_strict && !use_exec_resident && !is_ple_tensor(tensor))
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
        ++context->resident_hits;
    else {
        ++context->resident_misses;
        if (context->all_non_ple_resident) {
            ++context->persistent_misses;
            if (!is_ple_tensor(tensor)) ++context->non_ple_residency_miss;
        }
    }
    if (use_persistent_weight) ++context->persistent_hits;
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
    if (cudaEventRecord(upload_start, context->stream) != cudaSuccess ||
        (!use_resident_lm_head && !use_persistent_weight &&
         cudaMemcpyAsync(context->device_weights, data, (size_t)tensor->bytes,
                         cudaMemcpyHostToDevice, context->stream) !=
             cudaSuccess) ||
        cudaEventRecord(upload_stop, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_input, input, cols * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA forward matrix upload failed");
    if (cudaEventRecord(kernel_start, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA matrix timing failed");
    if (!use_resident_lm_head && !use_persistent_weight)
        context->matrix_upload_bytes += tensor->bytes;
    if (!use_resident_lm_head && !use_persistent_weight &&
        !is_ple_tensor(tensor))
        context->non_ple_upload_bytes_per_token += tensor->bytes;
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
    if (cudaEventRecord(kernel_stop, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA kernel timing failed");

    if (!launched ||
        cudaMemcpyAsync(output, context->device_output, rows * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return launched ? fail(error, error_len,
                               "CUDA forward matrix download failed")
                        : false;
    ++context->cuda_synchronizations;
    context->device_output_elements = rows;
    float upload_ms = event_elapsed(upload_start, upload_stop);
    float kernel_ms = event_elapsed(kernel_start, kernel_stop);
    emit_telemetry(context, model, tensor, rows, cols, (size_t)tensor->bytes,

                   use_resident_lm_head || use_persistent_weight,
                   !(use_resident_lm_head || use_persistent_weight),
                   (use_resident_lm_head || use_persistent_weight) ? 0 :
                       (size_t)tensor->bytes,
                   upload_ms, kernel_ms, host_now_ms() - host_started,
                   context->cuda_allocations - allocation_before, 1,
                   "matrix", use_resident_lm_head || use_persistent_weight
                       ? "resident_exec_tensor" : "gguf_host_upload");
    cudaEventDestroy(upload_start); cudaEventDestroy(upload_stop);
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_stop);
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
    const uint64_t allocations_before = context->cuda_allocations;
    const double started = host_now_ms();
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
        ++timing->allocations;
    }
    float *device_q = context->device_qsa_output;
    float *device_k = device_q + q_elements;
    float *device_v = device_k + k_elements;
    if (cudaMemcpyAsync(context->device_qsa_input, host_input, input_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        (context->qsa_candidate
             ? context->qsa_candidate(
                   q_exec->ptr, k_exec->ptr, v_exec->ptr,
                   context->device_qsa_input, device_q, device_k, device_v,
                   token_count, q_cols, (void *)context->stream) != 0
             : q38_qsa_cuda_project_main(
                   (const uint16_t *)q_exec->ptr, q_rows,
                   (const uint16_t *)k_exec->ptr, k_rows,
                   (const uint16_t *)v_exec->ptr, v_rows, q_cols,
                   context->device_qsa_input, token_count, device_q, device_k,
                   device_v, context->stream, error, error_len)) ||
        cudaMemcpyAsync(context->host_qsa_output, context->device_qsa_output,
                        output_bytes, cudaMemcpyDeviceToHost,
                        context->stream) != cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return fail(error, error_len, "QSA QKV CUDA execution failed");

    memcpy(host_q, context->host_qsa_output, q_elements * sizeof(float));
    memcpy(host_k, context->host_qsa_output + q_elements,
           k_elements * sizeof(float));
    memcpy(host_v, context->host_qsa_output + q_elements + k_elements,
           v_elements * sizeof(float));
    timing->qkv_projection_ms += host_now_ms() - started;
    timing->allocations += context->cuda_allocations - allocations_before;
    timing->kernel_launches += 3;
    timing->host_syncs++;
    timing->h2d_bytes += input_bytes;
    timing->d2h_bytes += output_bytes;
    ++context->persistent_hits;
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
    cudaEvent_t start = NULL, stop = NULL;
    if (cudaEventCreate(&start) == cudaSuccess &&
        cudaEventCreate(&stop) == cudaSuccess) {
        cudaEventRecord(start, context->stream);
        launched = q38_argmax_cuda(
            context->device_output, 1, context->device_output_elements,
            context->device_argmax, context->stream, error, error_len);
        cudaEventRecord(stop, context->stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        if (cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess)
            context->gpu_argmax_kernel_ms = elapsed;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    bool ok = launched &&
        cudaMemcpyAsync(token, context->device_argmax, sizeof(*token),
                        cudaMemcpyDeviceToHost, context->stream) == cudaSuccess &&
        cudaStreamSynchronize(context->stream) == cudaSuccess;
    if (!ok && (!error || !error_len || !error[0]))
        fail(error, error_len, "CUDA greedy argmax execution failed");
    if (ok) ++context->cuda_synchronizations;
    return ok;
}
