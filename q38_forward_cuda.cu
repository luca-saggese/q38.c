#include "q38_forward_cuda.h"

#include "q38_cuda_primitives.h"
#include "q38_gdn.h"
#include "q38_moe_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q38_forward_cuda_context {
    void *device_weights;
    size_t device_weights_bytes;
    float *device_input;
    size_t device_input_elements;
    float *device_output;
    size_t device_output_bytes;
    void *device_aux;
    size_t device_aux_bytes;
    float *device_moe_mid;
    size_t device_moe_mid_bytes;
    void *lm_head_device_weights;
    size_t lm_head_device_weights_bytes;
    const void *lm_head_host_data;
    bool lm_head_resident;
    cudaStream_t stream;
    q38_forward_cuda_allocation_observer allocation_observer;
    void *allocation_observer_user;
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
                          void *observer_user);

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
        gate_up->type != 10 || down->type != 10 ||
        gate_cols != 2560 || down_cols != 2560 ||
        gate_rows % 1280 != 0 || down_rows % 640 != 0 ||
        expert >= gate_rows / 1280 || expert >= down_rows / 640)
        return fail(error, error_len, "unsupported CUDA routed expert geometry");
    const size_t gate_row_bytes = (size_t)(gate_up->bytes / gate_rows);
    const size_t down_row_bytes = (size_t)(down->bytes / down_rows);
    const void *gate_data = q38_gguf_tensor_data(model, gate_up);
    const void *down_data = q38_gguf_tensor_data(model, down);
    const size_t gate_bytes = 1280u * gate_row_bytes;
    const size_t down_bytes = 640u * down_row_bytes;
    if (!gate_data || !down_data)
        return fail(error, error_len, "invalid CUDA routed expert payload");
    if (!ensure_buffer(&context->device_weights, &context->device_weights_bytes,
                       gate_bytes, context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer(&context->device_aux, &context->device_aux_bytes,
                       down_bytes, context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements, 2560u * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes, 2560u * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer((void **)&context->device_moe_mid,
                       &context->device_moe_mid_bytes,
                       Q38_MOE_INTERMEDIATE * sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user))
        return fail(error, error_len, "CUDA routed expert allocation failed");
    const unsigned char *gate_src =
        (const unsigned char *)gate_data + expert * 1280u * gate_row_bytes;
    const unsigned char *down_src =
        (const unsigned char *)down_data + expert * 640u * down_row_bytes;
    if (cudaMemcpyAsync(context->device_weights, gate_src, gate_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_aux, down_src, down_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_input, input, 2560u * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA routed expert upload failed");
    if (!q38_moe_cuda_expert_q2_workspace(
            context->device_weights, context->device_aux,
            context->device_input, context->device_output,
            context->device_moe_mid, context->stream,
            error, error_len) ||
        cudaMemcpyAsync(output, context->device_output, 2560u * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA routed expert execution failed");
    return true;
}

static bool ensure_buffer(void **buffer, size_t *capacity, size_t bytes,
                          q38_forward_cuda_allocation_observer observer,
                          void *observer_user) {
    if (*capacity >= bytes) return true;
    if (*buffer) cudaFree(*buffer);
    *buffer = NULL;
    *capacity = 0;
    if (cudaMalloc(buffer, bytes) != cudaSuccess) return false;
    if (observer) observer(bytes, observer_user);
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
    if (cudaStreamCreate(&context->stream) != cudaSuccess) {
        free(context);
        fail(error, error_len, "CUDA forward stream creation failed");
        return NULL;
    }
    return context;
}

extern "C" void
q38_forward_cuda_context_destroy(q38_forward_cuda_context *context) {
    if (!context) return;
    cudaFree(context->device_weights);
    cudaFree(context->device_input);
    cudaFree(context->device_output);
    cudaFree(context->device_aux);
    cudaFree(context->device_moe_mid);
    cudaFree(context->lm_head_device_weights);
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
    if (context->lm_head_device_weights &&
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
    return true;
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
    const void *data = q38_gguf_tensor_data(model, tensor);
    if (!data || !row_bytes || row > SIZE_MAX / row_bytes)
        return fail(error, error_len, "invalid CUDA forward tensor payload");
    const void *row_data = (const unsigned char *)data + row * row_bytes;

    size_t weight_bytes = row_bytes;
    if (tensor->type != 0 && tensor->type != 8 && tensor->type != 10 &&
        tensor->type != 30)
        return fail(error, error_len, "unsupported CUDA forward matvec type");
    if (cols > SIZE_MAX / sizeof(float))
        return fail(error, error_len, "CUDA forward matvec size overflow");
    if (!ensure_buffer(&context->device_weights,
                       &context->device_weights_bytes, weight_bytes,
                       context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       cols * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes, sizeof(float),
                       context->allocation_observer,
                       context->allocation_observer_user))
        return fail(error, error_len, "CUDA forward matvec allocation failed");

    if (cudaMemcpyAsync(context->device_weights, row_data, weight_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->device_input, input, cols * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA forward matvec upload failed");

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
    const void *data = q38_gguf_tensor_data(model, tensor);
    if (!data || !tensor->bytes || rows > SIZE_MAX / sizeof(float) ||
        cols > SIZE_MAX / sizeof(float))
        return fail(error, error_len, "invalid CUDA forward matrix payload");
    const bool use_resident_lm_head =
        is_lm_head_tensor(tensor) &&
        context->lm_head_resident && context->lm_head_host_data == data &&
        context->lm_head_device_weights_bytes == tensor->bytes;
    if ((!use_resident_lm_head &&
         !ensure_buffer(&context->device_weights,
                        &context->device_weights_bytes, (size_t)tensor->bytes,
                        context->allocation_observer,
                        context->allocation_observer_user)) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       cols * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes,
                       rows * sizeof(float), context->allocation_observer,
                       context->allocation_observer_user))
        return fail(error, error_len, "CUDA forward matrix allocation failed");
    if ((!use_resident_lm_head &&
         cudaMemcpyAsync(context->device_weights, data, (size_t)tensor->bytes,
                         cudaMemcpyHostToDevice, context->stream) !=
             cudaSuccess) ||
        cudaMemcpyAsync(context->device_input, input, cols * sizeof(float),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
        return fail(error, error_len, "CUDA forward matrix upload failed");
    bool launched = false;
    void *weight_storage = use_resident_lm_head
                               ? context->lm_head_device_weights
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

    if (!launched ||
        cudaMemcpyAsync(output, context->device_output, rows * sizeof(float),
                        cudaMemcpyDeviceToHost, context->stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(context->stream) != cudaSuccess)
        return launched ? fail(error, error_len,
                               "CUDA forward matrix download failed")
                        : false;
    return true;
}
