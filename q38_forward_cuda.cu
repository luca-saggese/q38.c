#include "q38_forward_cuda.h"

#include "q38_cuda_primitives.h"
#include "q38_gdn.h"

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
    cudaStream_t stream;
};

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
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

static bool ensure_buffer(void **buffer, size_t *capacity, size_t bytes) {
    if (*capacity >= bytes) return true;
    if (*buffer) cudaFree(*buffer);
    *buffer = NULL;
    *capacity = 0;
    if (cudaMalloc(buffer, bytes) != cudaSuccess) return false;
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
    if (context->stream) cudaStreamDestroy(context->stream);
    free(context);
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
    /*
     * The vocabulary head has 248,320 rows.  Keeping that diagnostic-only
     * matrix on the scalar path avoids turning a trace run into hundreds of
     * thousands of tiny synchronizing launches; all graph and MoE rows still
     * use CUDA.
     */
    if (rows > 8192) return false;

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
                       &context->device_weights_bytes, weight_bytes) ||
        !ensure_buffer((void **)&context->device_input,
                       &context->device_input_elements,
                       cols * sizeof(float)) ||
        !ensure_buffer((void **)&context->device_output,
                       &context->device_output_bytes, sizeof(float)))
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
