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

q38_forward_cuda_context *q38_forward_cuda_context_create(char *error,
                                                           size_t error_len);
void q38_forward_cuda_context_destroy(q38_forward_cuda_context *context);
void *q38_forward_cuda_stream(q38_forward_cuda_context *context);
void q38_forward_cuda_set_allocation_observer(
    q38_forward_cuda_context *context,
    q38_forward_cuda_allocation_observer observer, void *user);

bool q38_forward_cuda_matvec_backend(
    const q38_gguf *model, const q38_tensor *tensor, size_t row,
    const float *input, size_t cols, float *output, void *user, char *error,
    size_t error_len);

bool q38_forward_cuda_matrix_backend(
    const q38_gguf *model, const q38_tensor *tensor, const float *input,
    size_t rows, size_t cols, float *output, void *user, char *error,
    size_t error_len);

bool q38_forward_cuda_expert_backend(
    const q38_gguf *model, const q38_tensor *gate_up,
    const q38_tensor *down, size_t expert, const float *input, float *output,
    void *user, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
