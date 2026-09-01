#ifndef Q38_GDN_H
#define Q38_GDN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#include "q38_quant.h"
#include "q38_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Weight pointers are device pointers.  Matrix weights use the logical
 * output-by-input row-major shape [rows, cols], independently of GGUF's
 * physical dimension order.  The projection output is token-major
 * [token, row].  Q2_K and Q8_0 use their normal row-blocked physical layouts;
 * BF16 and F32 are dense row-major.
 */
#define Q38_GDN_WEIGHT_F32 0u
#define Q38_GDN_WEIGHT_Q8_0 8u
#define Q38_GDN_WEIGHT_BF16 30u

#define Q38_GDN_INPUT_DIM 2560u
#define Q38_GDN_KEY_HEADS 16u
#define Q38_GDN_KEY_CHANNELS (Q38_GDN_KEY_HEADS * Q38_GDN_HEAD_DIM)
#define Q38_GDN_VALUE_CHANNELS (Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM)
#define Q38_GDN_QKV_CHANNELS (2u * Q38_GDN_KEY_CHANNELS + Q38_GDN_VALUE_CHANNELS)
#define Q38_GDN_Z_CHANNELS Q38_GDN_VALUE_CHANNELS

/* GGML Q8_0's logical block representation (32 signed values per FP16 d). */
typedef struct {
    uint16_t d;
    int8_t qs[32];
} q38_gdn_q8_0_block;

/*
 * Project token-major F32 activations [tokens, cols] through a logical
 * output-by-input matrix and write [tokens, rows].  This is deliberately an
 * unfused reference path: Q2/BF16 dispatches to the existing matvec helpers,
 * while F32/Q8 use a simple one-thread-per-output kernel.
 */
bool q38_cuda_gdn_project(uint32_t weight_type, const void *weights,
                          size_t rows, size_t cols, const float *input,
                          size_t tokens, float *output, cudaStream_t stream,
                          char *error, size_t error_len);

/*
 * Causal depthwise convolution over logical input [tokens, channels] with
 * logical kernel [tap, channel].  history is persistent [kernel-1, channels]
 * and is updated to the final tail after the raw convolution completes.
 * kernel_type is Q38_GDN_WEIGHT_F32 or Q38_GDN_WEIGHT_BF16.  GGUF's
 * [channels, 1, kernel] storage must be explicitly converted by the caller;
 * this API never treats physical GGUF dimensions as model semantics.
 */
bool q38_cuda_gdn_conv(uint32_t kernel_type, const void *kernel,
                       const float *input, size_t tokens, size_t channels,
                       size_t kernel_size, float *history, float *output,
                       cudaStream_t stream, char *error, size_t error_len);

/* Compatibility-named entry point for callers that emphasize state update. */
bool q38_cuda_gdn_conv_update(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len);

/* Reference sequencing: raw convolution, then the existing standalone SiLU. */
bool q38_cuda_gdn_conv_silu(uint32_t kernel_type, const void *kernel,
                            const float *input, size_t tokens, size_t channels,
                            size_t kernel_size, float *history, float *output,
                            cudaStream_t stream, char *error,
                            size_t error_len);

/*
 * Split the frozen Qwen4Exp qkv stream, whose logical order is
 * [Q(16*128), K(16*128), V(48*128)].  Each output is token-major and remains
 * in the 16-head logical form; 16-to-48 repeat-interleave is a separate step.
 */
bool q38_cuda_gdn_split_qkv(const float *qkv, size_t tokens, float *q,
                            float *k, float *v, cudaStream_t stream,
                            char *error, size_t error_len);

/* Repeat key-head tensors using value_head h -> key_head h / 3. */
bool q38_cuda_gdn_repeat_key_heads(const float *key, size_t tokens,
                                   float *value, cudaStream_t stream,
                                   char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif /* Q38_GDN_H */
