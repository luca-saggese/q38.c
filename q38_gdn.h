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
 * Device-output BF16 matvec contract used by resident GDN chains.  Both
 * activations and weights are device-resident; this function only enqueues
 * the existing BF16 matvec kernel and never allocates, copies, or waits.
 */
bool q38_cuda_bf16_matvec_device(const uint16_t *weights, size_t rows,
                                 size_t cols, const float *input,
                                 float *output, cudaStream_t stream,
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
 * Fused reference-equivalent path: causal convolution, SiLU, and the
 * persistent history tail update execute in one kernel.  Each channel owns a
 * block; the block-wide barrier completes all output reads before thread zero
 * updates that channel's history tail.
 */
bool q38_cuda_gdn_conv_silu_fused(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len);

/* Same fused operation for GGUF GDN conv1d storage [channel, tap]. */
bool q38_cuda_gdn_conv_silu_fused_channel_major(
    uint32_t kernel_type, const void *kernel, const float *input,
    size_t tokens, size_t channels, size_t kernel_size, float *history,
    float *output, cudaStream_t stream, char *error, size_t error_len);

/* Update the persistent causal history after a fused single-token core. */
bool q38_cuda_gdn_history_update(
    const float *input, size_t tokens, size_t channels, size_t kernel_size,
    float *history, cudaStream_t stream, char *error, size_t error_len);

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

/*
 * Prepare normalized, repeated q/k heads plus v, decay, and beta directly
 * from the device-side convolution and projection outputs.
 */
bool q38_cuda_gdn_prepare_recurrence(
    const float *conv, const float *a, const float *b,
    uint32_t scalar_weight_type, const void *a_log, const void *dt_bias,
    size_t tokens, float *q, float *k, float *v, float *decay, float *beta,
    cudaStream_t stream, char *error, size_t error_len);

/* Apply FP32 per-head normalization and sigmoid(z) gating on device. */
bool q38_cuda_gdn_post_recurrence(
    const float *recurrent, const float *z, uint32_t norm_weight_type,
    const void *norm, size_t tokens, float *output, cudaStream_t stream,
    char *error, size_t error_len);

/*
 * Fuse convolution/SiLU, q/k/v preparation, FP32 recurrence, and
 * normalization/gating for the single-token decode shape.  The caller must
 * enqueue q38_cuda_gdn_history_update after this kernel so all convolution
 * reads observe the previous history.
 */
bool q38_cuda_gdn_fused_recurrent(
    const float *qkv, const float *z, const float *a, const float *b,
    uint32_t conv_weight_type, const void *conv_kernel,
    uint32_t scalar_weight_type, const void *a_log, const void *dt_bias,
    uint32_t norm_weight_type, const void *norm, float *state,
    const float *history, float *output, cudaStream_t stream, char *error,
    size_t error_len);

/*
 * Apply the scalar reference recurrence to one sequence.  State is the
 * contiguous logical [sequence=1, value_head, row, column] FP32 region.
 * q/k/v are token-major [tokens, value_head, dimension], decay and beta are
 * token-major [tokens, value_head], and output has the q/k/v shape.
 */
bool q38_cuda_gdn_recurrence(
    float *state, size_t tokens, const float *q, const float *k,
    const float *v, const float *decay, const float *beta, float scale,
    float *output, cudaStream_t stream, char *error, size_t error_len);

/* Asynchronously clear the logical single-sequence FP32 recurrent state. */
bool q38_cuda_gdn_recurrence_reset(float *state, cudaStream_t stream,
                                    char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif /* Q38_GDN_H */
