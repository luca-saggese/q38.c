#ifndef Q38_MOE_CUDA_H
#define Q38_MOE_CUDA_H

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q38_moe_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Naive reference router projection. The output is token-major F32 logits. */
bool q38_moe_cuda_router(const float *device_hidden, size_t token_count,
                         const float *device_router, float *device_logits,
                         cudaStream_t stream, char *error, size_t error_len);

/* Complete device-side top-k route selection.  Logits are selected using the
 * reference pre-cast ordering; returned weights use BF16 effective logits and
 * the canonical selected-set renormalization. */
bool q38_moe_cuda_route_weights(
    const float *device_logits, size_t token_count, uint32_t *device_indices,
    uint16_t *device_expert_ids, float *device_weights, cudaStream_t stream,
    char *error, size_t error_len);

bool q38_moe_cuda_route(const float *device_hidden, size_t token_count,
                        const float *device_router, float *device_logits,
                        q38_moe_route10 *host_routes, cudaStream_t stream,
                        char *error, size_t error_len);

bool q38_moe_cuda_expert_q2(const void *device_gate_up, const void *device_down,
                            const float *device_hidden, float *device_output,
                            cudaStream_t stream, char *error,
                            size_t error_len);

/* Variant for callers that reuse a persistent intermediate workspace. */
bool q38_moe_cuda_expert_q2_workspace(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, float *device_mid,
    cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_expert_q4(const void *device_gate_up, const void *device_down,
                            const float *device_hidden, float *device_output,
                            cudaStream_t stream, char *error,
                            size_t error_len);

bool q38_moe_cuda_expert_q4_workspace(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, float *device_output, float *device_mid,
    cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_q4_gate_up(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_q4_down(
    const void *device_down, const float *device_mid,
    float *device_output, cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_q2_gate_up(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_q2_gate_up_candidate(
    const void *device_gate_up, const float *device_hidden,
    float *device_mid, unsigned threads_per_block, cudaStream_t stream,
    char *error, size_t error_len);

bool q38_moe_cuda_q2_down(
    const void *device_down, const float *device_mid,
    float *device_output, cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_accumulate_weighted(
    float *device_accum, const float *device_expert, float weight,
    cudaStream_t stream, char *error, size_t error_len);

/* Native NVIDIA NVFP4 grouped top-k path.  All component arrays are already
 * resident on CUDA and use the canonical [layer][projection][expert] layout. */
bool q38_moe_cuda_nvfp4_grouped_indexed(
    const uint8_t *gate_weight, const uint8_t *gate_scale,
    const float *gate_scale_2, const float *gate_input_scale,
    const uint8_t *up_weight, const uint8_t *up_scale,
    const float *up_scale_2, const float *up_input_scale,
    const uint8_t *down_weight, const uint8_t *down_scale,
    const float *down_scale_2, const float *down_input_scale,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    float *device_output, float *device_mid,
    uint8_t *device_activation, uint8_t *device_activation_scale,
    uint8_t *device_down_activation,
    uint8_t *device_down_activation_scale, cudaStream_t stream,
    char *error, size_t error_len);

/* Structural candidate: execute all selected routed experts with one grouped
 * gate/up launch and one down/weighted-accumulation launch. */
bool q38_moe_cuda_q2_grouped(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const float *device_route_weights,
    size_t expert_count, float *device_output, float *device_mid,
    cudaStream_t stream, char *error, size_t error_len);

/* Indexed variant for resident full expert tensors. */
bool q38_moe_cuda_q2_grouped_indexed(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    size_t gate_expert_blocks, size_t down_expert_blocks,
    float *device_output, float *device_mid, cudaStream_t stream,
    char *error, size_t error_len);

/* Deterministic grouped variant.  Each selected expert writes one private
 * weighted output, followed by a canonical 0..expert_count reduction. */
bool q38_moe_cuda_q2_grouped_indexed_deterministic(
    const void *device_gate_up, const void *device_down,
    const float *device_hidden, const uint16_t *device_expert_ids,
    const float *device_route_weights, size_t expert_count,
    size_t gate_expert_blocks, size_t down_expert_blocks,
    float *device_output, float *device_mid, float *device_expert_outputs,
    cudaStream_t stream, char *error, size_t error_len);

bool q38_moe_cuda_shared_f32(const float *device_hidden, size_t token_count,
                             const float *device_gate_proj,
                             const float *device_up_proj,
                             const float *device_down_proj,
                             const float *device_gate_weight,
                             float *device_output, cudaStream_t stream,
                             char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
