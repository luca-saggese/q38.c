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
