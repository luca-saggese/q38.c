#ifndef Q38_MOE_REF_H
#define Q38_MOE_REF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_MOE_EXPERTS 512u
#define Q38_MOE_TOP_K 10u
#define Q38_MOE_HIDDEN 2560u
#define Q38_MOE_INTERMEDIATE 640u

typedef struct {
    uint16_t expert[Q38_MOE_TOP_K];
    float weight[Q38_MOE_TOP_K];
} q38_moe_route10;

typedef struct {
    uint64_t routed_pairs;
    uint64_t unique_experts;
    uint64_t bytes_touched;
} q38_moe_stats;

bool q38_moe_route_ref(const float *hidden, size_t token_count,
                       const float *router, q38_moe_route10 *routes,
                       char *error, size_t error_len);

bool q38_moe_expert_ref(const float *hidden, const float *gate_up,
                        const float *down, float *output, char *error,
                        size_t error_len);

bool q38_moe_shared_ref(const float *hidden, const float *gate_proj,
                        const float *up, const float *down,
                        const float *gate_weight, float *output, char *error,
                        size_t error_len);

bool q38_moe_forward_ref(const float *hidden, size_t token_count,
                         const float *router, const float *gate_up,
                         const float *down, const float *shared_gate_proj,
                         const float *shared_up, const float *shared_down,
                         const float *shared_gate_weight, float *output,
                         q38_moe_route10 *routes, char *error, size_t error_len);
bool q38_moe_forward_ref_stats(const float *hidden, size_t token_count,
                               const float *router, const float *gate_up,
                               const float *down, const float *shared_gate_proj,
                               const float *shared_up, const float *shared_down,
                               const float *shared_gate_weight, float *output,
                               q38_moe_route10 *routes, q38_moe_stats *stats,
                               char *error, size_t error_len);
bool q38_moe_decode_ref(const float *hidden, const float *router,
                        const float *gate_up, const float *down,
                        const float *shared_gate_proj, const float *shared_up,
                        const float *shared_down, const float *shared_gate_weight,
                        float *output, q38_moe_route10 *route,
                        char *error, size_t error_len);
bool q38_moe_combine_ref(const q38_moe_route10 *routes, size_t token_count,
                         const float *expert_outputs,
                         const float *shared_outputs, float *output,
                         char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
