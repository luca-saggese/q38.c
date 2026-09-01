#include "q38_moe_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void route_one(const float *x, const float *router,
                      q38_moe_route10 *route) {
    float logits[Q38_MOE_EXPERTS];
    float max_logit = -INFINITY;
    for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
        float value = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            value += router[e * Q38_MOE_HIDDEN + d] * x[d];
        logits[e] = value;
        if (value > max_logit) max_logit = value;
    }
    float sum = 0.0f;
    for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
        logits[e] = expf(logits[e] - max_logit);
        sum += logits[e];
    }
    for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) logits[e] /= sum;

    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        size_t best = Q38_MOE_EXPERTS;
        for (size_t e = 0; e < Q38_MOE_EXPERTS; ++e) {
            bool already = false;
            for (size_t j = 0; j < k; ++j)
                already |= route->expert[j] == e;
            if (already) continue;
            if (best == Q38_MOE_EXPERTS ||
                logits[e] > logits[best] ||
                (logits[e] == logits[best] && e < best))
                best = e;
        }
        route->expert[k] = (uint16_t)best;
        route->weight[k] = logits[best];
    }
    float selected_sum = 0.0f;
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) selected_sum += route->weight[k];
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) route->weight[k] /= selected_sum;
}

bool q38_moe_route_ref(const float *hidden, size_t token_count,
                       const float *router, q38_moe_route10 *routes,
                       char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !token_count || !router || !routes)
        return fail(error, error_len, "invalid MoE router arguments");
    for (size_t t = 0; t < token_count; ++t)
        route_one(hidden + t * Q38_MOE_HIDDEN, router, &routes[t]);
    return true;
}

bool q38_moe_expert_ref(const float *hidden, const float *gate_up,
                        const float *down, float *output, char *error,
                        size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !gate_up || !down || !output)
        return fail(error, error_len, "invalid routed expert arguments");
    float intermediate[Q38_MOE_INTERMEDIATE];
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        float gate = 0.0f, up = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
            gate += gate_up[i * Q38_MOE_HIDDEN + d] * hidden[d];
            up += gate_up[(Q38_MOE_INTERMEDIATE + i) * Q38_MOE_HIDDEN + d] *
                  hidden[d];
        }
        intermediate[i] = silu(gate) * up;
    }
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
        float value = 0.0f;
        for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
            value += down[d * Q38_MOE_INTERMEDIATE + i] * intermediate[i];
        output[d] = value;
    }
    return true;
}

bool q38_moe_shared_ref(const float *hidden, const float *gate_proj,
                        const float *up, const float *down,
                        const float *gate_weight, float *output, char *error,
                        size_t error_len) {
    if (!hidden || !gate_proj || !up || !down || !gate_weight || !output)
        return fail(error, error_len, "invalid shared expert arguments");
    float intermediate[Q38_MOE_INTERMEDIATE];
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        float g = 0.0f, u = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
            g += gate_proj[i * Q38_MOE_HIDDEN + d] * hidden[d];
            u += up[i * Q38_MOE_HIDDEN + d] * hidden[d];
        }
        intermediate[i] = silu(g) * u;
    }
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
        output[d] = 0.0f;
        for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
            output[d] += down[d * Q38_MOE_INTERMEDIATE + i] * intermediate[i];
        float shared_gate = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            shared_gate += gate_weight[d] * hidden[d];
        output[d] *= 1.0f / (1.0f + expf(-shared_gate));
    }
    return true;
}

bool q38_moe_forward_ref(const float *hidden, size_t token_count,
                         const float *router, const float *gate_up,
                         const float *down, const float *shared_gate_proj,
                         const float *shared_up, const float *shared_down,
                         const float *shared_gate_weight, float *output,
                         q38_moe_route10 *routes, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!hidden || !token_count || !router || !gate_up || !down ||
        !shared_gate_proj || !shared_up || !shared_down ||
        !shared_gate_weight || !output || !routes)
        return fail(error, error_len, "invalid MoE forward arguments");
    float routed[Q38_MOE_HIDDEN], shared[Q38_MOE_HIDDEN];
    for (size_t t = 0; t < token_count; ++t) {
        const float *x = hidden + t * Q38_MOE_HIDDEN;
        if (!q38_moe_route_ref(x, 1, router, &routes[t], error, error_len))
            return false;
        memset(output + t * Q38_MOE_HIDDEN, 0,
               Q38_MOE_HIDDEN * sizeof(float));
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
            const size_t e = routes[t].expert[k];
            if (!q38_moe_expert_ref(
                    x, gate_up + e * 2 * Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN,
                    down + e * Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE, routed,
                    error, error_len))
                return false;
            for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
                output[t * Q38_MOE_HIDDEN + d] += routes[t].weight[k] * routed[d];
        }
        if (!q38_moe_shared_ref(x, shared_gate_proj, shared_up, shared_down,
                                shared_gate_weight, shared, error, error_len))
            return false;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            output[t * Q38_MOE_HIDDEN + d] += shared[d];
    }
    return true;
}
