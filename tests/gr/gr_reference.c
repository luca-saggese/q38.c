#include "gr_reference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static float silu(float value) {
    return value * sigmoid(value);
}

static int normalize(const float *residual, const float *hc_norm,
                     float *normalized) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    if (!residual || !hc_norm || !normalized) return 0;
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; ++branch) {
        const float *source = residual + branch * Q38_GR_HIDDEN;
        float *destination = normalized + branch * Q38_GR_HIDDEN;
        double sum = 0.0;
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const double value = source[channel];
            sum += value * value;
        }
        const float scale = 1.0f /
            sqrtf((float)(sum / (double)Q38_GR_HIDDEN) + 1e-6f);
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const size_t index = branch * Q38_GR_HIDDEN + channel;
            destination[channel] = source[channel] * scale *
                (1.0f + hc_norm[index]);
        }
    }
    (void)width;
    return 1;
}

int gr_reference_read(const float *residual, const gr_reference_params *params,
                      float *input) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    float *normalized = NULL;
    float *bottleneck = NULL;
    float *gates = NULL;
    if (!residual || !params || !input || !params->hc_norm ||
        !params->input_mix_down || !params->input_mix_up)
        return 0;
    normalized = calloc(width, sizeof(*normalized));
    bottleneck = calloc(Q38_GR_RANK, sizeof(*bottleneck));
    gates = calloc(width, sizeof(*gates));
    if (!normalized || !bottleneck || !gates) {
        free(normalized);
        free(bottleneck);
        free(gates);
        return 0;
    }
    if (!normalize(residual, params->hc_norm, normalized)) {
        free(normalized);
        free(bottleneck);
        free(gates);
        return 0;
    }
    for (size_t rank = 0; rank < Q38_GR_RANK; ++rank) {
        float value = 0.0f;
        for (size_t i = 0; i < width; ++i)
            value += params->input_mix_down[rank * width + i] *
                     normalized[i];
        bottleneck[rank] = silu(value / (float)Q38_GR_HC_COUNT);
    }
    for (size_t i = 0; i < width; ++i) {
        float value = 0.0f;
        for (size_t rank = 0; rank < Q38_GR_RANK; ++rank)
            value += params->input_mix_up[i * Q38_GR_RANK + rank] *
                     bottleneck[rank];
        gates[i] = sigmoid(value);
    }
    for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
        float value = 0.0f;
        for (size_t branch = 0; branch < Q38_GR_BRANCHES; ++branch)
            value += gates[branch * Q38_GR_HIDDEN + channel] *
                     normalized[branch * Q38_GR_HIDDEN + channel];
        input[channel] = value / (float)Q38_GR_BRANCHES;
    }
    free(normalized);
    free(bottleneck);
    free(gates);
    return 1;
}

int gr_reference_write(const float *residual, const float *block_output,
                       const gr_reference_params *params, float *updated) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    float *normalized = NULL;
    if (!residual || !block_output || !params || !updated ||
        !params->hc_norm || !params->block_inject)
        return 0;
    normalized = calloc(width, sizeof(*normalized));
    if (!normalized) return 0;
    if (!normalize(residual, params->hc_norm, normalized)) {
        free(normalized);
        return 0;
    }
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; ++branch) {
        float value = 0.0f;
        for (size_t i = 0; i < width; ++i)
            value += params->block_inject[branch * width + i] *
                     normalized[i];
        const float scale = 2.0f *
            sigmoid(value / (float)Q38_GR_HC_COUNT);
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const size_t index = branch * Q38_GR_HIDDEN + channel;
            updated[index] = residual[index] + scale * block_output[channel];
        }
    }
    free(normalized);
    return 1;
}

int gr_reference_collapse(const float *residual, const float *block_output,
                          const gr_reference_params *params, float *input,
                          float *updated) {
    return gr_reference_read(residual, params, input) &&
           gr_reference_write(residual, block_output, params, updated);
}
