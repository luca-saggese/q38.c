#include "q38_gr_ref.h"

#include <math.h>

static float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static float silu(float value) {
    return value * sigmoid(value);
}

static void normalized_branches(const float *residual,
                                const q38_gr_ref_params *params,
                                float *normalized) {
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; branch++) {
        const float *source = residual + branch * Q38_GR_HIDDEN;
        float *destination = normalized + branch * Q38_GR_HIDDEN;
        float sum = 0.0f;
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; channel++)
            sum += source[channel] * source[channel];
        const float scale = 1.0f /
            sqrtf(sum / (float)Q38_GR_HIDDEN + 1e-6f);
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; channel++)
            destination[channel] = source[channel] * scale *
                params->gamma[branch * Q38_GR_HIDDEN + channel];
    }
}

void q38_gr_read(const float *residual, const q38_gr_ref_params *params,
                 float *input) {
    float normalized[Q38_GR_BRANCHES * Q38_GR_HIDDEN];
    float bottleneck[Q38_GR_RANK];
    float gates[Q38_GR_BRANCHES * Q38_GR_HIDDEN];
    normalized_branches(residual, params, normalized);
    for (size_t rank = 0; rank < Q38_GR_RANK; rank++) {
        float value = 0.0f;
        for (size_t i = 0; i < Q38_GR_BRANCHES * Q38_GR_HIDDEN; i++)
            value += params->input_mix_down[rank * Q38_GR_BRANCHES *
                                            Q38_GR_HIDDEN + i] * normalized[i];
        bottleneck[rank] = silu(value / (float)Q38_GR_HC_COUNT);
    }
    for (size_t i = 0; i < Q38_GR_BRANCHES * Q38_GR_HIDDEN; i++) {
        float value = 0.0f;
        for (size_t rank = 0; rank < Q38_GR_RANK; rank++)
            value += params->input_mix_up[i * Q38_GR_RANK + rank] *
                     bottleneck[rank];
        gates[i] = sigmoid(value);
    }
    for (size_t channel = 0; channel < Q38_GR_HIDDEN; channel++) {
        input[channel] = 0.0f;
        for (size_t branch = 0; branch < Q38_GR_BRANCHES; branch++)
            input[channel] += gates[branch * Q38_GR_HIDDEN + channel] *
                              normalized[branch * Q38_GR_HIDDEN + channel];
        input[channel] /= (float)Q38_GR_BRANCHES;
    }
}

void q38_gr_write(const float *residual, const float *block_output,
                  const q38_gr_ref_params *params, float *updated) {
    float normalized[Q38_GR_BRANCHES * Q38_GR_HIDDEN];
    normalized_branches(residual, params, normalized);
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; branch++) {
        float value = 0.0f;
        for (size_t i = 0; i < Q38_GR_BRANCHES * Q38_GR_HIDDEN; i++)
            value += params->block_inject[branch * Q38_GR_BRANCHES *
                                           Q38_GR_HIDDEN + i] * normalized[i];
        const float scale = 2.0f * sigmoid(value / (float)Q38_GR_HC_COUNT);
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; channel++)
            updated[branch * Q38_GR_HIDDEN + channel] =
                residual[branch * Q38_GR_HIDDEN + channel] +
                scale * block_output[channel];
    }
}

void q38_gr_collapse(const float *residual, const float *block_output,
                     const q38_gr_ref_params *params, float *input,
                     float *updated) {
    q38_gr_read(residual, params, input);
    q38_gr_write(residual, block_output, params, updated);
}
