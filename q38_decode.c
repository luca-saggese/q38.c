#include "q38_decode.h"

#include <math.h>
#include <stdio.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_decode(const q38_gguf *model, const q38_weights *weights,
                q38_forward_state *state, uint32_t token, float *logits,
                size_t logits_stride, uint32_t *next_token,
                q38_forward_diagnostics *diagnostics, char *error,
                size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!next_token)
        return fail(error, error_len, "decode output token is null");
    if (!q38_forward_full(model, weights, state, &token, 1, logits,
                          logits_stride, diagnostics, error, error_len))
        return false;
    size_t best = 0;
    float best_value = logits[0];
    if (!isfinite(best_value))
        return fail(error, error_len, "decode logits contain a non-finite value");
    for (size_t i = 1; i < 248320; ++i) {
        if (!isfinite(logits[i]))
            return fail(error, error_len, "decode logits contain a non-finite value");
        if (logits[i] > best_value) {
            best = i;
            best_value = logits[i];
        }
    }
    *next_token = (uint32_t)best;
    return true;
}
