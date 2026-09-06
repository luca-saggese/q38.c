#include "q38_directional_steering.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

void q38_directional_steering_init(q38_directional_steering *steering) {
    if (steering) memset(steering, 0, sizeof(*steering));
}

void q38_directional_steering_destroy(q38_directional_steering *steering) {
    if (!steering) return;
    free(steering->directions);
    memset(steering, 0, sizeof(*steering));
}

bool q38_directional_steering_validate(
    const float *directions, size_t count, char *error, size_t error_len) {
    if (!directions || count != Q38_DIRECTIONAL_STEERING_FLOATS)
        return fail(error, error_len, "invalid Q38 steering vector shape");
    for (uint32_t layer = 0; layer < Q38_DIRECTIONAL_STEERING_LAYERS;
         ++layer) {
        const float *row = directions +
                           (size_t)layer * Q38_DIRECTIONAL_STEERING_HIDDEN;
        double norm2 = 0.0;
        for (uint32_t i = 0; i < Q38_DIRECTIONAL_STEERING_HIDDEN; ++i) {
            if (!isfinite(row[i]))
                return fail(error, error_len,
                            "Q38 steering vector contains non-finite data");
            norm2 += (double)row[i] * row[i];
        }
        const double norm = sqrt(norm2);
        if (!(norm > 1.0e-6) || fabs(norm - 1.0) > 0.05)
            return fail(error, error_len,
                        "Q38 steering layer is not normalized");
    }
    return true;
}

static uint64_t fingerprint_bytes(const unsigned char *data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool q38_directional_steering_load(
    q38_directional_steering *steering, const char *path, float ffn_scale,
    float attn_scale, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!steering || !path || !path[0])
        return fail(error, error_len, "Q38 steering file path is required");
    if (!isfinite(ffn_scale) || !isfinite(attn_scale) ||
        fabsf(ffn_scale) > 100.0f || fabsf(attn_scale) > 100.0f)
        return fail(error, error_len, "Q38 steering scale is out of range");

    FILE *file = fopen(path, "rb");
    if (!file) {
        if (error && error_len)
            snprintf(error, error_len, "cannot open Q38 steering file %s: %s",
                     path, strerror(errno));
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return fail(error, error_len, "cannot seek Q38 steering file");
    }
    long length = ftell(file);
    if (length < 0 || (size_t)length != Q38_DIRECTIONAL_STEERING_BYTES) {
        fclose(file);
        if (error && error_len)
            snprintf(error, error_len,
                     "Q38 steering file must be exactly %zu bytes",
                     Q38_DIRECTIONAL_STEERING_BYTES);
        return false;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return fail(error, error_len, "cannot rewind Q38 steering file");
    }
    float *directions = (float *)malloc(Q38_DIRECTIONAL_STEERING_BYTES);
    if (!directions) {
        fclose(file);
        return fail(error, error_len, "Q38 steering allocation failed");
    }
    const size_t read = fread(directions, 1, Q38_DIRECTIONAL_STEERING_BYTES,
                              file);
    const bool closed = fclose(file) == 0;
    if (read != Q38_DIRECTIONAL_STEERING_BYTES || !closed) {
        free(directions);
        return fail(error, error_len, "Q38 steering file read failed");
    }
    if (!q38_directional_steering_validate(
            directions, Q38_DIRECTIONAL_STEERING_FLOATS, error, error_len)) {
        free(directions);
        return false;
    }

    q38_directional_steering_destroy(steering);
    steering->directions = directions;
    steering->layers = Q38_DIRECTIONAL_STEERING_LAYERS;
    steering->hidden_size = Q38_DIRECTIONAL_STEERING_HIDDEN;
    steering->ffn_scale = ffn_scale;
    steering->attn_scale = attn_scale;
    steering->enabled = ffn_scale != 0.0f || attn_scale != 0.0f;
    steering->fingerprint =
        fingerprint_bytes((const unsigned char *)directions,
                          Q38_DIRECTIONAL_STEERING_BYTES);
    return true;
}

bool q38_directional_steering_enabled(
    const q38_directional_steering *steering, float scale) {
    return steering && steering->directions && scale != 0.0f;
}

void q38_directional_steering_apply_cpu(
    float *values, size_t rows, const q38_directional_steering *steering,
    uint32_t layer, float scale) {
    if (!values || !q38_directional_steering_enabled(steering, scale) ||
        layer >= steering->layers)
        return;
    const float *direction =
        steering->directions +
        (size_t)layer * (size_t)steering->hidden_size;
    for (size_t row = 0; row < rows; ++row) {
        float *value = values + row * steering->hidden_size;
        float dot = 0.0f;
        for (uint32_t i = 0; i < steering->hidden_size; ++i)
            dot += direction[i] * value[i];
        const float coefficient = scale * dot;
        for (uint32_t i = 0; i < steering->hidden_size; ++i)
            value[i] -= coefficient * direction[i];
    }
}

void q38_directional_steering_effective(
    const q38_directional_steering *default_steering,
    const q38_directional_steering_override *override, float *ffn_scale,
    float *attn_scale) {
    if (!ffn_scale || !attn_scale) return;
    *ffn_scale = default_steering ? default_steering->ffn_scale : 0.0f;
    *attn_scale = default_steering ? default_steering->attn_scale : 0.0f;
    if (override && override->set) {
        *ffn_scale = override->ffn_scale;
        *attn_scale = override->attn_scale;
    }
}
