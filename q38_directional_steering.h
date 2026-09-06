#ifndef Q38_DIRECTIONAL_STEERING_H
#define Q38_DIRECTIONAL_STEERING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define Q38_DIRECTIONAL_STEERING_LAYERS 48u
#define Q38_DIRECTIONAL_STEERING_HIDDEN 2560u
#define Q38_DIRECTIONAL_STEERING_FLOATS \
    (Q38_DIRECTIONAL_STEERING_LAYERS * Q38_DIRECTIONAL_STEERING_HIDDEN)
#define Q38_DIRECTIONAL_STEERING_BYTES \
    ((size_t)Q38_DIRECTIONAL_STEERING_FLOATS * sizeof(float))

typedef struct {
    float *directions; /* [layers][hidden], normalized f32 */
    uint32_t layers;
    uint32_t hidden_size;
    float ffn_scale;
    float attn_scale;
    bool enabled;
    uint64_t fingerprint;
} q38_directional_steering;

typedef struct {
    bool set;
    float ffn_scale;
    float attn_scale;
} q38_directional_steering_override;

#ifdef __cplusplus
extern "C" {
#endif

void q38_directional_steering_init(q38_directional_steering *steering);
void q38_directional_steering_destroy(q38_directional_steering *steering);

bool q38_directional_steering_load(
    q38_directional_steering *steering, const char *path, float ffn_scale,
    float attn_scale, char *error, size_t error_len);

bool q38_directional_steering_validate(
    const float *directions, size_t count, char *error, size_t error_len);

bool q38_directional_steering_enabled(
    const q38_directional_steering *steering, float scale);

void q38_directional_steering_apply_cpu(
    float *values, size_t rows, const q38_directional_steering *steering,
    uint32_t layer, float scale);

void q38_directional_steering_effective(
    const q38_directional_steering *default_steering,
    const q38_directional_steering_override *override,
    float *ffn_scale, float *attn_scale);

#ifdef __cplusplus
}
#endif

#endif
