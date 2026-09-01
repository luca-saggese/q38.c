#ifndef Q38_ROPE_REF_H
#define Q38_ROPE_REF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float theta;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    uint32_t n_dims;
    uint32_t sections[4];
    bool interleaved;
} q38_rope_config;

bool q38_rope_apply_ref(const q38_rope_config *config,
                        const int64_t positions[4], const float *input,
                        float *output, size_t element_count, char *error,
                        size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
