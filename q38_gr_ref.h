#ifndef Q38_GR_REF_H
#define Q38_GR_REF_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_GR_HC_COUNT 4u
#define Q38_GR_BRANCHES Q38_GR_HC_COUNT
#define Q38_GR_HIDDEN 2560u
#define Q38_GR_RANK 320u

typedef struct {
    const float *gamma;
    const float *input_mix_down;
    const float *input_mix_up;
    const float *block_inject;
} q38_gr_ref_params;

void q38_gr_read(const float *residual, const q38_gr_ref_params *params,
                 float *input);
void q38_gr_write(const float *residual, const float *block_output,
                  const q38_gr_ref_params *params, float *updated);
void q38_gr_collapse(const float *residual, const float *block_output,
                     const q38_gr_ref_params *params, float *input,
                     float *updated);

#ifdef __cplusplus
}
#endif

#endif
