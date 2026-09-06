#ifndef Q38_GR_TEST_REFERENCE_H
#define Q38_GR_TEST_REFERENCE_H

#include <stddef.h>

#include "../../q38_gr_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *hc_norm;
    const float *input_mix_down;
    const float *input_mix_up;
    const float *block_inject;
} gr_reference_params;

int gr_reference_read(const float *residual, const gr_reference_params *params,
                      float *input);
int gr_reference_write(const float *residual, const float *block_output,
                       const gr_reference_params *params, float *updated);
int gr_reference_collapse(const float *residual, const float *block_output,
                          const gr_reference_params *params, float *input,
                          float *updated);

#ifdef __cplusplus
}
#endif

#endif
