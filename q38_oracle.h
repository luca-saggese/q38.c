#ifndef Q38_ORACLE_H
#define Q38_ORACLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float max_abs;
    float mean_abs;
    float rms;
    float relative;
    float cosine;
} q38_oracle_metrics;

void q38_oracle_compare(const float *expected, const float *actual,
                        size_t elements, float epsilon,
                        q38_oracle_metrics *metrics);

#ifdef __cplusplus
}
#endif

#endif
