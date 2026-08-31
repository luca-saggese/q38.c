#include "q38_oracle.h"

#include <math.h>
#include <string.h>

void q38_oracle_compare(const float *expected, const float *actual,
                        size_t elements, float epsilon,
                        q38_oracle_metrics *metrics) {
    if (!metrics) return;
    memset(metrics, 0, sizeof(*metrics));
    if (!expected || !actual || !elements) return;
    double abs_sum = 0.0, square_sum = 0.0, relative_sum = 0.0;
    double dot = 0.0, expected_norm = 0.0, actual_norm = 0.0;
    for (size_t i = 0; i < elements; i++) {
        double diff = (double)actual[i] - expected[i];
        double abs_diff = fabs(diff);
        double scale = fabs((double)expected[i]);
        if ((float)abs_diff > metrics->max_abs) metrics->max_abs = (float)abs_diff;
        abs_sum += abs_diff;
        square_sum += diff * diff;
        relative_sum += abs_diff / (scale + epsilon);
        dot += (double)expected[i] * actual[i];
        expected_norm += (double)expected[i] * expected[i];
        actual_norm += (double)actual[i] * actual[i];
    }
    metrics->mean_abs = (float)(abs_sum / elements);
    metrics->rms = (float)sqrt(square_sum / elements);
    metrics->relative = (float)(relative_sum / elements);
    metrics->cosine = expected_norm && actual_norm
        ? (float)(dot / sqrt(expected_norm * actual_norm)) : 0.0f;
}
