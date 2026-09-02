#include "q38_ple_ref.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const size_t hidden = 2560;
    const size_t elements = 4 * hidden;
    static float input[4 * 2560];
    static float actual[4 * 2560];
    static float weight[4 * 2560];
    for (size_t stream = 0; stream < 4; ++stream)
        for (size_t d = 0; d < hidden; ++d) {
            input[stream * hidden + d] =
                (float)(stream + 1) * (d % 2 ? 2.0f : 1.0f);
            weight[stream * hidden + d] = 1.0f;
        }
    for (size_t i = 0; i < elements; ++i) actual[i] = input[i];
    q38_ple_grouped_norm_inplace(actual, weight, 1, 4, hidden, 1e-6f);

    for (size_t stream = 0; stream < 4; ++stream) {
        const float *row = input + stream * hidden;
        float sum = 0.0f;
        for (size_t d = 0; d < hidden; ++d) sum += row[d] * row[d];
        const float rms = sqrtf(sum / hidden + 1e-6f);
        for (size_t d = 0; d < hidden; ++d) {
            const float expected = row[d] / rms;
            if (fabsf(actual[stream * 2 + d] - expected) > 2e-5f) {
                fprintf(stderr, "stream %zu grouped RMS mismatch\n", stream);
                return 1;
            }
        }
    }

    float global_sum = 0.0f;
    for (size_t i = 0; i < elements; ++i) global_sum += input[i] * input[i];
    const float global_rms = sqrtf(global_sum / elements + 1e-6f);
    if (fabsf(input[0] / sqrtf(2.5f + 1e-6f) - input[0] / global_rms) <
        1e-3f) {
        fprintf(stderr, "grouped RMS unexpectedly matched global RMS\n");
        return 1;
    }
    puts("PLE grouped normalization regression: PASS");
    return 0;
}
