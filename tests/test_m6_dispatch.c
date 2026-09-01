#include "q38_moe_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t tokens = 2;
    q38_moe_route10 routes[2] = {{{0}, {0}}, {{0}, {0}}};
    for (size_t t = 0; t < tokens; ++t)
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
            routes[t].expert[k] = (uint16_t)k;
            routes[t].weight[k] = 0.1f;
        }
    float *experts = calloc(tokens * Q38_MOE_TOP_K * Q38_MOE_HIDDEN,
                            sizeof(float));
    float *shared = calloc(tokens * Q38_MOE_HIDDEN, sizeof(float));
    float *output = calloc(tokens * Q38_MOE_HIDDEN, sizeof(float));
    for (size_t t = 0; t < tokens; ++t)
        for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
            experts[(t * Q38_MOE_TOP_K + k) * Q38_MOE_HIDDEN] =
                (float)(k + 1);
    char error[128];
    if (!experts || !shared || !output ||
        !q38_moe_combine_ref(routes, tokens, experts, shared, output, error,
                             sizeof(error)) ||
        fabsf(output[0] - 5.5f) > 1e-6f ||
        fabsf(output[Q38_MOE_HIDDEN] - 5.5f) > 1e-6f) return 1;
    free(experts); free(shared); free(output);
    puts("test_m6_dispatch: stable weighted routed/shared combine passed");
    return 0;
}
