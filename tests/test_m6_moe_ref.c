#include "q38_moe_ref.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    float *hidden = calloc(Q38_MOE_HIDDEN, sizeof(float));
    float *router = calloc(Q38_MOE_EXPERTS * Q38_MOE_HIDDEN, sizeof(float));
    q38_moe_route10 route;
    char error[128];
    if (!hidden || !router ||
        !q38_moe_route_ref(hidden, 1, router, &route, error, sizeof(error))) {
        fprintf(stderr, "router reference failed: %s\n", error);
        free(hidden);
        free(router);
        return 1;
    }
    for (size_t i = 0; i < Q38_MOE_TOP_K; ++i) {
        if (route.expert[i] != i ||
            route.weight[i] != 1.0f / (float)Q38_MOE_TOP_K) {
            fprintf(stderr, "router tie ordering/normalization mismatch\n");
            free(hidden);
            free(router);
            return 1;
        }
    }
    free(hidden);
    free(router);
    puts("test_m6_moe_ref: deterministic router top-10 and normalization passed");
    return 0;
}
