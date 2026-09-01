#include "q38_moe_ref.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const size_t special[] = {127,128,129,255,256,257,507,508,509,
                              511,512,513,1015,1016,1017};
    float *router = calloc(Q38_MOE_EXPERTS * Q38_MOE_HIDDEN, sizeof(float));
    float *hidden = calloc(1024 * Q38_MOE_HIDDEN, sizeof(float));
    q38_moe_route10 *routes = calloc(1024, sizeof(*routes));
    char error[128];
    if (!router || !hidden || !routes) return 1;
    if (!q38_moe_route_ref(hidden, 1017, router, routes, error, sizeof(error)))
        return 1;
    for (size_t n = 1; n <= 64; ++n)
        if (routes[n - 1].expert[0] != 0) return 1;
    for (size_t i = 0; i < sizeof(special)/sizeof(special[0]); ++i)
        if (routes[special[i] - 1].expert[0] != 0) return 1;
    free(router); free(hidden); free(routes);
    puts("test_m6_ubatch: router/route buffer tail fuzz passed");
    return 0;
}
