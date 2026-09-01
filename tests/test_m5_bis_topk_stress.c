#include "q38_topk_ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const size_t lengths[] = {1,2,3,4,5,6,7,8,127,128,129,511,512,513,
                              2047,2048,2049,4095,4096,4097};
    char error[128];
    for (size_t c = 0; c < sizeof(lengths)/sizeof(lengths[0]); ++c) {
        size_t n = lengths[c], k = n < 17 ? n : 17;
        float *scores = malloc(n * sizeof(float));
        uint32_t *a = malloc(k * sizeof(uint32_t));
        uint32_t *b = malloc(k * sizeof(uint32_t));
        if (!scores || !a || !b) return 1;
        for (size_t i = 0; i < n; ++i) scores[i] = (float)((i * 37) % 19);
        if (!q38_topk_select_ref(scores, n, k, a, error, sizeof(error)) ||
            !q38_topk_select_ref(scores, n, k, b, error, sizeof(error)) ||
            memcmp(a, b, k * sizeof(*a)) != 0) return 1;
        free(scores); free(a); free(b);
    }
    puts("test_m5_bis_topk_stress: exact deterministic top-k boundary stress passed");
    return 0;
}
