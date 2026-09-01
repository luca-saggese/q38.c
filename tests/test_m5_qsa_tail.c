#include "q38_qsa_ref.h"

#include <stdio.h>

int main(void) {
    const size_t cases[] = {1, 2, 3, 4, 5, 6, 7, 8, 2047, 2048, 2049};
    char error[128];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        size_t width = 0;
        if (!q38_qsa_selected_width_ref(cases[i], 2048, 4, &width, error,
                                        sizeof(error)))
            return 1;
        const size_t expected = cases[i] < 2051 ? cases[i] : 2051;
        if (width != expected) {
            fprintf(stderr, "tail width mismatch for %zu\n", cases[i]);
            return 1;
        }
    }
    puts("test_m5_qsa_tail: causal incomplete-block width matrix passed");
    return 0;
}
