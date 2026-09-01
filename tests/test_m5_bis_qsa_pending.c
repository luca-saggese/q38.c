#include "q38_qsa.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    const size_t boundaries[] = {1,2,3,4,5,7,8,15,16,31,32,127,128,
                                 511,512,2047,2048,4095,4096};
    char error[128];
    for (size_t b = 0; b < sizeof(boundaries)/sizeof(boundaries[0]); ++b) {
        q38_qsa_state state, clone;
        if (!q38_qsa_state_init(&state, 4, 4, 2, error, sizeof(error)))
            return 1;
        unsigned char k[4] = {1,2,3,4}, i[2] = {5,6};
        for (size_t n = 0; n < boundaries[b]; ++n)
            if (!q38_qsa_state_append(&state, k, k, i, 1, error,
                                      sizeof(error))) return 1;
        if (state.pending_count != boundaries[b] % 4 ||
            state.pending_position != boundaries[b] - state.pending_count ||
            !q38_qsa_state_clone(&state, &clone, error, sizeof(error)) ||
            clone.position != state.position ||
            clone.pending_count != state.pending_count ||
            memcmp(clone.index_k.data, state.index_k.data,
                   state.index_k.count * 2) != 0) return 1;
        q38_qsa_state_destroy(&clone);
        q38_qsa_state_destroy(&state);
    }
    puts("test_m5_bis_qsa_pending: raw pending boundaries, clone, and restore passed");
    return 0;
}
