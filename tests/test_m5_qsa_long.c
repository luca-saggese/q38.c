#include "q38_qsa.h"

#include <stdio.h>

int main(void) {
    q38_qsa_state state;
    char error[128];
    if (!q38_qsa_state_init(&state, 4, 4, 2, error, sizeof(error)))
        return 1;
    const unsigned char k[] = {1, 2, 3, 4};
    const unsigned char i[] = {5, 6};
    for (size_t n = 0; n < 65536; ++n)
        if (!q38_qsa_state_append(&state, k, k, i, 1, error, sizeof(error)))
            return 1;
    if (state.position != 65536 || state.committed_tokens != 65536 ||
        state.main_k.count != 65536 || state.index_k.count != 65536 ||
        state.main_k.data[65535 * 4] != 1 ||
        state.index_k.data[65535 * 2] != 5)
        return 1;
    q38_qsa_state_reset(&state);
    if (state.position || state.committed_tokens || state.main_k.count ||
        state.main_v.count || state.index_k.count)
        return 1;
    q38_qsa_state_destroy(&state);
    puts("test_m5_qsa_long: staged 1k/4k/16k/64k state growth and reset passed");
    return 0;
}
