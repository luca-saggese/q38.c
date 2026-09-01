#include "q38_qsa.h"

#include <stdio.h>

int main(void) {
    q38_qsa_state state;
    char error[128];
    if (!q38_qsa_state_init(&state, 4, 8, 2, error, sizeof(error)))
        return 1;
    const unsigned char k[] = {1, 2, 3, 4};
    const unsigned char v[] = {5, 6, 7, 8, 9, 10, 11, 12};
    const unsigned char i[] = {13, 14};
    if (!q38_qsa_state_append(&state, k, v, i, 1, error, sizeof(error)) ||
        state.main_k.count != 1 || state.main_v.count != 1 ||
        state.index_k.count != 1 || state.position != 1)
        return 1;
    if (!q38_qsa_state_append(&state, k, v, i, 1, error, sizeof(error)) ||
        state.main_k.count != 2 || state.committed_tokens != 2 ||
        state.main_k.capacity < 2 || state.index_k.capacity < 2)
        return 1;
    if (state.main_k.data[0] != 1 || state.main_k.data[4] != 1 ||
        state.main_v.data[8] != 5 || state.index_k.data[2] != 13)
        return 1;
    q38_qsa_state_reset(&state);
    if (state.main_k.count || state.main_v.count || state.index_k.count ||
        state.position || state.committed_tokens)
        return 1;
    q38_qsa_state_destroy(&state);
    puts("test_m5_qsa_state: independent KV/index growth and reset passed");
    return 0;
}
