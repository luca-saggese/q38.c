#include "q38_state.h"

#include <stdio.h>

int main(void) {
    q38_session_state state;
    char error[128];
    if (!q38_session_state_init(&state, 0, error, sizeof(error)) ||
        !q38_session_state_validate(&state, error, sizeof(error)) ||
        state.recurrent.slot_count != 36 ||
        state.conv_history.history_tokens != 3 ||
        state.memory.workspace_bytes != 0 ||
        state.memory.allocation_bytes !=
            state.memory.persistent_bytes + state.memory.activation_bytes)
        return 1;
    printf("test_m6_memory_baseline: persistent=%llu activation=%llu\n",
           (unsigned long long)state.memory.persistent_bytes,
           (unsigned long long)state.memory.activation_bytes);
    return 0;
}
