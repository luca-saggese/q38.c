#include "q38_state.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    q38_session_state layout;
    char error[256];
    if (!q38_session_state_init(&layout, 0, error, sizeof(error))) {
        fprintf(stderr, "state layout failed: %s\n", error);
        return 1;
    }
    if (layout.recurrent.value_heads != 48 ||
        layout.recurrent.head_dim != 128 ||
        layout.recurrent.dtype != Q38_STATE_DTYPE_F32 ||
        layout.recurrent.bytes != UINT64_C(3145728) ||
        layout.conv_history.channels != 10240 ||
        layout.conv_history.kernel != 4 ||
        layout.conv_history.history_tokens != 3 ||
        layout.conv_history.bytes != UINT64_C(122880) ||
        layout.gr.bytes != UINT64_C(40960) ||
        layout.memory.persistent_bytes != UINT64_C(3309568) ||
        layout.memory.allocation_bytes != UINT64_C(3309568) ||
        layout.memory.workspace_bytes != 0) {
        fprintf(stderr, "verified state shape or byte accounting mismatch\n");
        return 1;
    }
    q38_state_storage storage;
    if (!q38_state_alloc(&layout, &storage, error, sizeof(error))) {
        fprintf(stderr, "state allocation failed: %s\n", error);
        return 1;
    }
    storage.recurrent_state[0] = 1.0f;
    storage.conv_history[0] = 2.0f;
    storage.gr_state[0] = 3.0f;
    q38_state_reset(&storage);
    if (storage.recurrent_state[0] != 0.0f ||
        storage.conv_history[0] != 0.0f ||
        storage.gr_state[0] != 0.0f ||
        storage.workspace != NULL) {
        fprintf(stderr, "state reset did not clear all regions\n");
        q38_state_free(&storage);
        return 1;
    }
    q38_session_state copy = layout;
    if (!q38_session_state_validate(&copy, error, sizeof(error))) {
        fprintf(stderr, "serializable layout validation failed: %s\n", error);
        q38_state_free(&storage);
        return 1;
    }
    q38_state_free(&storage);
    if (storage.recurrent_state || storage.conv_history || storage.gr_state ||
        storage.workspace || memcmp(&copy, &layout, sizeof(layout)) != 0) {
        fprintf(stderr, "state release or layout copy mismatch\n");
        return 1;
    }
    puts("test_m3_state: verified FP32 GDN/conv/GR layout allocation and reset passed");
    return 0;
}
