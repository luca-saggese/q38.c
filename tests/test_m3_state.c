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
    if (layout.recurrent.slot_count != Q38_GDN_LAYER_COUNT ||
        layout.recurrent.elements_per_slot != UINT64_C(786432) ||
        layout.recurrent.bytes_per_slot != UINT64_C(3145728) ||
        layout.recurrent.bytes != UINT64_C(113246208) ||
        layout.conv_history.slot_count != Q38_GDN_LAYER_COUNT ||
        layout.conv_history.elements_per_slot != UINT64_C(30720) ||
        layout.conv_history.bytes_per_slot != UINT64_C(122880) ||
        layout.conv_history.bytes != UINT64_C(4423680) ||
        layout.gr_workspace.bytes != UINT64_C(40960) ||
        layout.memory.persistent_bytes != UINT64_C(117669888) ||
        layout.memory.activation_bytes != UINT64_C(40960) ||
        layout.memory.allocation_bytes != UINT64_C(117710848)) {
        fprintf(stderr, "verified multi-GDN shape or byte accounting mismatch\n");
        return 1;
    }
    for (uint32_t layer = 0, slot = 0; layer < Q38_STATE_MODEL_LAYERS; layer++) {
        int mapped = q38_gdn_slot_for_layer(&layout, layer);
        if (layer % 4u == 3u) {
            if (mapped != -1) {
                fprintf(stderr, "full-attention layer %u mapped to %d\n",
                        layer, mapped);
                return 1;
            }
        } else if (mapped != (int)slot++) {
            fprintf(stderr, "layer %u mapped to %d, expected %u\n",
                    layer, mapped, slot - 1);
            return 1;
        }
    }

    q38_state_storage storage;
    if (!q38_state_alloc(&layout, &storage, error, sizeof(error))) {
        fprintf(stderr, "state allocation failed: %s\n", error);
        return 1;
    }
    float *recurrent0 = q38_state_recurrent_slot(&storage, 0);
    float *recurrent1 = q38_state_recurrent_slot(&storage, 1);
    float *conv0 = q38_state_conv_history_slot(&storage, 0);
    float *conv1 = q38_state_conv_history_slot(&storage, 1);
    if (!recurrent0 || !recurrent1 || !conv0 || !conv1) {
        fprintf(stderr, "per-slot state access failed\n");
        q38_state_free(&storage);
        return 1;
    }
    recurrent0[0] = 1.0f;
    conv0[0] = 2.0f;
    recurrent1[0] = 3.0f;
    conv1[0] = 4.0f;
    if (recurrent0 == recurrent1 || conv0 == conv1 ||
        recurrent0[0] == recurrent1[0] || conv0[0] == conv1[0]) {
        fprintf(stderr, "per-layer state is not isolated\n");
        q38_state_free(&storage);
        return 1;
    }
    storage.gr_workspace[0] = 5.0f;
    q38_state_reset(&storage);
    if (recurrent0[0] != 0.0f || recurrent1[0] != 0.0f ||
        conv0[0] != 0.0f || conv1[0] != 0.0f ||
        storage.gr_workspace[0] != 0.0f || storage.workspace != NULL) {
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
    if (storage.recurrent_state || storage.conv_history ||
        storage.gr_workspace || storage.workspace ||
        memcmp(&copy, &layout, sizeof(layout)) != 0) {
        fprintf(stderr, "state release or layout copy mismatch\n");
        return 1;
    }
    puts("test_m3_state: 36-slot GDN isolation/reset and GR workspace accounting passed");
    return 0;
}
