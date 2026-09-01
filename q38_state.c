#include "q38_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
}

static bool checked_product(uint64_t a, uint64_t b, uint64_t *out) {
    if (b != 0 && a > UINT64_MAX / b) return false;
    *out = a * b;
    return true;
}

static bool checked_bytes(uint64_t elements, uint64_t *out) {
    return checked_product(elements, sizeof(float), out);
}

static bool checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool expected_layout(q38_session_state *state, uint64_t workspace_bytes,
                            char *error, size_t error_len) {
    uint64_t elements = 0;
    if (sizeof(float) != 4) {
        set_error(error, error_len, "F32 state requires four-byte float");
        return false;
    }
    if (!state || !checked_product(Q38_GDN_SEQUENCE_COUNT,
                                   Q38_GDN_VALUE_HEADS, &elements) ||
        !checked_product(elements, Q38_GDN_HEAD_DIM, &elements) ||
        !checked_product(elements, Q38_GDN_HEAD_DIM, &elements) ||
        !checked_bytes(elements, &state->recurrent.bytes)) {
        set_error(error, error_len, "recurrent state size overflow");
        return false;
    }
    state->recurrent.sequence_count = Q38_GDN_SEQUENCE_COUNT;
    state->recurrent.value_heads = Q38_GDN_VALUE_HEADS;
    state->recurrent.head_dim = Q38_GDN_HEAD_DIM;
    state->recurrent.dtype = Q38_STATE_DTYPE_F32;
    state->recurrent.elements = elements;

    if (!checked_product(Q38_GDN_SEQUENCE_COUNT, Q38_GDN_CONV_CHANNELS,
                         &elements) ||
        !checked_product(elements, Q38_GDN_CONV_KERNEL - 1u, &elements) ||
        !checked_bytes(elements, &state->conv_history.bytes)) {
        set_error(error, error_len, "convolution history size overflow");
        return false;
    }
    state->conv_history.sequence_count = Q38_GDN_SEQUENCE_COUNT;
    state->conv_history.channels = Q38_GDN_CONV_CHANNELS;
    state->conv_history.kernel = Q38_GDN_CONV_KERNEL;
    state->conv_history.history_tokens = Q38_GDN_CONV_KERNEL - 1u;
    state->conv_history.dtype = Q38_STATE_DTYPE_F32;
    state->conv_history.elements = elements;

    if (!checked_product(Q38_GDN_SEQUENCE_COUNT, Q38_GR_STATE_BRANCHES,
                         &elements) ||
        !checked_product(elements, Q38_GR_STATE_HIDDEN, &elements) ||
        !checked_bytes(elements, &state->gr.bytes)) {
        set_error(error, error_len, "GR state size overflow");
        return false;
    }
    state->gr.sequence_count = Q38_GDN_SEQUENCE_COUNT;
    state->gr.branches = Q38_GR_STATE_BRANCHES;
    state->gr.hidden_size = Q38_GR_STATE_HIDDEN;
    state->gr.dtype = Q38_STATE_DTYPE_F32;
    state->gr.elements = elements;

    state->memory.persistent_recurrent_state_bytes =
        state->recurrent.bytes;
    state->memory.conv_history_bytes = state->conv_history.bytes;
    state->memory.gr_state_bytes = state->gr.bytes;
    state->memory.workspace_bytes = workspace_bytes;
    if (!checked_add(state->memory.persistent_recurrent_state_bytes,
                     state->memory.conv_history_bytes,
                     &state->memory.persistent_bytes) ||
        !checked_add(state->memory.persistent_bytes,
                     state->memory.gr_state_bytes,
                     &state->memory.persistent_bytes) ||
        !checked_add(state->memory.persistent_bytes, workspace_bytes,
                     &state->memory.allocation_bytes)) {
        set_error(error, error_len, "state memory size overflow");
        return false;
    }
    return true;
}

static bool same_recurrent_desc(const q38_gdn_state_desc *a,
                                const q38_gdn_state_desc *b) {
    return a->sequence_count == b->sequence_count &&
           a->value_heads == b->value_heads &&
           a->head_dim == b->head_dim &&
           a->dtype == b->dtype &&
           a->elements == b->elements &&
           a->bytes == b->bytes;
}

static bool same_conv_desc(const q38_conv_history_desc *a,
                           const q38_conv_history_desc *b) {
    return a->sequence_count == b->sequence_count &&
           a->channels == b->channels &&
           a->kernel == b->kernel &&
           a->history_tokens == b->history_tokens &&
           a->dtype == b->dtype &&
           a->elements == b->elements &&
           a->bytes == b->bytes;
}

static bool same_gr_desc(const q38_gr_state_desc *a,
                         const q38_gr_state_desc *b) {
    return a->sequence_count == b->sequence_count &&
           a->branches == b->branches &&
           a->hidden_size == b->hidden_size &&
           a->dtype == b->dtype &&
           a->elements == b->elements &&
           a->bytes == b->bytes;
}

static bool same_memory(const q38_state_memory *a,
                        const q38_state_memory *b) {
    return a->persistent_recurrent_state_bytes ==
               b->persistent_recurrent_state_bytes &&
           a->conv_history_bytes == b->conv_history_bytes &&
           a->gr_state_bytes == b->gr_state_bytes &&
           a->workspace_bytes == b->workspace_bytes &&
           a->persistent_bytes == b->persistent_bytes &&
           a->allocation_bytes == b->allocation_bytes;
}

bool q38_session_state_init(q38_session_state *state,
                            uint64_t workspace_bytes,
                            char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!state) {
        set_error(error, error_len, "state layout is null");
        return false;
    }
    memset(state, 0, sizeof(*state));
    if (!expected_layout(state, workspace_bytes, error, error_len)) {
        memset(state, 0, sizeof(*state));
        return false;
    }
    return true;
}

bool q38_session_state_validate(const q38_session_state *state,
                                char *error, size_t error_len) {
    q38_session_state expected;
    if (error && error_len > 0) error[0] = '\0';
    if (!state) {
        set_error(error, error_len, "state layout is null");
        return false;
    }
    memset(&expected, 0, sizeof(expected));
    if (!expected_layout(&expected, state->memory.workspace_bytes,
                         error, error_len)) {
        if (!error || !error[0])
            set_error(error, error_len, "invalid state layout");
        return false;
    }
    if (!same_recurrent_desc(&state->recurrent, &expected.recurrent) ||
        !same_conv_desc(&state->conv_history, &expected.conv_history) ||
        !same_gr_desc(&state->gr, &expected.gr) ||
        !same_memory(&state->memory, &expected.memory)) {
        set_error(error, error_len, "state layout does not match verified shapes");
        return false;
    }
    return true;
}

static bool alloc_region(uint64_t bytes, float **out) {
    if (bytes > SIZE_MAX) return false;
    *out = bytes ? calloc(1, (size_t)bytes) : NULL;
    return bytes == 0 || *out != NULL;
}

bool q38_state_alloc(const q38_session_state *layout,
                     q38_state_storage *storage,
                     char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!layout || !storage ||
        !q38_session_state_validate(layout, error, error_len)) {
        if (!error || !error[0])
            set_error(error, error_len, "invalid state allocation arguments");
        return false;
    }
    memset(storage, 0, sizeof(*storage));
    storage->layout = *layout;
    if (!alloc_region(layout->memory.persistent_recurrent_state_bytes,
                      &storage->recurrent_state) ||
        !alloc_region(layout->memory.conv_history_bytes,
                      &storage->conv_history) ||
        !alloc_region(layout->memory.gr_state_bytes, &storage->gr_state)) {
        set_error(error, error_len, "state allocation failed");
        q38_state_free(storage);
        return false;
    }
    if (layout->memory.workspace_bytes > SIZE_MAX) {
        set_error(error, error_len, "workspace size exceeds size_t");
        q38_state_free(storage);
        return false;
    }
    if (layout->memory.workspace_bytes) {
        storage->workspace =
            calloc(1, (size_t)layout->memory.workspace_bytes);
        if (!storage->workspace) {
            set_error(error, error_len, "workspace allocation failed");
            q38_state_free(storage);
            return false;
        }
    }
    return true;
}

void q38_state_reset(q38_state_storage *storage) {
    if (!storage) return;
    if (storage->recurrent_state)
        memset(storage->recurrent_state, 0,
               (size_t)storage->layout.memory.persistent_recurrent_state_bytes);
    if (storage->conv_history)
        memset(storage->conv_history, 0,
               (size_t)storage->layout.memory.conv_history_bytes);
    if (storage->gr_state)
        memset(storage->gr_state, 0,
               (size_t)storage->layout.memory.gr_state_bytes);
    if (storage->workspace)
        memset(storage->workspace, 0,
               (size_t)storage->layout.memory.workspace_bytes);
}

void q38_state_free(q38_state_storage *storage) {
    if (!storage) return;
    free(storage->recurrent_state);
    free(storage->conv_history);
    free(storage->gr_state);
    free(storage->workspace);
    memset(storage, 0, sizeof(*storage));
}
