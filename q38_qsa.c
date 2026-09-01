#include "q38_qsa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_qsa_weights_validate(const q38_qsa_weights *weights,
                              char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!weights) return fail(error, error_len, "QSA weights are null");
    const q38_tensor *required[] = {
        weights->q_proj, weights->k_proj, weights->v_proj, weights->o_proj,
        weights->q_norm, weights->k_norm, weights->index_qk_proj,
        weights->index_q_norm, weights->index_k_norm,
    };
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        if (!required[i]) return fail(error, error_len, "QSA tensor set incomplete");
    }
    return true;
}

static bool cache_init(q38_qsa_cache *cache, size_t row_bytes,
                       char *error, size_t error_len) {
    if (!row_bytes) return fail(error, error_len, "QSA cache row size is zero");
    memset(cache, 0, sizeof(*cache));
    cache->row_bytes = row_bytes;
    return true;
}

static bool cache_reserve(q38_qsa_cache *cache, size_t extra,
                          char *error, size_t error_len) {
    if (extra > SIZE_MAX - cache->count)
        return fail(error, error_len, "QSA cache row count overflows");
    const size_t needed = cache->count + extra;
    if (needed <= cache->capacity) return true;
    size_t capacity = cache->capacity ? cache->capacity : 16;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / cache->row_bytes)
        return fail(error, error_len, "QSA cache byte size overflows");
    uint8_t *grown = (uint8_t *)realloc(
        cache->data, capacity * cache->row_bytes);
    if (!grown) return fail(error, error_len, "QSA cache growth failed");
    cache->data = grown;
    cache->capacity = capacity;
    return true;
}

bool q38_qsa_state_init(q38_qsa_state *state, size_t main_k_row_bytes,
                        size_t main_v_row_bytes, size_t index_k_row_bytes,
                        char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!state) return fail(error, error_len, "QSA state is null");
    memset(state, 0, sizeof(*state));
    if (!cache_init(&state->main_k, main_k_row_bytes, error, error_len) ||
        !cache_init(&state->main_v, main_v_row_bytes, error, error_len) ||
        !cache_init(&state->index_k, index_k_row_bytes, error, error_len)) {
        q38_qsa_state_destroy(state);
        return false;
    }
    return true;
}

bool q38_qsa_state_append(q38_qsa_state *state, const void *main_k,
                          const void *main_v, const void *index_k,
                          size_t row_count, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!state || !main_k || !main_v || !index_k || !row_count)
        return fail(error, error_len, "invalid QSA state append arguments");
    if (row_count > SIZE_MAX / state->main_k.row_bytes ||
        row_count > SIZE_MAX / state->main_v.row_bytes ||
        row_count > SIZE_MAX / state->index_k.row_bytes ||
        row_count > UINT64_MAX - state->position ||
        row_count > UINT64_MAX - state->committed_tokens)
        return fail(error, error_len, "QSA state append size overflows");
    if (!cache_reserve(&state->main_k, row_count, error, error_len) ||
        !cache_reserve(&state->main_v, row_count, error, error_len) ||
        !cache_reserve(&state->index_k, row_count, error, error_len))
        return false;
    memcpy(state->main_k.data + state->main_k.count * state->main_k.row_bytes,
           main_k, row_count * state->main_k.row_bytes);
    memcpy(state->main_v.data + state->main_v.count * state->main_v.row_bytes,
           main_v, row_count * state->main_v.row_bytes);
    memcpy(state->index_k.data + state->index_k.count * state->index_k.row_bytes,
           index_k, row_count * state->index_k.row_bytes);
    state->main_k.count += row_count;
    state->main_v.count += row_count;
    state->index_k.count += row_count;
    state->position += row_count;
    state->committed_tokens += row_count;
    return true;
}

void q38_qsa_state_reset(q38_qsa_state *state) {
    if (!state) return;
    state->main_k.count = 0;
    state->main_v.count = 0;
    state->index_k.count = 0;
    state->position = 0;
    state->committed_tokens = 0;
}

void q38_qsa_state_destroy(q38_qsa_state *state) {
    if (!state) return;
    free(state->main_k.data);
    free(state->main_v.data);
    free(state->index_k.data);
    memset(state, 0, sizeof(*state));
}
