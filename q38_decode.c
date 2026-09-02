#include "q38_decode.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static bool finite_floats(const float *values, size_t count) {
    if (!values && count) return false;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return false;
    return true;
}

static bool snapshot_state(const q38_forward_state *state,
                           q38_decode_step *snapshot) {
    if (!state || !snapshot) return false;
    memset(snapshot, 0, sizeof(*snapshot));
    const q38_state_storage *storage = &state->storage;
    snapshot->gdn_state_hash = hash_bytes(
        storage->recurrent_state, (size_t)storage->layout.recurrent.bytes);
    snapshot->conv_history_hash = hash_bytes(
        storage->conv_history, (size_t)storage->layout.conv_history.bytes);
    snapshot->ple_history_hash = hash_bytes(
        state->ple_history,
        state->ple_history_elements * sizeof(*state->ple_history));
    snapshot->finite =
        finite_floats(storage->recurrent_state,
                      (size_t)storage->layout.recurrent.elements) &&
        finite_floats(storage->conv_history,
                      (size_t)storage->layout.conv_history.elements) &&
        finite_floats(state->ple_history, state->ple_history_elements);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        const q38_qsa_state *qsa = &state->qsa[layer];
        q38_decode_qsa_snapshot *out = &snapshot->qsa[layer];
        out->position = qsa->position;
        out->committed_tokens = qsa->committed_tokens;
        out->pending_count = qsa->pending_count;
        out->pending_position = qsa->pending_position;
        out->main_k_count = qsa->main_k.count;
        out->main_v_count = qsa->main_v.count;
        out->index_k_count = qsa->index_k.count;
        out->main_k_hash = hash_bytes(
            qsa->main_k.data, qsa->main_k.count * qsa->main_k.row_bytes);
        out->main_v_hash = hash_bytes(
            qsa->main_v.data, qsa->main_v.count * qsa->main_v.row_bytes);
        out->index_k_hash = hash_bytes(
            qsa->index_k.data, qsa->index_k.count * qsa->index_k.row_bytes);
        if (!finite_floats((const float *)qsa->main_k.data,
                           qsa->main_k.count * qsa->main_k.row_bytes /
                               sizeof(float)) ||
            !finite_floats((const float *)qsa->main_v.data,
                           qsa->main_v.count * qsa->main_v.row_bytes /
                               sizeof(float)) ||
            !finite_floats((const float *)qsa->index_k.data,
                           qsa->index_k.count * qsa->index_k.row_bytes /
                               sizeof(float)))
            snapshot->finite = false;
    }
    return true;
}

bool q38_decode(const q38_gguf *model, const q38_weights *weights,
                q38_forward_state *state, uint32_t token, float *logits,
                size_t logits_stride, uint32_t *next_token,
                q38_forward_diagnostics *diagnostics, char *error,
                size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!next_token)
        return fail(error, error_len, "decode output token is null");
    if (!q38_forward_full(model, weights, state, &token, 1, logits,
                          logits_stride, diagnostics, error, error_len))
        return false;
    size_t best = 0;
    float best_value = logits[0];
    if (!isfinite(best_value))
        return fail(error, error_len, "decode logits contain a non-finite value");
    for (size_t i = 1; i < Q38_DECODE_VOCAB_SIZE; ++i) {
        if (!isfinite(logits[i]))
            return fail(error, error_len, "decode logits contain a non-finite value");
        if (logits[i] > best_value) {
            best = i;
            best_value = logits[i];
        }
    }
    *next_token = (uint32_t)best;
    return true;
}

bool q38_decode_stream(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_decode_trace trace, void *trace_user, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!prompt_count || !prompt ||
        (generated_count && !generated))
        return fail(error, error_len, "invalid decode stream arguments");
    if (!logits || logits_stride < Q38_DECODE_VOCAB_SIZE)
        return fail(error, error_len, "decode logits buffer is invalid");
    uint32_t current = 0;
    size_t step_index = 0;
    for (size_t i = 0; i < prompt_count; ++i) {
        uint32_t next = 0;
        if (!q38_decode(model, weights, state, prompt[i], logits,
                        logits_stride, &next, diagnostics, error, error_len))
            return false;
        current = next;
        if (trace) {
            q38_decode_step snapshot;
            if (!snapshot_state(state, &snapshot))
                return fail(error, error_len, "decode state snapshot failed");
            snapshot.step = step_index++;
            snapshot.generated = false;
            snapshot.input_token = prompt[i];
            snapshot.next_token = next;
            snapshot.committed_tokens = snapshot.step + 1;
            if (!snapshot.finite)
                return fail(error, error_len,
                            "decode state contains a non-finite value");
            if (!trace(&snapshot, trace_user, error, error_len))
                return false;
        }
    }
    for (size_t i = 0; i < generated_count; ++i) {
        const uint32_t input = current;
        uint32_t next = 0;
        if (!q38_decode(model, weights, state, input, logits, logits_stride,
                        &next, diagnostics, error, error_len))
            return false;
        generated[i] = next;
        current = next;
        if (trace) {
            q38_decode_step snapshot;
            if (!snapshot_state(state, &snapshot))
                return fail(error, error_len, "decode state snapshot failed");
            snapshot.step = step_index++;
            snapshot.generated = true;
            snapshot.input_token = input;
            snapshot.next_token = next;
            snapshot.committed_tokens = snapshot.step + 1;
            if (!snapshot.finite)
                return fail(error, error_len,
                            "decode state contains a non-finite value");
            if (!trace(&snapshot, trace_user, error, error_len))
                return false;
        }
    }
    return true;
}
