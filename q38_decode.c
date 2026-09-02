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

static uint64_t hash_qsa_pending(const q38_qsa_cache *cache,
                                 uint32_t pending_count) {
    if (!cache || !pending_count || pending_count > cache->count)
        return hash_bytes(NULL, 0);
    const size_t start = (cache->count - pending_count) * cache->row_bytes;
    return hash_bytes(cache->data + start, pending_count * cache->row_bytes);
}

static bool finite_floats(const float *values, size_t count) {
    if (!values && count) return false;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return false;
    return true;
}

static uint64_t hash_logits(const float *logits) {
    return hash_bytes(logits, Q38_DECODE_VOCAB_SIZE * sizeof(*logits));
}

static void snapshot_top10(const float *logits, q38_decode_step *snapshot) {
    for (size_t rank = 0; rank < 10; ++rank) {
        size_t top = Q38_DECODE_VOCAB_SIZE;
        float best = -INFINITY;
        for (size_t j = 0; j < Q38_DECODE_VOCAB_SIZE; ++j) {
            bool used = false;
            for (size_t previous = 0; previous < rank; ++previous)
                used |= snapshot->top_ids[previous] == j;
            if (!used && (top == Q38_DECODE_VOCAB_SIZE || logits[j] > best)) {
                top = j;
                best = logits[j];
            }
        }
        snapshot->top_ids[rank] = (uint32_t)top;
        snapshot->top_values[rank] = logits[top];
    }
}

static void snapshot_logits_metrics(const float *logits,
                                    q38_decode_step *snapshot) {
    double sum = 0.0;
    double squares = 0.0;
    float minimum = INFINITY;
    float maximum = -INFINITY;
    float maximum_abs = 0.0f;
    for (size_t i = 0; i < Q38_DECODE_VOCAB_SIZE; ++i) {
        minimum = fminf(minimum, logits[i]);
        maximum = fmaxf(maximum, logits[i]);
        maximum_abs = fmaxf(maximum_abs, fabsf(logits[i]));
        sum += logits[i];
        squares += (double)logits[i] * logits[i];
    }
    snapshot->logits_min = minimum;
    snapshot->logits_max = maximum;
    snapshot->logits_mean = (float)(sum / Q38_DECODE_VOCAB_SIZE);
    snapshot->logits_rms = (float)sqrt(
        squares / Q38_DECODE_VOCAB_SIZE);
    snapshot->logits_max_abs = maximum_abs;
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
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        const int slot = storage->layout.layer_to_gdn_slot[layer];
        if (slot < 0) continue;
        snapshot->gdn_layer_hash[layer] = hash_bytes(
            storage->recurrent_state +
                (size_t)slot * storage->layout.recurrent.elements_per_slot,
            (size_t)storage->layout.recurrent.bytes_per_slot);
        snapshot->conv_layer_hash[layer] = hash_bytes(
            storage->conv_history +
                (size_t)slot * storage->layout.conv_history.elements_per_slot,
            (size_t)storage->layout.conv_history.bytes_per_slot);
    }
    snapshot->ple_prev_token_1 = state->token_history.prev_token_1;
    snapshot->ple_prev_token_2 = state->token_history.prev_token_2;
    snapshot->ple_have_prev_1 = state->token_history.have_prev_1;
    snapshot->ple_have_prev_2 = state->token_history.have_prev_2;
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
        out->pending_main_k_hash = hash_qsa_pending(
            &qsa->main_k, qsa->pending_count);
        out->pending_main_v_hash = hash_qsa_pending(
            &qsa->main_v, qsa->pending_count);
        out->pending_index_k_hash = hash_qsa_pending(
            &qsa->index_k, qsa->pending_count);
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

static bool q38_decode_backend(const q38_gguf *model,
                const q38_weights *weights,
                q38_forward_state *state, uint32_t token, float *logits,
                size_t logits_stride, uint32_t *next_token,
                q38_forward_diagnostics *diagnostics, char *error,
                size_t error_len, q38_forward_matvec_backend backend,
                void *backend_user) {
    if (error && error_len) error[0] = '\0';
    if (!next_token)
        return fail(error, error_len, "decode output token is null");
    const bool ok = backend
        ? q38_forward_full_with_backend(model, weights, state, &token, 1,
                                        logits, logits_stride, diagnostics,
                                        backend, backend_user, error, error_len)
        : q38_forward_full(model, weights, state, &token, 1, logits,
                           logits_stride, diagnostics, error, error_len);
    if (!ok)
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

bool q38_decode_stream_with_backend(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_forward_matvec_backend backend, void *backend_user,
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
        if (!q38_decode_backend(model, weights, state, prompt[i], logits,
                                logits_stride, &next, diagnostics, error,
                                error_len, backend, backend_user))
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
            snapshot.argmax = next;
            snapshot.argmax_value = logits[next];
            snapshot_logits_metrics(logits, &snapshot);
            snapshot_top10(logits, &snapshot);
            snapshot.logits_hash = hash_logits(logits);
            snapshot.logits_finite =
                finite_floats(logits, Q38_DECODE_VOCAB_SIZE);
            if (!snapshot.finite)
                return fail(error, error_len,
                            "decode state contains a non-finite value");
            if (!snapshot.logits_finite)
                return fail(error, error_len,
                            "decode logits contain a non-finite value");
            if (!trace(&snapshot, trace_user, error, error_len))
                return false;
        }
    }
    for (size_t i = 0; i < generated_count; ++i) {
        const uint32_t input = current;
        uint32_t next = 0;
        if (!q38_decode_backend(model, weights, state, input, logits,
                                logits_stride, &next, diagnostics, error,
                                error_len, backend, backend_user))
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
            snapshot.argmax = next;
            snapshot.argmax_value = logits[next];
            snapshot_logits_metrics(logits, &snapshot);
            snapshot_top10(logits, &snapshot);
            snapshot.logits_hash = hash_logits(logits);
            snapshot.logits_finite =
                finite_floats(logits, Q38_DECODE_VOCAB_SIZE);
            if (!snapshot.finite)
                return fail(error, error_len,
                            "decode state contains a non-finite value");
            if (!snapshot.logits_finite)
                return fail(error, error_len,
                            "decode logits contain a non-finite value");
            if (!trace(&snapshot, trace_user, error, error_len))
                return false;
        }
    }
    return true;
}

bool q38_decode(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, uint32_t token, float *logits,
    size_t logits_stride, uint32_t *next_token,
    q38_forward_diagnostics *diagnostics, char *error, size_t error_len) {
    return q38_decode_backend(model, weights, state, token, logits,
                              logits_stride, next_token, diagnostics, error,
                              error_len, NULL, NULL);
}

bool q38_decode_stream(
    const q38_gguf *model, const q38_weights *weights,
    q38_forward_state *state, const uint32_t *prompt, size_t prompt_count,
    uint32_t *generated, size_t generated_count, float *logits,
    size_t logits_stride, q38_forward_diagnostics *diagnostics,
    q38_decode_trace trace, void *trace_user, char *error, size_t error_len) {
    return q38_decode_stream_with_backend(
        model, weights, state, prompt, prompt_count, generated,
        generated_count, logits, logits_stride, diagnostics, NULL, NULL, trace,
        trace_user, error, error_len);
}
