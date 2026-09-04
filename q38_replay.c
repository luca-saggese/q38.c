#include "q38_replay.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q38_REPLAY_MAGIC "Q38RPLY\0"
#define Q38_REPLAY_VERSION 1u
#define Q38_REPLAY_BOUNDARY 1u
#define Q38_REPLAY_STAGE 2u

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static bool io_write(FILE *file, const void *data, size_t size,
                     char *error, size_t error_len) {
    return !size || fwrite(data, 1, size, file) == size ||
           fail(error, error_len, "replay snapshot write failed");
}

static bool io_read(FILE *file, void *data, size_t size,
                    char *error, size_t error_len) {
    return !size || fread(data, 1, size, file) == size ||
           fail(error, error_len, "replay snapshot read failed");
}

static bool size_bytes(size_t count, size_t width, size_t *out) {
    if (width && count > SIZE_MAX / width) return false;
    *out = count * width;
    return true;
}

static bool write_cache(FILE *file, const q38_qsa_cache *cache,
                        char *error, size_t error_len) {
    uint64_t count = cache->count;
    uint64_t row_bytes = cache->row_bytes;
    if (!io_write(file, &row_bytes, sizeof(row_bytes), error, error_len) ||
        !io_write(file, &count, sizeof(count), error, error_len))
        return false;
    size_t bytes = 0;
    if (!size_bytes(cache->count, cache->row_bytes, &bytes))
        return fail(error, error_len, "replay cache size overflow");
    return io_write(file, cache->data, bytes, error, error_len);
}

static bool read_cache(FILE *file, q38_qsa_cache *cache, char *error,
                       size_t error_len) {
    uint64_t row_bytes = 0, count = 0;
    if (!io_read(file, &row_bytes, sizeof(row_bytes), error, error_len) ||
        !io_read(file, &count, sizeof(count), error, error_len) ||
        row_bytes != cache->row_bytes || count > SIZE_MAX)
        return fail(error, error_len, "replay QSA cache descriptor mismatch");
    size_t bytes = 0;
    if (!size_bytes((size_t)count, cache->row_bytes, &bytes))
        return fail(error, error_len, "replay cache size overflow");
    uint8_t *data = bytes ? (uint8_t *)malloc(bytes) : NULL;
    if (bytes && !data) return fail(error, error_len, "replay cache allocation failed");
    if (!io_read(file, data, bytes, error, error_len)) {
        free(data);
        return false;
    }
    free(cache->data);
    cache->data = data;
    cache->count = (size_t)count;
    cache->capacity = (size_t)count;
    return true;
}

bool q38_replay_snapshot_save(const char *path,
                              const q38_forward_state *state,
                              char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!path || !state || !state->initialized ||
        !q38_session_state_validate(&state->storage.layout, error, error_len))
        return fail(error, error_len, "invalid replay snapshot state");
    FILE *file = fopen(path, "wb");
    if (!file) return fail(error, error_len, "replay snapshot open failed");
    bool ok = io_write(file, Q38_REPLAY_MAGIC, 8, error, error_len);
    uint32_t version = Q38_REPLAY_VERSION;
    ok = ok && io_write(file, &version, sizeof(version), error, error_len);
    ok = ok && io_write(file, &state->storage.layout,
                        sizeof(state->storage.layout), error, error_len);
    ok = ok && io_write(file, &state->eos_token, sizeof(state->eos_token),
                        error, error_len);
    ok = ok && io_write(file, &state->token_history,
                        sizeof(state->token_history), error, error_len);
    ok = ok && io_write(file, &state->ple_history_elements,
                        sizeof(state->ple_history_elements), error, error_len);
    if (ok) {
        ok = io_write(file, state->storage.recurrent_state,
                      (size_t)state->storage.layout.recurrent.bytes,
                      error, error_len) &&
             io_write(file, state->storage.conv_history,
                      (size_t)state->storage.layout.conv_history.bytes,
                      error, error_len) &&
             io_write(file, state->storage.gr_workspace,
                      (size_t)state->storage.layout.gr_workspace.bytes,
                      error, error_len) &&
             io_write(file, state->storage.workspace,
                      (size_t)state->storage.layout.memory.workspace_bytes,
                      error, error_len) &&
             io_write(file, state->ple_history,
                      state->ple_history_elements * sizeof(*state->ple_history),
                      error, error_len);
    }
    for (size_t i = 0; ok && i < Q38_MODEL_LAYERS; ++i)
        ok = write_cache(file, &state->qsa[i].main_k, error, error_len) &&
             write_cache(file, &state->qsa[i].main_v, error, error_len) &&
             write_cache(file, &state->qsa[i].index_k, error, error_len) &&
             io_write(file, &state->qsa[i].position, sizeof(uint64_t), error, error_len) &&
             io_write(file, &state->qsa[i].committed_tokens, sizeof(uint64_t), error, error_len) &&
             io_write(file, &state->qsa[i].pending_count, sizeof(uint32_t), error, error_len) &&
             io_write(file, &state->qsa[i].pending_position, sizeof(uint64_t), error, error_len);
    if (fclose(file) != 0 && ok) ok = fail(error, error_len, "replay snapshot close failed");
    return ok;
}

bool q38_replay_snapshot_load(const char *path, q38_forward_state *state,
                              char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!path || !state || !state->initialized)
        return fail(error, error_len, "invalid replay snapshot load state");
    FILE *file = fopen(path, "rb");
    if (!file) return fail(error, error_len, "replay snapshot open failed");
    char magic[8] = {0};
    uint32_t version = 0;
    q38_session_state layout;
    size_t ple_elements = 0;
    bool ok = io_read(file, magic, sizeof(magic), error, error_len) &&
              io_read(file, &version, sizeof(version), error, error_len) &&
              io_read(file, &layout, sizeof(layout), error, error_len);
    if (ok && (memcmp(magic, Q38_REPLAY_MAGIC, 8) != 0 ||
               version != Q38_REPLAY_VERSION ||
               memcmp(&layout, &state->storage.layout, sizeof(layout)) != 0))
        ok = fail(error, error_len, "replay snapshot layout or version mismatch");
    uint32_t eos = 0;
    q38_ngram_history history;
    if (ok) ok = io_read(file, &eos, sizeof(eos), error, error_len) &&
                 io_read(file, &history, sizeof(history), error, error_len) &&
                 io_read(file, &ple_elements, sizeof(ple_elements), error, error_len);
    if (ok && ple_elements != state->ple_history_elements)
        ok = fail(error, error_len, "replay PLE descriptor mismatch");
    if (ok) {
        ok = io_read(file, state->storage.recurrent_state,
                     (size_t)layout.recurrent.bytes, error, error_len) &&
             io_read(file, state->storage.conv_history,
                     (size_t)layout.conv_history.bytes, error, error_len) &&
             io_read(file, state->storage.gr_workspace,
                     (size_t)layout.gr_workspace.bytes, error, error_len) &&
             io_read(file, state->storage.workspace,
                     (size_t)layout.memory.workspace_bytes, error, error_len) &&
             io_read(file, state->ple_history,
                     ple_elements * sizeof(*state->ple_history), error, error_len);
    }
    for (size_t i = 0; ok && i < Q38_MODEL_LAYERS; ++i) {
        ok = read_cache(file, &state->qsa[i].main_k, error, error_len) &&
             read_cache(file, &state->qsa[i].main_v, error, error_len) &&
             read_cache(file, &state->qsa[i].index_k, error, error_len) &&
             io_read(file, &state->qsa[i].position, sizeof(uint64_t), error, error_len) &&
             io_read(file, &state->qsa[i].committed_tokens, sizeof(uint64_t), error, error_len) &&
             io_read(file, &state->qsa[i].pending_count, sizeof(uint32_t), error, error_len) &&
             io_read(file, &state->qsa[i].pending_position, sizeof(uint64_t), error, error_len);
    }
    if (ok) {
        state->eos_token = eos;
        state->token_history = history;
    }
    if (fclose(file) != 0 && ok) ok = fail(error, error_len, "replay snapshot close failed");
    return ok;
}

bool q38_replay_restore_and_replay(const char *path, q38_forward_state *state,
                                   const uint32_t *tokens, size_t token_count,
                                   q38_replay_step_fn step, void *user,
                                   char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!q38_replay_snapshot_load(path, state, error, error_len))
        return false;
    if (!step && token_count)
        return fail(error, error_len, "replay step callback is required");
    for (size_t i = 0; i < token_count; ++i)
        if (!step(state, tokens[i], user, error, error_len))
            return false;
    return true;
}

static bool trace_event(q38_replay_trace *trace, uint32_t kind,
                        const void *payload, size_t payload_size,
                        char *error, size_t error_len) {
    uint64_t size = payload_size;
    if (trace->mode == Q38_REPLAY_RECORD)
        return io_write(trace->file, &kind, sizeof(kind), error, error_len) &&
               io_write(trace->file, &size, sizeof(size), error, error_len) &&
               io_write(trace->file, payload, payload_size, error, error_len);
    uint32_t expected_kind = 0;
    uint64_t expected_size = 0;
    unsigned char *expected = NULL;
    bool ok = io_read(trace->file, &expected_kind, sizeof(expected_kind), error, error_len) &&
              io_read(trace->file, &expected_size, sizeof(expected_size), error, error_len);
    if (ok && (expected_kind != kind || expected_size != payload_size))
        ok = fail(error, error_len, "replay trace event mismatch");
    if (ok && payload_size) {
        expected = (unsigned char *)malloc(payload_size);
        ok = expected && io_read(trace->file, expected, payload_size, error, error_len);
        if (ok && memcmp(expected, payload, payload_size) != 0)
            ok = fail(error, error_len, "replay trace payload mismatch");
    }
    free(expected);
    return ok;
}

bool q38_replay_trace_open(q38_replay_trace *trace, const char *path,
                           q38_replay_mode mode, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!trace || !path) return fail(error, error_len, "invalid replay trace arguments");
    memset(trace, 0, sizeof(*trace));
    trace->mode = mode;
    trace->file = fopen(path, mode == Q38_REPLAY_RECORD ? "wb" : "rb");
    return trace->file != NULL || fail(error, error_len, "replay trace open failed");
}

bool q38_replay_trace_close(q38_replay_trace *trace, char *error,
                            size_t error_len) {
    if (!trace || !trace->file) return fail(error, error_len, "replay trace is not open");
    bool ok = true;
    if (trace->mode == Q38_REPLAY_VERIFY) {
        if (fgetc(trace->file) != EOF || ferror(trace->file))
            ok = false;
    }
    if (fclose(trace->file) != 0) ok = false;
    trace->file = NULL;
    return ok || fail(error, error_len, "replay trace close failed");
}

bool q38_replay_boundary_trace(uint32_t layer, const char *boundary,
                               const float *values, size_t token_count,
                               size_t width, void *user, char *error,
                               size_t error_len) {
    q38_replay_trace *trace = (q38_replay_trace *)user;
    if (!trace || !trace->file || (!values && token_count && width))
        return fail(error, error_len, "invalid replay boundary trace");
    size_t bytes = 0, name_len = boundary ? strlen(boundary) : 0;
    if (!size_bytes(token_count, width * sizeof(float), &bytes) ||
        name_len > UINT32_MAX)
        return fail(error, error_len, "replay boundary size overflow");
    struct {
        uint32_t layer;
        uint32_t name_len;
        uint64_t token_count;
        uint64_t width;
    } header = {layer, (uint32_t)name_len, token_count, width};
    size_t total = sizeof(header) + name_len + bytes;
    unsigned char *payload = (unsigned char *)malloc(total);
    if (!payload) return fail(error, error_len, "replay boundary allocation failed");
    memcpy(payload, &header, sizeof(header));
    memcpy(payload + sizeof(header), boundary ? boundary : "", name_len);
    if (bytes) memcpy(payload + sizeof(header) + name_len, values, bytes);
    bool ok = trace_event(trace, Q38_REPLAY_BOUNDARY, payload, total, error, error_len);
    free(payload);
    if (ok) trace->event_count++;
    return ok;
}

bool q38_replay_stage_trace(const q38_forward_stage_usage *usage, void *user,
                            char *error, size_t error_len) {
    q38_replay_trace *trace = (q38_replay_trace *)user;
    if (!trace || !trace->file || !usage || !usage->name)
        return fail(error, error_len, "invalid replay stage trace");
    size_t name_len = strlen(usage->name);
    if (name_len > UINT32_MAX) return fail(error, error_len, "replay stage name overflow");
    size_t total = sizeof(uint32_t) + name_len + sizeof(*usage) - sizeof(usage->name);
    unsigned char *payload = (unsigned char *)malloc(total);
    if (!payload) return fail(error, error_len, "replay stage allocation failed");
    uint32_t length = (uint32_t)name_len;
    memcpy(payload, &length, sizeof(length));
    memcpy(payload + sizeof(length), usage->name, name_len);
    memcpy(payload + sizeof(length) + name_len, &usage->matrix_calls,
           sizeof(*usage) - sizeof(usage->name));
    bool ok = trace_event(trace, Q38_REPLAY_STAGE, payload, total, error, error_len);
    free(payload);
    if (ok) trace->event_count++;
    return ok;
}
