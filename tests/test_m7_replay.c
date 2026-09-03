#include "q38_replay.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(bool condition, const char *message) {
    if (!condition) fprintf(stderr, "test_m7_replay: %s\n", message);
    return condition ? 0 : 1;
}

int main(void) {
    q38_forward_state a, b;
    char error[256];
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    q38_weights weights;
    memset(&weights, 0, sizeof(weights));
    (void)weights;
    q38_session_state layout;
    if (check(q38_session_state_init(&layout, 0, error, sizeof(error)),
              "state layout init failed")) return 1;
    memset(&a, 0, sizeof(a));
    a.storage.layout = layout;
    if (check(q38_state_alloc(&layout, &a.storage, error, sizeof(error)),
              "state allocation failed")) return 1;
    a.ple_history_elements = 9u * 4u * Q38_GR_STATE_HIDDEN;
    a.ple_history = calloc(a.ple_history_elements, sizeof(float));
    a.eos_token = 99;
    a.initialized = true;
    a.storage.recurrent_state[0] = 3.25f;
    a.ple_history[7] = -2.0f;
    q38_qsa_state_init(&a.qsa[3], 8, 8, 4, error, sizeof(error));
    float k[8] = {1, 2}, v[8] = {3, 4}, i[4] = {5, 6};
    q38_qsa_state_append(&a.qsa[3], k, v, i, 1, error, sizeof(error));
    if (check(q38_replay_snapshot_save("tests/m7_replay.snapshot", &a,
                                       error, sizeof(error)),
              error)) return 1;
    memset(&b, 0, sizeof(b));
    b.storage.layout = layout;
    if (!q38_state_alloc(&layout, &b.storage, error, sizeof(error))) return 1;
    b.ple_history_elements = a.ple_history_elements;
    b.ple_history = calloc(b.ple_history_elements, sizeof(float));
    b.initialized = true;
    q38_qsa_state_init(&b.qsa[3], 8, 8, 4, error, sizeof(error));
    if (check(q38_replay_snapshot_load("tests/m7_replay.snapshot", &b,
                                       error, sizeof(error)),
              error)) return 1;
    int result = check(a.storage.recurrent_state[0] == b.storage.recurrent_state[0] &&
                       b.ple_history[7] == -2.0f &&
                       b.qsa[3].main_k.count == 1 &&
                       ((const float *)b.qsa[3].main_k.data)[0] == 1.0f,
                       "snapshot round trip mismatch");
    q38_replay_trace trace;
    float boundary[2] = {4.0f, -1.5f};
    q38_forward_stage_usage usage = {"embedding", 2, 3, 4, 5, 6.25};
    result |= check(q38_replay_trace_open(&trace, "tests/m7_replay.trace",
                                          Q38_REPLAY_RECORD, error,
                                          sizeof(error)),
                    "trace record open failed");
    result |= check(q38_replay_boundary_trace(2, "post_norm", boundary, 1, 2,
                                              &trace, error, sizeof(error)) &&
                    q38_replay_stage_trace(&usage, &trace, error, sizeof(error)) &&
                    q38_replay_trace_close(&trace, error, sizeof(error)),
                    "trace record failed");
    result |= check(q38_replay_trace_open(&trace, "tests/m7_replay.trace",
                                          Q38_REPLAY_VERIFY, error,
                                          sizeof(error)) &&
                    q38_replay_boundary_trace(2, "post_norm", boundary, 1, 2,
                                              &trace, error, sizeof(error)) &&
                    q38_replay_stage_trace(&usage, &trace, error, sizeof(error)) &&
                    q38_replay_trace_close(&trace, error, sizeof(error)),
                    "trace verify failed");
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        q38_qsa_state_destroy(&a.qsa[layer]);
        q38_qsa_state_destroy(&b.qsa[layer]);
    }
    q38_state_free(&a.storage);
    q38_state_free(&b.storage);
    free(a.ple_history);
    free(b.ple_history);
    remove("tests/m7_replay.snapshot");
    remove("tests/m7_replay.trace");
    return result;
}
