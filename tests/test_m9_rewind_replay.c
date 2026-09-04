#include "../q38_replay.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool step(q38_forward_state *state, uint32_t token, void *user,
                 char *error, size_t error_len) {
    (void)user;
    (void)error;
    (void)error_len;
    q38_ngram_history_append(&state->token_history, token, state->eos_token);
    state->qsa[0].position++;
    state->qsa[0].committed_tokens++;
    return true;
}

int main(void) {
    const char *path = "artifacts/m9/rewind_replay.snapshot";
    q38_forward_state source, restored;
    q38_session_state layout;
    char error[256];
    memset(&source, 0, sizeof(source));
    memset(&restored, 0, sizeof(restored));
    if (!q38_session_state_init(&layout, 0, error, sizeof(error))) return 1;
    source.storage.layout = layout;
    restored.storage.layout = layout;
    if (!q38_state_alloc(&layout, &source.storage, error, sizeof(error)) ||
        !q38_state_alloc(&layout, &restored.storage, error, sizeof(error)))
        return 1;
    source.initialized = true;
    restored.initialized = true;
    source.eos_token = 99;
    q38_ngram_history_append(&source.token_history, 7, source.eos_token);
    source.qsa[0].position = 1;
    source.qsa[0].committed_tokens = 1;
    if (!q38_replay_snapshot_save(path, &source, error, sizeof(error)) ||
        !q38_replay_restore_and_replay(path, &restored,
                                       (const uint32_t[]){8, 9}, 2, step,
                                       NULL, error, sizeof(error)) ||
        restored.qsa[0].position != 3 ||
        restored.qsa[0].committed_tokens != 3 ||
        !restored.token_history.have_prev_1 ||
        restored.token_history.prev_token_1 != 9) {
        fprintf(stderr, "test_m9_rewind_replay: %s\n", error);
        return 1;
    }
    q38_state_free(&source.storage);
    q38_state_free(&restored.storage);
    remove(path);
    puts("test_m9_rewind_replay: restore and deterministic token replay passed");
    return 0;
}
