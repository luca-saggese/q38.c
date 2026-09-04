#ifndef Q38_REPLAY_H
#define Q38_REPLAY_H

#include "q38_forward.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Versioned, deterministic host-state checkpointing for replay probes. */
bool q38_replay_snapshot_save(const char *path,
                              const q38_forward_state *state,
                              char *error, size_t error_len);
bool q38_replay_snapshot_load(const char *path, q38_forward_state *state,
                              char *error, size_t error_len);

typedef bool (*q38_replay_step_fn)(q38_forward_state *state, uint32_t token,
                                   void *user, char *error, size_t error_len);

/* Restore a checkpoint and apply the caller's deterministic token step for
 * every token after the checkpoint.  The callback owns model execution. */
bool q38_replay_restore_and_replay(const char *path, q38_forward_state *state,
                                   const uint32_t *tokens, size_t token_count,
                                   q38_replay_step_fn step, void *user,
                                   char *error, size_t error_len);

typedef enum {
    Q38_REPLAY_RECORD = 0,
    Q38_REPLAY_VERIFY = 1,
} q38_replay_mode;

typedef struct {
    FILE *file;
    q38_replay_mode mode;
    uint64_t event_count;
} q38_replay_trace;

bool q38_replay_trace_open(q38_replay_trace *trace, const char *path,
                           q38_replay_mode mode, char *error,
                           size_t error_len);
bool q38_replay_trace_close(q38_replay_trace *trace, char *error,
                            size_t error_len);

bool q38_replay_boundary_trace(uint32_t layer, const char *boundary,
                               const float *values, size_t token_count,
                               size_t width, void *user, char *error,
                               size_t error_len);
bool q38_replay_stage_trace(const q38_forward_stage_usage *usage, void *user,
                            char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif /* Q38_REPLAY_H */
