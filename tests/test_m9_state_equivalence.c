#include "q38_decode.h"
#include "q38_gguf.h"
#include "q38_replay.h"
#include "q38_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    q38_decode_step *steps;
    size_t count;
    size_t capacity;
} trace_capture;

static bool capture_only(const q38_decode_step *step, void *opaque,
                         char *error, size_t error_len) {
    trace_capture *capture = (trace_capture *)opaque;
    if (!capture || !step || capture->count == capture->capacity) {
        if (error && error_len)
            snprintf(error, error_len, "M9 trace fixture capacity exceeded");
        return false;
    }
    capture->steps[capture->count++] = *step;
    return true;
}

static bool same_step(const q38_decode_step *a, const q38_decode_step *b) {
    return memcmp(a, b, sizeof(*a)) == 0;
}

static int run_fixture(const q38_gguf *model, const q38_weights *weights,
                       size_t generated_count, const char *snapshot_path) {
    char error[256];
    q38_forward_state vanilla, replay;
    memset(&vanilla, 0, sizeof(vanilla));
    memset(&replay, 0, sizeof(replay));
    if (!q38_forward_state_init(&vanilla, weights, 248044, error,
                                sizeof(error)) ||
        !q38_forward_state_init(&replay, weights, 248044, error,
                                sizeof(error))) {
        fprintf(stderr, "M9 state-equivalence state init: %s\n", error);
        q38_forward_state_destroy(&vanilla);
        q38_forward_state_destroy(&replay);
        return 1;
    }

    const uint32_t prompt[] = {9419};
    uint32_t *generated = calloc(generated_count, sizeof(*generated));
    float *vanilla_logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*vanilla_logits));
    float *replay_logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*replay_logits));
    q38_decode_step *vanilla_steps =
        calloc(generated_count + 1, sizeof(*vanilla_steps));
    q38_decode_step *replay_steps =
        calloc(generated_count, sizeof(*replay_steps));
    trace_capture vanilla_trace = {
        .steps = vanilla_steps, .capacity = generated_count + 1
    };
    trace_capture replay_trace = {
        .steps = replay_steps, .capacity = generated_count
    };
    q38_forward_diagnostics diagnostics = {0};
    bool ok = generated && vanilla_logits && replay_logits &&
              vanilla_steps && replay_steps &&
              q38_decode_stream(model, weights, &vanilla, prompt, 1,
                                generated, generated_count, vanilla_logits,
                                Q38_DECODE_VOCAB_SIZE, &diagnostics,
                                capture_only, &vanilla_trace, error,
                                sizeof(error));
    if (!ok) {
        fprintf(stderr, "M9 state-equivalence vanilla[%zu]: %s\n",
                generated_count, error);
        goto done;
    }
    /*
     * Re-run the prompt alone in a fresh state so the saved checkpoint is
     * exactly the pre-verify semantic state.  This mirrors MTP rollback:
     * restore, then feed only the accepted prefix through canonical decode.
     */
    q38_forward_state checkpoint;
    memset(&checkpoint, 0, sizeof(checkpoint));
    if (!q38_forward_state_init(&checkpoint, weights, 248044, error,
                                sizeof(error)) ||
        !q38_decode_stream(model, weights, &checkpoint, prompt, 1, NULL, 0,
                           replay_logits, Q38_DECODE_VOCAB_SIZE, &diagnostics,
                           NULL, NULL, error, sizeof(error)) ||
        !q38_replay_snapshot_save(snapshot_path, &checkpoint, error,
                                  sizeof(error)) ||
        !q38_replay_snapshot_load(snapshot_path, &replay, error,
                                  sizeof(error))) {
        fprintf(stderr, "M9 state-equivalence checkpoint[%zu]: %s\n",
                generated_count, error);
        q38_forward_state_destroy(&checkpoint);
        goto done;
    }
    q38_forward_state_destroy(&checkpoint);

    const size_t accepted_prefix = generated_count - 1;
    ok = q38_decode_stream(model, weights, &replay, generated,
                           accepted_prefix, NULL, 0, replay_logits,
                           Q38_DECODE_VOCAB_SIZE, &diagnostics, capture_only,
                           &replay_trace, error, sizeof(error));
    if (!ok || replay_trace.count != accepted_prefix ||
        vanilla_trace.count != generated_count + 1) {
        fprintf(stderr, "M9 state-equivalence replay[%zu]: %s\n",
                generated_count, error);
        ok = false;
        goto done;
    }
    for (size_t i = 0; i < accepted_prefix; ++i) {
        const q38_decode_step *vanilla_step = &vanilla_steps[2 + i];
        if (!same_step(vanilla_step, &replay_steps[i]) ||
            vanilla_step->next_token != generated[i + 1]) {
            fprintf(stderr,
                    "M9 state-equivalence divergence at token %zu/%zu\n",
                    i + 1, generated_count);
            ok = false;
            goto done;
        }
    }
    printf("test_m9_state_equivalence: %zu-token canonical replay matched "
           "tokens, GDN/conv, QSA, PLE, counters, and logits\n",
           generated_count);

done:
    remove(snapshot_path);
    free(generated);
    free(vanilla_logits);
    free(replay_logits);
    free(vanilla_steps);
    free(replay_steps);
    q38_forward_state_destroy(&vanilla);
    q38_forward_state_destroy(&replay);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "M9 state-equivalence open: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
        fprintf(stderr, "M9 state-equivalence bind: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    int result = run_fixture(model, &weights, 32,
                             "artifacts/m9/state_equivalence_32.snapshot");
    if (!result)
        result = run_fixture(model, &weights, 128,
                             "artifacts/m9/state_equivalence_128.snapshot");
    q38_gguf_close(model);
    return result;
}
