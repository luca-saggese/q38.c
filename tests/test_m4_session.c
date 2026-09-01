#include "q38_session.h"

#include <stdio.h>

static int expect_context(const q38_ngram_history *history,
                          uint32_t current, uint32_t eos,
                          uint32_t a, uint32_t b, uint32_t c) {
    uint32_t context[3];
    q38_ngram_history_context(history, current, eos, context);
    if (context[0] != a || context[1] != b || context[2] != c) {
        fprintf(stderr, "context mismatch: got [%u,%u,%u], expected [%u,%u,%u]\n",
                context[0], context[1], context[2], a, b, c);
        return 0;
    }
    return 1;
}

int main(void) {
    const uint32_t eos = 99;
    q38_ngram_history history;
    q38_ngram_history_reset(&history);

    if (!expect_context(&history, 10, eos, 10, eos, eos)) return 1;
    q38_ngram_history_append(&history, 10, eos);
    if (!expect_context(&history, 20, eos, 20, 10, eos)) return 1;
    q38_ngram_history_append(&history, 20, eos);
    if (!expect_context(&history, 30, eos, 30, 20, 10)) return 1;

    q38_ngram_history_append(&history, eos, eos);
    if (!expect_context(&history, 40, eos, 40, eos, eos)) return 1;
    q38_ngram_history_append(&history, 50, eos);
    if (!expect_context(&history, 60, eos, 60, 50, eos)) return 1;

    q38_ngram_history_reset(&history);
    if (!expect_context(&history, 70, eos, 70, eos, eos)) return 1;
    puts("test_m4_session: n-gram history boundaries and EOS reset passed");
    return 0;
}
