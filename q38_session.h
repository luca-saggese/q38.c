#ifndef Q38_SESSION_H
#define Q38_SESSION_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t prev_token_1;
    uint32_t prev_token_2;
    bool have_prev_1;
    bool have_prev_2;
} q38_ngram_history;

void q38_ngram_history_reset(q38_ngram_history *history);
void q38_ngram_history_append(q38_ngram_history *history,
                              uint32_t token, uint32_t eos_token);
void q38_ngram_history_context(const q38_ngram_history *history,
                               uint32_t current_token, uint32_t eos_token,
                               uint32_t context[3]);

#ifdef __cplusplus
}
#endif

#endif
