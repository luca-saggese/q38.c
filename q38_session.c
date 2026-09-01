#include "q38_session.h"

void q38_ngram_history_reset(q38_ngram_history *history) {
    if (!history) return;
    history->prev_token_1 = 0;
    history->prev_token_2 = 0;
    history->have_prev_1 = false;
    history->have_prev_2 = false;
}

void q38_ngram_history_append(q38_ngram_history *history,
                              uint32_t token, uint32_t eos_token) {
    if (!history) return;
    if (token == eos_token) {
        history->prev_token_1 = eos_token;
        history->prev_token_2 = eos_token;
        history->have_prev_1 = true;
        history->have_prev_2 = true;
        return;
    }
    history->prev_token_2 = history->prev_token_1;
    history->have_prev_2 = history->have_prev_1;
    history->prev_token_1 = token;
    history->have_prev_1 = true;
}

void q38_ngram_history_context(const q38_ngram_history *history,
                               uint32_t current_token, uint32_t eos_token,
                               uint32_t context[3]) {
    context[0] = current_token;
    if (!history || !history->have_prev_1 || history->prev_token_1 == eos_token) {
        context[1] = eos_token;
        context[2] = eos_token;
        return;
    }
    context[1] = history->prev_token_1;
    context[2] = history->have_prev_2 ? history->prev_token_2 : eos_token;
    if (context[2] == eos_token) context[2] = eos_token;
}
