#include "q38_qsa.h"
#include "q38_session.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char error[128];
    q38_qsa_state source, copy;
    unsigned char k[8] = {1}, v[8] = {2}, i[4] = {3};
    if (!q38_qsa_state_init(&source, sizeof(k), sizeof(v), sizeof(i),
                            error, sizeof(error)) ||
        !q38_qsa_state_append(&source, k, v, i, 1, error, sizeof(error)) ||
        !q38_qsa_state_clone(&source, &copy, error, sizeof(error)) ||
        copy.position != source.position ||
        memcmp(copy.main_k.data, source.main_k.data, sizeof(k)) != 0) return 1;
    q38_qsa_state_reset(&source);
    if (source.position != 0 || copy.position != 1) return 1;
    q38_qsa_state_destroy(&source);
    q38_qsa_state_destroy(&copy);

    q38_ngram_history history;
    q38_ngram_history_reset(&history);
    q38_ngram_history_append(&history, 11, 99);
    q38_ngram_history_append(&history, 99, 99);
    uint32_t context[3];
    q38_ngram_history_context(&history, 12, 99, context);
    if (context[0] != 12 || context[1] != 99 || context[2] != 99) return 1;
    puts("test_m6_session_contamination: reset/clone domains remain isolated");
    return 0;
}
