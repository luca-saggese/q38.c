#include "q38_ple_ref.h"

#include <stdio.h>
#include <string.h>

static void init_config(q38_ple_hash_config *config) {
    memset(config, 0, sizeof(*config));
    config->ngram_size = 3;
    config->heads_per_ngram = 8;
    config->multipliers[0] = UINT64_C(6364136223846793005);
    config->multipliers[1] = UINT64_C(1442695040888963407);
    config->multipliers[2] = UINT64_C(3202034522624059733);
    for (uint32_t h = 0; h < Q38_PLE_MAX_HEADS; ++h) {
        config->head_offsets[h] = h * 1000u;
        config->head_vocab_sizes[h] = 997u + h;
    }
}

static int ids_for(const q38_ple_hash_config *config,
                   const q38_ngram_history *history, uint32_t token,
                   uint32_t eos, uint32_t ids[Q38_PLE_MAX_HEADS]) {
    char error[128];
    if (!q38_ple_ngram_ids_ref(config, history, token, eos, ids,
                               Q38_PLE_MAX_HEADS, error, sizeof(error))) {
        fprintf(stderr, "PLE oracle failed: %s\n", error);
        return 0;
    }
    return 1;
}

int main(void) {
    q38_ple_hash_config config;
    init_config(&config);
    q38_ngram_history history;
    q38_ngram_history_reset(&history);
    uint32_t ids[Q38_PLE_MAX_HEADS];
    uint32_t first[Q38_PLE_MAX_HEADS];
    static const uint32_t expected_initial[Q38_PLE_MAX_HEADS] = {
        969, 1429, 2938, 3375, 4240, 5167, 6072, 7879,
        8438, 9068, 10806, 11528, 12504, 13398, 14426, 15524
    };
    static const uint32_t expected_reset[Q38_PLE_MAX_HEADS] = {
        650, 1895, 2989, 3723, 4361, 5701, 6139, 7679,
        8562, 9284, 10181, 11724, 12755, 13632, 14187, 15500
    };

    if (!ids_for(&config, &history, 10, 99, ids)) return 1;
    if (memcmp(ids, expected_initial, sizeof(ids)) != 0) {
        fprintf(stderr, "initial golden IDs mismatch\n");
        return 1;
    }
    memcpy(first, ids, sizeof(first));
    q38_ngram_history_append(&history, 10, 99);
    if (!ids_for(&config, &history, 20, 99, ids)) return 1;
    q38_ngram_history_append(&history, 20, 99);
    if (!ids_for(&config, &history, 30, 99, ids)) return 1;
    if (memcmp(first, ids, sizeof(first)) == 0) {
        fprintf(stderr, "different contexts produced identical IDs\n");
        return 1;
    }

    q38_ngram_history_append(&history, 99, 99);
    if (!ids_for(&config, &history, 30, 99, ids)) return 1;
    q38_ngram_history_reset(&history);
    if (!ids_for(&config, &history, 30, 99, first) ||
        memcmp(ids, first, sizeof(first)) != 0 ||
        memcmp(ids, expected_reset, sizeof(ids)) != 0) {
        fprintf(stderr, "EOS reset does not match fresh history\n");
        return 1;
    }

    q38_ngram_history_reset(&history);
    uint32_t one_chunk[4][Q38_PLE_MAX_HEADS];
    uint32_t chunked[4][Q38_PLE_MAX_HEADS];
    const uint32_t tokens[] = {1, 2, 3, 4};
    for (size_t i = 0; i < 4; ++i) {
        if (!ids_for(&config, &history, tokens[i], 99, one_chunk[i])) return 1;
        q38_ngram_history_append(&history, tokens[i], 99);
    }
    q38_ngram_history_reset(&history);
    for (size_t i = 0; i < 4; ++i) {
        if (!ids_for(&config, &history, tokens[i], 99, chunked[i])) return 1;
        q38_ngram_history_append(&history, tokens[i], 99);
    }
    if (memcmp(one_chunk, chunked, sizeof(one_chunk)) != 0) {
        fprintf(stderr, "chunked history changed IDs\n");
        return 1;
    }
    puts("test_m4_ple_ref: verified scalar hash/index and history invariance");
    return 0;
}
