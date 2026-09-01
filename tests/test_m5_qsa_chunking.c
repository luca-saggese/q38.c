#include "q38_qsa.h"
#include "q38_qsa_ref.h"
#include "q38_topk_ref.h"

#include <stdio.h>
#include <string.h>

static int run_partition(const float *keys, size_t tokens,
                         const size_t *parts, size_t part_count,
                         float *scores, uint32_t *selected) {
    q38_qsa_state state;
    char error[128];
    if (!q38_qsa_state_init(&state, 4 * sizeof(float), 4 * sizeof(float),
                            4 * sizeof(float), error, sizeof(error)))
        return 0;
    size_t at = 0;
    for (size_t p = 0; p < part_count; ++p) {
        if (!parts[p] || parts[p] > tokens - at ||
            !q38_qsa_state_append(&state, keys + at * 4, keys + at * 4,
                                  keys + at * 4, parts[p], error,
                                  sizeof(error))) {
            q38_qsa_state_destroy(&state);
            return 0;
        }
        at += parts[p];
    }
    if (at != tokens ||
        !q38_qsa_index_scores_ref((const float *)state.index_k.data, tokens,
                                  keys, 1, 1, 4, 4, scores, error,
                                  sizeof(error)) ||
        !q38_topk_select_ref(scores, (tokens + 3) / 4, 2, selected, error,
                             sizeof(error))) {
        q38_qsa_state_destroy(&state);
        return 0;
    }
    q38_qsa_state_destroy(&state);
    return 1;
}

int main(void) {
    const size_t tokens = 12;
    float keys[tokens * 4];
    for (size_t i = 0; i < tokens * 4; ++i) keys[i] = (float)(i % 7) / 7.0f;
    const size_t one[] = {12};
    const size_t singles[] = {1,1,1,1,1,1,1,1,1,1,1,1};
    const size_t mixed[] = {3,5,4};
    float reference[3], actual[3];
    uint32_t ref_ids[2], ids[2];
    if (!run_partition(keys, tokens, one, 1, reference, ref_ids) ||
        !run_partition(keys, tokens, singles, 12, actual, ids) ||
        memcmp(reference, actual, sizeof(reference)) != 0 ||
        memcmp(ref_ids, ids, sizeof(ref_ids)) != 0 ||
        !run_partition(keys, tokens, mixed, 3, actual, ids) ||
        memcmp(reference, actual, sizeof(reference)) != 0 ||
        memcmp(ref_ids, ids, sizeof(ref_ids)) != 0) {
        fprintf(stderr, "QSA chunk invariance mismatch\n");
        return 1;
    }
    puts("test_m5_qsa_chunking: state, scores, and selected IDs are chunk invariant");
    return 0;
}
