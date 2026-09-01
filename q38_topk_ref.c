#include "q38_topk_ref.h"

#include <stdio.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_topk_select_ref(const float *scores, size_t count, size_t k,
                         uint32_t *indices, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!scores || !indices || !count || !k || k > count ||
        count > UINT32_MAX) {
        return fail(error, error_len, "invalid top-k arguments");
    }
    size_t used = 0;
    for (size_t candidate = 0; candidate < count; ++candidate) {
        size_t at = used;
        while (at > 0) {
            const uint32_t previous = indices[at - 1];
            if (scores[previous] > scores[candidate] ||
                (scores[previous] == scores[candidate] &&
                 previous < candidate))
                break;
            --at;
        }
        if (at < k) {
            if (used < k) ++used;
            for (size_t j = used; j > at + 1; --j)
                indices[j - 1] = indices[j - 2];
            indices[at] = (uint32_t)candidate;
        }
    }
    return true;
}
