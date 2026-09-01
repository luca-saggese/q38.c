#include "q38_qsa_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

static void normalize(float *vector, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) sum += (double)vector[i] * vector[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)count) + 1e-6f);
    for (size_t i = 0; i < count; ++i) vector[i] *= scale;
}

bool q38_qsa_index_scores_ref(const float *raw_keys, size_t token_count,
                              const float *queries, size_t query_count,
                              size_t heads, size_t head_dim, size_t ratio,
                              float *scores, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!raw_keys || !queries || !scores || !token_count || !query_count ||
        !heads || !head_dim || !ratio || heads > SIZE_MAX / head_dim ||
        token_count > SIZE_MAX / head_dim ||
        query_count > SIZE_MAX / (heads * head_dim)) {
        return fail(error, error_len, "invalid QSA index score arguments");
    }
    if (ratio > SIZE_MAX - token_count + 1)
        return fail(error, error_len, "QSA block count overflows");
    const size_t blocks = (token_count + ratio - 1) / ratio;
    if (blocks > SIZE_MAX / head_dim) {
        return fail(error, error_len, "QSA pooled-key size overflows");
    }
    float *pooled = (float *)calloc(blocks * head_dim, sizeof(*pooled));
    if (!pooled) return fail(error, error_len, "QSA pooled-key allocation failed");
    for (size_t block = 0; block < blocks; ++block) {
        const size_t begin = block * ratio;
        const size_t end = begin + ratio < token_count ? begin + ratio : token_count;
        for (size_t token = begin; token < end; ++token)
            for (size_t d = 0; d < head_dim; ++d)
                pooled[block * head_dim + d] += raw_keys[token * head_dim + d];
        const float scale = 1.0f / (float)(end - begin);
        for (size_t d = 0; d < head_dim; ++d)
            pooled[block * head_dim + d] *= scale;
        normalize(pooled + block * head_dim, head_dim);
    }
    for (size_t query = 0; query < query_count; ++query) {
        for (size_t block = 0; block < blocks; ++block) {
            float total = 0.0f;
            for (size_t head = 0; head < heads; ++head) {
                const float *q = queries + (query * heads + head) * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d)
                    dot += q[d] * pooled[block * head_dim + d];
                total += dot > 0.0f ? dot : 0.0f;
            }
            scores[query * blocks + block] = total;
        }
    }
    free(pooled);
    return true;
}

bool q38_qsa_expand_block_scores_ref(const float *block_scores,
                                     size_t block_count, size_t token_count,
                                     size_t ratio, float *token_scores,
                                     char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!block_scores || !token_scores || !block_count || !token_count ||
        !ratio || block_count != (token_count + ratio - 1) / ratio)
        return fail(error, error_len, "invalid QSA score expansion arguments");
    for (size_t token = 0; token < token_count; ++token)
        token_scores[token] = block_scores[token / ratio];
    return true;
}
