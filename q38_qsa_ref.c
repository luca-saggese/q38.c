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
                double q_norm = 0.0;
                for (size_t d = 0; d < head_dim; ++d)
                    q_norm += (double)q[d] * q[d];
                dot *= 1.0f / sqrtf((float)(q_norm / (double)head_dim) + 1e-6f);
                total += dot > 0.0f ? dot : 0.0f;
            }
            scores[query * blocks + block] = total / sqrtf((float)head_dim);
        }

    }
    free(pooled);
    return true;
}

bool q38_qsa_select_tokens_ref(const float *raw_keys, size_t token_count,
                               const float *queries, size_t query_count,
                               size_t heads, size_t head_dim, size_t ratio,
                               size_t budget, const uint32_t *visible,
                               const size_t *visible_offsets,
                               uint32_t *selected, size_t selected_stride,
                               size_t *selected_counts, char *error,
                               size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!raw_keys || !token_count || !queries || !query_count || !heads ||
        !head_dim || !ratio || !budget || !visible || !visible_offsets ||
        !selected || !selected_counts || !selected_stride ||
        visible_offsets[0] != 0) {
        return fail(error, error_len, "invalid QSA token selection arguments");
    }
    for (size_t query = 0; query < query_count; ++query) {
        const size_t begin = visible_offsets[query];
        const size_t end = visible_offsets[query + 1];
        if (end < begin || end - begin > token_count) {
            return fail(error, error_len, "QSA visible-cell range is invalid");
        }
        const size_t visible_count = end - begin;
        const size_t complete = visible_count / ratio;
        const size_t group_budget = budget / ratio;
        const size_t groups = complete < group_budget ? complete : group_budget;
        if (complete > UINT32_MAX || groups > SIZE_MAX / ratio ||
            groups * ratio > selected_stride ||
            visible_count - complete * ratio > selected_stride - groups * ratio) {
            return fail(error, error_len, "QSA selection output stride is too small");
        }
        float *group_scores = (float *)malloc(complete * sizeof(*group_scores));
        uint32_t *group_ids = (uint32_t *)malloc(complete * sizeof(*group_ids));
        if ((complete && (!group_scores || !group_ids))) {
            free(group_scores);
            free(group_ids);
            return fail(error, error_len, "QSA score allocation failed");
        }
        for (size_t group = 0; group < complete; ++group) {
            const size_t member = begin + group * ratio;
            float total = 0.0f;
            for (size_t head = 0; head < heads; ++head) {
                const float *q = queries + (query * heads + head) * head_dim;
                float dot = 0.0f;
                double norm = 0.0;
                for (size_t d = 0; d < head_dim; ++d) {
                    float pooled = 0.0f;
                    for (size_t j = 0; j < ratio; ++j)
                        pooled += raw_keys[(size_t)visible[member + j] *
                                           head_dim + d];
                    pooled /= (float)ratio;
                    dot += q[d] * pooled;
                    norm += (double)pooled * pooled;
                }
                dot /= sqrtf((float)(norm / (double)head_dim) + 1e-6f);
                double qnorm = 0.0;
                for (size_t d = 0; d < head_dim; ++d)
                    qnorm += (double)q[d] * q[d];
                dot *= 1.0f /
                       sqrtf((float)(qnorm / (double)head_dim) + 1e-6f);
                total += dot > 0.0f ? dot : 0.0f;
            }
            group_scores[group] = total / sqrtf((float)head_dim);
            group_ids[group] = (uint32_t)group;
        }
        size_t used = 0;
        for (size_t candidate = 0; candidate < complete; ++candidate) {
            size_t at = used;
            while (at > 0) {
                const size_t previous = group_ids[at - 1];
                const uint32_t previous_cell = visible[begin + previous * ratio];
                const uint32_t candidate_cell = visible[begin + candidate * ratio];
                if (group_scores[previous] > group_scores[candidate] ||
                    (group_scores[previous] == group_scores[candidate] &&
                     previous_cell < candidate_cell)) {
                    break;
                }
                --at;
            }
            if (at < groups) {
                if (used < groups) ++used;
                for (size_t j = used; j > at + 1; --j)
                    group_ids[j - 1] = group_ids[j - 2];
                group_ids[at] = (uint32_t)candidate;
            }
        }
        size_t out = 0;
        for (size_t rank = 0; rank < groups; ++rank) {
            const size_t group = group_ids[rank];
            for (size_t j = 0; j < ratio; ++j)
                selected[query * selected_stride + out++] =
                    visible[begin + group * ratio + j];
        }
        for (size_t i = complete * ratio; i < visible_count; ++i)
            selected[query * selected_stride + out++] = visible[begin + i];
        selected_counts[query] = out;
        free(group_scores);
        free(group_ids);
    }
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

bool q38_qsa_selected_width_ref(size_t kv_count, size_t top_k, size_t ratio,
                                size_t *width, char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!kv_count || !top_k || !ratio || !width)
        return fail(error, error_len, "invalid QSA selected-width arguments");
    if (top_k > SIZE_MAX - ratio + 1)
        return fail(error, error_len, "QSA selected width overflows");
    const size_t requested = top_k + ratio - 1;
    *width = requested < kv_count ? requested : kv_count;
    return true;
}
