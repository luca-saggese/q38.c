#include "q38_ple_ref.h"
#include "q38_quant.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) {
        snprintf(error, error_len, "%s", message);
    }
    return false;
}

bool q38_ple_hash_config_validate(const q38_ple_hash_config *config,
                                  char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!config) return fail(error, error_len, "PLE hash config is null");
    if (config->ngram_size < 2 || config->ngram_size > Q38_PLE_MAX_NGRAM) {
        return fail(error, error_len, "PLE ngram size is outside [2, 3]");
    }
    if (config->heads_per_ngram == 0 ||
        config->heads_per_ngram * (config->ngram_size - 1) > Q38_PLE_MAX_HEADS) {
        return fail(error, error_len, "PLE head count is invalid");
    }
    for (uint32_t h = 0; h < config->heads_per_ngram * (config->ngram_size - 1); ++h) {
        if (config->head_vocab_sizes[h] == 0) {
            return fail(error, error_len, "PLE head vocabulary size is zero");
        }
    }
    return true;
}

bool q38_ple_ngram_ids_ref(const q38_ple_hash_config *config,
                           const q38_ngram_history *history,
                           uint32_t current_token, uint32_t eos_token,
                           uint32_t *ids, size_t ids_count,
                           char *error, size_t error_len) {
    if (!q38_ple_hash_config_validate(config, error, error_len)) return false;
    const size_t n_heads =
        (size_t) config->heads_per_ngram * (config->ngram_size - 1);
    if (!ids || ids_count < n_heads) {
        return fail(error, error_len, "PLE ID output is too small");
    }

    uint32_t context[Q38_PLE_MAX_NGRAM];
    q38_ngram_history_context(history, current_token, eos_token, context);
    for (uint32_t n = 2; n <= config->ngram_size; ++n) {
        uint64_t mixed = (uint64_t) context[0] * config->multipliers[0];
        for (uint32_t j = 1; j < n; ++j) {
            mixed ^= (uint64_t) context[j] * config->multipliers[j];
        }
        const uint32_t base = (n - 2) * config->heads_per_ngram;
        for (uint32_t g = 0; g < config->heads_per_ngram; ++g) {
            const uint32_t h = base + g;
            const uint64_t row =
                mixed % config->head_vocab_sizes[h] + config->head_offsets[h];
            if (row > UINT32_MAX) {
                return fail(error, error_len, "PLE row index exceeds uint32");
            }
            ids[h] = (uint32_t) row;
        }
    }
    return true;
}

bool q38_ple_decode_row_ref(uint32_t qtype, const void *blocks,
                            uint32_t row_width, float *out,
                            size_t out_elements, char *error,
                            size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (qtype != Q38_QUANT_Q2_K && qtype != Q38_QUANT_Q4_K) {
        return fail(error, error_len, "unsupported PLE row quantization type");
    }

    if (!blocks || !out || row_width == 0 ||
        row_width % Q38_QUANT_QK_K != 0 ||
        out_elements != (size_t)row_width) {
        return fail(error, error_len, "invalid PLE quantized row arguments");
    }
    return q38_quant_dequantize_row(
        qtype, blocks, row_width / Q38_QUANT_QK_K, out, out_elements,
        error, error_len);
}

static void grouped_norm(float *x, const float *weight, size_t tokens,
                         size_t streams, size_t hidden, float eps) {
    for (size_t t = 0; t < tokens; ++t)
        for (size_t s = 0; s < streams; ++s) {
            float *row = x + (t * streams + s) * hidden;
            double sum = 0.0;
            for (size_t d = 0; d < hidden; ++d) sum += (double)row[d] * row[d];
            const float scale = 1.0f / sqrtf((float)(sum / hidden) + eps);
            for (size_t d = 0; d < hidden; ++d)
                row[d] *= scale * weight[s * hidden + d];
        }
}

static void matmul_rows(const float *x, size_t tokens, size_t in,
                        const float *w, size_t out, float *y) {
    for (size_t t = 0; t < tokens; ++t)
        for (size_t r = 0; r < out; ++r) {
            float value = 0.0f;
            for (size_t c = 0; c < in; ++c)
                value += x[t * in + c] * w[r * in + c];
            y[t * out + r] = value;
        }
}

bool q38_ple_forward_ref(const q38_ple_forward_config *config,
                         const float *hidden, size_t token_count,
                         const float *embedding, const float *key_proj,
                         const float *value_proj, const float *norm_key,
                         const float *norm_query, const float *norm_conv,
                         const float *conv, float *history, float *contribution,
                         float *after, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!config || !hidden || !token_count || !embedding || !key_proj ||
        !value_proj || !norm_key || !norm_query || !norm_conv || !conv ||
        !history || !contribution || !after || !config->hidden ||
        !config->streams || !config->heads || !config->row_width ||
        !config->kernel || !config->dilation || config->eps <= 0.0f)
        return fail(error, error_len, "invalid PLE forward arguments");
    const size_t emb_dim = config->heads * config->row_width;
    const size_t channels = config->streams * config->hidden;
    const size_t hist = (config->kernel - 1) * config->dilation;
    if (emb_dim > SIZE_MAX / sizeof(float) ||
        channels > SIZE_MAX / sizeof(float) ||
        config->kernel > SIZE_MAX / channels)
        return fail(error, error_len, "PLE forward dimensions overflow");
    float *key = calloc(token_count * channels, sizeof(float));
    float *value = calloc(token_count * config->hidden, sizeof(float));
    float *query = calloc(token_count * channels, sizeof(float));
    float *gated = calloc(token_count * channels, sizeof(float));
    float *normalized = calloc(token_count * channels, sizeof(float));
    float *padded = calloc((hist + token_count) * channels, sizeof(float));
    float *conv_out = calloc(token_count * channels, sizeof(float));
    if (!key || !value || !query || !gated || !normalized || !padded ||
        !conv_out) {
        free(key); free(value); free(query); free(gated); free(normalized);
        free(padded); free(conv_out);
        return fail(error, error_len, "PLE forward allocation failed");
    }
    matmul_rows(embedding, token_count, emb_dim, key_proj, channels, key);
    matmul_rows(embedding, token_count, emb_dim, value_proj, config->hidden,
                value);
    memcpy(query, hidden, token_count * channels * sizeof(float));
    grouped_norm(key, norm_key, token_count, config->streams, config->hidden,
                 config->eps);
    grouped_norm(query, norm_query, token_count, config->streams,
                 config->hidden, config->eps);
    for (size_t t = 0; t < token_count; ++t)
        for (size_t s = 0; s < config->streams; ++s) {
            float score = 0.0f;
            for (size_t d = 0; d < config->hidden; ++d)
                score += key[(t * config->streams + s) * config->hidden + d] *
                         query[(t * config->streams + s) * config->hidden + d];
            score /= sqrtf((float)config->hidden);
            const float mag = sqrtf(fmaxf(fabsf(score), 1e-6f));
            const float gate = 1.0f / (1.0f + expf(-copysignf(mag, score)));
            for (size_t d = 0; d < config->hidden; ++d)
                gated[(t * config->streams + s) * config->hidden + d] =
                    value[t * config->hidden + d] * gate;
        }
    memcpy(normalized, gated, token_count * channels * sizeof(float));
    grouped_norm(normalized, norm_conv, token_count, config->streams,
                 config->hidden, config->eps);
    memcpy(padded + hist * channels, normalized,
           token_count * channels * sizeof(float));
    if (hist) memcpy(padded, history, hist * channels * sizeof(float));
    for (size_t t = 0; t < token_count; ++t)
        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < config->kernel; ++k) {
                const size_t source = hist + t -
                    (config->kernel - 1 - k) * config->dilation;
                sum += conv[k * channels + c] * padded[source * channels + c];
            }
            conv_out[t * channels + c] = sum / (1.0f + expf(-sum));
        }
    for (size_t i = 0; i < token_count * channels; ++i) {
        contribution[i] = gated[i] + conv_out[i];
        after[i] = hidden[i] + contribution[i];
    }
    if (hist) {
        if (token_count >= hist)
            memcpy(history, normalized + (token_count - hist) * channels,
                   hist * channels * sizeof(float));
        else {
            memmove(history, history + token_count * channels,
                    (hist - token_count) * channels * sizeof(float));
            memcpy(history + (hist - token_count) * channels, normalized,
                   token_count * channels * sizeof(float));
        }
    }
    free(key); free(value); free(query); free(gated); free(normalized);
    free(padded); free(conv_out);
    return true;
}
