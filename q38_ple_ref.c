#include "q38_ple_ref.h"
#include "q38_quant.h"

#include <stdio.h>
#include <string.h>

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
