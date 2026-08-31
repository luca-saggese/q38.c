#include "q38_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

float q38_half_to_float(uint16_t bits) {
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x3ffu;
    uint32_t value;
    if (!exponent) {
        if (!fraction) value = sign;
        else {
            exponent = 1;
            while (!(fraction & 0x400u)) {
                fraction <<= 1;
                exponent--;
            }
            fraction &= 0x3ffu;
            value = sign | ((exponent + 112u) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (fraction << 13);
    } else {
        value = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static void dequant_q2(const q38_q2_k_block *block, float *out) {
    const float d = q38_half_to_float(block->d);
    const float min = q38_half_to_float(block->dmin);
    const uint8_t *q = block->qs;
    unsigned scale_index = 0;
    for (unsigned n = 0; n < Q38_QUANT_QK_K; n += 128) {
        (void)n;
        unsigned shift = 0;
        for (unsigned j = 0; j < 4; j++) {
            uint8_t scale = block->scales[scale_index++];
            float dl = d * (scale & 0xfu);
            float ml = min * (scale >> 4);
            for (unsigned l = 0; l < 16; l++)
                *out++ = dl * ((q[l] >> shift) & 3u) - ml;
            scale = block->scales[scale_index++];
            dl = d * (scale & 0xfu);
            ml = min * (scale >> 4);
            for (unsigned l = 0; l < 16; l++)
                *out++ = dl * ((q[l + 16] >> shift) & 3u) - ml;
            shift += 2;
        }
        q += 32;
    }
}

static void scale_min_q4(unsigned index, const uint8_t *scales,
                         uint8_t *scale, uint8_t *min) {
    if (index < 4) {
        *scale = scales[index] & 63u;
        *min = scales[index + 4] & 63u;
    } else {
        *scale = (scales[index + 4] & 0xfu) |
                 ((scales[index - 4] >> 6) << 4);
        *min = (scales[index + 4] >> 4) |
               ((scales[index] >> 6) << 4);
    }
}

static void dequant_q4(const q38_q4_k_block *block, float *out) {
    const float d = q38_half_to_float(block->d);
    const float min = q38_half_to_float(block->dmin);
    const uint8_t *q = block->qs;
    unsigned scale_index = 0;
    for (unsigned j = 0; j < Q38_QUANT_QK_K; j += 64) {
        uint8_t scale, minimum;
        scale_min_q4(scale_index++, block->scales, &scale, &minimum);
        const float d1 = d * scale;
        const float m1 = min * minimum;
        scale_min_q4(scale_index++, block->scales, &scale, &minimum);
        const float d2 = d * scale;
        const float m2 = min * minimum;
        for (unsigned l = 0; l < 32; l++)
            *out++ = d1 * (q[l] & 0xfu) - m1;
        for (unsigned l = 0; l < 32; l++)
            *out++ = d2 * (q[l] >> 4) - m2;
        q += 32;
    }
}

bool q38_quant_dequantize_row(uint32_t type, const void *blocks,
                              size_t block_count, float *out,
                              size_t out_elements, char *error,
                              size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!blocks || !out || !block_count ||
        block_count > SIZE_MAX / Q38_QUANT_QK_K ||
        out_elements != block_count * Q38_QUANT_QK_K) {
        set_error(error, error_len, "invalid quantized row arguments");
        return false;
    }
    if (type == Q38_QUANT_Q2_K) {
        const q38_q2_k_block *q = (const q38_q2_k_block *)blocks;
        for (size_t i = 0; i < block_count; i++) dequant_q2(&q[i], out + i * Q38_QUANT_QK_K);
        return true;
    }
    if (type == Q38_QUANT_Q4_K) {
        const q38_q4_k_block *q = (const q38_q4_k_block *)blocks;
        for (size_t i = 0; i < block_count; i++) dequant_q4(&q[i], out + i * Q38_QUANT_QK_K);
        return true;
    }
    set_error(error, error_len, "unsupported scalar quantization type");
    return false;
}
