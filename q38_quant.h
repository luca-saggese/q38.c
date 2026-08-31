#ifndef Q38_QUANT_H
#define Q38_QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    Q38_QUANT_Q2_K = 10,
    Q38_QUANT_Q4_K = 12,
};

#define Q38_QUANT_QK_K 256
#define Q38_QUANT_Q2_K_BLOCK_BYTES 84
#define Q38_QUANT_Q4_K_BLOCK_BYTES 144

typedef struct {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
} q38_q2_k_block;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} q38_q4_k_block;

float q38_half_to_float(uint16_t bits);

bool q38_quant_dequantize_row(uint32_t type, const void *blocks,
                              size_t block_count, float *out,
                              size_t out_elements, char *error,
                              size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
