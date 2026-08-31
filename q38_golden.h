#ifndef Q38_GOLDEN_H
#define Q38_GOLDEN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_GOLDEN_FORMAT_VERSION 1u

typedef enum {
    Q38_DTYPE_INVALID = 0,
    Q38_DTYPE_F32 = 1,
    Q38_DTYPE_BF16 = 2,
    Q38_DTYPE_U32 = 3,
    Q38_DTYPE_I32 = 4,
} q38_dtype;

typedef struct {
    char stage[64];
    int32_t layer;
    int32_t token_pos;
    q38_dtype dtype;
    uint64_t elements;
    uint64_t checksum;
} q38_golden_meta;

typedef struct {
    q38_golden_meta meta;
    void *data;
    size_t data_bytes;
} q38_golden_vector;

/* The checksum covers the canonical metadata fields and the payload. */
uint64_t q38_golden_checksum(const q38_golden_meta *meta,
                             const void *data, size_t data_bytes);

bool q38_golden_write(const char *path, const q38_golden_meta *meta,
                      const void *data, size_t data_bytes,
                      char *error, size_t error_len);

bool q38_golden_read(const char *path, q38_golden_vector *out,
                     char *error, size_t error_len);

void q38_golden_free(q38_golden_vector *vector);

bool q38_golden_equal(const q38_golden_vector *a,
                      const q38_golden_vector *b);

#ifdef __cplusplus
}
#endif

#endif
