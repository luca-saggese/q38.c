#ifndef Q38_TEST_GDN_REFERENCE_H
#define Q38_TEST_GDN_REFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    Q38_TEST_GDN_HIDDEN = 2560,
    Q38_TEST_GDN_QKV = 10240,
    Q38_TEST_GDN_Z = 6144,
    Q38_TEST_GDN_KEY_HEADS = 16,
    Q38_TEST_GDN_HEADS = 48,
    Q38_TEST_GDN_DIM = 128,
    Q38_TEST_GDN_HISTORY = 3,
    Q38_TEST_GDN_KERNEL = 4,
};

typedef struct {
    uint32_t type;
    size_t rows;
    size_t cols;
    const unsigned char *data;
} q38_test_gdn_weight;

bool q38_test_gdn_reference(
    const float *hidden, const float *history, const float *state,
    const q38_test_gdn_weight *qkv, const q38_test_gdn_weight *z,
    const q38_test_gdn_weight *a, const q38_test_gdn_weight *b,
    const q38_test_gdn_weight *conv, const q38_test_gdn_weight *a_log,
    const q38_test_gdn_weight *dt_bias, const q38_test_gdn_weight *norm,
    const q38_test_gdn_weight *out_proj, float *output, float *next_history,
    float *next_state, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
