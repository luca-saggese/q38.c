#ifndef Q38_TEST_QSA_REFERENCE_H
#define Q38_TEST_QSA_REFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t type;
    size_t rows;
    size_t cols;
    const unsigned char *data;
    size_t bytes;
} q38_test_qsa_weight;

bool q38_test_qsa_project(const q38_test_qsa_weight *weight,
                          const float *input, float *output,
                          char *error, size_t error_len);

bool q38_test_qsa_build_state(
    const float *q_projection, const float *keys, const float *values,
    const float *index_projection, const q38_test_qsa_weight *k_norm,
    float *state_main_k, float *state_main_v, float *state_index_k,
    char *error, size_t error_len);

bool q38_test_qsa_attention(
    const float *q_projection, const float *index_projection,
    const float *state_main_k, const float *state_main_v,
    const float *state_index_k, size_t state_count,
    const q38_test_qsa_weight *q_norm,
    const q38_test_qsa_weight *index_q_norm,
    const q38_test_qsa_weight *index_k_norm, float *attention,
    uint32_t *selected, size_t selected_stride, size_t *selected_count,
    char *error, size_t error_len);

bool q38_test_qsa_output_project(const q38_test_qsa_weight *o_proj,
                                  const float *attention, float *output,
                                  char *error, size_t error_len);

bool q38_test_qsa_post(
    const float *q_projection, const float *index_projection,
    const float *state_main_k, const float *state_main_v,
    const float *state_index_k, size_t state_count,
    const q38_test_qsa_weight *q_norm, const q38_test_qsa_weight *k_norm,
    const q38_test_qsa_weight *index_q_norm,
    const q38_test_qsa_weight *index_k_norm, const q38_test_qsa_weight *o_proj,
    float *attention, uint32_t *selected, size_t selected_stride,
    size_t *selected_count, float *output, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
