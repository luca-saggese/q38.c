#include "q38_gdn_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t state_index(size_t head, size_t row, size_t column) {
    return ((head * Q38_GDN_REF_HEAD_DIM + row) * Q38_GDN_REF_HEAD_DIM) +
           column;
}

static int near(float actual, float expected) {
    return fabsf(actual - expected) <= 2e-6f;
}

static void clear_vectors(float *q, float *k, float *v, float *decay,
                          float *beta) {
    memset(q, 0, Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM *
                   sizeof(*q));
    memset(k, 0, Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM *
                   sizeof(*k));
    memset(v, 0, Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM *
                   sizeof(*v));
    for (size_t head = 0; head < Q38_GDN_REF_VALUE_HEADS; head++) {
        decay[head] = 1.0f;
        beta[head] = 1.0f;
    }
}

static int test_zero_state(void) {
    const size_t n = Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM;
    float *state = calloc(q38_gdn_ref_state_elements(1), sizeof(float));
    float *q = calloc(n, sizeof(float)), *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float)), *output = calloc(n, sizeof(float));
    float decay[Q38_GDN_REF_VALUE_HEADS], beta[Q38_GDN_REF_VALUE_HEADS];
    if (!state || !q || !k || !v || !output) return 0;
    clear_vectors(q, k, v, decay, beta);
    q[0] = 1.0f; k[0] = 1.0f; v[0] = 2.0f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 0.5f,
                          output) ||
        !near(state[state_index(0, 0, 0)], 2.0f) ||
        !near(output[0], 1.0f))
        return 0;
    free(state); free(q); free(k); free(v); free(output);
    return 1;
}

static int test_beta_zero(void) {
    const size_t n = Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM;
    float *state = calloc(q38_gdn_ref_state_elements(1), sizeof(float));
    float *q = calloc(n, sizeof(float)), *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float)), *output = calloc(n, sizeof(float));
    float decay[Q38_GDN_REF_VALUE_HEADS], beta[Q38_GDN_REF_VALUE_HEADS];
    if (!state || !q || !k || !v || !output) return 0;
    clear_vectors(q, k, v, decay, beta);
    state[state_index(0, 0, 0)] = 3.0f;
    q[0] = 1.0f; k[0] = 1.0f; v[0] = 9.0f; beta[0] = 0.0f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 1.0f,
                          output) ||
        !near(state[state_index(0, 0, 0)], 3.0f) ||
        !near(output[0], 3.0f))
        return 0;
    free(state); free(q); free(k); free(v); free(output);
    return 1;
}

static int test_decay_ordering(void) {
    const size_t n = Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM;
    float *state = calloc(q38_gdn_ref_state_elements(1), sizeof(float));
    float *q = calloc(n, sizeof(float)), *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float)), *output = calloc(n, sizeof(float));
    float decay[Q38_GDN_REF_VALUE_HEADS], beta[Q38_GDN_REF_VALUE_HEADS];
    if (!state || !q || !k || !v || !output) return 0;
    clear_vectors(q, k, v, decay, beta);
    state[state_index(0, 0, 0)] = 2.0f;
    q[0] = 1.0f; k[0] = 1.0f; decay[0] = 0.5f; v[0] = 1.0f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 1.0f,
                          output) ||
        !near(state[state_index(0, 0, 0)], 1.0f) ||
        !near(output[0], 1.0f))
        return 0;
    free(state); free(q); free(k); free(v); free(output);
    return 1;
}

static int test_exact_prediction(void) {
    const size_t n = Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM;
    float *state = calloc(q38_gdn_ref_state_elements(1), sizeof(float));
    float *q = calloc(n, sizeof(float)), *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float)), *output = calloc(n, sizeof(float));
    float decay[Q38_GDN_REF_VALUE_HEADS], beta[Q38_GDN_REF_VALUE_HEADS];
    if (!state || !q || !k || !v || !output) return 0;
    clear_vectors(q, k, v, decay, beta);
    state[state_index(0, 0, 0)] = 2.0f;
    q[0] = 1.0f; k[0] = 1.0f; v[0] = 2.0f; beta[0] = 0.75f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 1.0f,
                          output) ||
        !near(state[state_index(0, 0, 0)], 2.0f) ||
        !near(output[0], 2.0f))
        return 0;
    free(state); free(q); free(k); free(v); free(output);
    return 1;
}

static int test_basis_orientation(void) {
    const size_t n = Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM;
    float *state = calloc(q38_gdn_ref_state_elements(1), sizeof(float));
    float *q = calloc(n, sizeof(float)), *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float)), *output = calloc(n, sizeof(float));
    float decay[Q38_GDN_REF_VALUE_HEADS], beta[Q38_GDN_REF_VALUE_HEADS];
    if (!state || !q || !k || !v || !output) return 0;
    clear_vectors(q, k, v, decay, beta);
    q[1] = 1.0f; k[1] = 1.0f; v[2] = 1.0f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 1.0f,
                          output) ||
        !near(state[state_index(0, 1, 2)], 1.0f) ||
        !near(state[state_index(0, 2, 1)], 0.0f) ||
        !near(output[2], 1.0f) || !near(output[1], 0.0f))
        return 0;
    free(state); free(q); free(k); free(v); free(output);
    return 1;
}

static int test_two_timesteps(void) {
    const size_t n = Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM;
    float *state = calloc(q38_gdn_ref_state_elements(1), sizeof(float));
    float *q = calloc(n, sizeof(float)), *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float)), *output = calloc(n, sizeof(float));
    float decay[Q38_GDN_REF_VALUE_HEADS], beta[Q38_GDN_REF_VALUE_HEADS];
    if (!state || !q || !k || !v || !output) return 0;
    clear_vectors(q, k, v, decay, beta);
    q[0] = 1.0f; k[0] = 1.0f; v[1] = 1.0f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 1.0f,
                          output))
        return 0;
    v[1] = 2.0f; beta[0] = 0.5f;
    if (!q38_gdn_ref_step(state, 1, 0, q, k, v, decay, beta, 1.0f,
                          output) ||
        !near(state[state_index(0, 0, 1)], 1.5f) ||
        !near(output[1], 1.5f))
        return 0;
    free(state); free(q); free(k); free(v); free(output);
    return 1;
}

static int test_head_mapping(void) {
    float key[Q38_GDN_REF_KEY_HEADS * Q38_GDN_REF_HEAD_DIM];
    float value[Q38_GDN_REF_VALUE_HEADS * Q38_GDN_REF_HEAD_DIM];
    for (size_t head = 0; head < Q38_GDN_REF_KEY_HEADS; head++)
        for (size_t dimension = 0; dimension < Q38_GDN_REF_HEAD_DIM;
             dimension++)
            key[head * Q38_GDN_REF_HEAD_DIM + dimension] =
                (float)(head + 1);
    memset(value, 0, sizeof(value));
    q38_gdn_ref_repeat_key_heads(key, value);
    for (size_t head = 0; head < Q38_GDN_REF_VALUE_HEADS; head++)
        for (size_t dimension = 0; dimension < Q38_GDN_REF_HEAD_DIM;
             dimension++)
            if (!near(value[head * Q38_GDN_REF_HEAD_DIM + dimension],
                      (float)(head / 3u + 1u)))
                return 0;
    return 1;
}

int main(void) {
    struct {
        const char *name;
        int (*run)(void);
    } tests[] = {
        {"zero_state", test_zero_state},
        {"beta_zero", test_beta_zero},
        {"decay_ordering", test_decay_ordering},
        {"exact_prediction", test_exact_prediction},
        {"basis_orientation", test_basis_orientation},
        {"two_timesteps", test_two_timesteps},
        {"head_mapping", test_head_mapping},
    };
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (!tests[i].run()) {
            fprintf(stderr, "test_m3_gdn_ref: %s failed\n", tests[i].name);
            return 1;
        }
    }
    puts("test_m3_gdn_ref: scalar F32 GDN recurrence microtests passed");
    return 0;
}
