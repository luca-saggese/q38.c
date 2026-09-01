#include "q38_forward.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char magic[4];
    uint32_t tokens, hidden, max_selected, weight_count;
} fixture_header;

static size_t matrix_elements(size_t rows, size_t cols) {
    return rows * cols;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s fixture.bin\n", argv[0]);
        return 2;
    }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) return 1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(fixture_header))
        return 1;
    const uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return 1;
    const fixture_header *header = (const fixture_header *)map;
    if (header->magic[0] != 'Q' || header->magic[1] != '3' ||
        header->magic[2] != '8' || header->magic[3] != 'F' ||
        header->hidden != 2560 || header->weight_count != 9) {
        munmap((void *)map, (size_t)st.st_size);
        return 1;
    }
    size_t offset = sizeof(*header);
    const size_t emb_bytes = (size_t)header->tokens * header->hidden * sizeof(float);
    const size_t out_bytes = emb_bytes;
    if (offset > (size_t)st.st_size || emb_bytes > (size_t)st.st_size - offset) return 1;
    const float *emb = (const float *)(map + offset); offset += emb_bytes;
    if (out_bytes > (size_t)st.st_size - offset) return 1;
    const float *expected = (const float *)(map + offset); offset += out_bytes;
    const size_t dims[9][2] = {
        {12288,2560}, {512,2560}, {512,2560}, {2560,6144},
        {640,2560}, {256,1}, {256,1}, {128,1}, {128,1},
    };
    q38_forward_matrix matrices[5];
    for (size_t i = 0; i < 5; ++i) {
        size_t bytes = matrix_elements(dims[i][0], dims[i][1]) * sizeof(float);
        if (bytes > (size_t)st.st_size - offset) return 1;
        matrices[i] = (q38_forward_matrix){map + offset, dims[i][0], dims[i][1],
                                           Q38_FORWARD_F32};
        offset += bytes;
    }
    const float *q_norm = (const float *)(map + offset); offset += 256 * sizeof(float);
    const float *k_norm = (const float *)(map + offset); offset += 256 * sizeof(float);
    const float *iq_norm = (const float *)(map + offset); offset += 128 * sizeof(float);
    const float *ik_norm = (const float *)(map + offset); offset += 128 * sizeof(float);
    if (offset > (size_t)st.st_size) return 1;

    q38_forward_qsa_weights w = {
        .q_proj = matrices[0], .k_proj = matrices[1], .v_proj = matrices[2],
        .o_proj = matrices[3], .index_qk_proj = matrices[4],
        .q_norm = q_norm, .k_norm = k_norm, .index_q_norm = iq_norm,
        .index_k_norm = ik_norm, .hidden = 2560, .query_heads = 24,
        .kv_heads = 2, .head_dim = 256, .index_heads = 4, .index_dim = 128,
        .ratio = 4, .budget = 2048, .rope_theta = 10000000.0f,
        .rotary_dims = 64,
    };
    q38_qsa_state state;
    char error[256];
    if (!q38_forward_qsa_state_init(&state, &w, error, sizeof(error))) return 1;
    float *actual = calloc((size_t)header->tokens * header->hidden, sizeof(float));
    uint32_t *ids = calloc((size_t)header->tokens * header->max_selected,
                           sizeof(uint32_t));
    size_t *counts = calloc(header->tokens, sizeof(size_t));
    if (!actual || !ids || !counts ||
        !q38_forward_qsa_ref(&w, &state, emb, header->tokens, actual, ids,
                             header->max_selected, counts, error,
                             sizeof(error))) {
        fprintf(stderr, "native QSA probe failed: %s\n", error);
        return 1;
    }
    for (size_t i = 0; i < (size_t)header->tokens * header->hidden; ++i)
        if (fabsf(actual[i] - expected[i]) > 3e-4f) {
            fprintf(stderr, "native/reference mismatch at %zu: %.8g %.8g\n",
                    i, actual[i], expected[i]);
            return 1;
        }
    q38_qsa_state_destroy(&state);
    free(actual); free(ids); free(counts);
    munmap((void *)map, (size_t)st.st_size);
    puts("test_m5_forward_probe: native q38 QSA graph matches independent checkpoint golden");
    return 0;
}
