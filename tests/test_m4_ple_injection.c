#include "q38_ple_ref.h"

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
    uint32_t tokens, channels, embedding_width, array_count;
} header;

static int equal(const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (fabsf(a[i] - b[i]) > 4e-4f) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) return 1;
    const uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ,
                              MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return 1;
    const header *h = (const header *)map;
    if (h->magic[0] != 'P' || h->magic[1] != '3' ||
        h->magic[2] != '8' || h->magic[3] != 'F' ||
        h->tokens != 4 || h->channels != 10240 ||
        h->embedding_width != 2560 || h->array_count != 10) return 1;
    size_t off = sizeof(*h), bytes;
    const size_t sizes[] = {
        4u * 10240u, 4u * 2560u, 10240u * 2560u, 2560u * 2560u,
        10240u, 10240u, 10240u, 4u * 10240u, 4u * 10240u, 4u * 10240u,
    };
    const float *a[10];
    for (size_t i = 0; i < 10; ++i) {
        bytes = sizes[i] * sizeof(float);
        if (off > (size_t)st.st_size || bytes > (size_t)st.st_size - off)
            return 1;
        a[i] = (const float *)(map + off);
        off += bytes;
    }
    q38_ple_forward_config config = {
        .hidden = 2560, .streams = 4, .heads = 16, .row_width = 160,
        .kernel = 4, .dilation = 3, .eps = 1e-6f,
    };
    float history[9u * 10240u] = {0};
    float contribution[4u * 10240u], after[4u * 10240u];
    float chunk_history[9u * 10240u] = {0};
    float chunk_contribution[4u * 10240u], chunk_after[4u * 10240u];
    char error[256];
    if (!q38_ple_forward_ref(&config, a[0], 4, a[1], a[2], a[3], a[4],
                             a[5], a[6], a[7], history, contribution, after,
                             error, sizeof(error)) ||
        !equal(contribution, a[8], 4u * 10240u) ||
        !equal(after, a[9], 4u * 10240u) ||
        !q38_ple_forward_ref(&config, a[0], 1, a[1], a[2], a[3], a[4],
                             a[5], a[6], a[7], chunk_history,
                             chunk_contribution, chunk_after, error,
                             sizeof(error)) ||
        !q38_ple_forward_ref(&config, a[0] + 10240, 3, a[1] + 2560,
                             a[2], a[3], a[4], a[5], a[6], a[7],
                             chunk_history, chunk_contribution + 10240,
                             chunk_after + 10240, error, sizeof(error)) ||
        !equal(chunk_after, after, 4u * 10240u) ||
        !equal(chunk_contribution, contribution, 4u * 10240u)) {
        fprintf(stderr, "PLE injection mismatch: %s\n", error);
        return 1;
    }
    munmap((void *)map, (size_t)st.st_size);
    puts("test_m4_ple_injection: independent hidden/contribution/after golden passed");
    return 0;
}
