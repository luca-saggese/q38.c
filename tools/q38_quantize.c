/*
 * Small streaming Safetensors -> GGUF block quantizer.
 *
 * The block writers are the reviewed DS4 donor implementation.  This wrapper
 * keeps the source mmap-backed and emits one tensor into a caller-supplied
 * file, so the Python GGUF writer never holds a dequantized model mirror.
 */
#define _POSIX_C_SOURCE 200809L

#include "../to_be_deleted/gguf-tools/quants.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *message) {
    fprintf(stderr, "q38_quantize: %s\n", message);
    exit(1);
}

static uint64_t parse_u64(const char *text, const char *what) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || *end || value > UINT64_MAX) {
        fprintf(stderr, "q38_quantize: invalid %s: %s\n", what, text);
        exit(2);
    }
    return (uint64_t)value;
}

static ds4q_type parse_type(const char *text) {
    if (strcmp(text, "Q2_K") == 0) return DS4Q_TYPE_Q2_K;
    if (strcmp(text, "IQ2_XXS") == 0) return DS4Q_TYPE_IQ2_XXS;
    if (strcmp(text, "Q4_K") == 0) return DS4Q_TYPE_Q4_K;
    if (strcmp(text, "Q8_0") == 0) return DS4Q_TYPE_Q8_0;
    fprintf(stderr, "q38_quantize: unsupported output type: %s\n", text);
    exit(2);
}

static uint16_t load_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint64_t product(const uint64_t *shape, int count) {
    uint64_t value = 1;
    for (int i = 0; i < count; i++) {
        if (shape[i] && value > UINT64_MAX / shape[i]) die("shape overflow");
        value *= shape[i];
    }
    return value;
}

int main(int argc, char **argv) {
    const char *input = NULL, *output = NULL, *type_name = NULL;
    uint64_t offset = 0, bytes = 0, shape[8] = {0};
    int ndim = 0;
    bool transpose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i];
        else if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc) offset = parse_u64(argv[++i], "offset");
        else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) bytes = parse_u64(argv[++i], "bytes");
        else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) type_name = argv[++i];
        else if (strcmp(argv[i], "--shape") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                if (ndim == 8) die("too many dimensions");
                shape[ndim++] = parse_u64(argv[++i], "dimension");
            }
        } else if (strcmp(argv[i], "--transpose-last-two") == 0) {
            transpose = true;
        } else {
            die("usage: --input FILE --output FILE --offset N --bytes N --type TYPE --shape D...");
        }
    }
    if (!input || !output || !type_name || !ndim || !bytes) die("missing argument");
    if (transpose && ndim < 2) die("transpose requires two dimensions");
    ds4q_type type = parse_type(type_name);
    uint64_t elements = product(shape, ndim);
    if (elements > bytes / 2) die("source bytes are smaller than BF16 tensor");

    int fd = open(input, O_RDONLY);
    if (fd < 0) die("cannot open input shard");
    struct stat st;
    if (fstat(fd, &st) < 0 || offset > (uint64_t)st.st_size ||
        bytes > (uint64_t)st.st_size - offset) die("source range is outside shard");
    const uint8_t *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) die("cannot mmap input shard");

    uint64_t src_rows = shape[ndim - 1];
    uint64_t src_cols = 1;
    if (ndim >= 2) {
        src_rows = shape[ndim - 2];
        src_cols = shape[ndim - 1];
    }
    uint64_t outer = product(shape, ndim - (ndim >= 2 ? 2 : 1));
    uint64_t rows = transpose ? src_cols : src_rows;
    uint64_t cols = transpose ? src_rows : src_cols;
    if (ndim == 1) rows = 1, cols = shape[0], outer = 1;
    int64_t block = ds4q_block_size(type);
    if (!block || cols % (uint64_t)block) die("output rows are not block aligned");
    size_t row_size = ds4q_row_size(type, (int64_t)cols);
    if (!row_size) die("cannot calculate output row size");

    FILE *stream = fopen(output, "wb");
    if (!stream) die("cannot open output");
    float *row = malloc((size_t)cols * sizeof(*row));
    float *weights = NULL;
    uint8_t *encoded = malloc(row_size);
    if (!row || !encoded) die("out of memory");
    if (ds4q_requires_imatrix(type)) {
        weights = malloc((size_t)cols * sizeof(*weights));
        if (!weights) die("out of memory");
        for (uint64_t c = 0; c < cols; c++) weights[c] = 1.0f;
    }
    ds4q_quantize_init(type);

    for (uint64_t matrix = 0; matrix < outer; matrix++) {
        for (uint64_t r = 0; r < rows; r++) {
            for (uint64_t c = 0; c < cols; c++) {
                uint64_t source_index;
                if (transpose) {
                    source_index = (matrix * src_rows + c) * src_cols + r;
                } else {
                    source_index = (matrix * src_rows + r) * src_cols + c;
                }
                row[c] = ds4q_bf16_to_f32(load_u16(mapped + offset + source_index * 2));
            }
            if (ds4q_quantize_chunk(type, row, encoded, 0, 1,
                                    (int64_t)cols, weights) != row_size) {
                die("quantization failed");
            }
            if (fwrite(encoded, 1, row_size, stream) != row_size) die("write failed");
        }
    }
    free(weights);
    free(encoded);
    free(row);
    fclose(stream);
    munmap((void *)mapped, (size_t)st.st_size);
    close(fd);
    return 0;
}
