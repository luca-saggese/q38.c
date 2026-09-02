#include "q38_gguf.h"
#include "q38_quant.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    uint64_t row;
} fixture_spec;

static const fixture_spec fixtures[] = {
    {"model.language_model.embed_tokens.weight", 9419},
    {"model.language_model.layers.1.ple.key_proj.weight", 0},
    {"model.language_model.layers.2.mlp.experts.gate_up_proj", 75 * 1280},
};

static uint64_t fnv1a(const unsigned char *bytes, size_t count) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static float bf16(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static bool decode_row(const q38_gguf *model, const q38_tensor *tensor,
                      uint64_t row, float *values, size_t count) {
    uint64_t rows = 1;
    for (uint32_t i = 0; i + 1 < tensor->ndim; ++i) rows *= tensor->dim[i];
    const size_t cols = (size_t)tensor->dim[tensor->ndim - 1];
    if (row >= rows || count != cols || tensor->bytes % rows != 0) return false;
    const size_t row_bytes = (size_t)(tensor->bytes / rows);
    const unsigned char *raw = q38_gguf_tensor_data(model, tensor);
    if (!raw) return false;
    raw += row * row_bytes;
    if (tensor->type == 30) {
        for (size_t i = 0; i < cols; ++i) {
            uint16_t bits;
            memcpy(&bits, raw + 2 * i, sizeof(bits));
            values[i] = bf16(bits);
        }
        return true;
    }
    if (tensor->type == 8) {
        if (cols % 32 != 0 || row_bytes != (cols / 32) * 34) return false;
        for (size_t i = 0; i < cols; ++i) {
            const unsigned char *block = raw + (i / 32) * 34;
            uint16_t bits;
            memcpy(&bits, block, sizeof(bits));
            values[i] = q38_half_to_float(bits) *
                (float)((const int8_t *)(block + 2))[i % 32];
        }
        return true;
    }
    if (tensor->type == 10) {
        if (cols % Q38_QUANT_QK_K != 0) return false;
        return q38_quant_dequantize_row(
            tensor->type, raw, cols / Q38_QUANT_QK_K, values, cols, NULL, 0);
    }
    return false;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf output.json\n", argv[0]);
        return 2;
    }
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open: %s\n", error);
        return 1;
    }
    FILE *out = fopen(argv[2], "w");
    if (!out) {
        perror("output");
        q38_gguf_close(model);
        return 1;
    }
    fputs("{\"format\":\"q38-m6-gguf-dequant-fixtures-v1\",\"fixtures\":[",
          out);
    for (size_t f = 0; f < sizeof(fixtures) / sizeof(fixtures[0]); ++f) {
        const fixture_spec *spec = &fixtures[f];
        const q38_tensor *tensor = NULL;
        for (uint64_t i = 0; i < model->n_tensors; ++i) {
            q38_tensor *candidate = &model->tensors[i];
            if (strlen(spec->name) == candidate->name.len &&
                memcmp(spec->name, candidate->name.ptr,
                       candidate->name.len) == 0) {
                tensor = candidate;
                break;
            }
        }
        if (!tensor || tensor->ndim == 0) {
            fprintf(stderr, "missing fixture tensor: %s\n", spec->name);
            fclose(out);
            q38_gguf_close(model);
            return 1;
        }
        const size_t cols = (size_t)tensor->dim[tensor->ndim - 1];
        uint64_t rows = 1;
        for (uint32_t i = 0; i + 1 < tensor->ndim; ++i) rows *= tensor->dim[i];
        const size_t row_bytes = (size_t)(tensor->bytes / rows);
        float *values = calloc(cols, sizeof(*values));
        const unsigned char *payload = q38_gguf_tensor_data(model, tensor);
        if (!values || !payload || !decode_row(model, tensor, spec->row,
                                                values, cols)) {
            fprintf(stderr, "cannot decode fixture tensor: %s\n", spec->name);
            free(values);
            fclose(out);
            q38_gguf_close(model);
            return 1;
        }
        if (f) fputc(',', out);
        fprintf(out, "{\"tensor\":\"%s\",\"type\":%u,\"ndim\":%u,\"shape\":[",
                spec->name, tensor->type, tensor->ndim);
        for (uint32_t i = 0; i < tensor->ndim; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%" PRIu64, tensor->dim[i]);
        }
        fprintf(out, "],\"row\":%" PRIu64 ",\"row_bytes\":%zu,"
                    "\"raw_fnv1a\":\"%016" PRIx64 "\",\"values\":[",
                spec->row, row_bytes, fnv1a(payload + spec->row * row_bytes,
                                            row_bytes));
        for (size_t i = 0; i < cols; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%.9g", values[i]);
        }
        fputs("]}", out);
        free(values);
    }
    fputs("]}\n", out);
    fclose(out);
    q38_gguf_close(model);
    return 0;
}
