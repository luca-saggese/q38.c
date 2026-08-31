#include "q38_golden.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q38_GOLDEN_MAGIC 0x314e4c47u /* GLN1 */

typedef struct {
    uint32_t magic;
    uint32_t version;
    q38_golden_meta meta;
    uint64_t data_bytes;
} q38_golden_header;

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

static bool dtype_size(q38_dtype dtype, size_t *size) {
    switch (dtype) {
    case Q38_DTYPE_F32:
    case Q38_DTYPE_U32:
    case Q38_DTYPE_I32:
        *size = 4;
        return true;
    case Q38_DTYPE_BF16:
        *size = 2;
        return true;
    default:
        return false;
    }
}

static uint64_t fnv1a(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t q38_golden_checksum(const q38_golden_meta *meta,
                             const void *data, size_t data_bytes) {
    q38_golden_meta empty;
    memset(&empty, 0, sizeof(empty));
    if (!meta) meta = &empty;
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = fnv1a(hash, meta->stage, sizeof(meta->stage));
    hash = fnv1a(hash, &meta->layer, sizeof(meta->layer));
    hash = fnv1a(hash, &meta->token_pos, sizeof(meta->token_pos));
    hash = fnv1a(hash, &meta->dtype, sizeof(meta->dtype));
    hash = fnv1a(hash, &meta->elements, sizeof(meta->elements));
    return fnv1a(hash, data, data_bytes);
}

bool q38_golden_write(const char *path, const q38_golden_meta *meta,
                      const void *data, size_t data_bytes,
                      char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    size_t item_size = 0;
    if (!path || !meta || (!data && data_bytes) ||
        !dtype_size(meta->dtype, &item_size) ||
        meta->elements > SIZE_MAX / item_size ||
        data_bytes != meta->elements * item_size) {
        set_error(error, error_len, "invalid golden vector");
        return false;
    }
    q38_golden_header header;
    memset(&header, 0, sizeof(header));
    header.magic = Q38_GOLDEN_MAGIC;
    header.version = Q38_GOLDEN_FORMAT_VERSION;
    header.meta = *meta;
    header.meta.checksum = q38_golden_checksum(&header.meta, data, data_bytes);
    header.data_bytes = data_bytes;

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        set_error(error, error_len, "cannot open golden vector for writing");
        return false;
    }
    bool ok = fwrite(&header, sizeof(header), 1, fp) == 1 &&
              (!data_bytes || fwrite(data, data_bytes, 1, fp) == 1);
    int close_status = fclose(fp);
    ok = ok && close_status == 0;
    if (!ok) {
        set_error(error, error_len, "cannot write golden vector");
    }
    return ok;
}

bool q38_golden_read(const char *path, q38_golden_vector *out,
                     char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!path || !out) {
        set_error(error, error_len, "invalid golden read arguments");
        return false;
    }
    memset(out, 0, sizeof(*out));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        set_error(error, error_len, "cannot open golden vector");
        return false;
    }
    q38_golden_header header;
    bool ok = fread(&header, sizeof(header), 1, fp) == 1;
    size_t item_size = 0;
    if (ok && (header.magic != Q38_GOLDEN_MAGIC ||
               header.version != Q38_GOLDEN_FORMAT_VERSION ||
               !dtype_size(header.meta.dtype, &item_size) ||
               header.meta.elements > SIZE_MAX / item_size ||
               header.data_bytes != header.meta.elements * item_size)) {
        ok = false;
        set_error(error, error_len, "invalid golden vector header");
    }
    if (ok && header.data_bytes) {
        out->data = malloc((size_t)header.data_bytes);
        ok = out->data && fread(out->data, (size_t)header.data_bytes, 1, fp) == 1;
    }
    if (ok && q38_golden_checksum(&header.meta, out->data,
                                  (size_t)header.data_bytes) != header.meta.checksum) {
        ok = false;
        set_error(error, error_len, "golden vector checksum mismatch");
    }
    if (ok && fgetc(fp) != EOF) {
        ok = false;
        set_error(error, error_len, "golden vector has trailing data");
    }
    fclose(fp);
    if (!ok) {
        free(out->data);
        memset(out, 0, sizeof(*out));
        if (!error || !error[0]) set_error(error, error_len, "cannot read golden vector");
        return false;
    }
    out->meta = header.meta;
    out->data_bytes = (size_t)header.data_bytes;
    return true;
}

void q38_golden_free(q38_golden_vector *vector) {
    if (!vector) return;
    free(vector->data);
    memset(vector, 0, sizeof(*vector));
}

bool q38_golden_equal(const q38_golden_vector *a,
                      const q38_golden_vector *b) {
    if (!a || !b || strncmp(a->meta.stage, b->meta.stage,
                             sizeof(a->meta.stage)) != 0 ||
        a->meta.layer != b->meta.layer ||
        a->meta.token_pos != b->meta.token_pos ||
        a->meta.dtype != b->meta.dtype ||
        a->meta.elements != b->meta.elements ||
        a->meta.checksum != b->meta.checksum ||
        a->data_bytes != b->data_bytes) return false;
    return a->data_bytes == 0 || memcmp(a->data, b->data, a->data_bytes) == 0;
}
