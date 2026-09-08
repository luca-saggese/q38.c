#include "q38_nvfp4_pack.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define Q38_NVFP4_REF_BYTES 32u
#define Q38_NVFP4_SOURCE_BYTES 56u
#define Q38_NVFP4_PLE_BYTES 72u
#define Q38_NVFP4_BF16_BYTES 80u
#define Q38_NVFP4_AUX_BYTES 88u
#define Q38_NVFP4_INVALID_OFFSET UINT64_MAX

static const unsigned char q38_nvfp4_magic[24] = {
    'Q', '3', '8', '_', 'N', 'V', 'F', 'P', '4', '_', 'P', 'A',
    'C', 'K', '_', 'V', '1',
};

#pragma pack(push, 1)
typedef struct {
    unsigned char magic[24];
    uint32_t version;
    uint32_t storage_mode;
    uint32_t flags;
    uint32_t source_count;
    uint32_t bf16_count;
    uint32_t aux_count;
    uint32_t ple_count;
    uint32_t reserved;
    uint64_t source_table_offset;
    uint64_t bf16_table_offset;
    uint64_t aux_table_offset;
    uint64_t expert_table_offset;
    uint64_t ple_table_offset;
    uint64_t string_table_offset;
    uint64_t string_table_bytes;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t main_resident_bytes;
    uint64_t ple_bytes;
    uint64_t metadata_bytes;
    uint64_t source_model_offset;
    uint64_t source_model_len;
    uint64_t source_revision_offset;
    uint64_t source_revision_len;
    uint64_t producer_offset;
    uint64_t producer_len;
    uint64_t architecture_offset;
    uint64_t architecture_len;
    unsigned char manifest_digest[32];
} q38_nvfp4_disk_header;

typedef struct {
    uint64_t path_offset;
    uint32_t path_len;
    uint32_t flags;
    uint64_t file_bytes;
    unsigned char sha256[32];
} q38_nvfp4_disk_source;

typedef struct {
    uint32_t source_id;
    uint32_t flags;
    uint64_t source_offset;
    uint64_t pack_offset;
    uint64_t bytes;
} q38_nvfp4_disk_ref;

typedef struct {
    uint64_t name_offset;
    uint32_t name_len;
    uint32_t source_id;
    uint32_t dtype;
    uint32_t ndim;
    uint64_t source_offset;
    uint64_t pack_offset;
    uint64_t bytes;
    uint64_t shape[4];
} q38_nvfp4_disk_bf16;

typedef struct {
    uint32_t shard_id;
    uint32_t source_id;
    uint32_t dtype;
    uint32_t ndim;
    uint64_t source_offset;
    uint64_t pack_offset;
    uint64_t bytes;
    uint64_t shape[4];
} q38_nvfp4_disk_ple;
#pragma pack(pop)

typedef struct {
    int fd;
    void *map;
    uint64_t bytes;
    char path[Q38_NVFP4_PACK_PATH_MAX];
} q38_nvfp4_source;

struct q38_nvfp4_pack {
    int fd;
    void *map;
    uint64_t bytes;
    q38_nvfp4_disk_header header;
    q38_nvfp4_source *sources;
    char *source_model;
    char *source_revision;
};

static bool set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static bool set_errorf(char *error, size_t error_len, const char *format,
                       uint64_t a, uint64_t b) {
    if (error && error_len) snprintf(error, error_len, format, a, b);
    return false;
}

static bool range_ok(uint64_t file_bytes, uint64_t offset, uint64_t bytes) {
    return offset <= file_bytes && bytes <= file_bytes - offset;
}

static bool table_ok(const q38_nvfp4_pack *pack, uint64_t offset,
                     uint64_t count, uint64_t stride) {
    if (count && stride > UINT64_MAX / count) return false;
    return range_ok(pack->bytes, offset, count * stride);
}

static const char *pack_string(const q38_nvfp4_pack *pack, uint64_t offset,
                               uint64_t length) {
    if (!range_ok(pack->header.string_table_bytes, offset, length))
        return NULL;
    return (const char *)pack->map + pack->header.string_table_offset + offset;
}

static bool copy_pack_string(const q38_nvfp4_pack *pack, uint64_t offset,
                             uint64_t length, char **out) {
    const char *value = pack_string(pack, offset, length);
    if (!value || length > SIZE_MAX - 1) return false;
    char *copy = malloc((size_t)length + 1);
    if (!copy) return false;
    memcpy(copy, value, (size_t)length);
    copy[length] = '\0';
    *out = copy;
    return true;
}

static bool join_source_path(const char *root, const char *relative,
                             char *out, size_t out_len) {
    if (!root || !root[0] || relative[0] == '/') {
        if (snprintf(out, out_len, "%s", relative) >= (int)out_len)
            return false;
    } else if (snprintf(out, out_len, "%s/%s", root, relative) >=
               (int)out_len) {
        return false;
    }
    return true;
}

static bool open_source(q38_nvfp4_source *source, const char *path,
                        uint64_t expected_bytes, char *error,
                        size_t error_len) {
    struct stat st;
    source->fd = open(path, O_RDONLY);
    if (source->fd < 0)
        return set_errorf(error, error_len, "cannot open NVFP4 source %"
                          PRIu64 " (errno=%" PRIu64 ")",
                          (uint64_t)errno, expected_bytes);
    if (fstat(source->fd, &st) != 0 ||
        st.st_size < 0 || (uint64_t)st.st_size != expected_bytes) {
        close(source->fd);
        source->fd = -1;
        return set_errorf(error, error_len,
                          "NVFP4 source size mismatch (%" PRIu64
                          " != %" PRIu64 ")",
                          (uint64_t)(st.st_size < 0 ? 0 : st.st_size),
                          expected_bytes);
    }
    source->bytes = expected_bytes;
    source->map = mmap(NULL, (size_t)expected_bytes, PROT_READ, MAP_PRIVATE,
                       source->fd, 0);
    if (source->map == MAP_FAILED) {
        close(source->fd);
        source->fd = -1;
        source->map = NULL;
        return set_error(error, error_len, "cannot map NVFP4 source file");
    }
    return true;
}

static bool resolve_ref(const q38_nvfp4_pack *pack,
                        const q38_nvfp4_disk_ref *ref,
                        q38_nvfp4_view *out, char *error, size_t error_len) {
    if (!ref || !out || ref->source_id >= pack->header.source_count)
        return set_error(error, error_len, "invalid NVFP4 source reference");
    const uint8_t *base = NULL;
    uint64_t file_bytes = 0;
    if (ref->pack_offset != Q38_NVFP4_INVALID_OFFSET) {
        if (!range_ok(pack->bytes, ref->pack_offset, ref->bytes))
            return set_error(error, error_len, "NVFP4 pack reference out of range");
        base = (const uint8_t *)pack->map;
        file_bytes = pack->bytes;
        (void)file_bytes;
    } else {
        const q38_nvfp4_source *source = &pack->sources[ref->source_id];
        if (!range_ok(source->bytes, ref->source_offset, ref->bytes))
            return set_error(error, error_len,
                             "NVFP4 source reference out of range");
        base = (const uint8_t *)source->map;
        file_bytes = source->bytes;
        (void)file_bytes;
    }
    out->data = base + (ref->pack_offset != Q38_NVFP4_INVALID_OFFSET
                            ? ref->pack_offset
                            : ref->source_offset);
    out->bytes = ref->bytes;
    out->dtype = 0;
    out->quant_type = 0;
    out->rows = 0;
    out->cols = 0;
    out->source_offset = ref->source_offset;
    out->pack_offset = ref->pack_offset;
    out->source_id = ref->source_id;
    return true;
}

bool q38_nvfp4_pack_open(const char *pack_path, const char *source_root,
                         q38_nvfp4_pack **out, char *error,
                         size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!pack_path || !out)
        return set_error(error, error_len, "invalid NVFP4 pack arguments");
    *out = NULL;
    q38_nvfp4_pack *pack = calloc(1, sizeof(*pack));
    if (!pack) return set_error(error, error_len, "NVFP4 pack allocation failed");
    pack->fd = open(pack_path, O_RDONLY);
    if (pack->fd < 0) {
        free(pack);
        return set_error(error, error_len, "cannot open NVFP4 pack");
    }
    struct stat st;
    if (fstat(pack->fd, &st) != 0 || st.st_size < Q38_NVFP4_PACK_HEADER_SIZE) {
        close(pack->fd);
        free(pack);
        return set_error(error, error_len, "invalid NVFP4 pack size");
    }
    pack->bytes = (uint64_t)st.st_size;
    pack->map = mmap(NULL, (size_t)pack->bytes, PROT_READ, MAP_PRIVATE,
                     pack->fd, 0);
    if (pack->map == MAP_FAILED) {
        close(pack->fd);
        free(pack);
        return set_error(error, error_len, "cannot map NVFP4 pack");
    }
    memcpy(&pack->header, pack->map, sizeof(pack->header));
    if (memcmp(pack->header.magic, q38_nvfp4_magic,
               sizeof(q38_nvfp4_magic)) != 0 ||
        pack->header.version != Q38_NVFP4_PACK_VERSION ||
        pack->header.reserved != 0) {
        q38_nvfp4_pack_close(pack);
        return set_error(error, error_len, "unsupported NVFP4 pack header");
    }
    if (!table_ok(pack, pack->header.source_table_offset,
                  pack->header.source_count, Q38_NVFP4_SOURCE_BYTES) ||
        !table_ok(pack, pack->header.bf16_table_offset,
                  pack->header.bf16_count, Q38_NVFP4_BF16_BYTES) ||
        !table_ok(pack, pack->header.aux_table_offset,
                  pack->header.aux_count, Q38_NVFP4_AUX_BYTES) ||
        !table_ok(pack, pack->header.expert_table_offset,
                  (uint64_t)Q38_NVFP4_LAYER_COUNT *
                      Q38_NVFP4_PROJECTION_COUNT *
                      Q38_NVFP4_EXPERT_COUNT *
                      Q38_NVFP4_COMPONENT_COUNT,
                  Q38_NVFP4_REF_BYTES) ||
        !table_ok(pack, pack->header.ple_table_offset,
                  pack->header.ple_count, Q38_NVFP4_PLE_BYTES) ||
        !range_ok(pack->bytes, pack->header.string_table_offset,
                  pack->header.string_table_bytes) ||
        !range_ok(pack->bytes, pack->header.data_offset,
                  pack->header.data_bytes)) {
        q38_nvfp4_pack_close(pack);
        return set_error(error, error_len, "NVFP4 pack table out of range");
    }
    if (!copy_pack_string(pack, pack->header.source_model_offset,
                          pack->header.source_model_len,
                          &pack->source_model) ||
        !copy_pack_string(pack, pack->header.source_revision_offset,
                          pack->header.source_revision_len,
                          &pack->source_revision)) {
        q38_nvfp4_pack_close(pack);
        return set_error(error, error_len, "NVFP4 pack metadata string failure");
    }
    pack->sources = calloc(pack->header.source_count, sizeof(*pack->sources));
    if (!pack->sources) {
        q38_nvfp4_pack_close(pack);
        return set_error(error, error_len, "NVFP4 source table allocation failed");
    }
    for (uint32_t i = 0; i < pack->header.source_count; ++i)
        pack->sources[i].fd = -1;
    const q38_nvfp4_disk_source *records =
        (const q38_nvfp4_disk_source *)((const uint8_t *)pack->map +
                                        pack->header.source_table_offset);
    for (uint32_t i = 0; i < pack->header.source_count; ++i) {
        const char *name = pack_string(pack, records[i].path_offset,
                                       records[i].path_len);
        if (!name || records[i].path_len >= Q38_NVFP4_PACK_PATH_MAX) {
            q38_nvfp4_pack_close(pack);
            return set_error(error, error_len, "invalid NVFP4 source path");
        }
        char path[Q38_NVFP4_PACK_PATH_MAX];
        memcpy(path, name, records[i].path_len);
        path[records[i].path_len] = '\0';
        if (!join_source_path(source_root, path, pack->sources[i].path,
                              sizeof(pack->sources[i].path))) {
            q38_nvfp4_pack_close(pack);
            return set_error(error, error_len, "NVFP4 source path too long");
        }
        if (!open_source(&pack->sources[i], pack->sources[i].path,
                         records[i].file_bytes, error, error_len)) {
            q38_nvfp4_pack_close(pack);
            return false;
        }
    }
    *out = pack;
    return true;
}

void q38_nvfp4_pack_close(q38_nvfp4_pack *pack) {
    if (!pack) return;
    if (pack->sources) {
        for (uint32_t i = 0; i < pack->header.source_count; ++i) {
            if (pack->sources[i].map && pack->sources[i].map != MAP_FAILED)
                munmap(pack->sources[i].map, (size_t)pack->sources[i].bytes);
            if (pack->sources[i].fd >= 0) close(pack->sources[i].fd);
        }
    }
    free(pack->sources);
    free(pack->source_model);
    free(pack->source_revision);
    if (pack->map && pack->map != MAP_FAILED)
        munmap(pack->map, (size_t)pack->bytes);
    if (pack->fd >= 0) close(pack->fd);
    free(pack);
}

bool q38_nvfp4_pack_get_expert_view(
    const q38_nvfp4_pack *pack, uint32_t layer, uint32_t projection,
    uint32_t expert, uint32_t component, q38_nvfp4_view *out,
    char *error, size_t error_len) {
    if (!pack || !out || layer >= Q38_NVFP4_LAYER_COUNT ||
        projection >= Q38_NVFP4_PROJECTION_COUNT ||
        expert >= Q38_NVFP4_EXPERT_COUNT ||
        component >= Q38_NVFP4_COMPONENT_COUNT)
        return set_error(error, error_len, "invalid NVFP4 expert coordinates");
    uint64_t slot = ((uint64_t)layer * Q38_NVFP4_PROJECTION_COUNT +
                     projection) *
                        Q38_NVFP4_EXPERT_COUNT +
                    expert;
    uint64_t index = slot * Q38_NVFP4_COMPONENT_COUNT + component;
    const q38_nvfp4_disk_ref *ref =
        (const q38_nvfp4_disk_ref *)((const uint8_t *)pack->map +
                                     pack->header.expert_table_offset +
                                     index * Q38_NVFP4_REF_BYTES);
    if (!resolve_ref(pack, ref, out, error, error_len)) return false;
    out->dtype = component == Q38_NVFP4_COMPONENT_WEIGHT
        ? 2
        : component == Q38_NVFP4_COMPONENT_WEIGHT_SCALE ? 3 : 4;
    out->quant_type = Q38_QUANT_NVIDIA_NVFP4;
    if (component == Q38_NVFP4_COMPONENT_WEIGHT) {
        out->rows = projection == Q38_NVFP4_PROJECTION_DOWN ? 2560 : 640;
        out->cols = projection == Q38_NVFP4_PROJECTION_DOWN ? 320 : 1280;
    } else if (component == Q38_NVFP4_COMPONENT_WEIGHT_SCALE) {
        out->rows = projection == Q38_NVFP4_PROJECTION_DOWN ? 2560 : 640;
        out->cols = projection == Q38_NVFP4_PROJECTION_DOWN ? 40 : 160;
    } else {
        out->rows = 1;
        out->cols = 1;
    }
    return true;
}

bool q38_nvfp4_pack_get_ple_view(const q38_nvfp4_pack *pack,
                                 uint32_t shard, q38_nvfp4_view *out,
                                 char *error, size_t error_len) {
    if (!pack || !out) return set_error(error, error_len, "invalid PLE view");
    const q38_nvfp4_disk_ple *records =
        (const q38_nvfp4_disk_ple *)((const uint8_t *)pack->map +
                                     pack->header.ple_table_offset);
    for (uint32_t i = 0; i < pack->header.ple_count; ++i) {
        if (records[i].shard_id != shard) continue;
        q38_nvfp4_disk_ref ref = {
            records[i].source_id, 0, records[i].source_offset,
            records[i].pack_offset, records[i].bytes,
        };
        if (!resolve_ref(pack, &ref, out, error, error_len)) return false;
        out->dtype = records[i].dtype;
        out->quant_type = 0;
        out->rows = records[i].shape[0];
        out->cols = records[i].shape[1];
        return true;
    }
    return set_error(error, error_len, "PLE shard is not present in pack");
}

bool q38_nvfp4_pack_get_bf16_view(const q38_nvfp4_pack *pack, uint32_t index,
                                  q38_nvfp4_view *out, const char **name,
                                  uint32_t *ndim, uint64_t shape[4],
                                  char *error, size_t error_len) {
    if (!pack || !out || index >= pack->header.bf16_count)
        return set_error(error, error_len, "invalid BF16 tensor index");
    const q38_nvfp4_disk_bf16 *record =
        (const q38_nvfp4_disk_bf16 *)((const uint8_t *)pack->map +
                                      pack->header.bf16_table_offset) +
        index;
    q38_nvfp4_disk_ref ref = {
        record->source_id, 0, record->source_offset,
        record->pack_offset, record->bytes,
    };
    if (!resolve_ref(pack, &ref, out, error, error_len)) return false;
    out->dtype = record->dtype;
    out->quant_type = 0;
    out->rows = record->shape[0];
    out->cols = record->shape[1];
    if (name) {
        *name = pack_string(pack, record->name_offset, record->name_len);
        if (!*name)
            return set_error(error, error_len, "invalid BF16 tensor name");
    }
    if (ndim) *ndim = record->ndim;
    if (shape) memcpy(shape, record->shape, sizeof(record->shape));
    return true;
}

bool q38_nvfp4_pack_get_bf16_count(const q38_nvfp4_pack *pack,
                                   uint32_t *count) {
    if (!pack || !count) return false;
    *count = pack->header.bf16_count;
    return true;
}

uint64_t q38_nvfp4_pack_main_resident_bytes(const q38_nvfp4_pack *pack) {
    return pack ? pack->header.main_resident_bytes : 0;
}

uint64_t q38_nvfp4_pack_ple_bytes(const q38_nvfp4_pack *pack) {
    return pack ? pack->header.ple_bytes : 0;
}

uint64_t q38_nvfp4_pack_file_bytes(const q38_nvfp4_pack *pack) {
    return pack ? pack->bytes : 0;
}

uint32_t q38_nvfp4_pack_storage_mode(const q38_nvfp4_pack *pack) {
    return pack ? pack->header.storage_mode : 0;
}

bool q38_nvfp4_pack_is_source_backed(const q38_nvfp4_pack *pack) {
    return q38_nvfp4_pack_storage_mode(pack) == 1;
}

const char *q38_nvfp4_pack_source_model(const q38_nvfp4_pack *pack) {
    return pack ? pack->source_model : NULL;
}

const char *q38_nvfp4_pack_source_revision(const q38_nvfp4_pack *pack) {
    return pack ? pack->source_revision : NULL;
}
