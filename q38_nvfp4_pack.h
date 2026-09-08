#ifndef Q38_NVFP4_PACK_H
#define Q38_NVFP4_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q38_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q38_NVFP4_PACK_VERSION 1u
#define Q38_NVFP4_PACK_HEADER_SIZE 512u
#define Q38_NVFP4_LAYER_COUNT 48u
#define Q38_NVFP4_EXPERT_COUNT 512u
#define Q38_NVFP4_PROJECTION_COUNT 3u
#define Q38_NVFP4_COMPONENT_COUNT 4u
#define Q38_NVFP4_PACK_PATH_MAX 4096u

typedef enum {
    Q38_NVFP4_PROJECTION_GATE = 0,
    Q38_NVFP4_PROJECTION_UP = 1,
    Q38_NVFP4_PROJECTION_DOWN = 2,
} q38_nvfp4_projection;

typedef enum {
    Q38_NVFP4_COMPONENT_WEIGHT = 0,
    Q38_NVFP4_COMPONENT_WEIGHT_SCALE = 1,
    Q38_NVFP4_COMPONENT_WEIGHT_SCALE_2 = 2,
    Q38_NVFP4_COMPONENT_INPUT_SCALE = 3,
} q38_nvfp4_component;

typedef enum {
    Q38_NVFP4_REGION_BF16 = 0,
    Q38_NVFP4_REGION_WEIGHT = 1,
    Q38_NVFP4_REGION_WEIGHT_SCALE = 2,
    Q38_NVFP4_REGION_WEIGHT_SCALE_2 = 3,
    Q38_NVFP4_REGION_INPUT_SCALE = 4,
    Q38_NVFP4_REGION_COUNT = 5,
} q38_nvfp4_region;

typedef struct {
    const void *data;
    uint64_t bytes;
    uint32_t dtype;
    uint32_t quant_type;
    uint64_t rows;
    uint64_t cols;
    uint64_t source_offset;
    uint64_t pack_offset;
    uint32_t source_id;
} q38_nvfp4_view;

typedef struct {
    const void *data;
    uint64_t bytes;
    uint64_t pack_offset;
} q38_nvfp4_region_view;

typedef struct q38_nvfp4_pack q38_nvfp4_pack;

bool q38_nvfp4_pack_open(const char *pack_path, const char *source_root,
                         q38_nvfp4_pack **out, char *error, size_t error_len);
void q38_nvfp4_pack_close(q38_nvfp4_pack *pack);

bool q38_nvfp4_pack_get_expert_view(
    const q38_nvfp4_pack *pack, uint32_t layer, uint32_t projection,
    uint32_t expert, uint32_t component, q38_nvfp4_view *out,
    char *error, size_t error_len);

bool q38_nvfp4_pack_get_ple_view(const q38_nvfp4_pack *pack,
                                 uint32_t shard, q38_nvfp4_view *out,
                                 char *error, size_t error_len);

bool q38_nvfp4_pack_get_bf16_view(const q38_nvfp4_pack *pack, uint32_t index,
                                  q38_nvfp4_view *out, const char **name,
                                  uint32_t *ndim, uint64_t shape[4],
                                  char *error, size_t error_len);
bool q38_nvfp4_pack_get_bf16_name_length(
    const q38_nvfp4_pack *pack, uint32_t index, uint32_t *name_len);

bool q38_nvfp4_pack_get_bf16_count(const q38_nvfp4_pack *pack,
                                   uint32_t *count);
bool q38_nvfp4_pack_get_aux_view(
    const q38_nvfp4_pack *pack, uint32_t index, q38_nvfp4_view *out,
    const char **name, uint32_t *class_id, uint32_t *ndim,
    uint64_t shape[4], char *error, size_t error_len);
bool q38_nvfp4_pack_get_aux_count(const q38_nvfp4_pack *pack,
                                  uint32_t *count);
bool q38_nvfp4_pack_get_aux_name_length(
    const q38_nvfp4_pack *pack, uint32_t index, uint32_t *name_len);
bool q38_nvfp4_pack_get_ple_tensor_view(
    const q38_nvfp4_pack *pack, uint32_t index, q38_nvfp4_view *out,
    const char **name, uint32_t *shard_id, uint32_t *ndim,
    uint64_t shape[4], char *error, size_t error_len);
bool q38_nvfp4_pack_get_ple_count(const q38_nvfp4_pack *pack,
                                  uint32_t *count);
bool q38_nvfp4_pack_get_region_view(
    const q38_nvfp4_pack *pack, uint32_t region,
    q38_nvfp4_region_view *out, char *error, size_t error_len);
uint64_t q38_nvfp4_pack_main_resident_bytes(const q38_nvfp4_pack *pack);
uint64_t q38_nvfp4_pack_ple_bytes(const q38_nvfp4_pack *pack);
uint64_t q38_nvfp4_pack_file_bytes(const q38_nvfp4_pack *pack);
uint32_t q38_nvfp4_pack_storage_mode(const q38_nvfp4_pack *pack);
bool q38_nvfp4_pack_is_source_backed(const q38_nvfp4_pack *pack);
const char *q38_nvfp4_pack_source_model(const q38_nvfp4_pack *pack);
const char *q38_nvfp4_pack_source_revision(const q38_nvfp4_pack *pack);

#ifdef __cplusplus
}
#endif

#endif
