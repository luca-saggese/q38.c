#ifndef Q38_NVFP4_RUNTIME_H
#define Q38_NVFP4_RUNTIME_H

#include "q38_gguf.h"
#include "q38_nvfp4_pack.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q38_nvfp4_runtime_model q38_nvfp4_runtime_model;

bool q38_nvfp4_path_is_pack(const char *path);
bool q38_nvfp4_runtime_model_open(
    const char *pack_path, const char *source_root,
    q38_nvfp4_runtime_model **out, char *error, size_t error_len);
void q38_nvfp4_runtime_model_close(q38_nvfp4_runtime_model *runtime);
q38_gguf *q38_nvfp4_runtime_model_gguf(
    q38_nvfp4_runtime_model *runtime);
const q38_nvfp4_pack *q38_nvfp4_runtime_model_pack(
    const q38_nvfp4_runtime_model *runtime);
const q38_tensor *q38_nvfp4_runtime_model_ple_scale(
    const q38_nvfp4_runtime_model *runtime);

#ifdef __cplusplus
}
#endif

#endif
