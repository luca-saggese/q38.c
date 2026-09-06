#ifndef Q38_SERVER_ENGINE_Q38_H
#define Q38_SERVER_ENGINE_Q38_H

#include "q38_server_engine.h"

#include <stddef.h>
#include <stdint.h>

q38_server_engine *q38_server_q38_engine_create(const char *model_path,
                                                const char *tokenizer_path,
                                                uint32_t ctx_size,
                                                char *error, size_t error_len);

q38_server_engine *q38_server_q38_engine_create_with_steering(
    const char *model_path, const char *tokenizer_path, uint32_t ctx_size,
    const char *steering_file, float steering_ffn, float steering_attn,
    char *error, size_t error_len);

#endif
