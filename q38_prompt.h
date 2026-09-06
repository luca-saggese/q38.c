#ifndef Q38_PROMPT_H
#define Q38_PROMPT_H

#include "q38_server_engine.h"

#include <stdbool.h>
#include <stddef.h>

bool q38_prompt_render_chat(const q38_server_request *request,
                            char **text, size_t *text_len,
                            char *error, size_t error_len);

bool q38_prompt_extract_tool_call(const char *text, size_t text_len,
                                  q38_server_tool_call *call,
                                  char *error, size_t error_len);

bool q38_prompt_is_reasoning_start(const char *text, size_t text_len);
bool q38_prompt_is_reasoning_end(const char *text, size_t text_len);

#endif
