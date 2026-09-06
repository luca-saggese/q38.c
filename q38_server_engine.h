#ifndef Q38_SERVER_ENGINE_H
#define Q38_SERVER_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct q38_server_engine q38_server_engine;

typedef enum {
    Q38_SERVER_API_OPENAI = 0,
    Q38_SERVER_API_RESPONSES,
    Q38_SERVER_API_ANTHROPIC,
} q38_server_api;

typedef struct {
    char *id;
    char *name;
    char *arguments_json;
} q38_server_tool_call;

typedef struct {
    q38_server_tool_call *items;
    size_t count;
} q38_server_tool_calls;

typedef struct {
    char *media_type;
    char *data;
    bool data_uri;
} q38_server_image;

typedef struct {
    char *role;
    char *content;
    char *reasoning;
    char *tool_call_id;
    q38_server_tool_calls tool_calls;
    q38_server_image *images;
    size_t image_count;
} q38_server_message;

typedef struct {
    q38_server_message *items;
    size_t count;
} q38_server_messages;

typedef struct {
    char *name;
    char *description;
    char *parameters_json;
} q38_server_tool;

typedef struct {
    q38_server_tool *items;
    size_t count;
} q38_server_tools;

typedef struct {
    q38_server_api api;
    char *model;
    char *prompt;
    q38_server_messages messages;
    q38_server_tools tools;
    char **stop;
    size_t stop_count;
    int max_tokens;
    int top_k;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool stream;
    bool stream_include_usage;
    bool thinking;
    char *reasoning_effort;
    uint32_t cache_read_tokens;
    uint32_t cache_write_tokens;
    bool has_image;
} q38_server_request;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t cached_tokens;
} q38_server_usage;

typedef enum {
    Q38_SERVER_EVENT_TEXT = 0,
    Q38_SERVER_EVENT_REASONING,
    Q38_SERVER_EVENT_TOOL_CALL,
    Q38_SERVER_EVENT_DONE,
} q38_server_event_kind;

typedef struct {
    q38_server_event_kind kind;
    const char *id;
    const char *name;
    const char *arguments_json;
    const char *text;
    size_t text_len;
} q38_server_event;

typedef bool (*q38_server_event_cb)(const q38_server_event *event,
                                    void *user, char *error, size_t error_len);

/* Compatibility callback for engines that only emit ordinary text tokens. */
typedef bool (*q38_server_token_cb)(const char *utf8, size_t len, void *user);

void q38_server_request_init(q38_server_request *request);
void q38_server_request_free(q38_server_request *request);

int q38_server_engine_generate(q38_server_engine *engine,
                               const q38_server_request *request,
                               q38_server_event_cb callback, void *callback_user,
                               q38_server_usage *usage,
                               char *error, size_t error_len);

const char *q38_server_engine_model_name(const q38_server_engine *engine);
bool q38_server_engine_is_mock(const q38_server_engine *engine);
void q38_server_engine_destroy(q38_server_engine *engine);

q38_server_engine *q38_server_mock_engine_create(const char *model_name,
                                                  char *error, size_t error_len);

#endif
