#include "q38_server_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q38_server_engine {
    char *model_name;
    bool mock;
};

static char *copy_string(const char *value) {
    const char *source = value ? value : "";
    const size_t len = strlen(source);
    char *copy = malloc(len + 1);
    if (copy) memcpy(copy, source, len + 1);
    return copy;
}

static void free_tool_calls(q38_server_tool_calls *calls) {
    if (!calls) return;
    for (size_t i = 0; i < calls->count; ++i) {
        free(calls->items[i].id);
        free(calls->items[i].name);
        free(calls->items[i].arguments_json);
    }
    free(calls->items);
    memset(calls, 0, sizeof(*calls));
}

void q38_server_request_init(q38_server_request *request) {
    if (!request) return;
    memset(request, 0, sizeof(*request));
    request->api = Q38_SERVER_API_OPENAI;
    request->max_tokens = 256;
    request->top_k = 0;
    request->temperature = 0.7f;
    request->top_p = 1.0f;
    request->min_p = 0.0f;
}

void q38_server_request_free(q38_server_request *request) {
    if (!request) return;
    free(request->model);
    free(request->prompt);
    for (size_t i = 0; i < request->messages.count; ++i) {
        q38_server_message *message = &request->messages.items[i];
        free(message->role);
        free(message->content);
        free(message->reasoning);
        free(message->tool_call_id);
        free_tool_calls(&message->tool_calls);
        for (size_t j = 0; j < message->image_count; ++j) {
            free(message->images[j].media_type);
            free(message->images[j].data);
        }
        free(message->images);
    }
    free(request->messages.items);
    for (size_t i = 0; i < request->tools.count; ++i) {
        free(request->tools.items[i].name);
        free(request->tools.items[i].description);
        free(request->tools.items[i].parameters_json);
    }
    free(request->tools.items);
    for (size_t i = 0; i < request->stop_count; ++i)
        free(request->stop[i]);
    free(request->stop);
    free(request->reasoning_effort);
    memset(request, 0, sizeof(*request));
}

static bool emit_event(q38_server_event_cb callback,
                       void *user, q38_server_event_kind kind,
                       const char *id, const char *name,
                       const char *arguments, const char *text,
                       char *error, size_t error_len) {
    if (!callback) return true;
    const q38_server_event event = {
        .kind = kind,
        .id = id,
        .name = name,
        .arguments_json = arguments,
        .text = text,
        .text_len = text ? strlen(text) : 0,
    };
    return callback(&event, user, error, error_len);
}

int q38_server_engine_generate(q38_server_engine *engine,
                               const q38_server_request *request,
                               q38_server_event_cb callback, void *callback_user,
                               q38_server_usage *usage,
                               char *error, size_t error_len) {
    static const char *const reasoning =
        "The Q38 mock engine is exercising the reasoning stream.";
    static const char *const answer =
        "This is a deterministic Q38 mock response.";
    if (error && error_len) error[0] = '\0';
    if (!engine || !request)
        goto invalid;
    if (usage) {
        usage->prompt_tokens = (uint32_t)(
            request->prompt ? strlen(request->prompt) / 4 : 0);
        usage->cached_tokens = request->cache_read_tokens;
        usage->completion_tokens = 0;
    }
    if (request->thinking &&
        !emit_event(callback, callback_user, Q38_SERVER_EVENT_REASONING,
                    NULL, NULL, NULL, reasoning, error, error_len))
        return -1;
    if (request->tools.count) {
        const char *tool_name = request->tools.items[0].name ?
                                request->tools.items[0].name : "tool";
        if (!emit_event(callback, callback_user, Q38_SERVER_EVENT_TOOL_CALL,
                        "call_q38_mock_1", tool_name, "{}", NULL,
                        error, error_len))
            return -1;
    } else if (!emit_event(callback, callback_user, Q38_SERVER_EVENT_TEXT,
                           NULL, NULL, NULL, answer, error, error_len)) {
        return -1;
    }
    if (usage) usage->completion_tokens = request->tools.count ? 1 : 8;
    if (!emit_event(callback, callback_user, Q38_SERVER_EVENT_DONE,
                    NULL, NULL, NULL, NULL, error, error_len))
        return -1;
    return 0;
invalid:
    if (error && error_len)
        snprintf(error, error_len, "invalid server engine request");
    return -1;
}

const char *q38_server_engine_model_name(const q38_server_engine *engine) {
    return engine && engine->model_name ? engine->model_name : "qwen3.8-flash-next";
}

bool q38_server_engine_is_mock(const q38_server_engine *engine) {
    return engine && engine->mock;
}

void q38_server_engine_destroy(q38_server_engine *engine) {
    if (!engine) return;
    free(engine->model_name);
    free(engine);
}

q38_server_engine *q38_server_mock_engine_create(const char *model_name,
                                                  char *error, size_t error_len) {
    q38_server_engine *engine = calloc(1, sizeof(*engine));
    if (error && error_len) error[0] = '\0';
    if (!engine) goto oom;
    engine->model_name = copy_string(model_name ? model_name :
                                      "qwen3.8-flash-next");
    if (!engine->model_name) {
        free(engine);
        goto oom;
    }
    engine->mock = true;
    return engine;
oom:
    if (error && error_len)
        snprintf(error, error_len, "mock engine allocation failed");
    return NULL;
}
