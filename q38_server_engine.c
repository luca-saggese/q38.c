#include "q38_server_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q38_server_engine {
    const q38_server_engine_ops *ops;
    void *impl;
};

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
    request->temperature = 0.7f;
    request->top_p = 1.0f;
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
    free(request->session_id);
    memset(request, 0, sizeof(*request));
}

q38_server_engine *q38_server_engine_wrap(const q38_server_engine_ops *ops,
                                          void *impl,
                                          char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!ops || !ops->generate || !ops->model_name || !ops->destroy)
        goto invalid;
    q38_server_engine *engine = calloc(1, sizeof(*engine));
    if (!engine) goto oom;
    engine->ops = ops;
    engine->impl = impl;
    return engine;
invalid:
    if (error && error_len) snprintf(error, error_len, "invalid server engine");
    return NULL;
oom:
    if (error && error_len) snprintf(error, error_len,
                                     "server engine allocation failed");
    return NULL;
}

int q38_server_engine_generate(q38_server_engine *engine,
                               const q38_server_request *request,
                               q38_server_event_cb callback, void *callback_user,
                               q38_server_usage *usage,
                               char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!engine || !engine->ops || !engine->ops->generate) {
        if (error && error_len) snprintf(error, error_len,
                                         "server engine is unavailable");
        return -1;
    }
    return engine->ops->generate(engine->impl, request, callback, callback_user,
                                 usage, error, error_len);
}

const char *q38_server_engine_model_name(const q38_server_engine *engine) {
    if (!engine || !engine->ops || !engine->ops->model_name)
        return "qwen3.8-flash-next";
    return engine->ops->model_name(engine->impl);
}

bool q38_server_engine_is_mock(const q38_server_engine *engine) {
    return engine && engine->ops && engine->ops->mock;
}

void q38_server_engine_destroy(q38_server_engine *engine) {
    if (!engine) return;
    if (engine->ops && engine->ops->destroy)
        engine->ops->destroy(engine->impl);
    free(engine);
}
