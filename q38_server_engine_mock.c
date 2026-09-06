#include "q38_server_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *model_name;
} mock_engine;

static char *copy_string(const char *value) {
    const char *source = value ? value : "";
    const size_t len = strlen(source);
    char *copy = malloc(len + 1);
    if (copy) memcpy(copy, source, len + 1);
    return copy;
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

static int mock_generate(void *opaque,
                               const q38_server_request *request,
                               q38_server_event_cb callback, void *callback_user,
                               q38_server_usage *usage,
                               char *error, size_t error_len) {
    static const char *const reasoning =
        "The Q38 mock engine is exercising the reasoning stream.";
    static const char *const answer =
        "This is a deterministic Q38 mock response.";
    if (error && error_len) error[0] = '\0';
    mock_engine *engine = opaque;
    if (!engine || !request)
        goto invalid;
    if (request->cancelled && request->cancelled(request->cancel_user)) {
        if (error && error_len) snprintf(error, error_len, "request cancelled");
        return -1;
    }
    if (usage) {
        usage->prompt_tokens = (uint32_t)(
            request->prompt ? strlen(request->prompt) / 4 : 0);
        usage->cached_tokens = request->cache_read_tokens;
        usage->completion_tokens = 0;
    }
    if (request->thinking &&
        (!request->cancelled ||
         !request->cancelled(request->cancel_user)) &&
        !emit_event(callback, callback_user, Q38_SERVER_EVENT_REASONING,
                    NULL, NULL, NULL, reasoning, error, error_len))
        return -1;
    if (request->tools.count) {
        const char *tool_name = request->tools.items[0].name ?
                                request->tools.items[0].name : "tool";
        if (request->cancelled && request->cancelled(request->cancel_user)) {
            if (error && error_len) snprintf(error, error_len,
                                             "request cancelled");
            return -1;
        }
        if (!emit_event(callback, callback_user, Q38_SERVER_EVENT_TOOL_CALL,
                        "call_q38_mock_1", tool_name, "{}", NULL,
                        error, error_len))
            return -1;
    } else if ((!request->cancelled ||
                !request->cancelled(request->cancel_user)) &&
               !emit_event(callback, callback_user, Q38_SERVER_EVENT_TEXT,
                           NULL, NULL, NULL, answer, error, error_len)) {
        return -1;
    }
    if (request->cancelled && request->cancelled(request->cancel_user)) {
        if (error && error_len) snprintf(error, error_len, "request cancelled");
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

static const char *mock_model_name(void *opaque) {
    mock_engine *engine = opaque;
    return engine && engine->model_name ? engine->model_name :
           "qwen3.8-flash-next";
}

static void mock_destroy(void *opaque) {
    mock_engine *engine = opaque;
    if (!engine) return;
    free(engine->model_name);
    free(engine);
}

q38_server_engine *q38_server_mock_engine_create(const char *model_name,
                                                  char *error, size_t error_len) {
    static const q38_server_engine_ops ops = {
        .generate = mock_generate,
        .model_name = mock_model_name,
        .destroy = mock_destroy,
        .mock = true,
    };
    mock_engine *impl = calloc(1, sizeof(*impl));
    if (error && error_len) error[0] = '\0';
    if (!impl) goto oom;
    impl->model_name = copy_string(model_name ? model_name :
                                   "qwen3.8-flash-next");
    if (!impl->model_name) {
        free(impl);
        goto oom;
    }
    q38_server_engine *engine = q38_server_engine_wrap(&ops, impl, error,
                                                        error_len);
    if (!engine) {
        mock_destroy(impl);
        return NULL;
    }
    return engine;
oom:
    if (error && error_len)
        snprintf(error, error_len, "mock engine allocation failed");
    return NULL;
}
