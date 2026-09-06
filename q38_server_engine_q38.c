#include "q38_server_engine_q38.h"

#include "q38_prompt.h"
#include "q38_session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    q38_runtime runtime;
    q38_session session;
    uint32_t ctx_size;
    char *model_name;
} q38_engine_impl;

typedef struct {
    q38_server_event_cb callback;
    void *callback_user;
    char *error;
    size_t error_len;
    bool defer_output;
    char *output;
    size_t output_len;
    size_t output_cap;
    bool failed;
} token_context;

static bool engine_error(char *error, size_t error_len,
                         const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static void emit_piece(uint32_t token, const char *piece, size_t len,
                       void *user) {
    (void)token;
    token_context *context = user;
    if (!context || context->failed) return;
    if (context->defer_output) {
        if (len > SIZE_MAX - context->output_len - 1) {
            context->failed = true;
            return;
        }
        size_t needed = context->output_len + len + 1;
        if (needed > context->output_cap) {
            size_t cap = context->output_cap ? context->output_cap : 256;
            while (cap < needed) {
                if (cap > SIZE_MAX / 2) {
                    cap = needed;
                    break;
                }
                cap *= 2;
            }
            char *grown = realloc(context->output, cap);
            if (!grown) {
                context->failed = true;
                return;
            }
            context->output = grown;
            context->output_cap = cap;
        }
        memcpy(context->output + context->output_len, piece, len);
        context->output_len += len;
        context->output[context->output_len] = '\0';
        return;
    }
    q38_server_event event = {
        .kind = Q38_SERVER_EVENT_TEXT,
        .text = piece,
        .text_len = len,
    };
    if (!context->callback ||
        !context->callback(&event, context->callback_user,
                            context->error, context->error_len))
        context->failed = true;
}

static bool emit_text(q38_server_event_cb callback, void *user,
                      const char *text, size_t len, char *error,
                      size_t error_len) {
    if (!text || !len) return true;
    const q38_server_event event = {
        .kind = Q38_SERVER_EVENT_TEXT,
        .text = text,
        .text_len = len,
    };
    return !callback || callback(&event, user, error, error_len);
}

static bool flush_deferred_output(const q38_server_request *request,
                                  token_context *context,
                                  char *error, size_t error_len) {
    const char *text = context->output ? context->output : "";
    size_t text_len = context->output_len;
    if (request->thinking) {
        const char *start = strstr(text, "<think>");
        if (start) {
            start += strlen("<think>");
            const char *end = strstr(start, "</think>");
            if (end) {
                q38_server_event event = {
                    .kind = Q38_SERVER_EVENT_REASONING,
                    .text = start,
                    .text_len = (size_t)(end - start),
                };
                if (context->callback &&
                    !context->callback(&event, context->callback_user,
                                       error, error_len))
                    return false;
                text = end + strlen("</think>");
                text_len = context->output_len - (size_t)(text -
                                                           context->output);
            }
        }
    }
    if (request->tools.count) {
        q38_server_tool_call call = {0};
        if (q38_prompt_extract_tool_call(text, text_len, &call,
                                          error, error_len)) {
            call.id = strdup("call_q38_runtime_1");
            q38_server_event event = {
                .kind = Q38_SERVER_EVENT_TOOL_CALL,
                .id = call.id,
                .name = call.name,
                .arguments_json = call.arguments_json,
            };
            bool ok = !context->callback ||
                       context->callback(&event, context->callback_user,
                                         error, error_len);
            free(call.id);
            free(call.name);
            free(call.arguments_json);
            return ok;
        }
    }
    return emit_text(context->callback, context->callback_user, text, text_len,
                     error, error_len);
}

static int real_generate(void *opaque, const q38_server_request *request,
                         q38_server_event_cb callback, void *callback_user,
                         q38_server_usage *usage, char *error,
                         size_t error_len) {
    q38_engine_impl *engine = opaque;
    q38_token_batch prompt = {0};
    char *rendered = NULL;
    size_t rendered_len = 0;
    float *logits = NULL;
    uint32_t next_token = 0;
    size_t step_index = 0;
    if (error && error_len) error[0] = '\0';
    if (!engine || !request) {
        engine_error(error, error_len, "invalid Q38 engine request");
        return -1;
    }
    if (request->has_image) {
        engine_error(error, error_len, "Q38 vision backend unavailable");
        return -1;
    }
    if (request->cache_restore || request->cache_save) {
        engine_error(error, error_len,
                     "Q38 session cache backend unavailable");
        return -1;
    }
    token_context token_user = {
        .callback = callback,
        .callback_user = callback_user,
        .error = error,
        .error_len = error_len,
        .defer_output = request->thinking || request->tools.count != 0,
    };
    q38_session_clear_directional_steering_override(&engine->session);
    if (request->steering_override &&
        !q38_session_set_directional_steering(
            &engine->session, request->steering_ffn, request->steering_attn,
            error, error_len))
        goto fail;
    if (request->cancelled && request->cancelled(request->cancel_user)) {
        engine_error(error, error_len, "request cancelled");
        goto fail;
    }
    if (request->legacy_completion) {
        if (!request->prompt) {
            engine_error(error, error_len, "completion prompt is required");
            goto fail;
        }
        rendered = strdup(request->prompt);
        if (rendered) rendered_len = strlen(rendered);
    } else if (!q38_prompt_render_chat(request, &rendered, &rendered_len,
                                       error, error_len)) {
        goto fail;
    }
    if (!rendered) {
        engine_error(error, error_len, "Q38 prompt allocation failed");
        goto fail;
    }
    if (!q38_tokenizer_encode(&engine->runtime.tokenizer, rendered, false,
                              &prompt, error, error_len)) {
        free(rendered);
        rendered = NULL;
        goto fail;
    }
    free(rendered);
    rendered = NULL;
    if (prompt.token_count == 0) {
        engine_error(error, error_len,
                     "Q38 prompt tokenization produced no tokens");
        goto fail;
    }
    logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*logits));
    if (!logits) {
        engine_error(error, error_len, "Q38 logits allocation failed");
        goto fail;
    }
    if (!q38_session_prefill(&engine->session, prompt.tokens, prompt.token_count,
                             logits, Q38_DECODE_VOCAB_SIZE, &next_token, NULL,
                             NULL, NULL, &step_index, error,
                             error_len)) {
        goto fail;
    }
    if (usage) {
        usage->prompt_tokens = prompt.token_count;
        usage->cached_tokens = request->cache_read_tokens;
        usage->completion_tokens = 0;
    }
    size_t generated = 0;
    const size_t limit = request->max_tokens > 0 ?
                         (size_t)request->max_tokens : 1;
    while (generated < limit && next_token != q38_session_eos_token(
               &engine->session)) {
        if (request->cancelled && request->cancelled(request->cancel_user)) {
            engine_error(error, error_len, "request cancelled");
            goto fail;
        }
        if (!q38_session_stream_token(&engine->session, next_token,
                                      emit_piece, &token_user, error,
                                      error_len) || token_user.failed)
            goto fail;
        generated++;
        if (usage) usage->completion_tokens++;
        if (generated == limit) break;
        if (request->cancelled && request->cancelled(request->cancel_user)) {
            engine_error(error, error_len, "request cancelled");
            goto fail;
        }
        if (!q38_session_eval(&engine->session, next_token, logits,
                              Q38_DECODE_VOCAB_SIZE, &next_token, NULL,
                              Q38_DECODE_TRACE_GENERATED_CONSUME,
                              next_token, next_token, NULL, NULL, &step_index,
                              error, error_len))
            goto fail;
    }
    if (token_user.defer_output &&
        !flush_deferred_output(request, &token_user, error, error_len))
        goto fail;
    if (callback) {
        const q38_server_event done = {.kind = Q38_SERVER_EVENT_DONE};
        if (!callback(&done, callback_user, error, error_len))
            goto fail;
    }
    free(logits);
    q38_token_batch_free(&prompt);
    q38_session_clear_directional_steering_override(&engine->session);
    return 0;
fail:
    q38_session_clear_directional_steering_override(&engine->session);
    free(token_user.output);
    free(rendered);
    free(logits);
    q38_token_batch_free(&prompt);
    return -1;
}

static const char *real_model_name(void *opaque) {
    q38_engine_impl *engine = opaque;
    return engine && engine->model_name ? engine->model_name :
           "qwen3.8-flash-next";
}

static void real_destroy(void *opaque) {
    q38_engine_impl *engine = opaque;
    if (!engine) return;
    q38_session_destroy(&engine->session);
    q38_runtime_destroy(&engine->runtime);
    free(engine->model_name);
    free(engine);
}

q38_server_engine *q38_server_q38_engine_create_with_steering(
    const char *model_path, const char *tokenizer_path, uint32_t ctx_size,
    const char *steering_file, float steering_ffn, float steering_attn,
    char *error, size_t error_len) {
    static const q38_server_engine_ops ops = {
        .generate = real_generate,
        .model_name = real_model_name,
        .destroy = real_destroy,
        .mock = false,
    };
    q38_engine_impl *impl = calloc(1, sizeof(*impl));
    if (error && error_len) error[0] = '\0';
    if (!impl) {
        engine_error(error, error_len, "Q38 engine allocation failed");
        return NULL;
    }
    impl->ctx_size = ctx_size ? ctx_size : 8192;
    if (!q38_runtime_init(&impl->runtime, model_path, tokenizer_path,
                          error, error_len) ||
        (steering_file &&
         !q38_runtime_load_directional_steering(
             &impl->runtime, steering_file, steering_ffn, steering_attn,
             error, error_len)) ||
        !q38_session_create(&impl->session, &impl->runtime, impl->ctx_size,
                            error, error_len)) {
        q38_runtime_destroy(&impl->runtime);
        free(impl);
        return NULL;
    }
    impl->model_name = strdup("qwen3.8-flash-next");
    if (!impl->model_name) {
        real_destroy(impl);
        engine_error(error, error_len, "Q38 model name allocation failed");
        return NULL;
    }
    q38_server_engine *engine = q38_server_engine_wrap(&ops, impl, error,
                                                        error_len);
    if (!engine) real_destroy(impl);
    return engine;
}

q38_server_engine *q38_server_q38_engine_create(const char *model_path,
                                                const char *tokenizer_path,
                                                uint32_t ctx_size,
                                                char *error, size_t error_len) {
    return q38_server_q38_engine_create_with_steering(
        model_path, tokenizer_path, ctx_size, NULL, 0.0f, 0.0f, error,
        error_len);
}
