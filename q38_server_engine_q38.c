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
    token_context token_user = {
        .callback = callback,
        .callback_user = callback_user,
        .error = error,
        .error_len = error_len,
    };
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
    if (request->legacy_completion) {
        if (!request->prompt) {
            engine_error(error, error_len, "completion prompt is required");
            return -1;
        }
        rendered = strdup(request->prompt);
        if (rendered) rendered_len = strlen(rendered);
    } else if (!q38_prompt_render_chat(request, &rendered, &rendered_len,
                                       error, error_len)) {
        return -1;
    }
    if (!rendered) {
        engine_error(error, error_len, "Q38 prompt allocation failed");
        return -1;
    }
    if (!q38_tokenizer_encode(&engine->runtime.tokenizer, rendered, false,
                              &prompt, error, error_len)) {
        free(rendered);
        return -1;
    }
    free(rendered);
    if (prompt.token_count == 0) {
        q38_token_batch_free(&prompt);
        engine_error(error, error_len,
                     "Q38 prompt tokenization produced no tokens");
        return -1;
    }
    logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*logits));
    if (!logits) {
        q38_token_batch_free(&prompt);
        engine_error(error, error_len, "Q38 logits allocation failed");
        return -1;
    }
    if (!q38_session_prefill(&engine->session, prompt.tokens, prompt.token_count,
                             logits, Q38_DECODE_VOCAB_SIZE, &next_token, NULL,
                             NULL, NULL, &step_index, error,
                             error_len)) {
        free(logits);
        q38_token_batch_free(&prompt);
        return -1;
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
        if (!q38_session_stream_token(&engine->session, next_token,
                                      emit_piece, &token_user, error,
                                      error_len) || token_user.failed)
            goto fail;
        generated++;
        if (usage) usage->completion_tokens++;
        if (generated == limit) break;
        if (!q38_session_eval(&engine->session, next_token, logits,
                              Q38_DECODE_VOCAB_SIZE, &next_token, NULL,
                              Q38_DECODE_TRACE_GENERATED_CONSUME,
                              next_token, next_token, NULL, NULL, &step_index,
                              error, error_len))
            goto fail;
    }
    if (callback) {
        const q38_server_event done = {.kind = Q38_SERVER_EVENT_DONE};
        if (!callback(&done, callback_user, error, error_len))
            goto fail;
    }
    free(logits);
    q38_token_batch_free(&prompt);
    return 0;
fail:
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

q38_server_engine *q38_server_q38_engine_create(const char *model_path,
                                                const char *tokenizer_path,
                                                uint32_t ctx_size,
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
