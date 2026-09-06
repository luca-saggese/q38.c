#include "q38_prompt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} prompt_buf;

static bool prompt_reserve(prompt_buf *buf, size_t extra) {
    if (!buf || extra > SIZE_MAX - buf->len - 1) return false;
    const size_t needed = buf->len + extra + 1;
    if (needed <= buf->cap) return true;
    size_t cap = buf->cap ? buf->cap : 256;
    while (cap < needed) {
        if (cap > SIZE_MAX / 2) {
            cap = needed;
            break;
        }
        cap *= 2;
    }
    char *grown = realloc(buf->data, cap);
    if (!grown) return false;
    buf->data = grown;
    buf->cap = cap;
    return true;
}

static bool prompt_puts(prompt_buf *buf, const char *text) {
    const size_t len = text ? strlen(text) : 0;
    if (!prompt_reserve(buf, len)) return false;
    if (len) memcpy(buf->data + buf->len, text, len);
    buf->len += len;
    buf->data[buf->len] = '\0';
    return true;
}

static bool prompt_put_json_line(prompt_buf *buf, const char *name,
                                 const char *value) {
    if (!prompt_puts(buf, name) || !prompt_puts(buf, ": ")) return false;
    if (!prompt_puts(buf, value ? value : "")) return false;
    return prompt_puts(buf, "\n");
}

bool q38_prompt_render_chat(const q38_server_request *request,
                            char **text, size_t *text_len,
                            char *error, size_t error_len) {
    prompt_buf buf = {0};
    if (error && error_len) error[0] = '\0';
    if (!request || !text || !text_len)
        goto invalid;
    if (!prompt_puts(&buf, "<|im_start|>system\n") ||
        !prompt_puts(&buf,
                     "You are Qwen3.8. Follow the user's instructions.\n")) {
        goto oom;
    }
    if (request->tools.count) {
        if (!prompt_puts(&buf, "Available tools:\n")) goto oom;
        for (size_t i = 0; i < request->tools.count; ++i) {
            const q38_server_tool *tool = &request->tools.items[i];
            if (!prompt_put_json_line(&buf, tool->name ? tool->name : "",
                                      tool->parameters_json))
                goto oom;
        }
    }
    if (request->thinking)
        if (!prompt_puts(&buf, "Reasoning is enabled for this request.\n"))
            goto oom;
    if (!prompt_puts(&buf, "<|im_end|>\n")) goto oom;
    for (size_t i = 0; i < request->messages.count; ++i) {
        const q38_server_message *message = &request->messages.items[i];
        const char *role = message->role ? message->role : "user";
        if (!prompt_puts(&buf, "<|im_start|>") ||
            !prompt_puts(&buf, role) || !prompt_puts(&buf, "\n"))
            goto oom;
        if ((message->reasoning &&
             !prompt_puts(&buf, "<think>\n")) ||
            (message->reasoning && !prompt_puts(&buf, message->reasoning)) ||
            (message->reasoning && !prompt_puts(&buf, "\n</think>\n")))
            goto oom;
        if (message->content && !prompt_puts(&buf, message->content))
            goto oom;
        for (size_t j = 0; j < message->image_count; ++j)
            if (!prompt_puts(&buf, "\n<image>\n"))
                goto oom;
        for (size_t j = 0; j < message->tool_calls.count; ++j) {
            const q38_server_tool_call *call = &message->tool_calls.items[j];
            if (!prompt_puts(&buf, "\nTool call ") ||
                !prompt_puts(&buf, call->name ? call->name : "") ||
                !prompt_puts(&buf, "(") ||
                !prompt_puts(&buf, call->arguments_json ?
                                   call->arguments_json : "{}") ||
                !prompt_puts(&buf, ")\n"))
                goto oom;
        }
        if (message->tool_call_id &&
            (!prompt_puts(&buf, "\nTool result for ") ||
             !prompt_puts(&buf, message->tool_call_id) ||
             !prompt_puts(&buf, "\n")))
            goto oom;
        if (!prompt_puts(&buf, "<|im_end|>\n")) goto oom;
    }
    if (!prompt_puts(&buf, "<|im_start|>assistant\n")) goto oom;
    *text = buf.data;
    *text_len = buf.len;
    return true;
invalid:
    if (error && error_len)
        snprintf(error, error_len, "invalid Qwen prompt request");
    return false;
oom:
    free(buf.data);
    if (error && error_len)
        snprintf(error, error_len, "Qwen prompt allocation failed");
    return false;
}

bool q38_prompt_extract_tool_call(const char *text, size_t text_len,
                                  q38_server_tool_call *call,
                                  char *error, size_t error_len) {
    (void)text;
    (void)text_len;
    if (error && error_len) error[0] = '\0';
    if (!call) {
        if (error && error_len)
            snprintf(error, error_len, "tool call output is null");
        return false;
    }
    memset(call, 0, sizeof(*call));
    return false;
}

bool q38_prompt_is_reasoning_start(const char *text, size_t text_len) {
    return text && text_len >= 7 && !memcmp(text, "<think>", 7);
}

bool q38_prompt_is_reasoning_end(const char *text, size_t text_len) {
    return text && text_len >= 8 && !memcmp(text, "</think>", 8);
}
