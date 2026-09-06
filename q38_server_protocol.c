#include "q38_server_protocol.h"

#include "q38_json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool protocol_fail(char *error, size_t error_len,
                          const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static bool validate_json_object(const char *body, char *error,
                                 size_t error_len) {
    const char *cursor = body;
    if (!q38_json_skip_value(&cursor, error, error_len))
        return false;
    while (*cursor && isspace((unsigned char)*cursor)) cursor++;
    if (*cursor || body[0] != '{')
        return protocol_fail(error, error_len, "request body must be one JSON object");
    return true;
}

static char *duplicate_range(const char *value, size_t len) {
    char *copy = malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, value, len);
    copy[len] = '\0';
    return copy;
}

static bool append_message(q38_server_request *request, const char *role,
                           const char *content, const char *reasoning,
                           char *error, size_t error_len) {
    q38_server_message *grown;
    q38_server_message *message;
    if (!request || !role)
        return protocol_fail(error, error_len, "invalid message");
    grown = realloc(request->messages.items,
                    (request->messages.count + 1) * sizeof(*grown));
    if (!grown)
        return protocol_fail(error, error_len, "message allocation failed");
    request->messages.items = grown;
    message = &request->messages.items[request->messages.count++];
    memset(message, 0, sizeof(*message));
    message->role = duplicate_range(role, strlen(role));
    message->content = content ? duplicate_range(content, strlen(content)) : NULL;
    message->reasoning = reasoning ?
        duplicate_range(reasoning, strlen(reasoning)) : NULL;
    if (!message->role || (content && !message->content) ||
        (reasoning && !message->reasoning))
        return protocol_fail(error, error_len, "message field allocation failed");
    return true;
}

static bool append_tool(q38_server_request *request, const char *name,
                        const char *description, const char *parameters,
                        char *error, size_t error_len) {
    q38_server_tool *grown;
    q38_server_tool *tool;
    if (!request || !name || !name[0])
        return protocol_fail(error, error_len, "tool name is required");
    grown = realloc(request->tools.items,
                    (request->tools.count + 1) * sizeof(*grown));
    if (!grown)
        return protocol_fail(error, error_len, "tool allocation failed");
    request->tools.items = grown;
    tool = &request->tools.items[request->tools.count++];
    memset(tool, 0, sizeof(*tool));
    tool->name = duplicate_range(name, strlen(name));
    tool->description = description ?
        duplicate_range(description, strlen(description)) : NULL;
    tool->parameters_json = parameters ?
        duplicate_range(parameters, strlen(parameters)) : duplicate_range(
            "{\"type\":\"object\",\"properties\":{}}", 34);
    if (!tool->name || (description && !tool->description) ||
        !tool->parameters_json)
        return protocol_fail(error, error_len, "tool field allocation failed");
    return true;
}

static bool append_tool_call(q38_server_message *message, const char *id,
                             const char *name, const char *arguments,
                             char *error, size_t error_len) {
    q38_server_tool_call *grown;
    q38_server_tool_call *call;
    if (!message || !name || !name[0])
        return protocol_fail(error, error_len, "tool call name is required");
    grown = realloc(message->tool_calls.items,
                    (message->tool_calls.count + 1) * sizeof(*grown));
    if (!grown)
        return protocol_fail(error, error_len, "tool call allocation failed");
    message->tool_calls.items = grown;
    call = &message->tool_calls.items[message->tool_calls.count++];
    memset(call, 0, sizeof(*call));
    call->id = id ? duplicate_range(id, strlen(id)) : NULL;
    call->name = duplicate_range(name, strlen(name));
    call->arguments_json = arguments ?
        duplicate_range(arguments, strlen(arguments)) : duplicate_range("{}", 2);
    if ((id && !call->id) || !call->name || !call->arguments_json)
        return protocol_fail(error, error_len, "tool call allocation failed");
    return true;
}

static bool parse_tool_call_item(const char *raw, size_t raw_len, void *user,
                                 char *error, size_t error_len) {
    q38_server_message *message = user;
    char *item = duplicate_range(raw, raw_len);
    char *function = NULL;
    char *id = NULL;
    char *name = NULL;
    char *arguments = NULL;
    char *raw_arguments = NULL;
    bool ok = false;
    if (!item || !message) goto done;
    if (!q38_json_get_string(item, "id", &id, error, error_len) ||
        !q38_json_object_field(item, "function", &function,
                               error, error_len))
        goto done;
    if (function) {
        if (!q38_json_get_string(function, "name", &name,
                                 error, error_len) ||
            !q38_json_get_string(function, "arguments", &arguments,
                                 error, error_len))
            goto done;
        if (!arguments &&
            !q38_json_object_field(function, "arguments", &raw_arguments,
                                   error, error_len))
            goto done;
    } else {
        if (!q38_json_get_string(item, "name", &name, error, error_len) ||
            !q38_json_get_string(item, "arguments", &arguments,
                                 error, error_len))
            goto done;
        if (!arguments &&
            !q38_json_object_field(item, "arguments", &raw_arguments,
                                   error, error_len))
            goto done;
    }
    ok = name && append_tool_call(message, id, name,
                                   arguments ? arguments : raw_arguments,
                                   error, error_len);
done:
    free(item);
    free(function);
    free(id);
    free(name);
    free(arguments);
    free(raw_arguments);
    return ok;
}

static bool append_image(q38_server_message *message, const char *media_type,
                         const char *data, bool data_uri,
                         char *error, size_t error_len) {
    q38_server_image *grown;
    q38_server_image *image;
    if (!message || !data)
        return protocol_fail(error, error_len, "image data is required");
    grown = realloc(message->images,
                    (message->image_count + 1) * sizeof(*grown));
    if (!grown)
        return protocol_fail(error, error_len, "image allocation failed");
    message->images = grown;
    image = &message->images[message->image_count++];
    memset(image, 0, sizeof(*image));
    image->media_type = media_type ?
        duplicate_range(media_type, strlen(media_type)) : NULL;
    image->data = duplicate_range(data, strlen(data));
    image->data_uri = data_uri;
    if ((media_type && !image->media_type) || !image->data)
        return protocol_fail(error, error_len, "image field allocation failed");
    return true;
}

static bool raw_string(const char *raw, char **value,
                       char *error, size_t error_len) {
    const char *cursor = raw;
    if (!raw || !value) return protocol_fail(error, error_len, "invalid string");
    return q38_json_parse_string(&cursor, value, error, error_len);
}

static bool parse_image_url(q38_server_message *message, const char *raw,
                            char *error, size_t error_len) {
    char *url = NULL;
    char *media_type = NULL;
    const char *comma;
    bool data_uri = false;
    if (!raw || !message) return protocol_fail(error, error_len, "invalid image");
    if (raw[0] == '"') {
        if (!raw_string(raw, &url, error, error_len)) return false;
    } else if (!q38_json_get_string(raw, "url", &url, error, error_len)) {
        return false;
    }
    if (!url || !url[0]) {
        free(url);
        return protocol_fail(error, error_len, "image URL is empty");
    }
    if (!strncmp(url, "data:", 5)) {
        data_uri = true;
        comma = strchr(url, ',');
        if (!comma || comma == url + 5) {
            free(url);
            return protocol_fail(error, error_len, "invalid image data URI");
        }
        media_type = duplicate_range(url + 5, (size_t)(comma - url - 5));
        if (!media_type || !append_image(message, media_type, comma + 1,
                                          true, error, error_len)) {
            free(media_type);
            free(url);
            return false;
        }
    } else if (!append_image(message, NULL, url, data_uri, error, error_len)) {
        free(url);
        return false;
    }
    free(media_type);
    free(url);
    return true;
}

typedef struct {
    q38_server_message *message;
} content_context;

static bool parse_content_item(const char *raw, size_t raw_len, void *user,
                               char *error, size_t error_len) {
    content_context *context = user;
    char *type = NULL;
    char *text = NULL;
    char *image_url = NULL;
    char *tool_id = NULL;
    char *name = NULL;
    char *arguments = NULL;
    char *source = NULL;
    char *media_type = NULL;
    char *data = NULL;
    char *item = duplicate_range(raw, raw_len);
    bool ok = false;
    if (!item || !context || !context->message) goto done;
    if (!q38_json_get_string(item, "type", &type, error, error_len))
        goto done;
    if (!type || !strcmp(type, "text") || !strcmp(type, "input_text")) {
        if (!q38_json_get_string(item, "text", &text, error, error_len))
            goto done;
        if (text) {
            size_t old_len = context->message->content ?
                             strlen(context->message->content) : 0;
            const size_t add_len = strlen(text);
            char *grown = realloc(context->message->content,
                                  old_len + add_len + (old_len ? 1 : 0) + 1);
            if (!grown) goto done;
            context->message->content = grown;
            if (old_len) context->message->content[old_len++] = '\n';
            memcpy(context->message->content + old_len, text, add_len + 1);
        }
        ok = true;
    } else if (!strcmp(type, "image_url")) {
        if (!q38_json_object_field(item, "image_url", &image_url,
                                   error, error_len))
            goto done;
        ok = image_url && parse_image_url(context->message, image_url,
                                          error, error_len);
    } else if (!strcmp(type, "image")) {
        if (!q38_json_object_field(item, "source", &source,
                                   error, error_len))
            goto done;
        if (source &&
            (!q38_json_get_string(source, "media_type", &media_type,
                                  error, error_len) ||
             !q38_json_get_string(source, "data", &data, error, error_len)))
            goto done;
        ok = source && data && append_image(context->message, media_type, data,
                                            true, error, error_len);
    } else if (!strcmp(type, "tool_use")) {
        if (!q38_json_get_string(item, "id", &tool_id, error, error_len) ||
            !q38_json_get_string(item, "name", &name, error, error_len) ||
            !q38_json_object_field(item, "input", &arguments,
                                   error, error_len))
            goto done;
        ok = tool_id && name && arguments &&
             append_tool_call(context->message, tool_id, name, arguments,
                              error, error_len);
    } else if (!strcmp(type, "tool_result")) {
        if (!q38_json_get_string(item, "tool_use_id", &tool_id,
                                 error, error_len) ||
            !q38_json_get_string(item, "content", &text, error, error_len))
            goto done;
        context->message->tool_call_id = tool_id;
        tool_id = NULL;
        if (text) {
            context->message->content = duplicate_range(text, strlen(text));
            ok = context->message->content != NULL;
        } else {
            ok = true;
        }
    } else {
        ok = protocol_fail(error, error_len, "unsupported content block type");
    }
done:
    free(type);
    free(text);
    free(image_url);
    free(tool_id);
    free(name);
    free(arguments);
    free(source);
    free(media_type);
    free(data);
    free(item);
    return ok;
}

static bool parse_message_item(const char *raw, size_t raw_len, void *user,
                               char *error, size_t error_len) {
    q38_server_request *request = user;
    char *item = duplicate_range(raw, raw_len);
    char *role = NULL;
    char *content = NULL;
    char *reasoning = NULL;
    char *tool_id = NULL;
    char *tool_calls = NULL;
    q38_server_message *message;
    bool ok = false;
    if (!item || !request) goto done;
    if (!q38_json_get_string(item, "role", &role, error, error_len))
        goto done;
    if (!role) role = duplicate_range("user", 4);
    if (!role) goto done;
    if (!q38_json_get_string(item, "reasoning_content", &reasoning,
                             error, error_len) ||
        (reasoning == NULL &&
         !q38_json_get_string(item, "reasoning", &reasoning,
                              error, error_len)) ||
        !q38_json_get_string(item, "tool_call_id", &tool_id,
                             error, error_len) ||
        !q38_json_object_field(item, "content", &content,
                               error, error_len) ||
        !q38_json_object_field(item, "tool_calls", &tool_calls,
                               error, error_len))
        goto done;
    if (!append_message(request, role, NULL, reasoning, error, error_len))
        goto done;
    message = &request->messages.items[request->messages.count - 1];
    message->tool_call_id = tool_id;
    tool_id = NULL;
    if (content && content[0] == '"') {
        if (!raw_string(content, &message->content, error, error_len))
            goto done;
    } else if (content && !q38_json_raw_is_null(content)) {
        content_context context = {.message = message};
        if (!q38_json_array_each(content, parse_content_item, &context,
                                 error, error_len))
            goto done;
    }
    if (tool_calls && !q38_json_raw_is_null(tool_calls)) {
        if (!q38_json_array_each(tool_calls, parse_tool_call_item, message,
                                 error, error_len))
            goto done;
    }
    ok = true;
done:
    free(item);
    free(role);
    free(content);
    free(reasoning);
    free(tool_id);
    free(tool_calls);
    return ok;
}

static bool parse_messages_raw(q38_server_request *request, const char *raw,
                               char *error, size_t error_len) {
    if (!raw) return protocol_fail(error, error_len, "messages are required");
    return q38_json_array_each(raw, parse_message_item, request,
                               error, error_len);
}

static bool parse_tools_item(const char *raw, size_t raw_len, void *user,
                             char *error, size_t error_len) {
    q38_server_request *request = user;
    char *item = duplicate_range(raw, raw_len);
    char *type = NULL;
    char *function = NULL;
    char *name = NULL;
    char *description = NULL;
    char *parameters = NULL;
    bool ok = false;
    if (!item || !request) goto done;
    if (!q38_json_get_string(item, "type", &type, error, error_len))
        goto done;
    if (!type || !strcmp(type, "function")) {
        if (!q38_json_object_field(item, "function", &function,
                                   error, error_len))
            goto done;
        if (function) {
            if (!q38_json_get_string(function, "name", &name,
                                     error, error_len) ||
                !q38_json_get_string(function, "description", &description,
                                     error, error_len) ||
                !q38_json_object_field(function, "parameters", &parameters,
                                       error, error_len))
                goto done;
        } else {
            if (!q38_json_get_string(item, "name", &name,
                                     error, error_len) ||
                !q38_json_get_string(item, "description", &description,
                                     error, error_len) ||
                !q38_json_object_field(item, "parameters", &parameters,
                                       error, error_len))
                goto done;
        }
        ok = name && append_tool(request, name, description, parameters,
                                 error, error_len);
    } else if (!strcmp(type, "custom")) {
        if (!q38_json_get_string(item, "name", &name, error, error_len) ||
            !q38_json_get_string(item, "description", &description,
                                 error, error_len) ||
            !q38_json_object_field(item, "input_schema", &parameters,
                                   error, error_len))
            goto done;
        ok = name && append_tool(request, name, description, parameters,
                                 error, error_len);
    } else if (!strcmp(type, "namespace")) {
        char *tools = NULL;
        if (!q38_json_object_field(item, "tools", &tools, error, error_len))
            goto done;
        ok = tools && q38_json_array_each(tools, parse_tools_item, request,
                                          error, error_len);
        free(tools);
        tools = NULL;
    } else {
        ok = protocol_fail(error, error_len, "unsupported tool type");
    }
done:
    free(item);
    free(type);
    free(function);
    free(name);
    free(description);
    free(parameters);
    return ok;
}

static bool parse_tools_raw(q38_server_request *request, const char *raw,
                            char *error, size_t error_len) {
    if (!raw || q38_json_raw_is_null(raw)) return true;
    return q38_json_array_each(raw, parse_tools_item, request,
                               error, error_len);
}

typedef struct {
    q38_server_request *request;
} stop_context;

static bool parse_stop_item(const char *raw, size_t raw_len, void *user,
                            char *error, size_t error_len) {
    stop_context *context = user;
    char *item = duplicate_range(raw, raw_len);
    char *value = NULL;
    char **grown;
    bool ok = false;
    if (!item || !context || !context->request) goto done;
    if (!raw_string(item, &value, error, error_len)) goto done;
    grown = realloc(context->request->stop,
                    (context->request->stop_count + 1) * sizeof(*grown));
    if (!grown) {
        protocol_fail(error, error_len, "stop allocation failed");
        goto done;
    }
    context->request->stop = grown;
    context->request->stop[context->request->stop_count++] = value;
    value = NULL;
    ok = true;
done:
    free(item);
    free(value);
    return ok;
}

static bool parse_common_fields(const char *body, q38_server_request *request,
                                char *error, size_t error_len) {
    char *model = NULL;
    char *tools = NULL;
    char *stop = NULL;
    char *reasoning = NULL;
    char *thinking = NULL;
    char *stream_options = NULL;
    char *session_id = NULL;
    double number;
    bool boolean;
    if (!q38_json_get_string(body, "model", &model, error, error_len) ||
        !q38_json_object_field(body, "tools", &tools, error, error_len) ||
        !q38_json_object_field(body, "stop", &stop, error, error_len) ||
        !q38_json_get_string(body, "reasoning_effort", &reasoning,
                             error, error_len) ||
        !q38_json_object_field(body, "thinking", &thinking,
                               error, error_len) ||
        !q38_json_object_field(body, "stream_options", &stream_options,
                               error, error_len) ||
        !q38_json_get_string(body, "session_id", &session_id,
                             error, error_len))
        goto fail;
    request->model = model;
    model = NULL;
    request->reasoning_effort = reasoning;
    reasoning = NULL;
    request->session_id = session_id;
    session_id = NULL;
    if (thinking) {
        if (thinking[0] == '{') {
            if (!q38_json_get_bool(thinking, "enabled", &boolean,
                                   error, error_len))
                goto fail;
            request->thinking = boolean;
        } else if (thinking[0] == 't') {
            request->thinking = true;
        }
    }
    if (!parse_tools_raw(request, tools, error, error_len))
        goto fail;
    if (stop && !q38_json_raw_is_null(stop)) {
        if (stop[0] == '"') {
            char *value = NULL;
            if (!raw_string(stop, &value, error, error_len)) goto fail;
            request->stop = malloc(sizeof(*request->stop));
            if (!request->stop) {
                free(value);
                goto fail;
            }
            request->stop[0] = value;
            request->stop_count = 1;
        } else {
            stop_context context = {.request = request};
            if (!q38_json_array_each(stop, parse_stop_item, &context,
                                     error, error_len))
                goto fail;
        }
    }
    number = -1.0;
    if (!q38_json_get_number(body, "max_tokens", &number, error, error_len))
        goto fail;
    if (number >= 0.0)
        request->max_tokens = (int)number;
    number = -1.0;
    if (!q38_json_get_number(body, "max_completion_tokens", &number,
                             error, error_len))
        goto fail;
    if (number >= 0.0)
        request->max_tokens = (int)number;
    number = -1.0;
    if (!q38_json_get_number(body, "temperature", &number, error, error_len))
        goto fail;
    if (number >= 0.0)
        request->temperature = (float)number;
    number = -1.0;
    if (!q38_json_get_number(body, "top_p", &number, error, error_len))
        goto fail;
    if (number >= 0.0)
        request->top_p = (float)number;
    number = -1.0;
    if (!q38_json_get_number(body, "top_k", &number, error, error_len))
        goto fail;
    if (number >= 0.0)
        request->top_k = (int)number;
    number = -1.0;
    if (!q38_json_get_number(body, "min_p", &number, error, error_len))
        goto fail;
    if (number >= 0.0)
        request->min_p = (float)number;
    number = -1.0;
    if (!q38_json_get_number(body, "seed", &number, error, error_len))
        goto fail;
    if (number >= 0.0)
        request->seed = (uint64_t)number;
    boolean = false;
    if (!q38_json_get_bool(body, "stream", &boolean, error, error_len))
        goto fail;
    if (boolean)
        request->stream = boolean;
    boolean = false;
    if (stream_options &&
        q38_json_get_bool(stream_options, "include_usage", &boolean,
                          error, error_len))
        request->stream_include_usage = boolean;
    number = -1.0;
    if (!q38_json_get_number(body, "cache_read_tokens", &number,
                             error, error_len))
        goto fail;
    if (number >= 0.0) request->cache_read_tokens = (uint32_t)number;
    number = -1.0;
    if (!q38_json_get_number(body, "cache_write_tokens", &number,
                             error, error_len))
        goto fail;
    if (number >= 0.0) request->cache_write_tokens = (uint32_t)number;
    boolean = false;
    if (!q38_json_get_bool(body, "cache_restore", &boolean,
                           error, error_len))
        goto fail;
    request->cache_restore = boolean;
    boolean = false;
    if (!q38_json_get_bool(body, "cache_save", &boolean, error, error_len))
        goto fail;
    request->cache_save = boolean;
    free(stream_options);
    free(session_id);
    stream_options = NULL;
    if (!request->model) request->model = duplicate_range("", 0);
    free(tools);
    free(stop);
    return request->model != NULL;
fail:
    free(model);
    free(tools);
    free(stop);
    free(reasoning);
    free(thinking);
    free(stream_options);
    free(session_id);
    return false;
}

static bool parse_chat_or_anthropic(q38_server_endpoint endpoint,
                                    const char *body,
                                    q38_server_request *request,
                                    char *error, size_t error_len) {
    char *messages = NULL;
    char *prompt = NULL;
    if (!parse_common_fields(body, request, error, error_len))
        return false;
    if (endpoint == Q38_SERVER_ENDPOINT_MESSAGES) {
        request->api = Q38_SERVER_API_ANTHROPIC;
        if (!q38_json_object_field(body, "system", &prompt,
                                   error, error_len))
            goto fail;
        if (prompt) {
            char *system = NULL;
            if (prompt[0] == '"' && !raw_string(prompt, &system,
                                                 error, error_len)) goto fail;
            if (system && !append_message(request, "system", system, NULL,
                                          error, error_len)) {
                free(system);
                goto fail;
            }
            free(system);
        }
        if (!q38_json_object_field(body, "messages", &messages,
                                   error, error_len) ||
            !parse_messages_raw(request, messages, error, error_len))
            goto fail;
    } else {
        request->api = Q38_SERVER_API_OPENAI;
        request->legacy_completion = true;
        if (!q38_json_object_field(body, "messages", &messages,
                                   error, error_len) ||
            !parse_messages_raw(request, messages, error, error_len))
            goto fail;
    }
    request->has_image = false;
    for (size_t i = 0; i < request->messages.count; ++i)
        request->has_image |= request->messages.items[i].image_count != 0;
    free(messages);
    free(prompt);
    return true;
fail:
    free(messages);
    free(prompt);
    return false;
}

static bool parse_completion(const char *body, q38_server_request *request,
                             char *error, size_t error_len) {
    char *prompt = NULL;
    if (!parse_common_fields(body, request, error, error_len))
        return false;
    request->api = Q38_SERVER_API_OPENAI;
    if (!q38_json_get_string(body, "prompt", &prompt, error, error_len))
        goto fail;
    request->prompt = prompt;
    return true;
fail:
    free(prompt);
    return false;
}

static bool parse_responses_input_item(const char *raw, size_t raw_len,
                                       void *user, char *error,
                                       size_t error_len) {
    q38_server_request *request = user;
    char *item = duplicate_range(raw, raw_len);
    char *type = NULL;
    char *role = NULL;
    char *content = NULL;
    char *name = NULL;
    char *call_id = NULL;
    char *arguments = NULL;
    char *output = NULL;
    bool ok = false;
    if (!item || !request) goto done;
    if (!q38_json_get_string(item, "type", &type, error, error_len))
        goto done;
    if (!type || !strcmp(type, "message")) {
        if (!q38_json_get_string(item, "role", &role, error, error_len) ||
            !q38_json_object_field(item, "content", &content,
                                   error, error_len))
            goto done;
        if (!role) role = duplicate_range("user", 4);
        if (content && content[0] == '"') {
            if (!raw_string(content, &output, error, error_len)) goto done;
        } else if (content) {
            content_context context;
            if (!append_message(request, role, NULL, NULL, error, error_len))
                goto done;
            context.message = &request->messages.items[
                request->messages.count - 1];
            ok = q38_json_array_each(content, parse_content_item, &context,
                                     error, error_len);
            goto done;
        }
        ok = append_message(request, role, output, NULL, error, error_len);
    } else if (!strcmp(type, "input_text")) {
        if (!q38_json_get_string(item, "text", &output, error, error_len))
            goto done;
        ok = append_message(request, "user", output, NULL, error, error_len);
    } else if (!strcmp(type, "function_call")) {
        if (!q38_json_get_string(item, "call_id", &call_id, error, error_len) ||
            !q38_json_get_string(item, "name", &name, error, error_len) ||
            !q38_json_get_string(item, "arguments", &arguments,
                                 error, error_len) ||
            !append_message(request, "assistant", NULL, NULL,
                            error, error_len))
            goto done;
        ok = append_tool_call(&request->messages.items[
                                  request->messages.count - 1],
                              call_id, name, arguments, error, error_len);
    } else if (!strcmp(type, "function_call_output")) {
        if (!q38_json_get_string(item, "call_id", &call_id, error, error_len) ||
            !q38_json_get_string(item, "output", &output, error, error_len) ||
            !append_message(request, "tool", output, NULL, error, error_len))
            goto done;
        request->messages.items[request->messages.count - 1].tool_call_id =
            call_id;
        call_id = NULL;
        ok = true;
    } else if (!strcmp(type, "reasoning")) {
        if (!q38_json_get_string(item, "summary", &output, error, error_len))
            goto done;
        ok = append_message(request, "assistant", NULL, output,
                            error, error_len);
    } else {
        ok = protocol_fail(error, error_len, "unsupported Responses input item");
    }
done:
    free(item);
    free(type);
    free(role);
    free(content);
    free(name);
    free(call_id);
    free(arguments);
    free(output);
    return ok;
}

static bool parse_responses(const char *body, q38_server_request *request,
                            char *error, size_t error_len) {
    char *input = NULL;
    char *instructions = NULL;
    char *reasoning = NULL;
    if (!parse_common_fields(body, request, error, error_len))
        return false;
    request->api = Q38_SERVER_API_RESPONSES;
    if (!q38_json_object_field(body, "input", &input, error, error_len) ||
        !q38_json_object_field(body, "instructions", &instructions,
                               error, error_len) ||
        !q38_json_object_field(body, "reasoning", &reasoning,
                               error, error_len))
        goto fail;
    if (instructions) {
        char *text = NULL;
        if (!raw_string(instructions, &text, error, error_len) ||
            !append_message(request, "system", text, NULL, error, error_len)) {
            free(text);
            goto fail;
        }
        free(text);
    }
    if (!input) goto fail;
    if (input[0] == '"') {
        char *text = NULL;
        if (!raw_string(input, &text, error, error_len) ||
            !append_message(request, "user", text, NULL, error, error_len)) {
            free(text);
            goto fail;
        }
        free(text);
    } else if (!q38_json_array_each(input, parse_responses_input_item, request,
                                    error, error_len)) {
        goto fail;
    }
    if (reasoning && !q38_json_raw_is_null(reasoning)) {
        char *effort = NULL;
        if (reasoning[0] == '{' &&
            q38_json_get_string(reasoning, "effort", &effort,
                                error, error_len)) {
            request->reasoning_effort = effort;
            request->thinking = true;
        } else if (reasoning[0] == '{') {
            goto fail;
        }
    }
    free(input);
    free(instructions);
    free(reasoning);
    return true;
fail:
    free(input);
    free(instructions);
    free(reasoning);
    return false;
}

bool q38_server_parse_request(q38_server_endpoint endpoint,
                              const char *body,
                              q38_server_request *request,
                              char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!body || !request)
        return protocol_fail(error, error_len, "invalid request body");
    if (!validate_json_object(body, error, error_len)) return false;
    q38_server_request_init(request);
    if (endpoint == Q38_SERVER_ENDPOINT_COMPLETIONS)
        return parse_completion(body, request, error, error_len);
    if (endpoint == Q38_SERVER_ENDPOINT_RESPONSES)
        return parse_responses(body, request, error, error_len);
    return parse_chat_or_anthropic(endpoint, body, request, error, error_len);
}

const char *q38_server_endpoint_name(q38_server_endpoint endpoint) {
    switch (endpoint) {
    case Q38_SERVER_ENDPOINT_CHAT_COMPLETIONS: return "/v1/chat/completions";
    case Q38_SERVER_ENDPOINT_COMPLETIONS: return "/v1/completions";
    case Q38_SERVER_ENDPOINT_RESPONSES: return "/v1/responses";
    case Q38_SERVER_ENDPOINT_MESSAGES: return "/v1/messages";
    }
    return "unknown";
}
