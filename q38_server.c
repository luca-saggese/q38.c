#include "q38_server.h"

#include "q38_json.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

struct q38_server {
    q38_server_engine *engine;
    bool owns_engine;
    pthread_mutex_t inference_mutex;
    uint64_t next_request_id;
    char *model_json;
    q38_kvstore kvstore;
    bool kvstore_initialized;
};

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} response_buffer;

typedef struct {
    q38_server *server;
    int fd;
    q38_server_request *request;
    bool stream;
    bool first_stream_event;
    char id[64];
    response_buffer text;
    response_buffer reasoning;
    response_buffer tool;
    response_buffer tool_arguments;
    char *tool_id;
    char *tool_name;
    pthread_mutex_t stream_mutex;
    volatile bool cancelled;
    q38_server_usage usage;
    char error[256];
} generation_context;

static bool set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static char *copy_string(const char *value) {
    const char *source = value ? value : "";
    size_t len = strlen(source);
    char *copy = malloc(len + 1);
    if (copy) memcpy(copy, source, len + 1);
    return copy;
}

static char *duplicate_range(const char *value, size_t len) {
    char *copy = malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, value, len);
    copy[len] = '\0';
    return copy;
}

static bool buffer_reserve(response_buffer *buffer, size_t extra) {
    if (!buffer || extra > SIZE_MAX - buffer->len - 1) return false;
    size_t needed = buffer->len + extra + 1;
    if (needed <= buffer->cap) return true;
    size_t cap = buffer->cap ? buffer->cap : 256;
    while (cap < needed) {
        if (cap > SIZE_MAX / 2) {
            cap = needed;
            break;
        }
        cap *= 2;
    }
    char *grown = realloc(buffer->data, cap);
    if (!grown) return false;
    buffer->data = grown;
    buffer->cap = cap;
    return true;
}

static bool buffer_append(response_buffer *buffer, const char *value,
                          size_t len) {
    if (!value) len = 0;
    if (!buffer_reserve(buffer, len)) return false;
    if (len) memcpy(buffer->data + buffer->len, value, len);
    buffer->len += len;
    buffer->data[buffer->len] = '\0';
    return true;
}

static bool buffer_append_cstr(response_buffer *buffer, const char *value) {
    return buffer_append(buffer, value, value ? strlen(value) : 0);
}

static void buffer_free(response_buffer *buffer) {
    if (!buffer) return;
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static bool send_all(int fd, const char *data, size_t len) {
    while (len) {
        ssize_t written = send(fd, data, len, MSG_NOSIGNAL);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (!written) return false;
        data += written;
        len -= (size_t)written;
    }
    return true;
}

static bool send_cstr(int fd, const char *data) {
    return send_all(fd, data, data ? strlen(data) : 0);
}

static bool json_append_escaped(response_buffer *buffer, const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value ? value : "");
    if (!buffer_append_cstr(buffer, "\"")) return false;
    while (*cursor) {
        char escaped[7];
        switch (*cursor) {
        case '"': if (!buffer_append_cstr(buffer, "\\\"")) return false; break;
        case '\\': if (!buffer_append_cstr(buffer, "\\\\")) return false; break;
        case '\b': if (!buffer_append_cstr(buffer, "\\b")) return false; break;
        case '\f': if (!buffer_append_cstr(buffer, "\\f")) return false; break;
        case '\n': if (!buffer_append_cstr(buffer, "\\n")) return false; break;
        case '\r': if (!buffer_append_cstr(buffer, "\\r")) return false; break;
        case '\t': if (!buffer_append_cstr(buffer, "\\t")) return false; break;
        default:
            if (*cursor < 0x20) {
                snprintf(escaped, sizeof(escaped), "\\u%04x", *cursor);
                if (!buffer_append_cstr(buffer, escaped)) return false;
            } else if (!buffer_append(buffer, (const char *)cursor, 1)) {
                return false;
            }
            break;
        }
        cursor++;
    }
    return buffer_append_cstr(buffer, "\"");
}

static char *trimmed_copy(const char *start, size_t len) {
    while (len && isspace((unsigned char)start[0])) {
        start++;
        len--;
    }
    while (len && isspace((unsigned char)start[len - 1])) len--;
    return duplicate_range(start, len);
}

static const char *find_header_end(const char *data, size_t len) {
    if (len < 4) return NULL;
    for (size_t i = 0; i + 3 < len; ++i)
        if (!memcmp(data + i, "\r\n\r\n", 4)) return data + i;
    return NULL;
}

static bool header_value(const char *headers, const char *name,
                         char **value) {
    const size_t name_len = strlen(name);
    const char *cursor = headers;
    *value = NULL;
    while (*cursor) {
        const char *line_end = strstr(cursor, "\r\n");
        if (!line_end) line_end = cursor + strlen(cursor);
        if ((size_t)(line_end - cursor) > name_len &&
            !strncasecmp(cursor, name, name_len) &&
            cursor[name_len] == ':') {
            const char *start = cursor + name_len + 1;
            return (*value = trimmed_copy(start, (size_t)(line_end - start))) != NULL;
        }
        if (!*line_end) return true;
        cursor = line_end + 2;
    }
    return true;
}

static bool parse_http_request(const char *raw, size_t raw_len,
                               q38_http_request *request,
                               size_t *header_bytes,
                               char *error, size_t error_len) {
    const char *header_end = find_header_end(raw, raw_len);
    const char *line_end;
    char *headers = NULL;
    char *content_length = NULL;
    size_t body_len = 0;
    if (!header_end)
        return set_error(error, error_len, "incomplete HTTP headers");
    *header_bytes = (size_t)(header_end - raw) + 4;
    line_end = strstr(raw, "\r\n");
    if (!line_end)
        return set_error(error, error_len, "invalid HTTP request line");
    headers = duplicate_range(line_end + 2,
                              (size_t)(header_end - (line_end + 2)));
    if (!headers)
        return set_error(error, error_len, "HTTP header allocation failed");
    char *line = duplicate_range(raw, (size_t)(line_end - raw));
    if (!line) {
        free(headers);
        return set_error(error, error_len, "HTTP request line allocation failed");
    }
    char *method = strtok(line, " ");
    char *target = strtok(NULL, " ");
    char *version = strtok(NULL, " ");
    if (!method || !target || !version || strtok(NULL, " ")) {
        free(line);
        free(headers);
        return set_error(error, error_len, "invalid HTTP request line");
    }
    if (!header_value(headers, "Content-Length", &content_length)) {
        free(line);
        free(headers);
        return set_error(error, error_len, "invalid HTTP headers");
    }
    if (content_length) {
        char *end = NULL;
        unsigned long parsed = strtoul(content_length, &end, 10);
        if (*end || parsed > Q38_SERVER_MAX_BODY_BYTES) {
            free(content_length);
            free(line);
            free(headers);
            return set_error(error, error_len, "invalid or oversized content length");
        }
        body_len = (size_t)parsed;
    }
    if (raw_len - *header_bytes < body_len) {
        free(content_length);
        free(line);
        free(headers);
        return set_error(error, error_len, "incomplete HTTP body");
    }
    request->method = copy_string(method);
    request->target = copy_string(target);
    request->version = copy_string(version);
    request->body = duplicate_range(raw + *header_bytes, body_len);
    request->body_len = body_len;
    free(content_length);
    free(line);
    free(headers);
    if (!request->method || !request->target || !request->version ||
        !request->body) {
        q38_http_request_free(request);
        return set_error(error, error_len, "HTTP request allocation failed");
    }
    return true;
}

void q38_http_request_free(q38_http_request *request) {
    if (!request) return;
    free(request->method);
    free(request->target);
    free(request->version);
    free(request->body);
    memset(request, 0, sizeof(*request));
}

static bool read_http_request(int fd, q38_http_request *request,
                              char *error, size_t error_len) {
    char *raw = NULL;
    size_t len = 0;
    size_t cap = 0;
    size_t header_bytes = 0;
    bool parsed = false;
    while (!parsed) {
        if (len == cap) {
            size_t next = cap ? cap * 2 : 4096;
            if (next > Q38_SERVER_MAX_HEADER_BYTES + Q38_SERVER_MAX_BODY_BYTES)
                next = Q38_SERVER_MAX_HEADER_BYTES + Q38_SERVER_MAX_BODY_BYTES;
            char *grown = realloc(raw, next);
            if (!grown) {
                free(raw);
                return set_error(error, error_len, "HTTP read allocation failed");
            }
            raw = grown;
            cap = next;
        }
        ssize_t received = recv(fd, raw + len, cap - len, 0);
        if (received < 0) {
            if (errno == EINTR) continue;
            free(raw);
            return set_error(error, error_len, "HTTP receive failed");
        }
        if (!received) {
            free(raw);
            return set_error(error, error_len, "client disconnected");
        }
        len += (size_t)received;
        const char *header_end = find_header_end(raw, len);
        if (header_end &&
            (size_t)(header_end - raw) + 4 <= Q38_SERVER_MAX_HEADER_BYTES) {
            char *length = NULL;
            const char *line_end = strstr(raw, "\r\n");
            if (!line_end) continue;
            char *headers = duplicate_range(line_end + 2,
                (size_t)(header_end - (line_end + 2)));
            if (!headers || !header_value(headers, "Content-Length", &length)) {
                free(headers);
                free(length);
                free(raw);
                return set_error(error, error_len, "invalid HTTP headers");
            }
            size_t body_len = 0;
            if (length) {
                char *end = NULL;
                unsigned long value = strtoul(length, &end, 10);
                if (*end || value > Q38_SERVER_MAX_BODY_BYTES) {
                    free(headers);
                    free(length);
                    free(raw);
                    return set_error(error, error_len,
                                     "invalid or oversized content length");
                }
                body_len = (size_t)value;
            }
            free(headers);
            free(length);
            if (len < (size_t)(header_end - raw) + 4 + body_len)
                continue;
            parsed = parse_http_request(raw, len, request, &header_bytes,
                                        error, error_len);
            free(raw);
            return parsed;
        }
        if (header_end && (size_t)(header_end - raw) + 4 >
                            Q38_SERVER_MAX_HEADER_BYTES) {
            free(raw);
            return set_error(error, error_len, "HTTP headers too large");
        }
        if (len >= Q38_SERVER_MAX_HEADER_BYTES + Q38_SERVER_MAX_BODY_BYTES) {
            free(raw);
            return set_error(error, error_len, "HTTP request too large");
        }
    }
    free(raw);
    return false;
}

static bool send_http_response(int fd, int status, const char *type,
                               const char *body, bool keep_alive) {
    char header[512];
    size_t body_len = body ? strlen(body) : 0;
    const char *reason = status == 200 ? "OK" :
                         status == 201 ? "Created" :
                         status == 204 ? "No Content" :
                         status == 400 ? "Bad Request" :
                         status == 404 ? "Not Found" :
                         status == 405 ? "Method Not Allowed" :
                         status == 413 ? "Payload Too Large" :
                         status == 500 ? "Internal Server Error" : "Error";
    int length = snprintf(header, sizeof(header),
                          "HTTP/1.1 %d %s\r\n"
                          "Content-Type: %s\r\n"
                          "Content-Length: %zu\r\n"
                          "Access-Control-Allow-Origin: *\r\n"
                          "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                          "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                          "Connection: %s\r\n\r\n",
                          status, reason, type ? type : "application/json",
                          body_len, keep_alive ? "keep-alive" : "close");
    return length > 0 && (size_t)length < sizeof(header) &&
           send_all(fd, header, (size_t)length) &&
           (!body_len || send_all(fd, body, body_len));
}

const char *q38_server_health_json(void) {
    return "{\"status\":\"ok\"}";
}

const char *q38_server_models_json(const q38_server *server) {
    return server && server->model_json ? server->model_json :
           "{\"object\":\"list\",\"data\":[]}";
}

static bool append_error_json(response_buffer *body, const char *message) {
    return buffer_append_cstr(body, "{\"error\":{\"message\":") &&
           json_append_escaped(body, message) &&
           buffer_append_cstr(body, ",\"type\":\"invalid_request_error\"}}");
}

static bool stream_event_json(generation_context *context,
                              const q38_server_event *event) {
    response_buffer body = {0};
    bool ok = true;
    const char *api = context->request->api == Q38_SERVER_API_ANTHROPIC ?
                      "anthropic" :
                      context->request->api == Q38_SERVER_API_RESPONSES ?
                      "responses" : "openai";
    if (!event) return false;
    if (event->kind == Q38_SERVER_EVENT_DONE) {
        if (!strcmp(api, "openai")) {
            ok = buffer_append_cstr(&body, "data: [DONE]\n\n");
        } else if (!strcmp(api, "anthropic")) {
            ok = buffer_append_cstr(&body,
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        } else {
            ok = buffer_append_cstr(&body,
                "event: response.completed\n"
                "data: {\"type\":\"response.completed\"}\n\n");
        }
    } else if (!strcmp(api, "openai")) {
        if (!buffer_append_cstr(&body, "data: {\"id\":") ||
            !json_append_escaped(&body, context->id) ||
            !buffer_append_cstr(&body,
                ",\"object\":\"chat.completion.chunk\",\"choices\":["
                "{\"index\":0,\"delta\":{"))
            ok = false;
        if (ok && event->kind == Q38_SERVER_EVENT_REASONING)
            ok = buffer_append_cstr(&body, "\"reasoning_content\":") &&
                 json_append_escaped(&body, event->text);
        else if (ok && event->kind == Q38_SERVER_EVENT_TOOL_CALL)
            ok = buffer_append_cstr(&body, "\"tool_calls\":[{\"index\":0,\"id\":") &&
                 json_append_escaped(&body, event->id) &&
                 buffer_append_cstr(&body, ",\"type\":\"function\",\"function\":{\"name\":") &&
                 json_append_escaped(&body, event->name) &&
                 buffer_append_cstr(&body, ",\"arguments\":") &&
                 json_append_escaped(&body, event->arguments_json) &&
                 buffer_append_cstr(&body, "}}]");
        else if (ok)
            ok = buffer_append_cstr(&body, "\"content\":") &&
                 json_append_escaped(&body, event->text);
        if (ok)
            ok = buffer_append_cstr(&body, "},\"finish_reason\":null}]}\n\n");
        if (ok && body.len >= 2 && !send_all(context->fd, body.data, body.len))
            ok = false;
        buffer_free(&body);
        return ok;
    } else if (!strcmp(api, "anthropic")) {
        if (event->kind == Q38_SERVER_EVENT_REASONING)
            ok = buffer_append_cstr(&body,
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\","
                "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":") &&
                 json_append_escaped(&body, event->text) &&
                 buffer_append_cstr(&body, "}}\n\n");
        else if (event->kind == Q38_SERVER_EVENT_TOOL_CALL)
            ok = buffer_append_cstr(&body,
                "event: content_block_start\ndata: {\"type\":\"content_block_start\","
                "\"content_block\":{\"type\":\"tool_use\",\"id\":") &&
                 json_append_escaped(&body, event->id) &&
                 buffer_append_cstr(&body, ",\"name\":") &&
                 json_append_escaped(&body, event->name) &&
                 buffer_append_cstr(&body, ",\"input\":") &&
                 buffer_append_cstr(&body, event->arguments_json) &&
                 buffer_append_cstr(&body, "}}\n\n");
        else
            ok = buffer_append_cstr(&body,
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\","
                "\"delta\":{\"type\":\"text_delta\",\"text\":") &&
                 json_append_escaped(&body, event->text) &&
                 buffer_append_cstr(&body, "}}\n\n");
    } else {
        ok = buffer_append_cstr(&body,
            "event: response.output_text.delta\ndata: {\"type\":"
            "\"response.output_text.delta\",\"delta\":") &&
             json_append_escaped(&body, event->text) &&
             buffer_append_cstr(&body, "}\n\n");
    }
    if (ok) ok = send_all(context->fd, body.data, body.len);
    buffer_free(&body);
    return ok;
}

static bool collect_event(const q38_server_event *event, void *user,
                          char *error, size_t error_len) {
    generation_context *context = user;
    response_buffer *target = NULL;
    if (!event || !context) return set_error(error, error_len, "invalid event");
    if (event->kind == Q38_SERVER_EVENT_TEXT) target = &context->text;
    else if (event->kind == Q38_SERVER_EVENT_REASONING)
        target = &context->reasoning;
    else if (event->kind == Q38_SERVER_EVENT_TOOL_CALL) {
        free(context->tool_id);
        free(context->tool_name);
        context->tool_id = copy_string(event->id);
        context->tool_name = copy_string(event->name);
        if (!context->tool_id || !context->tool_name)
            return set_error(error, error_len, "response allocation failed");
        if (!buffer_append_cstr(&context->tool, event->name) ||
            !buffer_append_cstr(&context->tool_arguments,
                                 event->arguments_json))
            return set_error(error, error_len, "response allocation failed");
    }
    if (target && !buffer_append(target, event->text, event->text_len))
        return set_error(error, error_len, "response allocation failed");
    if (context->stream) {
        bool sent;
        pthread_mutex_lock(&context->stream_mutex);
        sent = stream_event_json(context, event);
        pthread_mutex_unlock(&context->stream_mutex);
        if (!sent) {
            context->cancelled = true;
            return set_error(error, error_len, "client disconnected");
        }
    }
    return true;
}

static bool generation_cancelled(void *user) {
    generation_context *context = user;
    return context && context->cancelled;
}

typedef struct {
    generation_context *context;
    bool done;
    int result;
    pthread_cond_t condition;
    pthread_mutex_t state_mutex;
} generation_job;

static void *generation_worker(void *user) {
    generation_job *job = user;
    job->result = q38_server_engine_generate(
        job->context->server->engine, job->context->request, collect_event,
        job->context, &job->context->usage, job->context->error,
        sizeof(job->context->error));
    pthread_mutex_lock(&job->state_mutex);
    job->done = true;
    pthread_cond_signal(&job->condition);
    pthread_mutex_unlock(&job->state_mutex);
    return NULL;
}

static bool build_nonstream_response(generation_context *context,
                                     response_buffer *body) {
    const char *finish = context->tool.len ? "tool_calls" : "stop";
    char usage[128];
    char legacy_usage[128];
    if (context->request->api == Q38_SERVER_API_ANTHROPIC) {
        if (!buffer_append_cstr(body,
            "{\"id\":") || !json_append_escaped(body, context->id) ||
            !buffer_append_cstr(body,
                ",\"type\":\"message\",\"role\":\"assistant\",\"content\":["))
            return false;
        if (context->reasoning.len &&
            (!buffer_append_cstr(body,
                "{\"type\":\"thinking\",\"thinking\":") ||
             !json_append_escaped(body, context->reasoning.data) ||
             !buffer_append_cstr(body, "},")))
            return false;
        if (context->tool.len &&
            (!buffer_append_cstr(body,
                "{\"type\":\"tool_use\",\"id\":\"call_q38_mock_1\","
                "\"name\":") ||
             !json_append_escaped(body, context->tool_name) ||
             !buffer_append_cstr(body, ",\"input\":") ||
             buffer_append_cstr(body, context->tool_arguments.data) == false ||
             !buffer_append_cstr(body, "},")))
            return false;
        if (!buffer_append_cstr(body, "{\"type\":\"text\",\"text\":") ||
            !json_append_escaped(body, context->text.data) ||
            !buffer_append_cstr(body, "}],\"stop_reason\":\"") ||
            !buffer_append_cstr(body, finish) ||
            !buffer_append_cstr(body, "\"}"))
            return false;
        return true;
    }
    if (context->request->api == Q38_SERVER_API_RESPONSES) {
        return buffer_append_cstr(body, "{\"id\":") &&
               json_append_escaped(body, context->id) &&
               buffer_append_cstr(body, ",\"object\":\"response\",\"output_text\":") &&
               json_append_escaped(body, context->text.data) &&
               buffer_append_cstr(body, "}");
    }
    snprintf(usage, sizeof(usage),
             "\"}],\"usage\":{\"prompt_tokens\":%u,"
             "\"completion_tokens\":%u,\"total_tokens\":%u}}",
             context->usage.prompt_tokens, context->usage.completion_tokens,
             context->usage.prompt_tokens + context->usage.completion_tokens);
    snprintf(legacy_usage, sizeof(legacy_usage),
             "\"}],\"usage\":{\"prompt_tokens\":%u,"
             "\"completion_tokens\":%u,\"total_tokens\":%u}}",
             context->usage.prompt_tokens, context->usage.completion_tokens,
             context->usage.prompt_tokens + context->usage.completion_tokens);
    if (context->request->legacy_completion) {
        return buffer_append_cstr(body,
            "{\"id\":") && json_append_escaped(body, context->id) &&
            buffer_append_cstr(body,
                ",\"object\":\"text_completion\",\"choices\":[{\"index\":0,"
                "\"text\":") &&
            json_append_escaped(body, context->text.data) &&
            buffer_append_cstr(body, ",\"finish_reason\":\"") &&
            buffer_append_cstr(body, finish) &&
            buffer_append_cstr(body, legacy_usage);
    }
    if (context->tool.len) {
        return buffer_append_cstr(body,
        "{\"id\":") && json_append_escaped(body, context->id) &&
        buffer_append_cstr(body,
            ",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
            "\"message\":{\"role\":\"assistant\",\"content\":null,"
            "\"tool_calls\":[{\"id\":") &&
        json_append_escaped(body, context->tool_id) &&
        buffer_append_cstr(body, ",\"type\":\"function\",\"function\":{\"name\":") &&
        json_append_escaped(body, context->tool_name) &&
        buffer_append_cstr(body, ",\"arguments\":") &&
        json_append_escaped(body, context->tool_arguments.data) &&
        buffer_append_cstr(body, "}}]},\"finish_reason\":\"") &&
        buffer_append_cstr(body, finish) &&
        buffer_append_cstr(body, usage);
    }
    return buffer_append_cstr(body,
        "{\"id\":") && json_append_escaped(body, context->id) &&
        buffer_append_cstr(body,
            ",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
            "\"message\":{\"role\":\"assistant\",\"content\":") &&
        json_append_escaped(body, context->text.data) &&
        buffer_append_cstr(body, ",\"reasoning_content\":") &&
        json_append_escaped(body, context->reasoning.data) &&
        buffer_append_cstr(body, "},\"finish_reason\":\"") &&
        buffer_append_cstr(body, finish) &&
        buffer_append_cstr(body, usage);
}

static bool route_target(const char *target, q38_server_endpoint *endpoint) {
    char *path = copy_string(target);
    char *query;
    bool matched = false;
    if (!path) return false;
    query = strchr(path, '?');
    if (query) *query = '\0';
    if (!strcmp(path, "/v1/chat/completions")) {
        *endpoint = Q38_SERVER_ENDPOINT_CHAT_COMPLETIONS;
        matched = true;
    } else if (!strcmp(path, "/v1/completions")) {
        *endpoint = Q38_SERVER_ENDPOINT_COMPLETIONS;
        matched = true;
    } else if (!strcmp(path, "/v1/responses")) {
        *endpoint = Q38_SERVER_ENDPOINT_RESPONSES;
        matched = true;
    } else if (!strcmp(path, "/v1/messages")) {
        *endpoint = Q38_SERVER_ENDPOINT_MESSAGES;
        matched = true;
    }
    free(path);
    return matched;
}

static bool handle_generation(q38_server *server, int fd,
                              q38_server_request *request,
                              char *error, size_t error_len) {
    generation_context context;
    response_buffer body = {0};
    int result;
    memset(&context, 0, sizeof(context));
    context.server = server;
    context.fd = fd;
    context.request = request;
    context.stream = request->stream;
    context.error[0] = '\0';
    request->cancelled = generation_cancelled;
    request->cancel_user = &context;
    snprintf(context.id, sizeof(context.id), "q38-%llu",
             (unsigned long long)server->next_request_id++);
    if (request->stream) {
        const char *headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";
        if (!send_cstr(fd, headers)) {
            set_error(error, error_len, "client disconnected");
            return false;
        }
        if (!send_cstr(fd, ": q38 prefill started\n\n")) {
            set_error(error, error_len, "client disconnected");
            return false;
        }
    }
    generation_job job;
    memset(&job, 0, sizeof(job));
    job.context = &context;
    if (request->stream) {
        pthread_t worker;
        pthread_mutex_init(&context.stream_mutex, NULL);
        pthread_mutex_init(&job.state_mutex, NULL);
        pthread_cond_init(&job.condition, NULL);
        if (pthread_create(&worker, NULL, generation_worker, &job) != 0) {
            pthread_cond_destroy(&job.condition);
            pthread_mutex_destroy(&job.state_mutex);
            pthread_mutex_destroy(&context.stream_mutex);
            set_error(error, error_len, "generation thread creation failed");
            return false;
        }
        pthread_mutex_lock(&job.state_mutex);
        while (!job.done) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_sec += 5;
            int wait_result = pthread_cond_timedwait(
                &job.condition, &job.state_mutex, &deadline);
            if (!job.done && wait_result == ETIMEDOUT) {
                pthread_mutex_unlock(&job.state_mutex);
                pthread_mutex_lock(&context.stream_mutex);
                bool sent = send_cstr(fd, ": q38 keepalive\n\n");
                pthread_mutex_unlock(&context.stream_mutex);
                pthread_mutex_lock(&job.state_mutex);
                if (!sent) {
                    context.cancelled = true;
                    pthread_mutex_unlock(&job.state_mutex);
                    pthread_join(worker, NULL);
                    pthread_cond_destroy(&job.condition);
                    pthread_mutex_destroy(&job.state_mutex);
                    pthread_mutex_destroy(&context.stream_mutex);
                    set_error(error, error_len, "client disconnected");
                    return false;
                }
            }
        }
        pthread_mutex_unlock(&job.state_mutex);
        pthread_join(worker, NULL);
        result = job.result;
        pthread_cond_destroy(&job.condition);
        pthread_mutex_destroy(&job.state_mutex);
    } else {
        result = q38_server_engine_generate(server->engine, request, collect_event,
                                            &context, &context.usage,
                                            error, error_len);
    }
    if (context.error[0] && error && error_len)
        snprintf(error, error_len, "%s", context.error);
    if (result != 0) {
        if (request->stream)
            send_cstr(fd, "event: error\ndata: {\"error\":{\"message\":\"generation failed\"}}\n\n");
        buffer_free(&context.text);
        buffer_free(&context.reasoning);
        buffer_free(&context.tool);
        buffer_free(&context.tool_arguments);
        free(context.tool_id);
        free(context.tool_name);
        if (request->stream) pthread_mutex_destroy(&context.stream_mutex);
        return false;
    }
    if (!request->stream) {
        if (!build_nonstream_response(&context, &body) ||
            !send_http_response(fd, 200, "application/json", body.data, false))
            set_error(error, error_len, "response send failed");
    }
    buffer_free(&body);
    buffer_free(&context.text);
    buffer_free(&context.reasoning);
    buffer_free(&context.tool);
    buffer_free(&context.tool_arguments);
    free(context.tool_id);
    free(context.tool_name);
    if (request->stream) pthread_mutex_destroy(&context.stream_mutex);
    return !error || error[0] == '\0';
}

q38_server *q38_server_create(q38_server_engine *engine, bool owns_engine,
                              char *error, size_t error_len) {
    q38_server *server;
    if (error && error_len) error[0] = '\0';
    if (!engine) {
        set_error(error, error_len, "server engine is required");
        return NULL;
    }
    server = calloc(1, sizeof(*server));
    if (!server) {
        set_error(error, error_len, "server allocation failed");
        return NULL;
    }
    server->engine = engine;
    server->owns_engine = owns_engine;
    server->next_request_id = 1;
    if (pthread_mutex_init(&server->inference_mutex, NULL) != 0) {
        free(server);
        set_error(error, error_len, "server mutex initialization failed");
        return NULL;
    }
    const char *model_name = q38_server_engine_model_name(engine);
    if (asprintf(&server->model_json,
                 "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
                 "\"object\":\"model\",\"owned_by\":\"q38\"}]}",
                 model_name ? model_name : "qwen3.8-flash-next") < 0) {
        pthread_mutex_destroy(&server->inference_mutex);
        free(server);
        set_error(error, error_len, "model metadata allocation failed");
        return NULL;
    }
    return server;
}

bool q38_server_enable_kvstore(q38_server *server, const char *root,
                               char *error, size_t error_len) {
    if (!server) return set_error(error, error_len, "server is null");
    if (server->kvstore_initialized)
        q38_kvstore_destroy(&server->kvstore);
    if (!q38_kvstore_init(&server->kvstore, root, error, error_len))
        return false;
    server->kvstore_initialized = true;
    return true;
}

void q38_server_destroy(q38_server *server) {
    if (!server) return;
    if (server->owns_engine) q38_server_engine_destroy(server->engine);
    if (server->kvstore_initialized) q38_kvstore_destroy(&server->kvstore);
    pthread_mutex_destroy(&server->inference_mutex);
    free(server->model_json);
    free(server);
}

int q38_server_handle_connection(q38_server *server, int fd,
                                 char *error, size_t error_len) {
    q38_http_request http = {0};
    q38_server_request request;
    q38_server_endpoint endpoint;
    int status = 200;
    bool ok = false;
    if (error && error_len) error[0] = '\0';
    if (!server || fd < 0) {
        set_error(error, error_len, "invalid server connection");
        return -1;
    }
    if (!read_http_request(fd, &http, error, error_len)) {
        response_buffer body = {0};
        append_error_json(&body, error && error[0] ? error : "bad request");
        send_http_response(fd, 400, "application/json", body.data, false);
        buffer_free(&body);
        return -1;
    }
    if (!strcmp(http.method, "OPTIONS")) {
        send_http_response(fd, 204, "text/plain", "", false);
        q38_http_request_free(&http);
        return 0;
    }
    if (!strcmp(http.method, "GET")) {
        if (!strcmp(http.target, "/health")) {
            ok = send_http_response(fd, 200, "application/json",
                                    q38_server_health_json(), false);
        } else if (!strcmp(http.target, "/v1/models") ||
                   !strncmp(http.target, "/v1/models?", 11)) {
            ok = send_http_response(fd, 200, "application/json",
                                    q38_server_models_json(server), false);
        } else {
            status = 404;
        }
    } else if (strcmp(http.method, "POST")) {
        status = 405;
    } else if (route_target(http.target, &endpoint)) {
        q38_server_request_init(&request);
        if (!q38_server_parse_request(endpoint, http.body, &request,
                                      error, error_len)) {
            response_buffer body = {0};
            append_error_json(&body, error);
            send_http_response(fd, 400, "application/json", body.data, false);
            buffer_free(&body);
            q38_server_request_free(&request);
            q38_http_request_free(&http);
            return -1;
        }
        if ((request.cache_restore || request.cache_save) &&
            (!server->kvstore_initialized || !server->kvstore.enabled)) {
            response_buffer body = {0};
            append_error_json(&body,
                              "disk KV/session cache is unavailable for Q38 runtime");
            send_http_response(fd, 501, "application/json", body.data, false);
            buffer_free(&body);
            q38_server_request_free(&request);
            q38_http_request_free(&http);
            return -1;
        }
        pthread_mutex_lock(&server->inference_mutex);
        ok = handle_generation(server, fd, &request, error, error_len);
        pthread_mutex_unlock(&server->inference_mutex);
        q38_server_request_free(&request);
    } else {
        status = 404;
    }
    if (status != 200) {
        response_buffer body = {0};
        append_error_json(&body, status == 405 ? "method not allowed" :
                          "endpoint not found");
        send_http_response(fd, status, "application/json", body.data, false);
        buffer_free(&body);
    }
    q38_http_request_free(&http);
    return ok ? 0 : -1;
}

static int create_listener(const char *host, uint16_t port,
                           char *error, size_t error_len) {
    struct addrinfo hints;
    struct addrinfo *results = NULL;
    char service[16];
    int fd = -1;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    snprintf(service, sizeof(service), "%u", (unsigned)port);
    int rc = getaddrinfo(host, service, &hints, &results);
    if (rc != 0) {
        set_error(error, error_len, gai_strerror(rc));
        return -1;
    }
    for (struct addrinfo *item = results; item; item = item->ai_next) {
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) continue;
        int enabled = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
        if (!bind(fd, item->ai_addr, item->ai_addrlen) &&
            !listen(fd, 16))
            break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0) set_error(error, error_len, "unable to bind server listener");
    return fd;
}

typedef struct {
    q38_server *server;
    int fd;
} connection_context;

static void *serve_connection_thread(void *user) {
    connection_context *context = user;
    char error[256] = {0};
    if (context) {
        q38_server_handle_connection(context->server, context->fd,
                                     error, sizeof(error));
        close(context->fd);
        free(context);
    }
    return NULL;
}

int q38_server_listen_and_serve(q38_server *server, const char *host,
                                uint16_t port, char *error, size_t error_len) {
    int listener;
    if (!server) {
        set_error(error, error_len, "server is null");
        return -1;
    }
    signal(SIGPIPE, SIG_IGN);
    listener = create_listener(host ? host : Q38_SERVER_DEFAULT_HOST, port,
                               error, error_len);
    if (listener < 0) return -1;
    for (;;) {
        int client = accept(listener, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            close(listener);
            set_error(error, error_len, "accept failed");
            return -1;
        }
        connection_context *context = malloc(sizeof(*context));
        pthread_t thread;
        if (!context) {
            close(client);
            set_error(error, error_len, "connection allocation failed");
            continue;
        }
        context->server = server;
        context->fd = client;
        if (pthread_create(&thread, NULL, serve_connection_thread, context) != 0) {
            close(client);
            free(context);
            set_error(error, error_len, "connection thread creation failed");
            continue;
        }
        pthread_detach(thread);
    }
}
