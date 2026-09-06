#include "q38_json.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct {
    char *role;
    char *content;
} cli_message;

typedef struct {
    cli_message *items;
    size_t count;
} cli_messages;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} cli_buffer;

static bool buffer_append(cli_buffer *buffer, const char *value, size_t len) {
    if (!buffer || len > SIZE_MAX - buffer->len - 1) return false;
    size_t needed = buffer->len + len + 1;
    if (needed > buffer->cap) {
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
    }
    memcpy(buffer->data + buffer->len, value, len);
    buffer->len += len;
    buffer->data[buffer->len] = '\0';
    return true;
}

static bool buffer_cstr(cli_buffer *buffer, const char *value) {
    return buffer_append(buffer, value, value ? strlen(value) : 0);
}

static bool append_json_string(cli_buffer *buffer, const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value ? value : "");
    if (!buffer_cstr(buffer, "\"")) return false;
    while (*cursor) {
        switch (*cursor) {
        case '"': if (!buffer_cstr(buffer, "\\\"")) return false; break;
        case '\\': if (!buffer_cstr(buffer, "\\\\")) return false; break;
        case '\n': if (!buffer_cstr(buffer, "\\n")) return false; break;
        case '\r': if (!buffer_cstr(buffer, "\\r")) return false; break;
        case '\t': if (!buffer_cstr(buffer, "\\t")) return false; break;
        default:
            if (*cursor < 0x20) return false;
            if (!buffer_append(buffer, (const char *)cursor, 1)) return false;
        }
        cursor++;
    }
    return buffer_cstr(buffer, "\"");
}

static void free_messages(cli_messages *messages) {
    if (!messages) return;
    for (size_t i = 0; i < messages->count; ++i) {
        free(messages->items[i].role);
        free(messages->items[i].content);
    }
    free(messages->items);
    memset(messages, 0, sizeof(*messages));
}

static bool append_message(cli_messages *messages, const char *role,
                           const char *content) {
    cli_message *grown = realloc(messages->items,
                                  (messages->count + 1) * sizeof(*grown));
    if (!grown) return false;
    messages->items = grown;
    cli_message *message = &messages->items[messages->count++];
    message->role = strdup(role);
    message->content = strdup(content);
    if (!message->role || !message->content) {
        free(message->role);
        free(message->content);
        messages->count--;
        return false;
    }
    return true;
}

static int connect_server(const char *host, unsigned short port) {
    struct addrinfo hints;
    struct addrinfo *results = NULL;
    char service[16];
    int fd = -1;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(service, sizeof(service), "%u", (unsigned)port);
    if (getaddrinfo(host, service, &hints, &results) != 0) return -1;
    for (struct addrinfo *item = results; item; item = item->ai_next) {
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) continue;
        if (!connect(fd, item->ai_addr, item->ai_addrlen)) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    return fd;
}

static bool send_all(int fd, const char *data, size_t len) {
    while (len) {
        ssize_t written = send(fd, data, len, MSG_NOSIGNAL);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        data += written;
        len -= (size_t)written;
    }
    return true;
}

static char *read_response(int fd) {
    cli_buffer response = {0};
    char chunk[4096];
    for (;;) {
        ssize_t received = recv(fd, chunk, sizeof(chunk), 0);
        if (received < 0 && errno == EINTR) continue;
        if (received <= 0) break;
        if (!buffer_append(&response, chunk, (size_t)received)) {
            free(response.data);
            return NULL;
        }
    }
    return response.data;
}

static bool server_ready(const char *host, unsigned short port) {
    int fd = connect_server(host, port);
    if (fd < 0) return false;
    const char *request =
        "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    bool ok = send_all(fd, request, strlen(request));
    shutdown(fd, SHUT_WR);
    char *response = ok ? read_response(fd) : NULL;
    if (response) ok = strstr(response, "200 OK") != NULL;
    free(response);
    close(fd);
    return ok;
}

static bool ensure_server(const char *host, unsigned short port,
                          bool autostart) {
    if (server_ready(host, port)) return true;
    if (!autostart) return false;
    pid_t child = fork();
    if (child < 0) return false;
    if (!child) {
        char port_text[16];
        snprintf(port_text, sizeof(port_text), "%u", (unsigned)port);
        execl("./q38-server", "q38-server", "--host", host,
              "--port", port_text, (char *)NULL);
        execlp("q38-server", "q38-server", "--host", host,
               "--port", port_text, (char *)NULL);
        _exit(127);
    }
    for (int i = 0; i < 300; ++i) {
        if (server_ready(host, port)) return true;
        usleep(100000);
    }
    return false;
}

static bool build_chat_body(const cli_messages *messages, int max_tokens,
                            bool stream, char **body, size_t *body_len) {
    cli_buffer buffer = {0};
    if (!buffer_cstr(&buffer, "{\"model\":\"qwen3.8-flash-next\",\"stream\":") ||
        !buffer_cstr(&buffer, stream ? "true" : "false") ||
        !buffer_cstr(&buffer, ",\"max_tokens\":") ) goto fail;
    char number[32];
    snprintf(number, sizeof(number), "%d", max_tokens);
    if (!buffer_cstr(&buffer, number) ||
        !buffer_cstr(&buffer, ",\"messages\":[")) goto fail;
    for (size_t i = 0; i < messages->count; ++i) {
        if (i && !buffer_cstr(&buffer, ",")) goto fail;
        if (!buffer_cstr(&buffer, "{\"role\":") ||
            !append_json_string(&buffer, messages->items[i].role) ||
            !buffer_cstr(&buffer, ",\"content\":")) {
            goto fail;
        }
        if (!append_json_string(&buffer, messages->items[i].content) ||
            !buffer_cstr(&buffer, "}"))
            goto fail;
    }
    if (!buffer_cstr(&buffer, "]}")) goto fail;
    *body = buffer.data;
    *body_len = buffer.len;
    return true;
fail:
    free(buffer.data);
    return false;
}

static char *response_body(char *response) {
    char *separator = response ? strstr(response, "\r\n\r\n") : NULL;
    return separator ? separator + 4 : NULL;
}

static char *extract_string_field(const char *json, const char *field) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":", field);
    const char *value = strstr(json ? json : "", needle);
    if (!value) return NULL;
    value += strlen(needle);
    while (*value == ' ' || *value == '\t') value++;
    char *result = NULL;
    if (!q38_json_parse_string(&value, &result, NULL, 0)) {
        free(result);
        return NULL;
    }
    return result;
}

static char *print_sse_content(const char *body) {
    cli_buffer collected = {0};
    const char *cursor = body;
    while (cursor && *cursor) {
        const char *line_end = strstr(cursor, "\n");
        size_t line_len = line_end ? (size_t)(line_end - cursor) :
                                     strlen(cursor);
        if (line_len >= 6 && !strncmp(cursor, "data: ", 6)) {
            char *line = strndup(cursor + 6, line_len - 6);
            if (line && strcmp(line, "[DONE]")) {
                char *content = extract_string_field(line, "content");
                char *reasoning = extract_string_field(line,
                                                        "reasoning_content");
                char *tool = extract_string_field(line, "name");
                if (content) {
                    buffer_append(&collected, content, strlen(content));
                    fputs(content, stdout);
                    fflush(stdout);
                }
                if (reasoning) {
                    buffer_append(&collected, reasoning, strlen(reasoning));
                    fputs(reasoning, stdout);
                    fflush(stdout);
                }
                if (tool) {
                    printf("\n[tool: %s]\n", tool);
                }
                free(content);
                free(reasoning);
                free(tool);
            }
            free(line);
        }
        if (!line_end) break;
        cursor = line_end + 1;
    }
    putchar('\n');
    return collected.data;
}

static bool send_chat(const char *host, unsigned short port,
                      const cli_messages *messages, int max_tokens,
                      bool stream, char **assistant_text) {
    char *body = NULL;
    size_t body_len = 0;
    int fd = -1;
    char header[512];
    char *response = NULL;
    bool ok = false;
    if (!build_chat_body(messages, max_tokens, stream, &body, &body_len))
        return false;
    fd = connect_server(host, port);
    if (fd < 0) goto done;
    int header_len = snprintf(header, sizeof(header),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", body_len);
    if (header_len <= 0 || (size_t)header_len >= sizeof(header) ||
        !send_all(fd, header, (size_t)header_len) ||
        !send_all(fd, body, body_len))
        goto done;
    shutdown(fd, SHUT_WR);
    response = read_response(fd);
    if (!response) goto done;
    if (stream) {
        char *payload = response_body(response);
        char *collected = payload ? print_sse_content(payload) : NULL;
        if (assistant_text) *assistant_text = collected;
        else free(collected);
        ok = strstr(response, "200 OK") != NULL;
    } else {
        char *payload = response_body(response);
        char *content = payload ? extract_string_field(payload, "content") :
                                  NULL;
        if (content) {
            puts(content);
        } else if (payload) {
            puts(payload);
        }
        if (assistant_text) *assistant_text = content;
        else free(content);
        ok = strstr(response, "200 OK") != NULL;
    }
done:
    free(response);
    if (fd >= 0) close(fd);
    free(body);
    return ok;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s [--host HOST] [--port PORT] [--max-tokens N] "
            "[--no-stream] [--no-autostart] [-p PROMPT]\n", program);
}

int main(int argc, char **argv) {
    const char *host = "127.0.0.1";
    unsigned long port = 8000;
    int max_tokens = 256;
    bool stream = true;
    bool autostart = true;
    const char *one_shot = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            char *end = NULL;
            port = strtoul(argv[++i], &end, 10);
            if (!end || *end || port > 65535) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
            if (max_tokens <= 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--no-stream")) stream = false;
        else if (!strcmp(argv[i], "--no-autostart")) autostart = false;
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) one_shot = argv[++i];
        else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!ensure_server(host, (unsigned short)port, autostart)) {
        fprintf(stderr, "q38-cli: q38-server is not available\n");
        return 1;
    }
    cli_messages messages = {0};
    if (one_shot) {
        if (!append_message(&messages, "user", one_shot) ||
            !send_chat(host, (unsigned short)port, &messages, max_tokens,
                       stream, NULL)) {
            free_messages(&messages);
            return 1;
        }
        free_messages(&messages);
        return 0;
    }
    printf("Connected to q38-server at http://%s:%lu\n", host, port);
    printf("Type /clear, /model, /status, or /quit.\n");
    char *line = NULL;
    size_t line_cap = 0;
    for (;;) {
        fputs("> ", stdout);
        fflush(stdout);
        ssize_t length = getline(&line, &line_cap, stdin);
        if (length < 0) break;
        while (length && (line[length - 1] == '\n' ||
                          line[length - 1] == '\r'))
            line[--length] = '\0';
        if (!strcmp(line, "/quit")) break;
        if (!strcmp(line, "/clear")) {
            free_messages(&messages);
            puts("conversation cleared");
            continue;
        }
        if (!strcmp(line, "/model")) {
            puts("qwen3.8-flash-next");
            continue;
        }
        if (!strcmp(line, "/status")) {
            puts(server_ready(host, (unsigned short)port) ? "ready" :
                 "unavailable");
            continue;
        }
        if (!line[0]) continue;
        char *assistant = NULL;
        if (!append_message(&messages, "user", line) ||
            !send_chat(host, (unsigned short)port, &messages, max_tokens,
                       stream, &assistant)) {
            fprintf(stderr, "q38-cli: request failed\n");
            free(assistant);
            break;
        }
        if (!assistant) assistant = strdup("");
        if (!assistant ||
            !append_message(&messages, "assistant", assistant)) {
            fprintf(stderr, "q38-cli: conversation allocation failed\n");
            free(assistant);
            break;
        }
        free(assistant);
    }
    free(line);
    free_messages(&messages);
    return 0;
}
