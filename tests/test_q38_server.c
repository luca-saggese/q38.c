#include "q38_server.h"

#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int failures;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static char *exchange(q38_server *server, const char *request) {
    int sockets[2];
    char *response = NULL;
    size_t len = 0;
    size_t cap = 4096;
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) return NULL;
    if (send(sockets[0], request, strlen(request), 0) < 0) goto done;
    shutdown(sockets[0], SHUT_WR);
    char error[256] = {0};
    if (q38_server_handle_connection(server, sockets[1], error,
                                     sizeof(error)) != 0 &&
        !error[0]) {
        fprintf(stderr, "server exchange failed without error\n");
        goto done;
    }
    shutdown(sockets[1], SHUT_WR);
    response = malloc(cap);
    if (!response) goto done;
    for (;;) {
        ssize_t received = recv(sockets[0], response + len, cap - len - 1, 0);
        if (received <= 0) break;
        len += (size_t)received;
        if (cap - len <= 1) {
            size_t next = cap * 2;
            char *grown = realloc(response, next);
            if (!grown) {
                free(response);
                response = NULL;
                goto done;
            }
            response = grown;
            cap = next;
        }
    }
    response[len] = '\0';
done:
    close(sockets[0]);
    close(sockets[1]);
    return response;
}

int main(void) {
    char error[256] = {0};
    q38_server_engine *engine = q38_server_mock_engine_create(NULL, error,
                                                                sizeof(error));
    q38_server *server = q38_server_create(engine, true, error, sizeof(error));
    check(server != NULL, "server creation");

    char *response = exchange(server,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    check(response && strstr(response, "200 OK") &&
              strstr(response, "{\"status\":\"ok\"}"),
          "health endpoint");
    free(response);

    response = exchange(server,
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
    check(response && strstr(response, "\"object\":\"list\"") &&
              strstr(response, "\"object\":\"model\""),
          "models endpoint");
    free(response);

    const char *body =
        "{\"model\":\"qwen3.8-flash-next\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hello\"}]}";
    char request[1024];
    snprintf(request, sizeof(request),
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "Host: localhost\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\n\r\n%s", strlen(body), body);
    response = exchange(server, request);
    check(response && strstr(response, "200 OK") &&
              strstr(response, "deterministic Q38 mock response"),
          "OpenAI non-stream chat");
    free(response);

    const char *stream_body =
        "{\"model\":\"qwen3.8-flash-next\",\"stream\":true,\"thinking\":true,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}";
    snprintf(request, sizeof(request),
             "POST /v1/chat/completions HTTP/1.1\r\n"
             "Host: localhost\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\n\r\n%s",
             strlen(stream_body), stream_body);
    response = exchange(server, request);
    check(response && strstr(response, "text/event-stream") &&
              strstr(response, "reasoning_content") &&
              strstr(response, "data: [DONE]"),
          "OpenAI streaming chat");
    free(response);

    const char *anthropic_body =
        "{\"model\":\"qwen3.8-flash-next\",\"max_tokens\":16,"
        "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}";
    snprintf(request, sizeof(request),
             "POST /v1/messages HTTP/1.1\r\n"
             "Host: localhost\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\n\r\n%s",
             strlen(anthropic_body), anthropic_body);
    response = exchange(server, request);
    check(response && strstr(response, "\"type\":\"message\"") &&
              strstr(response, "Q38 mock response"),
          "Anthropic message endpoint");
    free(response);

    const char *completion_body =
        "{\"model\":\"qwen3.8-flash-next\",\"prompt\":\"hello\","
        "\"max_tokens\":4}";
    snprintf(request, sizeof(request),
             "POST /v1/completions HTTP/1.1\r\n"
             "Host: localhost\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\n\r\n%s",
             strlen(completion_body), completion_body);
    response = exchange(server, request);
    check(response && strstr(response, "\"object\":\"text_completion\"") &&
              strstr(response, "\"text\":\"This is a deterministic"),
          "OpenAI legacy completion endpoint");
    free(response);

    const char *responses_body =
        "{\"model\":\"qwen3.8-flash-next\",\"input\":\"hello\"}";
    snprintf(request, sizeof(request),
             "POST /v1/responses HTTP/1.1\r\n"
             "Host: localhost\r\nContent-Type: application/json\r\n"
             "Content-Length: %zu\r\n\r\n%s",
             strlen(responses_body), responses_body);
    response = exchange(server, request);
    check(response && strstr(response, "\"object\":\"response\"") &&
              strstr(response, "\"output_text\":\"This is a deterministic"),
          "OpenAI Responses endpoint");
    free(response);

    response = exchange(server,
        "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n");
    check(response && strstr(response, "204 No Content") &&
              strstr(response, "Access-Control-Allow-Origin: *"),
          "CORS preflight");
    free(response);

    response = exchange(server,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
        "Content-Length: 2\r\n\r\n{}");
    check(response && strstr(response, "400 Bad Request"), "invalid request");
    free(response);

    q38_server_destroy(server);
    if (failures) return 1;
    puts("test_q38_server: all tests passed");
    return 0;
}
