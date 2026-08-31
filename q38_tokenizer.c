#include "q38_tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void set_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
}

static bool write_all(int fd, const char *data, size_t size) {
    while (size) {
        ssize_t written = write(fd, data, size);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        data += written;
        size -= (size_t)written;
    }
    return true;
}

static char *read_all(int fd, size_t *size_out) {
    size_t size = 0, capacity = 256;
    char *data = malloc(capacity + 1);
    if (!data) return NULL;
    for (;;) {
        if (size == capacity) {
            capacity *= 2;
            char *grown = realloc(data, capacity + 1);
            if (!grown) {
                free(data);
                return NULL;
            }
            data = grown;
        }
        ssize_t got = read(fd, data + size, capacity - size);
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) {
            free(data);
            return NULL;
        }
        if (!got) break;
        size += (size_t)got;
    }
    data[size] = '\0';
    if (size_out) *size_out = size;
    return data;
}

static bool json_string(const char *text, char **out) {
    size_t size = 2;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++)
        size += *p < 0x20 ? 6 : (*p == '"' || *p == '\\') ? 2 : 1;
    char *result = malloc(size + 1);
    if (!result) return false;
    char *q = result;
    *q++ = '"';
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"' || *p == '\\') {
            *q++ = '\\';
            *q++ = (char)*p;
        } else if (*p < 0x20) {
            static const char hex[] = "0123456789abcdef";
            *q++ = '\\';
            *q++ = 'u';
            *q++ = '0';
            *q++ = '0';
            *q++ = hex[*p >> 4];
            *q++ = hex[*p & 15];
        } else {
            *q++ = (char)*p;
        }
    }
    *q++ = '"';
    *q = '\0';
    *out = result;
    return true;
}

static bool run_reference(const q38_tokenizer *tokenizer, const char *request,
                          char **response, char *error, size_t error_len) {
    int input_pipe[2], output_pipe[2];
    if (pipe(input_pipe) || pipe(output_pipe)) {
        set_error(error, error_len, "cannot create tokenizer pipes");
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(input_pipe[0]); close(input_pipe[1]);
        close(output_pipe[0]); close(output_pipe[1]);
        set_error(error, error_len, "cannot start tokenizer reference");
        return false;
    }
    if (pid == 0) {
        dup2(input_pipe[0], STDIN_FILENO);
        dup2(output_pipe[1], STDOUT_FILENO);
        close(input_pipe[0]); close(input_pipe[1]);
        close(output_pipe[0]); close(output_pipe[1]);
        execlp("python3", "python3", tokenizer->reference_script,
               "--model-dir", tokenizer->model_dir, (char *)NULL);
        _exit(127);
    }
    close(input_pipe[0]);
    close(output_pipe[1]);
    bool wrote = write_all(input_pipe[1], request, strlen(request));
    close(input_pipe[1]);
    size_t response_size = 0;
    char *reply = wrote ? read_all(output_pipe[0], &response_size) : NULL;
    close(output_pipe[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (!wrote || !reply || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        response_size == 0) {
        free(reply);
        set_error(error, error_len, "tokenizer reference process failed");
        return false;
    }
    *response = reply;
    return true;
}

static bool parse_ids(const char *response, q38_token_batch *out,
                      char *error, size_t error_len) {
    const char *error_field = strstr(response, "\"error\":\"");
    if (error_field) {
        error_field += strlen("\"error\":\"");
        char message[256];
        size_t i = 0;
        while (error_field[i] && error_field[i] != '"' && i + 1 < sizeof(message)) {
            message[i] = error_field[i];
            i++;
        }
        message[i] = '\0';
        set_error(error, error_len, message);
        return false;
    }
    const char *p = strstr(response, "\"ids\":[");
    if (!p) {
        set_error(error, error_len, "tokenizer reference returned no IDs");
        return false;
    }
    p += strlen("\"ids\":[");
    uint32_t capacity = 16, count = 0;
    uint32_t *tokens = malloc(capacity * sizeof(*tokens));
    if (!tokens) {
        set_error(error, error_len, "out of memory for token IDs");
        return false;
    }
    while (*p && *p != ']') {
        char *end = NULL;
        errno = 0;
        unsigned long value = strtoul(p, &end, 10);
        if (end == p || errno == ERANGE || value > UINT32_MAX) {
            free(tokens);
            set_error(error, error_len, "invalid token ID response");
            return false;
        }
        if (count == capacity) {
            capacity *= 2;
            uint32_t *grown = realloc(tokens, capacity * sizeof(*tokens));
            if (!grown) {
                free(tokens);
                set_error(error, error_len, "out of memory for token IDs");
                return false;
            }
            tokens = grown;
        }
        tokens[count++] = (uint32_t)value;
        p = end;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == ',') p++;
        else if (*p != ']') {
            free(tokens);
            set_error(error, error_len, "invalid token ID separator");
            return false;
        }
    }
    if (*p != ']') {
        free(tokens);
        set_error(error, error_len, "unterminated token ID response");
        return false;
    }
    out->tokens = tokens;
    out->token_count = count;
    return true;
}

bool q38_tokenizer_init(q38_tokenizer *tokenizer, const char *model_dir,
                        const char *reference_script, char *error,
                        size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!tokenizer || !model_dir || !reference_script) {
        set_error(error, error_len, "invalid tokenizer arguments");
        return false;
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
    tokenizer->model_dir = strdup(model_dir);
    tokenizer->reference_script = strdup(reference_script);
    if (!tokenizer->model_dir || !tokenizer->reference_script) {
        q38_tokenizer_destroy(tokenizer);
        set_error(error, error_len, "out of memory for tokenizer paths");
        return false;
    }
    return true;
}

void q38_tokenizer_destroy(q38_tokenizer *tokenizer) {
    if (!tokenizer) return;
    free(tokenizer->model_dir);
    free(tokenizer->reference_script);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

bool q38_tokenizer_encode(const q38_tokenizer *tokenizer, const char *text,
                          bool add_special_tokens, q38_token_batch *out,
                          char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!tokenizer || !text || !out) {
        set_error(error, error_len, "invalid tokenizer encode arguments");
        return false;
    }
    q38_token_batch_free(out);
    char *encoded = NULL;
    if (!json_string(text, &encoded)) {
        set_error(error, error_len, "out of memory for tokenizer request");
        return false;
    }
    size_t request_size = strlen(encoded) + 64;
    char *request = malloc(request_size);
    if (!request) {
        free(encoded);
        set_error(error, error_len, "out of memory for tokenizer request");
        return false;
    }
    snprintf(request, request_size, "{\"op\":\"encode\",\"text\":%s,\"add_special_tokens\":%s}\n",
             encoded, add_special_tokens ? "true" : "false");
    free(encoded);
    char *response = NULL;
    bool ok = run_reference(tokenizer, request, &response, error, error_len);
    free(request);
    if (ok) ok = parse_ids(response, out, error, error_len);
    free(response);
    return ok;
}

bool q38_tokenizer_encode_chat_json(const q38_tokenizer *tokenizer,
                                    const char *messages_json,
                                    bool add_generation_prompt,
                                    bool enable_thinking,
                                    q38_token_batch *out, char *error,
                                    size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!tokenizer || !messages_json || !out) {
        set_error(error, error_len, "invalid chat tokenizer arguments");
        return false;
    }
    q38_token_batch_free(out);
    size_t request_size = strlen(messages_json) + 160;
    char *request = malloc(request_size);
    if (!request) {
        set_error(error, error_len, "out of memory for tokenizer request");
        return false;
    }
    snprintf(request, request_size,
             "{\"op\":\"chat\",\"messages\":%s,\"add_generation_prompt\":%s,\"enable_thinking\":%s}\n",
             messages_json, add_generation_prompt ? "true" : "false",
             enable_thinking ? "true" : "false");
    char *response = NULL;
    bool ok = run_reference(tokenizer, request, &response, error, error_len);
    free(request);
    if (ok) ok = parse_ids(response, out, error, error_len);
    free(response);
    return ok;
}

void q38_token_batch_free(q38_token_batch *batch) {
    if (!batch) return;
    free(batch->tokens);
    memset(batch, 0, sizeof(*batch));
}
