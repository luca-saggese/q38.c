#include "q38_server.h"
#include "q38_server_engine_q38.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --model GGUF --tokenizer DIR "
            "[--ctx-size N] [--host HOST] [--port PORT] "
            "[--dir-steering-file FILE] [--dir-steering-ffn F] "
            "[--dir-steering-attn F]\n", program);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *tokenizer = NULL;
    const char *host = Q38_SERVER_DEFAULT_HOST;
    uint32_t ctx_size = 8192;
    unsigned long port = Q38_SERVER_DEFAULT_PORT;
    const char *steering_file = NULL;
    float steering_ffn = 0.0f;
    float steering_attn = 0.0f;
    bool steering_scale_set = false;
    char error[256] = {0};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model = argv[++i];
        } else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) {
            tokenizer = argv[++i];
        } else if (!strcmp(argv[i], "--ctx-size") && i + 1 < argc) {
            char *end = NULL;
            unsigned long value = strtoul(argv[++i], &end, 10);
            if (!end || *end || !value || value > UINT32_MAX) {
                usage(argv[0]);
                return 2;
            }
            ctx_size = (uint32_t)value;
        } else if (!strcmp(argv[i], "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            char *end = NULL;
            port = strtoul(argv[++i], &end, 10);
            if (!end || *end || port > 65535) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--dir-steering-file") &&
                   i + 1 < argc) {
            steering_file = argv[++i];
        } else if ((!strcmp(argv[i], "--dir-steering-ffn") ||
                    !strcmp(argv[i], "--dir-steering-attn")) &&
                   i + 1 < argc) {
            char *end = NULL;
            const float value = strtof(argv[++i], &end);
            if (!end || *end || !isfinite(value) ||
                value < -100.0f || value > 100.0f) {
                usage(argv[0]);
                return 2;
            }
            steering_scale_set = true;
            if (!strcmp(argv[i - 1], "--dir-steering-ffn"))
                steering_ffn = value;
            else
                steering_attn = value;
        } else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model || !tokenizer) {
        usage(argv[0]);
        return 2;
    }
    if (steering_file && !steering_scale_set) steering_ffn = 1.0f;
    fprintf(stderr,
            "q38-server-real: loading one resident model; do not run this "
            "while a full benchmark worker is active\n");
    q38_server_engine *engine = q38_server_q38_engine_create_with_steering(
        model, tokenizer, ctx_size, steering_file, steering_ffn, steering_attn,
        error, sizeof(error));
    if (!engine) {
        fprintf(stderr, "q38-server-real: %s\n", error);
        return 1;
    }
    q38_server *server = q38_server_create(engine, true, error, sizeof(error));
    if (!server) {
        fprintf(stderr, "q38-server-real: %s\n", error);
        q38_server_engine_destroy(engine);
        return 1;
    }
    int result = q38_server_listen_and_serve(server, host, (uint16_t)port,
                                             error, sizeof(error));
    if (result != 0) fprintf(stderr, "q38-server-real: %s\n", error);
    q38_server_destroy(server);
    return result == 0 ? 0 : 1;
}
