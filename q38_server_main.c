#include "q38_server.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr, "usage: %s [--host HOST] [--port PORT]\n", program);
}

int main(int argc, char **argv) {
    const char *host = Q38_SERVER_DEFAULT_HOST;
    unsigned long port = Q38_SERVER_DEFAULT_PORT;
    char error[256] = {0};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            char *end = NULL;
            port = strtoul(argv[++i], &end, 10);
            if (!end || *end || port > 65535) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    q38_server_engine *engine = q38_server_mock_engine_create(
        "qwen3.8-flash-next", error, sizeof(error));
    if (!engine) {
        fprintf(stderr, "q38-server: %s\n", error);
        return 1;
    }
    q38_server *server = q38_server_create(engine, true, error, sizeof(error));
    if (!server) {
        fprintf(stderr, "q38-server: %s\n", error);
        q38_server_engine_destroy(engine);
        return 1;
    }
    fprintf(stderr, "q38-server listening on %s:%lu (mock engine)\n",
            host, port);
    int result = q38_server_listen_and_serve(server, host, (uint16_t)port,
                                             error, sizeof(error));
    if (result != 0) fprintf(stderr, "q38-server: %s\n", error);
    q38_server_destroy(server);
    return result == 0 ? 0 : 1;
}
