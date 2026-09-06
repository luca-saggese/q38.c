#ifndef Q38_SERVER_H
#define Q38_SERVER_H

#include "q38_server_engine.h"
#include "q38_server_protocol.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define Q38_SERVER_DEFAULT_HOST "127.0.0.1"
#define Q38_SERVER_DEFAULT_PORT 8000
#define Q38_SERVER_MAX_HEADER_BYTES (64U * 1024U)
#define Q38_SERVER_MAX_BODY_BYTES (16U * 1024U * 1024U)

typedef struct q38_server q38_server;

typedef struct {
    char *method;
    char *target;
    char *version;
    char *body;
    size_t body_len;
} q38_http_request;

void q38_http_request_free(q38_http_request *request);

q38_server *q38_server_create(q38_server_engine *engine,
                              bool owns_engine,
                              char *error, size_t error_len);
void q38_server_destroy(q38_server *server);

int q38_server_handle_connection(q38_server *server, int fd,
                                 char *error, size_t error_len);
int q38_server_listen_and_serve(q38_server *server, const char *host,
                                uint16_t port, char *error, size_t error_len);

const char *q38_server_health_json(void);
const char *q38_server_models_json(const q38_server *server);

#endif
