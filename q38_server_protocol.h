#ifndef Q38_SERVER_PROTOCOL_H
#define Q38_SERVER_PROTOCOL_H

#include "q38_server_engine.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    Q38_SERVER_ENDPOINT_CHAT_COMPLETIONS = 0,
    Q38_SERVER_ENDPOINT_COMPLETIONS,
    Q38_SERVER_ENDPOINT_RESPONSES,
    Q38_SERVER_ENDPOINT_MESSAGES,
} q38_server_endpoint;

bool q38_server_parse_request(q38_server_endpoint endpoint,
                              const char *body,
                              q38_server_request *request,
                              char *error, size_t error_len);

const char *q38_server_endpoint_name(q38_server_endpoint endpoint);

#endif
