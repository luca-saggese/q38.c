#include "q38_kvstore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool kv_error(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_kvstore_init(q38_kvstore *store, const char *root,
                      char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!store) return kv_error(error, error_len, "KV store is null");
    memset(store, 0, sizeof(*store));
    if (root && root[0]) {
        store->root = strdup(root);
        if (!store->root)
            return kv_error(error, error_len, "KV store root allocation failed");
    }
    /*
     * The disk format is intentionally not enabled until Q38 session state
     * serialization is available. Requests can still carry cache hints.
     */
    store->enabled = false;
    return true;
}

void q38_kvstore_destroy(q38_kvstore *store) {
    if (!store) return;
    free(store->root);
    memset(store, 0, sizeof(*store));
}

q38_kvstore_result q38_kvstore_read(q38_kvstore *store, const char *session_id,
                                    void *data, size_t capacity, size_t *size,
                                    char *error, size_t error_len) {
    (void)session_id;
    (void)data;
    (void)capacity;
    if (size) *size = 0;
    if (error && error_len) error[0] = '\0';
    if (!store || !session_id) {
        kv_error(error, error_len, "invalid KV read request");
        return Q38_KVSTORE_ERROR;
    }
    if (!store->enabled) {
        kv_error(error, error_len,
                 "disk KV/session cache is unavailable for Q38 runtime");
        return Q38_KVSTORE_DISABLED;
    }
    return Q38_KVSTORE_NOT_FOUND;
}

q38_kvstore_result q38_kvstore_write(q38_kvstore *store, const char *session_id,
                                     const void *data, size_t size,
                                     char *error, size_t error_len) {
    (void)data;
    (void)size;
    if (error && error_len) error[0] = '\0';
    if (!store || !session_id) {
        kv_error(error, error_len, "invalid KV write request");
        return Q38_KVSTORE_ERROR;
    }
    if (!store->enabled) {
        kv_error(error, error_len,
                 "disk KV/session cache is unavailable for Q38 runtime");
        return Q38_KVSTORE_DISABLED;
    }
    return Q38_KVSTORE_OK;
}

q38_kvstore_result q38_kvstore_remove(q38_kvstore *store,
                                      const char *session_id,
                                      char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!store || !session_id) {
        kv_error(error, error_len, "invalid KV remove request");
        return Q38_KVSTORE_ERROR;
    }
    if (!store->enabled) {
        kv_error(error, error_len,
                 "disk KV/session cache is unavailable for Q38 runtime");
        return Q38_KVSTORE_DISABLED;
    }
    return Q38_KVSTORE_OK;
}
