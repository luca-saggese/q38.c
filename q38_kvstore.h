#ifndef Q38_KVSTORE_H
#define Q38_KVSTORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    Q38_KVSTORE_OK = 0,
    Q38_KVSTORE_DISABLED,
    Q38_KVSTORE_NOT_FOUND,
    Q38_KVSTORE_ERROR,
} q38_kvstore_result;

typedef struct {
    char *root;
    bool enabled;
} q38_kvstore;

typedef struct {
    uint64_t steering_fingerprint;
    float steering_ffn_scale;
    float steering_attn_scale;
    uint32_t steering_abi;
} q38_kvstore_steering_metadata;

bool q38_kvstore_steering_compatible(
    const q38_kvstore_steering_metadata *saved,
    const q38_kvstore_steering_metadata *requested);

bool q38_kvstore_init(q38_kvstore *store, const char *root,
                      char *error, size_t error_len);
void q38_kvstore_destroy(q38_kvstore *store);

q38_kvstore_result q38_kvstore_read(q38_kvstore *store, const char *session_id,
                                    void *data, size_t capacity, size_t *size,
                                    char *error, size_t error_len);
q38_kvstore_result q38_kvstore_write(q38_kvstore *store, const char *session_id,
                                     const void *data, size_t size,
                                     char *error, size_t error_len);
q38_kvstore_result q38_kvstore_remove(q38_kvstore *store,
                                      const char *session_id,
                                      char *error, size_t error_len);

#endif
