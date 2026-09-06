#include "q38_kvstore.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char error[256] = {0};
    q38_kvstore store;
    if (!q38_kvstore_init(&store, "/tmp/q38-kv-test", error,
                          sizeof(error))) {
        fprintf(stderr, "KV init failed: %s\n", error);
        return 1;
    }
    size_t size = 0;
    if (q38_kvstore_read(&store, "session-1", NULL, 0, &size,
                         error, sizeof(error)) != Q38_KVSTORE_DISABLED ||
        !strstr(error, "unavailable")) {
        fprintf(stderr, "KV disabled read gate failed\n");
        q38_kvstore_destroy(&store);
        return 1;
    }
    q38_kvstore_steering_metadata saved = {
        .steering_fingerprint = 7,
        .steering_ffn_scale = -1.0f,
        .steering_attn_scale = 0.0f,
        .steering_abi = 1,
    };
    q38_kvstore_steering_metadata requested = saved;
    if (!q38_kvstore_steering_compatible(&saved, &requested)) {
        fprintf(stderr, "KV steering metadata compatibility failed\n");
        q38_kvstore_destroy(&store);
        return 1;
    }
    requested.steering_ffn_scale = 0.0f;
    if (q38_kvstore_steering_compatible(&saved, &requested)) {
        fprintf(stderr, "KV accepted incompatible steering metadata\n");
        q38_kvstore_destroy(&store);
        return 1;
    }
    q38_kvstore_destroy(&store);
    puts("test_q38_kvstore: disabled capability gate passed");
    return 0;
}
