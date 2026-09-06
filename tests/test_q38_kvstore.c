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
    q38_kvstore_destroy(&store);
    puts("test_q38_kvstore: disabled capability gate passed");
    return 0;
}
