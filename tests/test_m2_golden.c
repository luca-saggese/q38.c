#include "q38_golden.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    static const uint32_t values[] = {0, 1, 0xdeadbeefU, 42};
    q38_golden_meta meta = {
        .stage = "embedding",
        .layer = -1,
        .token_pos = 3,
        .dtype = Q38_DTYPE_U32,
        .elements = sizeof(values) / sizeof(values[0]),
    };
    char error[256];
    if (!q38_golden_write("artifacts/m2-test-golden.bin", &meta, values,
                          sizeof(values), error, sizeof(error))) {
        fprintf(stderr, "write failed: %s\n", error);
        return 1;
    }
    q38_golden_vector got;
    if (!q38_golden_read("artifacts/m2-test-golden.bin", &got, error,
                         sizeof(error))) {
        fprintf(stderr, "read failed: %s\n", error);
        return 1;
    }
    q38_golden_vector expected = {
        .meta = got.meta, .data = (void *)values, .data_bytes = sizeof(values)
    };
    if (!q38_golden_equal(&got, &expected) ||
        got.meta.checksum != q38_golden_checksum(&got.meta, values, sizeof(values))) {
        fprintf(stderr, "golden roundtrip mismatch\n");
        q38_golden_free(&got);
        return 1;
    }
    q38_golden_free(&got);
    remove("artifacts/m2-test-golden.bin");
    puts("test_m2_golden: binary roundtrip and checksum passed");
    return 0;
}
