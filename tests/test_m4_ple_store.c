#include "q38_ple.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    q38_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.ndim = 2;
    tensor.dim[0] = 2560;
    tensor.dim[1] = 1234;
    tensor.type = 1;
    tensor.abs_offset = UINT64_C(4096);
    tensor.bytes = tensor.dim[0] * tensor.dim[1] * 2;

    q38_ple_store store;
    char error[128];
    if (!q38_ple_store_bind(&tensor, 2560, &store, error, sizeof(error))) {
        fprintf(stderr, "PLE bind failed: %s\n", error);
        return 1;
    }
    uint64_t offset;
    if (store.rows != 1234 || store.row_width != 2560 ||
        store.row_bytes != 5120 ||
        !q38_ple_store_row_range(&store, 7, &offset, error, sizeof(error)) ||
        offset != UINT64_C(4096) + 7 * UINT64_C(5120)) {
        fprintf(stderr, "PLE row geometry/range mismatch: %s\n", error);
        return 1;
    }
    if (q38_ple_store_row_range(&store, store.rows, &offset,
                                error, sizeof(error))) {
        fprintf(stderr, "out-of-range PLE row was accepted\n");
        return 1;
    }
    tensor.dim[0] = 128;
    if (q38_ple_store_bind(&tensor, 2560, &store, error, sizeof(error))) {
        fprintf(stderr, "invalid PLE row width was accepted\n");
        return 1;
    }
    puts("test_m4_ple_store: strict table geometry and row addressing passed");
    return 0;
}
