#include "q38_cuda.h"

#include <stdio.h>

int main(int argc, char **argv) {
    q38_cuda_shared_memory_info info;
    if (q38_cuda_get_shared_memory_info(&info) != 0 ||
        info.cc_major != 12 || info.cc_minor != 1 ||
        info.warp_size <= 0 || info.multiprocessor_count <= 0 ||
        info.max_shared_memory_per_block <= 0 ||
        info.max_shared_memory_per_block_optin <
            info.max_shared_memory_per_block)
        return 1;
    FILE *out = argc > 1 ? fopen(argv[1], "w") : NULL;
    if (!out) return 2;
    fprintf(out, "{\"cc\":\"sm_%d%d\",\"max_shared_memory_per_block\":%d,"
            "\"max_shared_memory_per_block_optin\":%d,\"warp_size\":%d,"
            "\"multiprocessor_count\":%d,\"variants\":[{\"tile\":\"small\","
            "\"shared_bytes\":0},{\"tile\":\"large\",\"shared_bytes\":%d}],"
            "\"status\":\"pass\"}\n", info.cc_major, info.cc_minor,
            info.max_shared_memory_per_block,
            info.max_shared_memory_per_block_optin, info.warp_size,
            info.multiprocessor_count, info.max_shared_memory_per_block_optin);
    fclose(out);
    puts("test_m5_bis_shared: GB10 shared-memory limits and kernel variants recorded");
    return 0;
}
