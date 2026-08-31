#include "q38_gguf.h"
#include "q38_weights.h"

#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

static size_t fd_count(void) {
    size_t count = 0;
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) return 0;
    while (readdir(dir)) count++;
    closedir(dir);
    return count;
}

static long rss_kb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    const size_t baseline_fds = fd_count();
    const long baseline_rss = rss_kb();
    long max_rss = baseline_rss;
    for (unsigned iteration = 0; iteration < 20; iteration++) {
        char error[256];
        q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
        if (!model) {
            fprintf(stderr, "iteration %u open failed: %s\n", iteration, error);
            return 1;
        }
        q38_weights weights;
        if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
            fprintf(stderr, "iteration %u bind failed: %s\n", iteration, error);
            q38_gguf_close(model);
            return 1;
        }
        q38_gguf_close(model);
        size_t fds = fd_count();
        long rss = rss_kb();
        if (rss > max_rss) max_rss = rss;
        if (fds != baseline_fds) {
            fprintf(stderr, "FD count grew at iteration %u: %zu vs %zu\n",
                    iteration, fds, baseline_fds);
            return 1;
        }
    }
    if (max_rss - baseline_rss > 256 * 1024) {
        fprintf(stderr, "RSS grew by %ld KiB\n", max_rss - baseline_rss);
        return 1;
    }
    printf("{\"gate\":\"M2-C10\",\"iterations\":20,\"baseline_fds\":%zu,"
           "\"final_fds\":%zu,\"maxrss_delta_kb\":%ld,\"status\":\"pass\"}\n",
           baseline_fds, fd_count(), max_rss - baseline_rss);
    return 0;
}
