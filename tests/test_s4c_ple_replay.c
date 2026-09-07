#include "q38_gguf.h"
#include "q38_ple.h"
#include "q38_ple_prefetch.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static const uint64_t captured_rows[] = {
    12109452, 28254228, 43544568, 75552284, 80806652, 109462696,
    138237652, 156144216, 168329152, 190666072, 201666848, 225260208,
    254098352, 272035424, 292035584, 319593848,
    18499624, 22365860, 54738584, 64328860, 90760884, 100379048,
    130401008, 151656052, 160449792, 192426524, 209290172, 239256296,
    245223472, 278778304, 298778464, 303161068,
    1064772, 39306444, 50712440, 73711868, 81125400, 116961256,
    132812660, 144562284, 172067744, 189246916, 200841072, 239936376,
    254640328, 264481376, 284481536, 310326380,
    1403032, 30638676, 54340568, 61757356, 87331412, 109191656,
    131052980, 154778860, 160985664, 190280976, 209434008, 227779860,
    257651064, 277413056, 297413216, 307105688
};

int main(int argc, char **argv) {
    char error[256];
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.json\n", argv[0]);
        return 2;
    }
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    q38_ple_store store;
    const char *prefix =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
    if (!q38_ple_store_bind_gguf(model, prefix, 128, 160, &store,
                                 error, sizeof(error))) {
        fprintf(stderr, "PLE bind failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    q38_ple_scheduler *scheduler =
        q38_ple_scheduler_create(&store, error, sizeof(error));
    if (!scheduler) {
        fprintf(stderr, "scheduler failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    if (!q38_ple_scheduler_submit(scheduler, captured_rows,
                                  sizeof(captured_rows) / sizeof(*captured_rows),
                                  error, sizeof(error)) ||
        !q38_ple_scheduler_wait(scheduler, error, sizeof(error))) {
        fprintf(stderr, "replay failed: %s\n", error);
        q38_ple_scheduler_destroy(scheduler);
        q38_gguf_close(model);
        return 1;
    }
    q38_ple_scheduler_stats stats;
    if (!q38_ple_scheduler_get_stats(scheduler, &stats)) {
        fprintf(stderr, "stats unavailable\n");
        q38_ple_scheduler_destroy(scheduler);
        q38_gguf_close(model);
        return 1;
    }
    FILE *out = fopen(argv[2], "w");
    if (!out) {
        perror("fopen");
        q38_ple_scheduler_destroy(scheduler);
        q38_gguf_close(model);
        return 1;
    }
    fprintf(out,
            "{\"format\":\"q38_s4c_ple_replay_v1\","
            "\"status\":\"pass\",\"source\":\"artifacts/m4/ple_injection_vectors.json\","
            "\"accesses\":%" PRIu64 ",\"unique_rows\":%" PRIu64
            ",\"bytes\":%" PRIu64 ",\"file_read_ops\":%" PRIu64
            ",\"file_read_min_bytes\":%" PRIu64
            ",\"file_read_max_bytes\":%" PRIu64
            ",\"file_io_ms\":%.6f,\"worker_elapsed_ms\":%.6f,"
            "\"worker_cpu_ms\":%.6f,\"cache_hits\":%" PRIu64
            ",\"cache_misses\":%" PRIu64
            ",\"sequential\":%s,\"mode\":\"buffered_pread_transient\"}\n",
            stats.logical_accesses, stats.unique_rows, stats.logical_bytes,
            stats.file_read_ops, stats.file_read_min_bytes,
            stats.file_read_max_bytes, stats.file_io_ms, stats.elapsed_ms,
            stats.worker_cpu_ms, stats.cache_hits, stats.cache_misses,
            stats.file_reads_sequential ? "true" : "false");
    fclose(out);
    q38_ple_scheduler_destroy(scheduler);
    q38_gguf_close(model);
    puts("test_s4c_ple_replay: captured PLE file-backed replay passed");
    return 0;
}
