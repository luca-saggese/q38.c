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
    const uint64_t token_position = 7;
    q38_ple_scheduler_record_timeline(
        scheduler, Q38_PLE_T0_TOKEN_FORWARD_BEGIN, 0, token_position);
    if (!q38_ple_scheduler_submit(scheduler, captured_rows,
                                  sizeof(captured_rows) / sizeof(*captured_rows),
                                  error, sizeof(error)) ||
        !q38_ple_scheduler_record_timeline(
            scheduler, Q38_PLE_T2_LAYER0_BEGIN, 0, token_position) ||
        !q38_ple_scheduler_record_timeline(
            scheduler, Q38_PLE_T3_LAYER0_END, 0, token_position) ||
        !q38_ple_scheduler_record_timeline(
            scheduler, Q38_PLE_T4_LAYER1_BEGIN, 0, token_position) ||
        !q38_ple_scheduler_wait(scheduler, error, sizeof(error)) ||
        !q38_ple_scheduler_record_injection_begin(scheduler) ||
        !q38_ple_scheduler_record_injection_timing(scheduler, 0.0, 0.0, 0.0) ||
        !q38_ple_scheduler_record_timeline(
            scheduler, Q38_PLE_T5_LAYER1_END, 0, token_position) ||
        !q38_ple_scheduler_record_timeline(
            scheduler, Q38_PLE_T11_TOKEN_FORWARD_END, 0, token_position)) {
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
    const double token_wall =
        stats.t11_token_forward_end_ms - stats.t0_token_forward_begin_ms;
    const double true_overlap =
        stats.t6_ple_injection_arrival_ms -
        stats.t1_ple_request_submit_ms;
    if (stats.request_id == 0 || stats.token_position != token_position ||
        stats.submit_position != token_position ||
        stats.injection_position != token_position ||
        stats.t0_token_forward_begin_ms > stats.t1_ple_request_submit_ms ||
        stats.t1_ple_request_submit_ms >
            stats.t6_ple_injection_arrival_ms ||
        stats.t6_ple_injection_arrival_ms >
            stats.t11_token_forward_end_ms ||
        stats.t8_ple_wait_end_ms < stats.t7_ple_wait_begin_ms ||
        stats.t10_ple_injection_end_ms <
            stats.t9_ple_injection_begin_ms ||
        true_overlap < 0.0 || token_wall <= 0.0 ||
        true_overlap >= token_wall) {
        fprintf(stderr, "invalid PLE timeline boundaries\n");
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
            ",\"sequential\":%s,\"mode\":\"buffered_pread_transient\","
            "\"timeline\":{\"request_id\":%" PRIu64
            ",\"token_position\":%" PRIu64
            ",\"submit_position\":%" PRIu64
            ",\"injection_position\":%" PRIu64
            ",\"T0_token_forward_begin_ms\":%.6f"
            ",\"T1_ple_request_submit_ms\":%.6f"
            ",\"T2_layer0_begin_ms\":%.6f"
            ",\"T3_layer0_end_ms\":%.6f"
            ",\"T4_layer1_begin_ms\":%.6f"
            ",\"T5_layer1_end_ms\":%.6f"
            ",\"T6_ple_injection_arrival_ms\":%.6f"
            ",\"T7_ple_wait_begin_ms\":%.6f"
            ",\"T8_ple_wait_end_ms\":%.6f"
            ",\"T9_ple_injection_begin_ms\":%.6f"
            ",\"T10_ple_injection_end_ms\":%.6f"
            ",\"T11_token_forward_end_ms\":%.6f"
            ",\"true_ple_overlap_window_ms\":%.6f"
            ",\"token_wall_ms\":%.6f,\"wait_at_injection_ms\":%.6f"
            ",\"synchronous_injection_ms\":%.6f,\"overlap_fraction\":%.6f"
            ",\"overlap_small\":%s,\"boundaries_valid\":true}}\n",
            stats.logical_accesses, stats.unique_rows, stats.logical_bytes,
            stats.file_read_ops, stats.file_read_min_bytes,
            stats.file_read_max_bytes, stats.file_io_ms, stats.elapsed_ms,
            stats.worker_cpu_ms, stats.cache_hits, stats.cache_misses,
            stats.file_reads_sequential ? "true" : "false",
            stats.request_id, stats.token_position, stats.submit_position,
            stats.injection_position, stats.t0_token_forward_begin_ms,
            stats.t1_ple_request_submit_ms, stats.t2_layer0_begin_ms,
            stats.t3_layer0_end_ms, stats.t4_layer1_begin_ms,
            stats.t5_layer1_end_ms, stats.t6_ple_injection_arrival_ms,
            stats.t7_ple_wait_begin_ms, stats.t8_ple_wait_end_ms,
            stats.t9_ple_injection_begin_ms,
            stats.t10_ple_injection_end_ms,
            stats.t11_token_forward_end_ms, true_overlap, token_wall,
            stats.wait_at_injection_ms, stats.injection_ms,
            token_wall > 0.0 ? true_overlap / token_wall : 0.0,
            token_wall > 0.0 && true_overlap / token_wall < 0.25
                ? "true" : "false");
    fclose(out);
    q38_ple_scheduler_destroy(scheduler);
    q38_gguf_close(model);
    puts("test_s4c_ple_replay: captured PLE file-backed replay passed");
    return 0;
}
