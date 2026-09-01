#include "q38_ple_cache.h"
#include "q38_ple_ref.h"
#include "q38_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *message) {
    fprintf(stderr, "test_m4_ple_cache: %s\n", message);
    return 1;
}

static void make_row(q38_q2_k_block *row, uint8_t value) {
    memset(row, 0, sizeof(*row));
    row->d = 0x3c00;
    for (size_t i = 0; i < sizeof(row->scales); ++i) row->scales[i] = 1;
    for (size_t i = 0; i < sizeof(row->qs); ++i) row->qs[i] = value;
}

static int decode_equal(const q38_q2_k_block *left,
                        const q38_q2_k_block *right) {
    float left_values[Q38_QUANT_QK_K];
    float right_values[Q38_QUANT_QK_K];
    char error[128];
    if (!q38_ple_decode_row_ref(Q38_QUANT_Q2_K, left, Q38_QUANT_QK_K,
                                left_values, Q38_QUANT_QK_K, error,
                                sizeof(error)) ||
        !q38_ple_decode_row_ref(Q38_QUANT_Q2_K, right, Q38_QUANT_QK_K,
                                right_values, Q38_QUANT_QK_K, error,
                                sizeof(error))) {
        fprintf(stderr, "test_m4_ple_cache: row decode failed: %s\n", error);
        return 0;
    }
    for (size_t i = 0; i < Q38_QUANT_QK_K; ++i) {
        if (fabsf(left_values[i] - right_values[i]) > 1e-6f) return 0;
    }
    return 1;
}

static int write_stats(const char *path, const char *mode,
                       const q38_ple_cache *cache) {
    const q38_ple_cache_stats *stats = q38_ple_cache_get_stats(cache);
    FILE *fp = fopen(path, "w");
    if (!fp) return 0;
    fprintf(fp,
            "{\"gate\":\"M4-C08\",\"mode\":\"%s\","
            "\"lookup_count\":%llu,\"cache_hits\":%llu,"
            "\"cache_misses\":%llu,\"insert_count\":%llu,"
            "\"eviction_count\":%llu,\"resident_bytes\":%zu,"
            "\"status\":\"pass\"}\n",
            mode, (unsigned long long)stats->lookup_count,
            (unsigned long long)stats->cache_hits,
            (unsigned long long)stats->cache_misses,
            (unsigned long long)stats->insert_count,
            (unsigned long long)stats->eviction_count,
            q38_ple_cache_resident_bytes(cache));
    fclose(fp);
    return 1;
}

static int run_cache(const char *cold_report, const char *warm_report) {
    enum { ROW_COUNT = 5 };
    q38_q2_k_block source[ROW_COUNT];
    for (size_t row = 0; row < ROW_COUNT; ++row)
        make_row(&source[row], (uint8_t)(0xe4u ^ (uint8_t)row));

    q38_ple_cache cache;
    char error[128];
    if (!q38_ple_cache_init(&cache, 3 * sizeof(source[0]),
                            sizeof(source[0]), error, sizeof(error))) {
        fprintf(stderr, "test_m4_ple_cache: init failed: %s\n", error);
        return 0;
    }
    if (q38_ple_cache_resident_bytes(&cache) > 3 * sizeof(source[0])) {
        q38_ple_cache_destroy(&cache);
        fail("cache resident bytes exceed configured budget");
        return 0;
    }

    for (uint64_t row = 0; row < ROW_COUNT; ++row) {
        q38_q2_k_block actual;
        if (q38_ple_cache_lookup(&cache, row, &actual, error,
                                 sizeof(error))) {
            q38_ple_cache_destroy(&cache);
            return fail("cold cache unexpectedly hit") == 0;
        }
        if (!q38_ple_cache_insert(&cache, row, &source[row], error,
                                  sizeof(error))) {
            fprintf(stderr, "test_m4_ple_cache: insert failed: %s\n", error);
            q38_ple_cache_destroy(&cache);
            return 0;
        }
    }
    const q38_ple_cache_stats *cold = q38_ple_cache_get_stats(&cache);
    if (cold->cache_hits != 0 || cold->cache_misses != ROW_COUNT ||
        cold->insert_count != ROW_COUNT || cold->eviction_count != 2) {
        q38_ple_cache_destroy(&cache);
        fail("cold cache statistics are incorrect");
        return 0;
    }
    if (!write_stats(cold_report, "cold", &cache)) {
        q38_ple_cache_destroy(&cache);
        fail("cannot write cold cache report");
        return 0;
    }

    for (uint64_t row = 2; row < ROW_COUNT; ++row) {
        q38_q2_k_block actual;
        if (!q38_ple_cache_lookup(&cache, row, &actual, error,
                                  sizeof(error)) ||
            !decode_equal(&actual, &source[row])) {
            q38_ple_cache_destroy(&cache);
            return fail("warm cache row differs from direct decode") == 0;
        }
    }
    const q38_ple_cache_stats *warm = q38_ple_cache_get_stats(&cache);
    if (warm->cache_hits != 3 || warm->cache_misses != ROW_COUNT) {
        q38_ple_cache_destroy(&cache);
        fail("warm cache statistics are incorrect");
        return 0;
    }
    if (!write_stats(warm_report, "warm", &cache)) {
        q38_ple_cache_destroy(&cache);
        fail("cannot write warm cache report");
        return 0;
    }

    q38_ple_cache other;
    if (!q38_ple_cache_init(&other, 3 * sizeof(source[0]),
                            sizeof(source[0]), error, sizeof(error))) {
        q38_ple_cache_destroy(&cache);
        fprintf(stderr, "test_m4_ple_cache: second init failed: %s\n", error);
        return 0;
    }
    for (uint64_t row = 0; row < ROW_COUNT; ++row) {
        if (!q38_ple_cache_insert(&other, row, &source[row], error,
                                  sizeof(error))) {
            q38_ple_cache_destroy(&other);
            q38_ple_cache_destroy(&cache);
            return 0;
        }
    }
    for (uint64_t row = 0; row < ROW_COUNT; ++row) {
        q38_q2_k_block left, right;
        const int left_hit =
            q38_ple_cache_lookup(&cache, row, &left, error, sizeof(error));
        const int right_hit =
            q38_ple_cache_lookup(&other, row, &right, error, sizeof(error));
        if (left_hit != right_hit || (left_hit && memcmp(&left, &right,
                                                         sizeof(left)) != 0)) {
            q38_ple_cache_destroy(&other);
            q38_ple_cache_destroy(&cache);
            fail("replacement is not deterministic");
            return 0;
        }
    }
    q38_ple_cache_destroy(&other);
    q38_ple_cache_destroy(&cache);
    return 1;
}

int main(int argc, char **argv) {
    const char *cold_report =
        argc > 1 ? argv[1] : "artifacts/m4/ple_cache_stats_cold.json";
    const char *warm_report =
        argc > 2 ? argv[2] : "artifacts/m4/ple_cache_stats_warm.json";
    if (!run_cache(cold_report, warm_report)) return 1;
    puts("test_m4_ple_cache: bounded deterministic quantized-row cache passed");
    puts("test_m4_ple_cache: cold and warm rows decode identically");
    return 0;
}
