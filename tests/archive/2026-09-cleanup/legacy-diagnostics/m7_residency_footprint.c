#include "q38_gguf.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int has(const q38_tensor *t, const char *needle) {
    size_t n = strlen(needle);
    if (n > t->name.len) return 0;
    for (size_t i = 0; i + n <= t->name.len; ++i)
        if (memcmp(t->name.ptr + i, needle, n) == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    char error[256] = {0};
    q38_gguf *m = q38_gguf_open(path, error, sizeof(error));
    if (!m) { fprintf(stderr, "%s\n", error); return 1; }
    const char *names[] = {"embedding", "GR", "GDN", "QSA", "router",
                           "shared_experts", "routed_experts_Q2", "LM-head"};
    uint64_t sums[8] = {0};
    uint64_t excluded_ple = 0, unclassified = 0;
    for (uint64_t i = 0; i < m->n_tensors; ++i) {
        const q38_tensor *t = &m->tensors[i];
        if (has(t, ".ple") || has(t, "ngram_heads") ||
            has(t, "layer_multipliers")) {
            excluded_ple += t->bytes;
            continue;
        }
        int category = -1;
        if (has(t, "embed_tokens")) category = 0;
        else if (has(t, "hyper_connection")) category = 1;
        else if (has(t, "linear_attn")) category = 2;
        else if (has(t, "self_attn") || has(t, "indexer")) category = 3;
        else if (has(t, ".mlp.gate.weight")) category = 4;
        else if (has(t, "shared_expert")) category = 5;
        else if (has(t, "experts.")) category = 6;
        else if (has(t, "lm_head.weight")) category = 7;
        if (category < 0) unclassified += t->bytes;
        else sums[category] += t->bytes;
    }
    uint64_t total = 0;
    for (int i = 0; i < 8; ++i) total += sums[i];
    printf("{\"format\":\"q38-m7-all-non-ple-footprint-v1\","
           "\"model_bytes\":%" PRIu64 ",\"non_ple_bytes\":%" PRIu64
           ",\"excluded_ple_bytes\":%" PRIu64 ",\"unclassified_bytes\":%" PRIu64
           ",\"categories\":{", m->size, total, excluded_ple, unclassified);
    for (int i = 0; i < 8; ++i)
        printf("%s\"%s\":%" PRIu64, i ? "," : "", names[i], sums[i]);
    puts("}}");
    q38_gguf_close(m);
    return 0;
}
