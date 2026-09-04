#include "q38_forward.h"
#include "q38_gguf.h"
#include "q38_moe_ref.h"
#include "q38_weights.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 248320u
#define QSA_WIDTH 2051u

typedef struct {
    uint16_t route_ids[Q38_MODEL_LAYERS][Q38_MOE_TOP_K];
    float route_weights[Q38_MODEL_LAYERS][Q38_MOE_TOP_K];
    uint32_t qsa[Q38_MODEL_LAYERS][QSA_WIDTH];
    size_t qsa_count[Q38_MODEL_LAYERS];
} trace_state;

static bool route_trace(uint32_t layer, const uint16_t *experts,
                        const float *weights, size_t count, void *opaque,
                        char *error, size_t error_len) {
    trace_state *trace = opaque;
    if (!trace || layer >= Q38_MODEL_LAYERS || count != Q38_MOE_TOP_K) {
        snprintf(error, error_len, "invalid route trace");
        return false;
    }
    memcpy(trace->route_ids[layer], experts, count * sizeof(*experts));
    memcpy(trace->route_weights[layer], weights, count * sizeof(*weights));
    return true;
}

static bool qsa_trace(uint32_t layer, const uint32_t *selected, size_t count,
                      void *opaque, char *error, size_t error_len) {
    trace_state *trace = opaque;
    if (!trace || layer >= Q38_MODEL_LAYERS || count > QSA_WIDTH) {
        snprintf(error, error_len, "invalid QSA trace");
        return false;
    }
    memcpy(trace->qsa[layer], selected, count * sizeof(*selected));
    trace->qsa_count[layer] = count;
    return true;
}

static void json_float(float value) {
    if (isfinite(value))
        printf("%.9g", value);
    else
        fputs("null", stdout);
}

static bool json_string(const char *line, const char *key, char *value,
                        size_t value_size) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    char *start = strstr(line, needle);
    if (!start) return false;
    start += strlen(needle);
    while (*start == ' ' || *start == '\t') ++start;
    if (*start++ != ':') return false;
    while (*start == ' ' || *start == '\t') ++start;
    if (*start++ != '"') return false;
    char *end = strchr(start, '"');
    if (!end || (size_t)(end - start) + 1 > value_size) return false;
    memcpy(value, start, (size_t)(end - start));
    value[end - start] = '\0';
    return true;
}

static size_t parse_tokens(char *line, uint32_t *tokens, size_t capacity) {
    char *start = strstr(line, "\"token_ids\"");
    if (!start) return 0;
    start += strlen("\"token_ids\"");
    while (*start == ' ' || *start == '\t') ++start;
    if (*start++ != ':') return 0;
    while (*start == ' ' || *start == '\t') ++start;
    if (*start++ != '[') return 0;
    size_t count = 0;
    while (*start && *start != ']' && count < capacity) {
        char *end;
        errno = 0;
        unsigned long value = strtoul(start, &end, 10);
        if (end == start || errno || value >= VOCAB) return 0;
        tokens[count++] = (uint32_t)value;
        start = end;
        while (*start == ' ' || *start == ',') ++start;
    }
    return *start == ']' ? count : 0;
}

static void top20(const float *logits, uint32_t *ids, float *values) {
    for (size_t rank = 0; rank < 20; ++rank) {
        size_t best = VOCAB;
        for (size_t i = 0; i < VOCAB; ++i) {
            bool duplicate = false;
            for (size_t j = 0; j < rank; ++j)
                duplicate |= ids[j] == i;
            if (duplicate) continue;
            if (best == VOCAB || logits[i] > logits[best] ||
                (logits[i] == logits[best] && i < best))
                best = i;
        }
        ids[rank] = (uint32_t)best;
        values[rank] = logits[best];
    }
}

static void write_record(const char *id, const uint32_t *tokens, size_t count,
                         const float *logits, const trace_state *trace) {
    const float *last = logits + (count - 1) * VOCAB;
    uint32_t top_ids[20];
    float top_values[20];
    top20(last, top_ids, top_values);
    printf("{\"id\":\"%s\",\"tokens\":[", id);
    for (size_t i = 0; i < count; ++i)
        printf("%s%" PRIu32, i ? "," : "", tokens[i]);
    fputs("],\"logits\":[", stdout);
    for (size_t i = 0; i < VOCAB; ++i) {
        if (i) putchar(',');
        json_float(last[i]);
    }
    fputs("],\"target_logit\":", stdout);
    json_float(last[tokens[count - 1]]);
    fputs(",\"top20\":[", stdout);
    for (size_t i = 0; i < 20; ++i) {
        if (i) putchar(',');
        printf("{\"id\":%" PRIu32 ",\"value\":", top_ids[i]);
        json_float(top_values[i]);
        putchar('}');
    }
    fputs("],\"router_top10\":[", stdout);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) putchar(',');
        putchar('[');
        for (size_t i = 0; i < Q38_MOE_TOP_K; ++i) {
            if (i) putchar(',');
            printf("{\"id\":%u,\"weight\":", trace->route_ids[layer][i]);
            json_float(trace->route_weights[layer][i]);
            putchar('}');
        }
        putchar(']');
    }
    fputs("],\"qsa_selected\":[", stdout);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) putchar(',');
        putchar('[');
        for (size_t i = 0; i < trace->qsa_count[layer]; ++i)
            printf("%s%" PRIu32, i ? "," : "", trace->qsa[layer][i]);
        putchar(']');
    }
    fputs("]}\n", stdout);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf corpus.jsonl\n", argv[0]);
        return 2;
    }
    char error[256] = {0};
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
        fprintf(stderr, "bind: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    FILE *input = fopen(argv[2], "r");
    if (!input) {
        fprintf(stderr, "corpus: %s\n", strerror(errno));
        q38_gguf_close(model);
        return 1;
    }
    char line[8192];
    while (fgets(line, sizeof(line), input)) {
        if (!strchr(line, '\n')) {
            fprintf(stderr, "corpus line too long\n");
            fclose(input);
            q38_gguf_close(model);
            return 1;
        }
        char id[256];
        uint32_t tokens[256];
        size_t count = parse_tokens(line, tokens, 256);
        if (!json_string(line, "id", id, sizeof(id)) || !count) {
            fprintf(stderr, "invalid corpus record\n");
            fclose(input);
            q38_gguf_close(model);
            return 1;
        }
        q38_forward_state state;
        if (!q38_forward_state_init(&state, &weights, 248044, error,
                                    sizeof(error))) {
            fprintf(stderr, "state: %s\n", error);
            fclose(input);
            q38_gguf_close(model);
            return 1;
        }
        float *logits = calloc(count * (size_t)VOCAB, sizeof(*logits));
        trace_state trace = {0};
        q38_forward_diagnostics diagnostics = {0};
        diagnostics.route_trace = route_trace;
        diagnostics.qsa_trace = qsa_trace;
        diagnostics.trace_user = &trace;
        bool ok = logits &&
                  q38_forward_full(model, &weights, &state, tokens, count,
                                   logits, VOCAB, &diagnostics, error,
                                   sizeof(error));
        if (ok)
            write_record(id, tokens, count, logits, &trace);
        else {
            fprintf(stderr, "%s: forward: %s\n", id, error);
            free(logits);
            q38_forward_state_destroy(&state);
            fclose(input);
            q38_gguf_close(model);
            return 1;
        }
        free(logits);
        q38_forward_state_destroy(&state);
    }
    fclose(input);
    q38_gguf_close(model);
    return 0;
}
