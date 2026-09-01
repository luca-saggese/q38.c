#include "q38_forward.h"
#include "q38_gguf.h"
#include "q38_moe_ref.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t fixed_coordinates[] = {0, 1, 2, 3, 1024, 4096, 8191,
                                           10239};

typedef struct {
    FILE *out;
    bool first;
    uint16_t routed[Q38_MODEL_LAYERS][Q38_MOE_TOP_K];
    float route_weights[Q38_MODEL_LAYERS][Q38_MOE_TOP_K];
    size_t route_count[Q38_MODEL_LAYERS];
    uint32_t qsa_selected[Q38_MODEL_LAYERS][2051];
    size_t qsa_count[Q38_MODEL_LAYERS];
} trace_context;

static void json_float(FILE *out, float value) {
    if (isfinite(value))
        fprintf(out, "%.9g", value);
    else
        fputs("null", out);
}

static uint64_t checksum(const float *values, size_t count) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void write_stats(FILE *out, const float *values, size_t count) {
    double sum = 0.0, squares = 0.0;
    float minimum = INFINITY, maximum = -INFINITY, maximum_abs = 0.0f;
    size_t nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < count; ++i) {
        const float value = values[i];
        if (isnan(value)) {
            ++nan_count;
            continue;
        }
        if (isinf(value)) {
            ++inf_count;
            continue;
        }
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        if (fabsf(value) > maximum_abs) maximum_abs = fabsf(value);
        sum += value;
        squares += (double)value * value;
    }
    fprintf(out, "\"min\":");
    json_float(out, minimum);
    fprintf(out, ",\"max\":");
    json_float(out, maximum);
    fprintf(out, ",\"mean\":%.17g,\"rms\":%.17g,\"max_abs\":%.9g,"
                "\"nan_count\":%zu,\"inf_count\":%zu,\"checksum\":\"%016" PRIx64
                "\"",
            count ? sum / (double)count : NAN,
            count ? sqrt(squares / (double)count) : NAN, maximum_abs,
            nan_count, inf_count, checksum(values, count));
}

static void write_fixed(FILE *out, const float *values, size_t count) {
    fputs("\"fixed\":[", out);
    for (size_t i = 0; i < sizeof(fixed_coordinates) /
                               sizeof(fixed_coordinates[0]);
         ++i) {
        const size_t coordinate = fixed_coordinates[i];
        if (i) fputc(',', out);
        fprintf(out, "{\"index\":%zu,\"value\":", coordinate);
        if (coordinate < count)
            json_float(out, values[coordinate]);
        else
            fputs("null", out);
        fputc('}', out);
    }
    fputc(']', out);
}

static bool trace(uint32_t layer, const float *hidden, size_t tokens,
                  size_t width, void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || !context->out || tokens != 1 || !hidden)
        return false;
    if (layer != UINT32_MAX &&
        layer != 0 && layer != 3 && layer != 7 && layer != 15 &&
        layer != 31 && layer != 47)
        return true;
    FILE *out = context->out;
    fprintf(out, "%s{\"stage\":\"%s\",\"layer\":%u,\"width\":%zu,"
                "\"elements\":%zu,\"stats\":{",
            context->first ? "" : ",\n",
            layer == UINT32_MAX ? "final_norm" : "layer",
            layer == UINT32_MAX ? 0 : layer, width, width);
    write_stats(out, hidden, width);
    fputs("},", out);
    write_fixed(out, hidden, width);
    fputc('}', out);
    context->first = false;
    (void)error;
    (void)error_len;
    return true;
}

static bool route_trace(uint32_t layer, const uint16_t *experts,
                        const float *weights, size_t count, void *opaque,
                        char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || layer >= Q38_MODEL_LAYERS || count > Q38_MOE_TOP_K)
        return false;
    context->route_count[layer] = count;
    memcpy(context->routed[layer], experts, count * sizeof(*experts));
    memcpy(context->route_weights[layer], weights, count * sizeof(*weights));
    (void)error;
    (void)error_len;
    return true;
}

static bool qsa_trace(uint32_t layer, const uint32_t *selected, size_t count,
                      void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || layer >= Q38_MODEL_LAYERS || count > 2051)
        return false;
    context->qsa_count[layer] = count;
    memcpy(context->qsa_selected[layer], selected, count * sizeof(*selected));
    (void)error;
    (void)error_len;
    return true;
}

static void write_logits(FILE *out, const float *logits, size_t count) {
    fprintf(out, ",\n{\"stage\":\"logits\",\"layer\":0,\"width\":%zu,"
                "\"elements\":%zu,\"stats\":{",
            count, count);
    write_stats(out, logits, count);
    fputs("},", out);
    write_fixed(out, logits, count);
    fputs(",\"top\":[", out);
    float *copy = malloc(count * sizeof(*copy));
    if (!copy) return;
    memcpy(copy, logits, count * sizeof(*copy));
    for (size_t rank = 0; rank < 10; ++rank) {
        size_t best = 0;
        for (size_t i = 1; i < count; ++i)
            if (copy[i] > copy[best]) best = i;
        if (rank) fputc(',', out);
        fprintf(out, "{\"id\":%zu,\"value\":", best);
        json_float(out, copy[best]);
        fputc('}', out);
        copy[best] = -INFINITY;
    }
    fputs("]}", out);
    free(copy);
}

static void write_decisions(FILE *out, const trace_context *context) {
    fputs(",\n\"routing\":[", out);
    bool first = true;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (!context->route_count[layer]) continue;
        if (!first) fputc(',', out);
        first = false;
        fprintf(out, "{\"layer\":%zu,\"experts\":[", layer);
        for (size_t i = 0; i < context->route_count[layer]; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%u", context->routed[layer][i]);
        }
        fputs("],\"weights\":[", out);
        for (size_t i = 0; i < context->route_count[layer]; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->route_weights[layer][i]);
        }
        fputs("]}", out);
    }
    fputs("],\n\"qsa_selection\":[", out);
    first = true;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (!context->qsa_count[layer]) continue;
        if (!first) fputc(',', out);
        first = false;
        fprintf(out, "{\"layer\":%zu,\"selected\":[", layer);
        for (size_t i = 0; i < context->qsa_count[layer]; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%u", context->qsa_selected[layer][i]);
        }
        fputs("]}", out);
    }
    fputs("]", out);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf output.json\n", argv[0]);
        return 2;
    }
    char error[256];
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
    q38_forward_state state;
    if (!q38_forward_state_init(&state, &weights, 248044, error,
                                sizeof(error))) {
        fprintf(stderr, "state: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    FILE *out = fopen(argv[2], "w");
    if (!out) {
        perror("output");
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    fputs("{\"format\":\"q38-m6-full-trace-v2\",\"tokens\":[9419],"
          "\"stages\":[\n",
          out);
    trace_context context = {.out = out, .first = true};
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.trace = trace;
    diagnostics.route_trace = route_trace;
    diagnostics.qsa_trace = qsa_trace;
    diagnostics.trace_user = &context;
    float *logits = calloc(248320, sizeof(float));
    const uint32_t token = 9419;
    if (!logits || !q38_forward_full(model, &weights, &state, &token, 1,
                                     logits, 248320, &diagnostics, error,
                                     sizeof(error))) {
        fprintf(stderr, "forward: %s\n", error);
        free(logits);
        fclose(out);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    write_logits(out, logits, 248320);
    write_decisions(out, &context);
    fputs("\n]}\n", out);
    fclose(out);
    free(logits);
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
