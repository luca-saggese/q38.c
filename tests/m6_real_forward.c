#include "q38_forward.h"
#include "q38_gguf.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *out;
    bool first;
} trace_context;

static bool trace(uint32_t layer, const float *hidden, size_t tokens,
                  size_t width, void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || !context->out || tokens != 1 || !hidden)
        return false;
    if (layer != UINT32_MAX &&
        layer != 0 && layer != 3 && layer != 7 && layer != 15 &&
        layer != 31 && layer != 47)
        return true;
    fprintf(context->out, "%s{\"stage\":\"%s\",\"layer\":%u,\"width\":%zu,"
            "\"values\":[", context->first ? "" : ",\n",
            layer == UINT32_MAX ? "final_norm" : "layer",
            layer == UINT32_MAX ? 0 : layer, width);
    for (size_t i = 0; i < (width < 8 ? width : 8); ++i)
        fprintf(context->out, "%s%.9g", i ? "," : "", hidden[i]);
    fprintf(context->out, "]}");
    context->first = false;
    (void)error;
    (void)error_len;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf output.json\n", argv[0]);
        return 2;
    }
    char error[256];
    fprintf(stderr, "open\n");
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open: %s\n", error);
        return 1;
    }
    q38_weights weights;
    fprintf(stderr, "bind\n");
    if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
        fprintf(stderr, "bind: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    q38_forward_state state;
    fprintf(stderr, "state\n");
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
    fputs("[\n", out);
    fprintf(stderr, "forward\n");
    trace_context context = {.out = out, .first = true};
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.trace = trace;
    diagnostics.trace_user = &context;
    uint32_t token = 9419;
    float *logits = calloc(248320, sizeof(float));
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
    fprintf(out, "%s{\"stage\":\"logits\",\"top\":[", context.first ? "" : ",\n");
    context.first = false;
    for (size_t n = 0; n < 10; ++n) {
        size_t best = 0;
        for (size_t i = 1; i < 248320; ++i)
            if (logits[i] > logits[best]) best = i;
        fprintf(out, "%s{\"id\":%zu,\"value\":%.9g}", n ? "," : "",
                best, logits[best]);
        logits[best] = -INFINITY;
    }
    fprintf(out, "]}\n]\n");
    fclose(out);
    free(logits);
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
