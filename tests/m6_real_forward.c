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

static const size_t fixed_coordinates[] = {0, 1, 2, 3};

typedef struct {
    FILE *out;
    bool first;
    uint16_t routed[Q38_MODEL_LAYERS][Q38_MOE_TOP_K];
    float route_weights[Q38_MODEL_LAYERS][Q38_MOE_TOP_K];
    size_t route_count[Q38_MODEL_LAYERS];
    float router_logits[Q38_MODEL_LAYERS][Q38_MOE_EXPERTS];
    size_t router_count[Q38_MODEL_LAYERS];
    float router_input_by_layer[Q38_MODEL_LAYERS][Q38_MOE_HIDDEN];
    float routed_output_by_layer[Q38_MODEL_LAYERS][Q38_MOE_HIDDEN];
    bool moe_trace_by_layer[Q38_MODEL_LAYERS];
    bool moe_trace_valid;
    float router_input[Q38_MOE_HIDDEN];
    float router_logits_pre_cast[Q38_MOE_EXPERTS];
    float router_logits_effective[Q38_MOE_EXPERTS];
    uint16_t top15_rank[15];
    float top15_value[15];
    float selected_weights_pre_cast[Q38_MOE_TOP_K];
    float selected_weights_effective[Q38_MOE_TOP_K];
    uint16_t selected_experts[Q38_MOE_TOP_K];
    float routed_output[Q38_MOE_HIDDEN];
    float margin_rank10_rank11;
    q38_forward_dtype router_dtype;
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
    size_t minimum_index = 0, maximum_index = 0, maximum_abs_index = 0;
    size_t finite_count = 0, nan_count = 0, inf_count = 0;
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
        ++finite_count;
        if (value < minimum) {
            minimum = value;
            minimum_index = i;
        }
        if (value > maximum) {
            maximum = value;
            maximum_index = i;
        }
        if (fabsf(value) > maximum_abs) {
            maximum_abs = fabsf(value);
            maximum_abs_index = i;
        }
        sum += value;
        squares += (double)value * value;
    }
    fprintf(out, "\"min\":");
    json_float(out, minimum);
    fprintf(out, ",\"max\":");
    json_float(out, maximum);
    fprintf(out, ",\"finite_count\":%zu,\"mean\":%.17g,\"rms\":%.17g,\"max_abs\":%.9g,"
                "\"min_index\":%zu,\"max_index\":%zu,\"max_abs_index\":%zu,"
                "\"nan_count\":%zu,\"inf_count\":%zu,\"checksum\":\"%016" PRIx64
                "\"",
            finite_count, finite_count ? sum / (double)finite_count : NAN,
            finite_count ? sqrt(squares / (double)finite_count) : NAN,
            maximum_abs,
            minimum_index, maximum_index, maximum_abs_index, nan_count, inf_count,
            checksum(values, count));
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
    FILE *out = context->out;
    fprintf(out, "%s{\"stage\":\"%s\",\"layer\":%u,\"width\":%zu,"
                "\"elements\":%zu,\"stats\":{",
            context->first ? "" : ",\n",
            layer == UINT32_MAX ? "final_norm" : "layer",
            layer == UINT32_MAX ? 0 : layer, width, width);
    write_stats(out, hidden, width);
    fputs("},", out);
    write_fixed(out, hidden, width);
    if (layer == 7) {
        fputs(",\"values\":[", out);
        for (size_t i = 0; i < width; ++i) {
            if (i) fputc(',', out);
            json_float(out, hidden[i]);
        }
        fputc(']', out);
    }
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
    if (!context || layer >= Q38_MODEL_LAYERS || count > Q38_MOE_TOP_K ||
        (count && (!experts || !weights)))
        return false;
    context->route_count[layer] = count;
    memcpy(context->routed[layer], experts, count * sizeof(*experts));
    memcpy(context->route_weights[layer], weights, count * sizeof(*weights));
    (void)error;
    (void)error_len;
    return true;
}

static bool router_trace(uint32_t layer, const float *logits, size_t count,
                         void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || layer >= Q38_MODEL_LAYERS || count > Q38_MOE_EXPERTS ||
        (count && !logits))
        return false;
    context->router_count[layer] = count;
    memcpy(context->router_logits[layer], logits, count * sizeof(*logits));
    (void)error;
    (void)error_len;
    return true;
}

static bool moe_trace(uint32_t layer, const q38_moe_trace *trace,
                      void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || !trace ||
        layer >= Q38_MODEL_LAYERS ||
        !trace->router_input || !trace->router_logits_pre_cast ||
        !trace->router_logits_effective || !trace->top15_rank ||
        !trace->top15_value || !trace->selected_experts ||
        !trace->selected_weights_pre_cast ||
        !trace->selected_weights_effective || !trace->routed_output ||
        trace->router_input_count != Q38_MOE_HIDDEN ||
        trace->router_logits_count != Q38_MOE_EXPERTS ||
        trace->top15_count != 15 ||
        trace->selected_count != Q38_MOE_TOP_K ||
        trace->routed_output_count != Q38_MOE_HIDDEN)
        return false;
    if (layer == 2) {
        memcpy(context->router_input, trace->router_input,
               sizeof(context->router_input));
        memcpy(context->router_logits_pre_cast, trace->router_logits_pre_cast,
               sizeof(context->router_logits_pre_cast));
        memcpy(context->router_logits_effective, trace->router_logits_effective,
               sizeof(context->router_logits_effective));
        memcpy(context->top15_rank, trace->top15_rank,
               sizeof(context->top15_rank));
        memcpy(context->top15_value, trace->top15_value,
               sizeof(context->top15_value));
        memcpy(context->selected_experts, trace->selected_experts,
               sizeof(context->selected_experts));
        memcpy(context->selected_weights_pre_cast,
               trace->selected_weights_pre_cast,
               sizeof(context->selected_weights_pre_cast));
        memcpy(context->selected_weights_effective,
               trace->selected_weights_effective,
               sizeof(context->selected_weights_effective));
        memcpy(context->routed_output, trace->routed_output,
               sizeof(context->routed_output));
        context->margin_rank10_rank11 = trace->margin_rank10_rank11;
        context->router_dtype = trace->router_dtype;
    }
    memcpy(context->router_input_by_layer[layer], trace->router_input,
           sizeof(context->router_input_by_layer[layer]));
    memcpy(context->routed_output_by_layer[layer], trace->routed_output,
           sizeof(context->routed_output_by_layer[layer]));
    context->moe_trace_by_layer[layer] = true;
    context->moe_trace_valid = true;
    (void)error;
    (void)error_len;
    return true;
}

static bool qsa_trace(uint32_t layer, const uint32_t *selected, size_t count,
                      void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || layer >= Q38_MODEL_LAYERS || count > 2051 ||
        (count && !selected))
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
    fputs("\n\"routing\":[", out);
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
        fputs("],\"hidden_input\":[", out);
        for (size_t i = 0; i < Q38_MOE_HIDDEN; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->router_input_by_layer[layer][i]);
        }
        fputs("],\"routed_output\":[", out);
        for (size_t i = 0; i < Q38_MOE_HIDDEN; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->routed_output_by_layer[layer][i]);
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
    fputs(",\n\"router_logits\":[", out);
    first = true;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (!context->router_count[layer]) continue;
        if (!first) fputc(',', out);
        first = false;
        fprintf(out, "{\"layer\":%zu,\"logits\":[", layer);
        for (size_t i = 0; i < context->router_count[layer]; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->router_logits[layer][i]);
        }
        fputs("],\"top\":[", out);
        size_t order[Q38_MOE_EXPERTS];
        for (size_t i = 0; i < context->router_count[layer]; ++i)
            order[i] = i;
        for (size_t i = 1; i < context->router_count[layer]; ++i) {
            size_t at = i;
            size_t candidate = order[i];
            while (at > 0) {
                size_t previous = order[at - 1];
                if (context->router_logits[layer][previous] >
                        context->router_logits[layer][candidate] ||
                    (context->router_logits[layer][previous] ==
                         context->router_logits[layer][candidate] &&
                     previous < candidate))
                    break;
                order[at] = order[at - 1];
                --at;
            }
            order[at] = candidate;
        }
        const size_t top_count = context->router_count[layer] < 15
            ? context->router_count[layer] : 15;
        for (size_t i = 0; i < top_count; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "{\"id\":%zu,\"value\":", order[i]);
            json_float(out, context->router_logits[layer][order[i]]);
            fputc('}', out);
        }
        fputs("],\"rank10\":", out);
        if (top_count > 9) {
            fprintf(out, "{\"expert\":%zu,\"score\":", order[9]);
            json_float(out, context->router_logits[layer][order[9]]);
            fputc('}', out);
        } else fputs("null", out);
        fputs(",\"rank11\":", out);
        if (top_count > 10) {
            fprintf(out, "{\"expert\":%zu,\"score\":", order[10]);
            json_float(out, context->router_logits[layer][order[10]]);
            fputc('}', out);
        } else fputs("null", out);
        fputs(",\"margin_rank10_rank11\":", out);
        if (top_count > 10)
            json_float(out, context->router_logits[layer][order[9]] -
                       context->router_logits[layer][order[10]]);
        else fputs("null", out);
        fputs("}", out);
    }
    fputs("]", out);
    if (context->moe_trace_valid) {
        fputs(",\n\"layer2_moe_trace\":{\"router_dtype\":\"", out);
        fputs(context->router_dtype == Q38_FORWARD_BF16 ? "bf16" : "f32",
              out);
        fputs("\",\"router_input\":[", out);
        for (size_t i = 0; i < Q38_MOE_HIDDEN; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->router_input[i]);
        }
        fputs("],\"router_logits_pre_cast\":[", out);
        for (size_t i = 0; i < Q38_MOE_EXPERTS; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->router_logits_pre_cast[i]);
        }
        fputs("],\"router_logits_effective\":[", out);
        for (size_t i = 0; i < Q38_MOE_EXPERTS; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->router_logits_effective[i]);
        }
        fputs("],\"top15_rank\":[", out);
        for (size_t i = 0; i < 15; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "{\"rank\":%zu,\"expert\":%u,\"value\":",
                    i + 1, context->top15_rank[i]);
            json_float(out, context->top15_value[i]);
            fputc('}', out);
        }
        fputs("],\"margin_rank10_rank11\":", out);
        json_float(out, context->margin_rank10_rank11);
        fputs(",\"selected_experts\":[", out);
        for (size_t i = 0; i < Q38_MOE_TOP_K; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%u", context->selected_experts[i]);
        }
        fputs("],\"selected_weights_pre_cast\":[", out);
        for (size_t i = 0; i < Q38_MOE_TOP_K; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->selected_weights_pre_cast[i]);
        }
        fputs("],\"selected_weights_effective\":[", out);
        for (size_t i = 0; i < Q38_MOE_TOP_K; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->selected_weights_effective[i]);
        }
        fputs("],\"routed_output\":[", out);
        for (size_t i = 0; i < Q38_MOE_HIDDEN; ++i) {
            if (i) fputc(',', out);
            json_float(out, context->routed_output[i]);
        }
        fputs("]}", out);
    }
}

static bool trace_complete(const trace_context *context) {
    if (!context) return false;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer)
        if (context->route_count[layer] != Q38_MOE_TOP_K ||
            context->router_count[layer] != Q38_MOE_EXPERTS ||
            !context->moe_trace_by_layer[layer])
            return false;
    return true;
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
    fputs("{\"format\":\"q38-m6-full-trace-v4\",\"tokens\":[9419],"
          "\"stages\":[\n",
          out);
    trace_context context = {.out = out, .first = true};
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.trace = trace;
    diagnostics.route_trace = route_trace;
    diagnostics.router_trace = router_trace;
    diagnostics.moe_trace = moe_trace;
    diagnostics.qsa_trace = qsa_trace;
    diagnostics.trace_user = &context;
    float *logits = calloc(248320, sizeof(float));
    const uint32_t token = 9419;
    if (!logits || !q38_forward_full(model, &weights, &state, &token, 1,
                                     logits, 248320, &diagnostics, error,
                                     sizeof(error)) ||
        !trace_complete(&context)) {
        if (!error[0])
            snprintf(error, sizeof(error),
                     "forward trace is incomplete; refusing partial output");
        fprintf(stderr, "forward: %s\n", error);
        remove(argv[2]);
        free(logits);
        fclose(out);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    write_logits(out, logits, 248320);
    fputs("\n],", out);
    write_decisions(out, &context);
    fputs("\n}\n", out);
    fclose(out);
    free(logits);
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
