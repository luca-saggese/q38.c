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
static const char *const layer9_boundary_order[] = {
    "layer_input",
    "core_pre_norm",
    "gr_core_read",
    "gdn_qsa_input",
    "gdn_qsa_output",
    "core_residual_gr_write",
    "mlp_pre_norm",
    "mlp_gr_read",
    "router_input",
    "router_logits_pre_cast",
    "router_logits_effective",
    "routed_output",
    "shared_expert",
    "final_mlp_gr_write",
    "layer_output",
};
static const char *const ple_boundary_order[] = {
    "ple_embedding",
    "ple_key_projection",
    "ple_value_projection",
    "ple_key_normed",
    "ple_query_normed",
    "ple_gated_value",
    "ple_gated_value_normed",
    "ple_conv_output",
    "ple_contribution",
};

typedef struct {
    char name[48];
    float *values;
    size_t count;
    size_t width;
} boundary_record;

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
    float layer9_router_input[Q38_MOE_HIDDEN];
    float layer9_gr_output[Q38_MOE_HIDDEN];
    float layer9_router_pre_cast[Q38_MOE_EXPERTS];
    float layer9_router_effective[Q38_MOE_EXPERTS];
    const q38_tensor *layer9_router;
    bool layer9_pre_router_valid;
    bool layer9_router_vectors_valid;
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
    boundary_record layer9_boundaries[
        sizeof(layer9_boundary_order) / sizeof(layer9_boundary_order[0]) +
        sizeof(ple_boundary_order) / sizeof(ple_boundary_order[0])];
    size_t layer9_boundary_count;
    boundary_record layer1_boundaries[
        sizeof(layer9_boundary_order) / sizeof(layer9_boundary_order[0]) +
        sizeof(ple_boundary_order) / sizeof(ple_boundary_order[0])];
    size_t layer1_boundary_count;
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

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
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

static void write_vector_field(FILE *out, const char *name,
                               const float *values, size_t count) {
    fprintf(out, "\"%s\":[", name);
    for (size_t i = 0; i < count; ++i) {
        if (i) fputc(',', out);
        json_float(out, values[i]);
    }
    fprintf(out, "],\"%s_stats\":{", name);
    write_stats(out, values, count);
    fputc('}', out);
}

static void write_boundary_set(FILE *out, const trace_context *context,
                               const boundary_record *records,
                               size_t record_count, uint32_t layer,
                               bool selected) {
    fprintf(out, "\"layer%u_boundaries\":[", layer);
    bool first = true;
    for (size_t order = 0; order < sizeof(ple_boundary_order) /
                                      sizeof(ple_boundary_order[0]);
         ++order) {
        const boundary_record *record = NULL;
        for (size_t i = 0; i < record_count; ++i)
            if (strcmp(records[i].name, ple_boundary_order[order]) == 0) {
                record = &records[i];
                break;
            }
        if (record) {
            fprintf(out, "{\"name\":\"%s\",\"width\":%zu,\"elements\":%zu,"
                        "\"values\":[",
                    record->name, record->width, record->count);
            for (size_t j = 0; j < record->count; ++j) {
                if (j) fputc(',', out);
                json_float(out, record->values[j]);
            }
            fputs("],\"stats\":{", out);
            write_stats(out, record->values, record->count);
            fputs("}}", out);
            first = false;
        }
        if (record && order + 1 < sizeof(ple_boundary_order) /
                                  sizeof(ple_boundary_order[0]))
            fputc(',', out);
    }
    for (size_t order = 0; order < sizeof(layer9_boundary_order) /
                                      sizeof(layer9_boundary_order[0]);
         ++order) {
        const boundary_record *record = NULL;
        for (size_t i = 0; i < record_count; ++i)
            if (strcmp(records[i].name, layer9_boundary_order[order]) == 0) {
                record = &records[i];
                break;
            }
        if (!record) continue;
        if (!first) fputc(',', out);
        first = false;
        fprintf(out, "{\"name\":\"%s\",\"width\":%zu,\"elements\":%zu,"
                    "\"values\":[",
                record->name, record->width, record->count);
        for (size_t i = 0; i < record->count; ++i) {
            if (i) fputc(',', out);
            json_float(out, record->values[i]);
        }
        fputs("],\"stats\":{", out);
        write_stats(out, record->values, record->count);
        fputs("}}", out);
    }
    if (selected && context->route_count[layer] == Q38_MOE_TOP_K) {
        if (!first) fputc(',', out);
        fputs("{\"name\":\"selected_experts\",\"width\":10,\"elements\":10,"
              "\"ids\":[",
              out);
        for (size_t i = 0; i < Q38_MOE_TOP_K; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%u", context->routed[layer][i]);
        }
        fputs("]}", out);
    }
    fputc(']', out);
}

static bool store_boundary(boundary_record *records, size_t *record_count,
                           uint32_t layer, const char *boundary,
                           const float *values, size_t token_count,
                           size_t width, char *error, size_t error_len) {
    if (!records || !record_count || !boundary || !values || !token_count ||
        !width || token_count > SIZE_MAX / width)
        return false;
    const size_t count = token_count * width;
    size_t index = 0;
    for (; index < *record_count; ++index)
        if (strcmp(records[index].name, boundary) == 0)
            break;
    if (index == *record_count) {
        if (index >= sizeof(layer9_boundary_order) /
                         sizeof(layer9_boundary_order[0]) +
                         sizeof(ple_boundary_order) /
                         sizeof(ple_boundary_order[0])) {
            if (error && error_len)
                snprintf(error, error_len, "too many layer-%u boundaries",
                         layer);
            return false;
        }
        snprintf(records[index].name, sizeof(records[index].name), "%s",
                 boundary);
        (*record_count)++;
    }
    float *copy = realloc(records[index].values, count * sizeof(*copy));
    if (!copy) {
        if (error && error_len)
            snprintf(error, error_len, "layer-%u boundary allocation failed",
                     layer);
        return false;
    }
    memcpy(copy, values, count * sizeof(*copy));
    records[index].values = copy;
    records[index].count = count;
    records[index].width = width;
    return true;
}

static bool boundary_trace(uint32_t layer, const char *boundary,
                           const float *values, size_t token_count,
                           size_t width, void *opaque, char *error,
                           size_t error_len) {
    trace_context *context = opaque;
    if (!context || (layer != 1 && layer != 9))
        return true;
    boundary_record *records = layer == 9 ? context->layer9_boundaries
                                          : context->layer1_boundaries;
    size_t *record_count = layer == 9 ? &context->layer9_boundary_count
                                      : &context->layer1_boundary_count;
    return store_boundary(records, record_count, layer, boundary, values,
                          token_count, width, error, error_len);
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
    if (layer != UINT32_MAX) {
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
    if (layer == 9) {
        memcpy(context->layer9_router_pre_cast,
               trace->router_logits_pre_cast,
               sizeof(context->layer9_router_pre_cast));
        memcpy(context->layer9_router_effective,
               trace->router_logits_effective,
               sizeof(context->layer9_router_effective));
        context->layer9_router_vectors_valid = true;
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

static bool pre_router_trace(uint32_t layer,
                             const q38_pre_router_trace *trace,
                             void *opaque, char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || !trace || layer != 9 || !trace->router ||
        !trace->router_input || !trace->gr_output ||
        trace->router_input_count != Q38_MOE_HIDDEN ||
        trace->gr_output_count != Q38_MOE_HIDDEN ||
        trace->router->ndim != 2 || trace->router->dim[0] != Q38_MOE_EXPERTS ||
        trace->router->dim[1] != Q38_MOE_HIDDEN)
        return false;
    memcpy(context->layer9_router_input, trace->router_input,
           sizeof(context->layer9_router_input));
    memcpy(context->layer9_gr_output, trace->gr_output,
           sizeof(context->layer9_gr_output));
    context->layer9_router = trace->router;
    context->layer9_pre_router_valid = true;
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
    fputs(",\n", out);
    write_boundary_set(out, context, context->layer1_boundaries,
                       context->layer1_boundary_count, 1, true);
    fputs(",\n", out);
    write_boundary_set(out, context, context->layer9_boundaries,
                       context->layer9_boundary_count, 9, true);
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
        fputs("],\"effective_bits\":[", out);
        for (size_t i = 0; i < context->router_count[layer]; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "%u",
                    bf16_bits(context->router_logits[layer][i]));
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
        const size_t top_count = context->router_count[layer] < 20
            ? context->router_count[layer] : 20;
        for (size_t i = 0; i < top_count; ++i) {
            if (i) fputc(',', out);
            fprintf(out, "{\"id\":%zu,\"value\":", order[i]);
            json_float(out, context->router_logits[layer][order[i]]);
            fputc('}', out);
        }
        if (layer == 9 && context->layer9_router_vectors_valid) {
            fputs("],\"matvec_pre_cast\":[", out);
            for (size_t i = 0; i < Q38_MOE_EXPERTS; ++i) {
                if (i) fputc(',', out);
                json_float(out, context->layer9_router_pre_cast[i]);
            }
            fputs("],\"matvec_pre_cast_stats\":{", out);
            write_stats(out, context->layer9_router_pre_cast,
                        Q38_MOE_EXPERTS);
            fputs("},\"effective_bf16\":[", out);
            for (size_t i = 0; i < Q38_MOE_EXPERTS; ++i) {
                if (i) fputc(',', out);
                json_float(out, context->layer9_router_effective[i]);
            }
            fputs("],\"effective_bf16_stats\":{", out);
            write_stats(out, context->layer9_router_effective,
                        Q38_MOE_EXPERTS);
            fputs("},\"effective_bf16_bits\":[", out);
            for (size_t i = 0; i < Q38_MOE_EXPERTS; ++i) {
                if (i) fputc(',', out);
                fprintf(out, "%u",
                        bf16_bits(context->layer9_router_effective[i]));
            }
            fputs("]", out);
        } else {
            fputs("]", out);
        }
        fputs(",\"rank10\":", out);
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
        if (layer == 9 && context->layer9_pre_router_valid &&
            context->layer9_router) {
            const q38_tensor *router = context->layer9_router;
            const uint64_t row_bytes = router->bytes / Q38_MOE_EXPERTS;
            fputs(",\"router_weight_rows\":{\"tensor\":\"", out);
            fprintf(out, "%.*s", (int)router->name.len, router->name.ptr);
            fprintf(out, "\",\"type\":%u,\"rows\":%" PRIu64
                        ",\"cols\":%" PRIu64 ",\"abs_offset\":%" PRIu64
                        ",\"row_bytes\":%" PRIu64 ",\"used_rows\":[",
                    router->type, router->dim[0], router->dim[1],
                    router->abs_offset, row_bytes);
            for (size_t i = 0; i < context->route_count[9]; ++i) {
                if (i) fputc(',', out);
                fprintf(out, "%u", context->routed[9][i]);
            }
            fputs("]}", out);
        }
        fputs("}", out);
    }
    fputs("]", out);
    if (context->layer9_pre_router_valid) {
        fputs(",\n\"layer9_pre_router\":{\"layer\":9,\"router_input\":{",
              out);
        write_vector_field(out, "values", context->layer9_router_input,
                           Q38_MOE_HIDDEN);
        fputs("},\"gr_output\":{", out);
        write_vector_field(out, "values", context->layer9_gr_output,
                           Q38_MOE_HIDDEN);
        fputs("}}", out);
    }
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
    if (!context->layer9_pre_router_valid ||
        !context->layer9_router_vectors_valid)
        return false;
    const boundary_record *sets[] = {
        context->layer1_boundaries, context->layer9_boundaries,
    };
    const size_t counts[] = {
        context->layer1_boundary_count, context->layer9_boundary_count,
    };
    for (size_t set = 0; set < sizeof(sets) / sizeof(sets[0]); ++set)
        for (size_t i = 0; i < sizeof(layer9_boundary_order) /
                                sizeof(layer9_boundary_order[0]); ++i) {
            bool found = false;
            for (size_t j = 0; j < counts[set]; ++j)
                found |= strcmp(sets[set][j].name,
                                layer9_boundary_order[i]) == 0;
            if (!found) return false;
        }
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
    fputs("{\"format\":\"q38-m6-full-trace-v5\",\"tokens\":[9419],"
          "\"stages\":[\n",
          out);
    trace_context context = {.out = out, .first = true};
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.trace = trace;
    diagnostics.route_trace = route_trace;
    diagnostics.router_trace = router_trace;
    diagnostics.moe_trace = moe_trace;
    diagnostics.pre_router_trace = pre_router_trace;
    diagnostics.boundary_trace = boundary_trace;
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
        for (size_t i = 0; i < context.layer1_boundary_count; ++i)
            free(context.layer1_boundaries[i].values);
        for (size_t i = 0; i < context.layer9_boundary_count; ++i)
            free(context.layer9_boundaries[i].values);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    write_logits(out, logits, 248320);
    fputs("\n],", out);
    fputs("\"layer9_ple_contribution\":{\"status\":\"not_applicable\","
          "\"reason\":\"layer 9 has no PLE\"},", out);
    write_decisions(out, &context);
    fputs("\n}\n", out);
    fclose(out);
    for (size_t i = 0; i < context.layer1_boundary_count; ++i)
        free(context.layer1_boundaries[i].values);
    for (size_t i = 0; i < context.layer9_boundary_count; ++i)
        free(context.layer9_boundaries[i].values);
    free(logits);
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
