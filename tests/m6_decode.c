#include "q38_decode.h"
#include "q38_gguf.h"
#include "q38_weights.h"
#ifdef Q38_DECODE_CUDA
#include "q38_forward_cuda.h"
#else
#endif

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *out;
    bool first;
    q38_decode_stats layer_stats[Q38_MODEL_LAYERS];
} trace_context;

static q38_decode_stats capture_stats(const float *values, size_t count) {
    q38_decode_stats result = {0};
    result.min = INFINITY;
    result.max = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
        const float value = values[i];
        if (isnan(value)) {
            result.nan_count++;
            continue;
        }
        if (isinf(value)) {
            result.inf_count++;
            continue;
        }
        result.finite_count++;
        result.min = fminf(result.min, value);
        result.max = fmaxf(result.max, value);
        result.max_abs = fmaxf(result.max_abs, fabsf(value));
        result.mean += value;
        result.rms += (double)value * value;
    }
    if (result.finite_count) {
        result.mean /= (float)result.finite_count;
        result.rms = (float)sqrt(result.rms / (double)result.finite_count);
    } else {
        result.min = result.max = result.mean = result.rms = 0.0f;
    }
    result.checksum = 1469598103934665603ULL;
    for (size_t i = 0; i < count * sizeof(*values); ++i) {
        result.checksum ^= ((const unsigned char *)values)[i];
        result.checksum *= 1099511628211ULL;
    }
    return result;
}

static void write_stats(FILE *out, const q38_decode_stats *stats) {
    fprintf(out,
            "{\"min\":%.9g,\"max\":%.9g,\"mean\":%.17g,\"rms\":%.17g,"
            "\"max_abs\":%.9g,\"finite_count\":%zu,\"nan_count\":%zu,"
            "\"inf_count\":%zu,\"checksum\":\"%016" PRIx64 "\"}",
            stats->min, stats->max, stats->mean, stats->rms, stats->max_abs,
            stats->finite_count, stats->nan_count, stats->inf_count,
            stats->checksum);
}

static bool layer_trace(uint32_t layer, const float *hidden, size_t tokens,
                        size_t width, void *opaque, char *error,
                        size_t error_len) {
    trace_context *context = opaque;
    if (!context || !hidden || tokens != 1 || width == 0 ||
        (layer != UINT32_MAX && layer >= Q38_MODEL_LAYERS))
        return false;
    if (layer != UINT32_MAX)
        context->layer_stats[layer] = capture_stats(hidden, width);
    (void)error;
    (void)error_len;
    return true;
}

static bool trace_step(const q38_decode_step *step, void *opaque, char *error,
                       size_t error_len) {
    trace_context *context = opaque;
    if (!context || !context->out || !step) {
        if (error && error_len) snprintf(error, error_len, "invalid trace context");
        return false;
    }
    FILE *out = context->out;
    fprintf(out, "%s{\"step\":%zu,\"generated\":%s,\"input_token\":%u,"
                "\"next_token\":%u,\"committed_tokens\":%" PRIu64
                ",\"logits_hash\":\"%016" PRIx64 "\",\"logits_finite\":%s,"
                "\"gdn_state_hash\":\"%016" PRIx64
                "\",\"conv_history_hash\":\"%016" PRIx64
                "\",\"ple_history_hash\":\"%016" PRIx64 "\",\"finite\":%s,"
                "\"gdn_state_stats\":",
            context->first ? "" : ",\n", step->step,
            step->generated ? "true" : "false", step->input_token,
            step->next_token, step->committed_tokens, step->logits_hash,
            step->logits_finite ? "true" : "false", step->gdn_state_hash,
            step->conv_history_hash, step->ple_history_hash,
            step->finite ? "true" : "false");
    write_stats(out, &step->gdn_state_stats);
    fputs(",\"conv_history_stats\":", out);
    write_stats(out, &step->conv_history_stats);
    fputs(",\"ple_history_stats\":", out);
    write_stats(out, &step->ple_history_stats);
    fputs(",\"logits_stats\":{\"min\":", out);
    fprintf(out, "%.9g,\"max\":%.9g,\"mean\":%.17g,\"rms\":%.17g,"
                "\"max_abs\":%.9g}", step->logits_min, step->logits_max,
            step->logits_mean, step->logits_rms, step->logits_max_abs);
    fputs(",\"layer_outputs\":[", out);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) fputc(',', out);
        fprintf(out, "{\"layer\":%zu,\"stats\":", layer);
        write_stats(out, &context->layer_stats[layer]);
        fputc('}', out);
    }
    fputs("],\"gdn_layers\":[", out);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) fputc(',', out);
        fprintf(out, "{\"layer\":%zu,\"hash\":\"%016" PRIx64
                    "\",\"stats\":", layer, step->gdn_layer_hash[layer]);
        write_stats(out, &step->gdn_layer_stats[layer]);
        fputc('}', out);
    }
    fputs("],\"conv_layers\":[", out);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) fputc(',', out);
        fprintf(out, "{\"layer\":%zu,\"hash\":\"%016" PRIx64
                    "\",\"stats\":", layer, step->conv_layer_hash[layer]);
        write_stats(out, &step->conv_layer_stats[layer]);
        fputc('}', out);
    }
    fputs("],\"qsa\":[", out);
    context->first = false;
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        const q38_decode_qsa_snapshot *qsa = &step->qsa[layer];
        if (layer) fputc(',', out);
        fprintf(out, "{\"layer\":%zu,\"position\":%" PRIu64
                    ",\"committed_tokens\":%" PRIu64
                    ",\"pending_count\":%u,\"pending_position\":%" PRIu64
                    ",\"main_k_count\":%zu,\"main_v_count\":%zu,"
                    "\"index_k_count\":%zu,\"main_k_hash\":\"%016" PRIx64
                    "\",\"main_v_hash\":\"%016" PRIx64
                    "\",\"index_k_hash\":\"%016" PRIx64
                    "\",\"main_k_stats\":",
                layer, qsa->position, qsa->committed_tokens,
                qsa->pending_count, qsa->pending_position, qsa->main_k_count,
                qsa->main_v_count, qsa->index_k_count, qsa->main_k_hash,
                qsa->main_v_hash, qsa->index_k_hash);
        write_stats(out, &qsa->main_k_stats);
        fputs(",\"main_v_stats\":", out);
        write_stats(out, &qsa->main_v_stats);
        fputs(",\"index_k_stats\":", out);
        write_stats(out, &qsa->index_k_stats);
        fputc('}', out);
    }
    fputs("]}", out);
    return true;
}

static size_t parse_count(const char *text) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    return end && *end == '\0' && value <= 128 ? (size_t)value : 0;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s model.gguf output.json [generated_count]\n",
                argv[0]);
        return 2;
    }
    const size_t generated_count = argc == 4 ? parse_count(argv[3]) : 8;
    if (!generated_count && argc == 4) {
        fprintf(stderr, "generated_count must be 1..128\n");
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
    const uint32_t prompt[] = {9419};
    uint32_t *generated = calloc(generated_count, sizeof(*generated));
    float *logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*logits));
    trace_context context = {.out = out, .first = true};
    q38_forward_diagnostics diagnostics = {0};
    diagnostics.trace = layer_trace;
    diagnostics.trace_user = &context;
#ifdef Q38_DECODE_CUDA
    q38_forward_cuda_context *cuda =
        q38_forward_cuda_context_create(error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda: %s\n", error);
        fclose(out);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
#endif
    fprintf(out, "{\"format\":\"q38-m6-decode-v1\",\"steps\":[\n");
    bool ok = generated && logits &&
#ifdef Q38_DECODE_CUDA
              q38_decode_stream_with_matrix_backend(
               model, &weights, &state, prompt, 1, generated, generated_count,
               logits, Q38_DECODE_VOCAB_SIZE, &diagnostics,
               q38_forward_cuda_matvec_backend,
               q38_forward_cuda_matrix_backend,
               q38_forward_cuda_expert_backend, cuda,
#else
              q38_decode_stream(
               model, &weights, &state, prompt, 1, generated, generated_count,
               logits, Q38_DECODE_VOCAB_SIZE, &diagnostics,
#endif
               trace_step, &context, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "decode: %s\n", error);
        free(generated);
        free(logits);
        fclose(out);
#ifdef Q38_DECODE_CUDA
        q38_forward_cuda_context_destroy(cuda);
#endif
        remove(argv[2]);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    fprintf(out, "\n],\"prompt\":[9419],\"generated\":[");
    for (size_t i = 0; i < generated_count; ++i)
        fprintf(out, "%s%u", i ? "," : "", generated[i]);
    fprintf(out, "],\"generated_count\":%zu,\"temperature\":0,\"sampling\":\"greedy\","
                "\"vocab_size\":%u}\n",
            generated_count, Q38_DECODE_VOCAB_SIZE);
    fclose(out);
    free(generated);
    free(logits);
#ifdef Q38_DECODE_CUDA
    q38_forward_cuda_context_destroy(cuda);
#endif
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
