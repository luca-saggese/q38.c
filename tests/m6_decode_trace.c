#include "q38_decode.h"
#include "q38_gguf.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_qsa(FILE *out, const q38_decode_qsa_snapshot *qsa) {
    fprintf(out,
            "\"position\":%" PRIu64 ",\"committed_tokens\":%" PRIu64
            ",\"pending_count\":%u,\"pending_position\":%" PRIu64
            ",\"main_k_count\":%zu,\"main_v_count\":%zu,\"index_k_count\":%zu"
            ",\"main_k_hash\":\"%016" PRIx64
            "\",\"main_v_hash\":\"%016" PRIx64
            "\",\"index_k_hash\":\"%016" PRIx64
            "\",\"pending_main_k_hash\":\"%016" PRIx64
            "\",\"pending_main_v_hash\":\"%016" PRIx64
            "\",\"pending_index_k_hash\":\"%016" PRIx64,
            qsa->position, qsa->committed_tokens, qsa->pending_count,
            qsa->pending_position, qsa->main_k_count, qsa->main_v_count,
            qsa->index_k_count, qsa->main_k_hash, qsa->main_v_hash,
            qsa->index_k_hash, qsa->pending_main_k_hash,
            qsa->pending_main_v_hash, qsa->pending_index_k_hash);
}

static void write_step(FILE *out, const q38_decode_step *step,
                       bool first) {
    if (!first) fputs(",\n", out);
    const char *kind = step->kind == Q38_DECODE_TRACE_PROMPT_PREDICTION
        ? "prompt_prediction"
        : step->kind == Q38_DECODE_TRACE_GENERATED_EMIT
            ? "generated_emit" : "generated_consume";
    fprintf(out,
            "{\"step\":%zu,\"kind\":\"%s\",\"generated\":%s"
            ",\"state_committed\":%s,\"input_token\":%u"
            ",\"next_token\":%u,\"emitted_token\":%u,\"consumed_token\":%u"
            ",\"argmax\":%u,\"argmax_value\":%.9g"
            ",\"logits_hash\":\"%016" PRIx64
            "\",\"logits_stats\":{\"min\":%.9g,\"max\":%.9g"
            ",\"mean\":%.17g,\"rms\":%.17g,\"max_abs\":%.9g}"
            ",\"committed_tokens\":%" PRIu64
            ",\"finite\":%s,\"ple_history_hash\":\"%016" PRIx64
            "\",\"ple_history\":{\"prev_token_1\":%u,\"prev_token_2\":%u"
            ",\"have_prev_1\":%s,\"have_prev_2\":%s}"
            ",\"gdn_state_hash\":\"%016" PRIx64
            "\",\"conv_history_hash\":\"%016" PRIx64 "\",\"gdn_layers\":[",
            step->step, kind, step->generated ? "true" : "false",
            step->state_committed ? "true" : "false", step->input_token,
            step->next_token, step->emitted_token, step->consumed_token,
            step->argmax,
            step->argmax_value, step->logits_hash, step->logits_min,
            step->logits_max, step->logits_mean, step->logits_rms,
            step->logits_max_abs, step->committed_tokens,
            step->finite ? "true" : "false", step->ple_history_hash,
            step->ple_prev_token_1, step->ple_prev_token_2,
            step->ple_have_prev_1 ? "true" : "false",
            step->ple_have_prev_2 ? "true" : "false", step->gdn_state_hash,
            step->conv_history_hash);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) fputc(',', out);
        fprintf(out, "{\"layer\":%zu,\"recurrent_hash\":\"%016" PRIx64
                    "\",\"conv_hash\":\"%016" PRIx64 "\"}",
                layer, step->gdn_layer_hash[layer],
                step->conv_layer_hash[layer]);
    }
    fputs("],\"top\":[", out);
    for (size_t rank = 0; rank < 20; ++rank) {
        if (rank) fputc(',', out);
        fprintf(out, "{\"id\":%u,\"value\":%.9g}", step->top_ids[rank],
                step->top_values[rank]);
    }
    fputs("],\"qsa\":[", out);
    for (size_t layer = 0; layer < Q38_MODEL_LAYERS; ++layer) {
        if (layer) fputc(',', out);
        fprintf(out, "{\"layer\":%zu,", layer);
        write_qsa(out, &step->qsa[layer]);
        fputc('}', out);
    }
    fputs("]}", out);
}

typedef struct {
    FILE *out;
    bool first;
} trace_context;

static bool trace_step(const q38_decode_step *step, void *opaque,
                       char *error, size_t error_len) {
    trace_context *context = opaque;
    if (!context || !context->out || !step) {
        if (error && error_len)
            snprintf(error, error_len, "invalid decode trace callback");
        return false;
    }
    write_step(context->out, step, context->first);
    context->first = false;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf generated_count\n", argv[0]);
        return 2;
    }
    char error[256];
    char *end = NULL;
    const unsigned long requested = strtoul(argv[2], &end, 10);
    if (!end || *end || requested == 0 || requested > 4096) {
        fprintf(stderr, "generated_count must be in [1,4096]\n");
        return 2;
    }
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
    float *logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*logits));
    uint32_t *generated = calloc(requested, sizeof(*generated));
    if (!logits || !generated) {
        fprintf(stderr, "decode allocation failed\n");
        free(logits);
        free(generated);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    trace_context context = {.out = stdout, .first = true};
    fputs("{\"format\":\"q38-m6-decode-trace-v2\",\"prompt\":[9419],"
          "\"generated_count\":",
          stdout);
    fprintf(stdout, "%lu,\"steps\":[\n", requested);
    if (!q38_decode_stream(model, &weights, &state, (uint32_t[]){9419}, 1,
                           generated, requested, logits, Q38_DECODE_VOCAB_SIZE,
                           NULL, trace_step, &context, error,
                           sizeof(error))) {
        fprintf(stderr, "decode: %s\n", error);
        free(logits);
        free(generated);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    fputs("],\"generated\":[", stdout);
    for (size_t i = 0; i < requested; ++i) {
        if (i) fputc(',', stdout);
        fprintf(stdout, "%u", generated[i]);
    }
    fputs("]}\n", stdout);
    free(logits);
    free(generated);
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
