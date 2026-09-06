#include "q38_forward_cuda.h"
#include "q38_session.h"

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    VOCAB_SIZE = Q38_DECODE_VOCAB_SIZE,
    MAX_GENERATED = 4096,
    MAX_PREFILL_CASES = 8
};

typedef struct {
    double wall_ms;
    double forward_ms;
    double argmax_ms;
    double bookkeeping_ms;
    double ple_critical_stall_ms;
    double ple_elapsed_ms;
    double ple_overlap_ms;
    double qsa_ms;
    double qsa_qkv_ms;
    double qsa_output_projection_ms;
    double qsa_attention_ms;
    double qsa_index_compress_ms;
    double qsa_state_glue_ms;
    double moe_ms;
    double gdn_ms;
    double gr_ms;
    double lm_head_ms;
    double norms_residual_glue_ms;
    double host_scalar_ms;
    double cuda_dispatch_ms;
    double cuda_sync_wait_ms;
    double memcpy_ms;
    uint64_t telemetry_callbacks;
    uint64_t kernel_launches;
    uint64_t host_syncs;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint64_t d2d_bytes;
    uint64_t non_ple_upload_bytes;
    uint64_t non_ple_residency_misses;
} q2_sample;

typedef struct {
    q2_sample sample;
    q38_forward_qsa_timing qsa_timing;
} q2_capture;

typedef struct {
    uint64_t callbacks;
    uint64_t kernel_launches;
    uint64_t host_syncs;
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint64_t d2d_bytes;
    uint64_t non_ple_upload_bytes;
    uint64_t non_ple_residency_misses;
    double kernel_ms;
    double backend_overhead_ms;
    double upload_ms;
} q2_telemetry;

typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *prompt;
    const char *mode;
    size_t generated_count;
    size_t context_size;
    size_t prefill_chunk;
    size_t measure_first;
    size_t measure_last;
    size_t prefill_sizes[MAX_PREFILL_CASES];
    size_t prefill_count;
} q2_options;

typedef struct {
    uint64_t logits_hash;
    uint32_t argmax;
    bool finite;
} q2_trace_capture;

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 +
           (double)ts.tv_nsec / 1000000.0;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void json_string(const char *value) {
    putchar('"');
    if (value) {
        for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
            switch (*p) {
            case '\\': fputs("\\\\", stdout); break;
            case '"': fputs("\\\"", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*p < 0x20)
                    printf("\\u%04x", *p);
                else
                    putchar(*p);
                break;
            }
        }
    }
    putchar('"');
}

static void print_ids(const uint32_t *ids, size_t count) {
    putchar('[');
    for (size_t i = 0; i < count; ++i)
        printf("%s%u", i ? "," : "", ids[i]);
    putchar(']');
}

static void zero_sample(q2_sample *sample) {
    if (sample) memset(sample, 0, sizeof(*sample));
}

static void telemetry_observer(const q38_forward_cuda_telemetry *record,
                               void *opaque) {
    q2_telemetry *telemetry = (q2_telemetry *)opaque;
    if (!record || !telemetry) return;
    telemetry->callbacks++;
    telemetry->kernel_launches++;
    telemetry->host_syncs += record->host_syncs;
    telemetry->h2d_bytes += record->upload_bytes +
                            record->activation_read_bytes;
    telemetry->d2h_bytes += record->d2h_bytes;
    telemetry->kernel_ms += record->kernel_ms;
    telemetry->backend_overhead_ms += record->backend_overhead_ms;
    telemetry->upload_ms += record->upload_ms;
    if (!record->ple_file_backed_access) {
        telemetry->non_ple_upload_bytes += record->upload_bytes;
        if (record->non_ple_residency_miss)
            telemetry->non_ple_residency_misses++;
    }
}

static double *stage_slot(q2_sample *sample, const char *name) {
    if (!sample || !name) return NULL;
    if (strstr(name, "moe_")) return &sample->moe_ms;
    if (strstr(name, "gdn_")) return &sample->gdn_ms;
    if (strstr(name, "gr_")) return &sample->gr_ms;
    if (strstr(name, "lm_head")) return &sample->lm_head_ms;
    if (strstr(name, "norm") || strstr(name, "residual"))
        return &sample->norms_residual_glue_ms;
    return NULL;
}

static bool stage_trace(const q38_forward_stage_usage *usage, void *opaque,
                        char *error, size_t error_len) {
    q2_capture *capture = (q2_capture *)opaque;
    if (!capture || !usage) {
        if (error && error_len)
            snprintf(error, error_len, "invalid canonical stage telemetry");
        return false;
    }
    double *slot = stage_slot(&capture->sample, usage->logical_stage);
    if (!slot) slot = stage_slot(&capture->sample, usage->name);
    if (slot) *slot += usage->elapsed_ms;
    return true;
}

static bool trace_step(const q38_decode_step *step, void *opaque,
                       char *error, size_t error_len) {
    q2_trace_capture *capture = (q2_trace_capture *)opaque;
    if (!capture || !step) {
        if (error && error_len)
            snprintf(error, error_len, "invalid canonical decode trace");
        return false;
    }
    capture->logits_hash = step->logits_hash;
    capture->argmax = step->argmax;
    capture->finite = step->finite && step->logits_finite;
    return true;
}

static void add_sample(q2_sample *sum, const q2_sample *sample) {
    if (!sum || !sample) return;
#define ADD(field) sum->field += sample->field
    ADD(wall_ms);
    ADD(forward_ms);
    ADD(argmax_ms);
    ADD(bookkeeping_ms);
    ADD(ple_critical_stall_ms);
    ADD(ple_elapsed_ms);
    ADD(ple_overlap_ms);
    ADD(qsa_ms);
    ADD(qsa_qkv_ms);
    ADD(qsa_output_projection_ms);
    ADD(qsa_attention_ms);
    ADD(qsa_index_compress_ms);
    ADD(qsa_state_glue_ms);
    ADD(moe_ms);
    ADD(gdn_ms);
    ADD(gr_ms);
    ADD(lm_head_ms);
    ADD(norms_residual_glue_ms);
    ADD(host_scalar_ms);
    ADD(cuda_dispatch_ms);
    ADD(cuda_sync_wait_ms);
    ADD(memcpy_ms);
    sum->telemetry_callbacks += sample->telemetry_callbacks;
    sum->kernel_launches += sample->kernel_launches;
    sum->host_syncs += sample->host_syncs;
    sum->h2d_bytes += sample->h2d_bytes;
    sum->d2h_bytes += sample->d2h_bytes;
    sum->d2d_bytes += sample->d2d_bytes;
    sum->non_ple_upload_bytes += sample->non_ple_upload_bytes;
    sum->non_ple_residency_misses += sample->non_ple_residency_misses;
#undef ADD
}

static void divide_sample(q2_sample *sample, double divisor) {
    if (!sample || divisor <= 0.0) return;
#define DIV(field) sample->field /= divisor
    DIV(wall_ms);
    DIV(forward_ms);
    DIV(argmax_ms);
    DIV(bookkeeping_ms);
    DIV(ple_critical_stall_ms);
    DIV(ple_elapsed_ms);
    DIV(ple_overlap_ms);
    DIV(qsa_ms);
    DIV(qsa_qkv_ms);
    DIV(qsa_output_projection_ms);
    DIV(qsa_attention_ms);
    DIV(qsa_index_compress_ms);
    DIV(qsa_state_glue_ms);
    DIV(moe_ms);
    DIV(gdn_ms);
    DIV(gr_ms);
    DIV(lm_head_ms);
    DIV(norms_residual_glue_ms);
    DIV(host_scalar_ms);
    DIV(cuda_dispatch_ms);
    DIV(cuda_sync_wait_ms);
    DIV(memcpy_ms);
#undef DIV
    sample->telemetry_callbacks = (uint64_t)(
        (double)sample->telemetry_callbacks / divisor);
    sample->kernel_launches = (uint64_t)(
        (double)sample->kernel_launches / divisor);
    sample->host_syncs = (uint64_t)(
        (double)sample->host_syncs / divisor);
    sample->h2d_bytes = (uint64_t)((double)sample->h2d_bytes / divisor);
    sample->d2h_bytes = (uint64_t)((double)sample->d2h_bytes / divisor);
    sample->d2d_bytes = (uint64_t)((double)sample->d2d_bytes / divisor);
    sample->non_ple_upload_bytes = (uint64_t)(
        (double)sample->non_ple_upload_bytes / divisor);
    sample->non_ple_residency_misses = (uint64_t)(
        (double)sample->non_ple_residency_misses / divisor);
}

static bool parse_size_list(const char *value, q2_options *options) {
    char *copy = NULL;
    char *cursor = NULL;
    if (!value || !options) return false;
    copy = strdup(value);
    if (!copy) return false;
    cursor = copy;
    while (cursor && *cursor && options->prefill_count < MAX_PREFILL_CASES) {
        char *comma = strchr(cursor, ',');
        if (comma) *comma = '\0';
        unsigned long long parsed = strtoull(cursor, NULL, 10);
        if (!parsed || parsed > SIZE_MAX) {
            free(copy);
            return false;
        }
        options->prefill_sizes[options->prefill_count++] = (size_t)parsed;
        cursor = comma ? comma + 1 : NULL;
    }
    free(copy);
    return options->prefill_count != 0;
}

static void usage(FILE *stream) {
    fprintf(stream,
            "usage: q2_canonical_bench --mode decode|prefill "
            "--model MODEL --tokenizer DIR --prompt TEXT [options]\n"
            "  --generated N       decode generated token count (default 128)\n"
            "  --ctx N             session context (default 4096)\n"
            "  --prefill-chunk N   prefill chunk (default 128)\n"
            "  --measure-first N   first measured generated index (default 16)\n"
            "  --measure-last N    last measured generated index (default 127)\n"
            "  --prefill-sizes A,B,C  prefill sizes (default 128,512,2048)\n");
}

static bool parse_options(int argc, char **argv, q2_options *options) {
    memset(options, 0, sizeof(*options));
    options->generated_count = 128;
    options->context_size = 4096;
    options->prefill_chunk = 128;
    options->measure_first = 16;
    options->measure_last = 127;
    options->prefill_sizes[0] = 128;
    options->prefill_sizes[1] = 512;
    options->prefill_sizes[2] = 2048;
    options->prefill_count = 3;
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--model") && i + 1 < argc)
            options->model_path = argv[++i];
        else if (!strcmp(arg, "--tokenizer") && i + 1 < argc)
            options->tokenizer_path = argv[++i];
        else if (!strcmp(arg, "--prompt") && i + 1 < argc)
            options->prompt = argv[++i];
        else if (!strcmp(arg, "--mode") && i + 1 < argc)
            options->mode = argv[++i];
        else if (!strcmp(arg, "--generated") && i + 1 < argc)
            options->generated_count = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(arg, "--ctx") && i + 1 < argc)
            options->context_size = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(arg, "--prefill-chunk") && i + 1 < argc)
            options->prefill_chunk = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(arg, "--measure-first") && i + 1 < argc)
            options->measure_first = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(arg, "--measure-last") && i + 1 < argc)
            options->measure_last = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(arg, "--prefill-sizes") && i + 1 < argc) {
            options->prefill_count = 0;
            if (!parse_size_list(argv[++i], options)) return false;
        } else if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            exit(0);
        } else {
            return false;
        }
    }
    return options->model_path && options->tokenizer_path && options->prompt &&
           options->mode && (!strcmp(options->mode, "decode") ||
                             !strcmp(options->mode, "prefill")) &&
           options->generated_count && options->context_size &&
           options->prefill_chunk;
}

static bool make_repeated_tokens(const q38_token_batch *seed, size_t count,
                                 uint32_t **output) {
    uint32_t *tokens;
    if (!seed || !seed->token_count || !count || !output) return false;
    tokens = calloc(count, sizeof(*tokens));
    if (!tokens) return false;
    for (size_t i = 0; i < count; ++i)
        tokens[i] = seed->tokens[i % seed->token_count];
    *output = tokens;
    return true;
}

static void apply_qsa_timing(q2_sample *sample,
                             const q38_forward_qsa_timing *timing) {
    if (!sample || !timing) return;
    sample->qsa_ms = timing->total_ms;
    sample->qsa_qkv_ms = timing->qkv_projection_ms;
    sample->qsa_output_projection_ms = timing->output_projection_ms;
    sample->qsa_attention_ms = timing->attention_ms;
    sample->qsa_index_compress_ms = timing->indexer_compression_ms;
    sample->qsa_state_glue_ms = timing->score_ms +
        timing->exact_top_k_ms + timing->selected_kv_gather_ms +
        timing->state_update_ms + timing->allocation_cleanup_ms;
}

static void add_telemetry_delta(q2_sample *sample,
                                const q2_telemetry *before,
                                const q2_telemetry *after) {
    if (!sample || !before || !after) return;
    sample->telemetry_callbacks = after->callbacks - before->callbacks;
    sample->kernel_launches = after->kernel_launches -
                              before->kernel_launches;
    sample->host_syncs = after->host_syncs - before->host_syncs;
    sample->h2d_bytes = after->h2d_bytes - before->h2d_bytes;
    sample->d2h_bytes = after->d2h_bytes - before->d2h_bytes;
    sample->d2d_bytes = after->d2d_bytes - before->d2d_bytes;
    sample->non_ple_upload_bytes = after->non_ple_upload_bytes -
                                   before->non_ple_upload_bytes;
    sample->non_ple_residency_misses =
        after->non_ple_residency_misses - before->non_ple_residency_misses;
    sample->cuda_dispatch_ms = after->backend_overhead_ms -
                               before->backend_overhead_ms;
    sample->memcpy_ms = after->upload_ms - before->upload_ms;
}

static bool run_decode(q38_session *session, const q2_options *options,
                       const q38_token_batch *prompt, float *logits,
                       q2_telemetry *telemetry, q2_sample *summary,
                       q2_sample **samples_out, uint32_t **generated_out,
                       uint64_t *final_hash, bool *final_finite, char *error,
                       size_t error_len) {
    q38_forward_diagnostics diagnostics;
    q2_trace_capture trace_capture = {0};
    q2_capture prefill_capture;
    q2_sample *samples = NULL;
    q2_telemetry telemetry_before;
    uint32_t *generated = NULL;
    uint32_t next_token = 0;
    size_t step_index = 0;
    size_t sample_count = 0;
    if (!session || !options || !prompt || !logits || !telemetry ||
        !summary || !samples_out || !generated_out || !final_hash ||
        !final_finite)
        return false;
    if (options->generated_count > MAX_GENERATED ||
        options->measure_last >= options->generated_count ||
        options->measure_first > options->measure_last)
        return false;
    samples = calloc(options->generated_count, sizeof(*samples));
    generated = calloc(options->generated_count, sizeof(*generated));
    if (!samples || !generated) {
        free(samples);
        free(generated);
        return false;
    }
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.stage_trace = stage_trace;
    memset(&prefill_capture, 0, sizeof(prefill_capture));
    diagnostics.trace_user = &prefill_capture;
    diagnostics.qsa_timing = &prefill_capture.qsa_timing;
    q38_forward_cuda_set_telemetry_observer(
        session->runtime->cuda, telemetry_observer, telemetry);
    if (!q38_session_prefill_chunked(
            session, prompt->tokens, prompt->token_count,
            options->prefill_chunk, logits, VOCAB_SIZE, &next_token,
            &diagnostics, trace_step, &trace_capture, &step_index,
            error, error_len))
        goto fail;
    generated[0] = next_token;
    for (size_t index = 1; index < options->generated_count; ++index) {
        q2_capture capture;
        q38_decode_timing timing = {0};
        q38_ple_scheduler_stats ple = {0};
        const double started = now_ms();
        memset(&capture, 0, sizeof(capture));
        diagnostics.trace_user = &capture;
        diagnostics.qsa_timing = &capture.qsa_timing;
        telemetry_before = *telemetry;
        if (!q38_session_eval_timed(
                session, generated[index - 1], logits, VOCAB_SIZE,
                &next_token, &diagnostics,
                Q38_DECODE_TRACE_GENERATED_CONSUME, next_token,
                generated[index - 1], NULL, NULL, &step_index, &timing, &ple,
                error, error_len))
            goto fail;
        capture.sample.wall_ms = now_ms() - started;
        capture.sample.forward_ms = timing.forward_core_ms;
        capture.sample.argmax_ms = timing.argmax_ms;
        capture.sample.bookkeeping_ms = capture.sample.wall_ms -
                                        timing.total_ms;
        capture.sample.ple_critical_stall_ms = ple.wait_ms;
        capture.sample.ple_elapsed_ms = ple.elapsed_ms;
        capture.sample.ple_overlap_ms = ple.overlap_ms;
        apply_qsa_timing(&capture.sample, &capture.qsa_timing);
        add_telemetry_delta(&capture.sample, &telemetry_before, telemetry);
        generated[index] = next_token;
        if (index >= options->measure_first &&
            index <= options->measure_last) {
            add_sample(summary, &capture.sample);
            sample_count++;
            samples[index] = capture.sample;
        }
    }
    *final_hash = hash_bytes(logits, VOCAB_SIZE * sizeof(*logits));
    *final_finite = true;
    for (size_t i = 0; i < VOCAB_SIZE; ++i)
        if (!isfinite(logits[i])) *final_finite = false;
    *samples_out = samples;
    *generated_out = generated;
    divide_sample(summary, (double)sample_count);
    return true;
fail:
    free(samples);
    free(generated);
    return false;
}

static bool run_prefill_case(
    q38_session *session, const q2_options *options, const uint32_t *tokens,
    size_t token_count, float *logits, q2_telemetry *telemetry,
    q2_sample *sample, uint32_t *next_token, uint64_t *logits_hash,
    bool *finite, char *error, size_t error_len) {
    q38_forward_diagnostics diagnostics;
    q38_forward_qsa_timing qsa_timing = {0};
    q2_capture capture;
    q2_telemetry before;
    size_t step_index = 0;
    const double started = now_ms();
    memset(&diagnostics, 0, sizeof(diagnostics));
    memset(&capture, 0, sizeof(capture));
    diagnostics.stage_trace = stage_trace;
    diagnostics.trace_user = &capture;
    diagnostics.qsa_timing = &qsa_timing;
    q38_forward_cuda_set_telemetry_observer(
        session->runtime->cuda, telemetry_observer, telemetry);
    before = *telemetry;
    if (!q38_session_prefill_chunked(
            session, tokens, token_count, options->prefill_chunk, logits,
            VOCAB_SIZE, next_token, &diagnostics, NULL, NULL, &step_index,
            error, error_len))
        return false;
    sample->wall_ms = now_ms() - started;
    apply_qsa_timing(sample, &qsa_timing);
    add_telemetry_delta(sample, &before, telemetry);
    *logits_hash = hash_bytes(logits, VOCAB_SIZE * sizeof(*logits));
    *finite = true;
    for (size_t i = 0; i < VOCAB_SIZE; ++i)
        if (!isfinite(logits[i])) *finite = false;
    return true;
}

static void print_sample(const q2_sample *sample) {
    printf("{\"wall_ms\":%.6f,\"forward_core_ms\":%.6f,"
           "\"argmax_ms\":%.6f,\"bookkeeping_ms\":%.6f,"
           "\"ple_critical_stall_ms\":%.6f,\"ple_elapsed_ms\":%.6f,"
           "\"ple_overlap_ms\":%.6f,\"categories\":{"
           "\"QSA\":{\"ms\":%.6f,\"qkv_ms\":%.6f,"
           "\"output_projection_ms\":%.6f,\"attention_ms\":%.6f,"
           "\"index_compress_ms\":%.6f,\"state_glue_ms\":%.6f},"
           "\"MoE\":{\"ms\":%.6f},\"GDN\":{\"ms\":%.6f},"
           "\"GR\":{\"ms\":%.6f},\"norms_residual_glue\":{\"ms\":%.6f},"
           "\"LM_head\":{\"ms\":%.6f},\"host_scalar\":{\"ms\":%.6f},"
           "\"cuda_dispatch\":{\"ms\":%.6f},"
           "\"cuda_sync_wait\":{\"ms\":%.6f},"
           "\"memcpy\":{\"ms\":%.6f},"
           "\"PLE_critical_stall\":{\"ms\":%.6f}},"
           "\"traffic\":{\"kernel_launches\":%" PRIu64
           ",\"host_syncs\":%" PRIu64 ",\"h2d_bytes\":%" PRIu64
           ",\"d2h_bytes\":%" PRIu64 ",\"d2d_bytes\":%" PRIu64 "}}",
           sample->wall_ms, sample->forward_ms, sample->argmax_ms,
           sample->bookkeeping_ms, sample->ple_critical_stall_ms,
           sample->ple_elapsed_ms, sample->ple_overlap_ms, sample->qsa_ms,
           sample->qsa_qkv_ms, sample->qsa_output_projection_ms,
           sample->qsa_attention_ms, sample->qsa_index_compress_ms,
           sample->qsa_state_glue_ms, sample->moe_ms, sample->gdn_ms,
           sample->gr_ms, sample->norms_residual_glue_ms,
           sample->lm_head_ms, sample->host_scalar_ms,
           sample->cuda_dispatch_ms, sample->cuda_sync_wait_ms,
           sample->memcpy_ms, sample->ple_critical_stall_ms,
           sample->kernel_launches, sample->host_syncs, sample->h2d_bytes,
           sample->d2h_bytes, sample->d2d_bytes);
}

static int run(const q2_options *options) {
    char error[256] = {0};
    q38_runtime runtime = {0};
    q38_session session = {0};
    q38_token_batch seed = {0};
    float *logits = NULL;
    q2_telemetry telemetry = {0};
    bool session_ready = false;
    int result = 1;
    if (!q38_runtime_init(&runtime, options->model_path,
                          options->tokenizer_path, error, sizeof(error)) ||
        !q38_tokenizer_encode(&runtime.tokenizer, options->prompt, false,
                              &seed, error, sizeof(error)) ||
        !seed.token_count) {
        fprintf(stderr, "canonical benchmark initialization failed: %s\n",
                error);
        goto cleanup;
    }
    if (!q38_session_create(&session, &runtime, (uint32_t)options->context_size,
                            error, sizeof(error))) {
        fprintf(stderr, "canonical benchmark session failed: %s\n", error);
        goto cleanup;
    }
    session_ready = true;
    logits = calloc(VOCAB_SIZE, sizeof(*logits));
    if (!logits) goto cleanup;
    if (!strcmp(options->mode, "decode")) {
        q2_sample summary = {0};
        q2_sample *samples = NULL;
        uint32_t *generated = NULL;
        uint64_t final_hash = 0;
        bool finite = false;
        q38_forward_cuda_residency_stats residency = {0};
        if (!run_decode(
                &session, options, &seed, logits, &telemetry, &summary,
                &samples, &generated, &final_hash, &finite, error,
                sizeof(error)))
            goto cleanup;
        q38_forward_cuda_get_residency_stats(runtime.cuda, &residency);
        printf("{\"format\":\"q2-canonical-bench-raw-v1\","
               "\"mode\":\"decode\",\"prompt\":");
        json_string(options->prompt);
        printf(",\"prompt_ids\":");
        print_ids(seed.tokens, seed.token_count);
        printf(",\"generated_ids\":");
        print_ids(generated, options->generated_count);
        printf(",\"context_size\":%zu,\"generated_count\":%zu,"
               "\"measure_first\":%zu,\"measure_last\":%zu,"
               "\"position\":{\"prefill_final\":%zu,\"measured_first\":%zu,"
               "\"measured_last\":%zu},"
               "\"summary\":",
               options->context_size, options->generated_count,
               options->measure_first, options->measure_last,
               (size_t)seed.token_count, options->measure_first,
               options->measure_last);
        print_sample(&summary);
        printf(",\"samples\":[");
        for (size_t i = options->measure_first;
             i <= options->measure_last; ++i) {
            if (i != options->measure_first) putchar(',');
            print_sample(&samples[i]);
        }
        printf("],\"correctness\":{\"final_logits_hash\":\"%016" PRIx64
               "\",\"argmax\":%u,\"nan_inf\":%s},"
               "\"residency\":{\"all_non_ple_resident\":%s,"
               "\"persistent_resident_bytes\":%zu,"
               "\"persistent_resident_tensors\":%" PRIu64
               ",\"persistent_ple_entries\":%" PRIu64
               ",\"non_ple_upload_bytes\":%" PRIu64
               ",\"non_ple_residency_misses\":%" PRIu64
               "},\"telemetry\":{\"callbacks\":%" PRIu64
               ",\"kernel_ms\":%.6f,\"backend_overhead_ms\":%.6f,"
               "\"upload_ms\":%.6f,\"h2d_bytes\":%" PRIu64
               ",\"d2h_bytes\":%" PRIu64 ",\"host_syncs\":%" PRIu64 "}}\n",
               final_hash, generated[options->generated_count - 1],
               finite ? "false" : "true",
               residency.all_non_ple_resident ? "true" : "false",
               residency.persistent_resident_bytes,
               residency.persistent_resident_tensors,
               residency.persistent_ple_entries,
               summary.non_ple_upload_bytes,
               summary.non_ple_residency_misses,
               telemetry.callbacks, telemetry.kernel_ms,
               telemetry.backend_overhead_ms, telemetry.upload_ms,
               telemetry.h2d_bytes, telemetry.d2h_bytes,
               telemetry.host_syncs);
        free(samples);
        free(generated);
    } else {
        q38_forward_cuda_residency_stats residency = {0};
        printf("{\"format\":\"q2-canonical-bench-raw-v1\","
               "\"mode\":\"prefill\",\"prompt\":");
        json_string(options->prompt);
        printf(",\"seed_prompt_ids\":");
        print_ids(seed.tokens, seed.token_count);
        printf(",\"context_size\":%zu,\"cases\":[", options->context_size);
        for (size_t c = 0; c < options->prefill_count; ++c) {
            uint32_t *tokens = NULL;
            q2_sample sample = {0};
            uint32_t next_token = 0;
            uint64_t logits_hash = 0;
            bool finite = false;
            if (!make_repeated_tokens(&seed, options->prefill_sizes[c],
                                      &tokens) ||
                options->prefill_sizes[c] > options->context_size ||
                !run_prefill_case(
                    &session, options, tokens, options->prefill_sizes[c],
                    logits, &telemetry, &sample, &next_token, &logits_hash,
                    &finite, error, sizeof(error))) {
                free(tokens);
                fprintf(stderr, "canonical prefill failed: %s\n", error);
                goto cleanup;
            }
            if (c) putchar(',');
            printf("{\"token_count\":%zu,\"next_token\":%u,"
                   "\"logits_hash\":\"%016" PRIx64
                   "\",\"nan_inf\":%s,\"position\":%zu,\"sample\":",
                   options->prefill_sizes[c], next_token, logits_hash,
                   finite ? "false" : "true", options->prefill_sizes[c]);
            print_sample(&sample);
            putchar('}');
            free(tokens);
        }
        q38_forward_cuda_get_residency_stats(runtime.cuda, &residency);
        printf("],\"residency\":{\"all_non_ple_resident\":%s,"
               "\"persistent_resident_bytes\":%zu,"
               "\"persistent_resident_tensors\":%" PRIu64
               ",\"persistent_ple_entries\":%" PRIu64
               ",\"non_ple_upload_bytes\":%" PRIu64
               ",\"non_ple_residency_misses\":%" PRIu64
               "},\"telemetry\":{\"callbacks\":%" PRIu64
               ",\"kernel_ms\":%.6f,\"backend_overhead_ms\":%.6f,"
               "\"upload_ms\":%.6f,\"h2d_bytes\":%" PRIu64
               ",\"d2h_bytes\":%" PRIu64 ",\"host_syncs\":%" PRIu64 "}}\n",
               residency.all_non_ple_resident ? "true" : "false",
               residency.persistent_resident_bytes,
               residency.persistent_resident_tensors,
               residency.persistent_ple_entries,
               telemetry.non_ple_upload_bytes,
               telemetry.non_ple_residency_misses,
               telemetry.callbacks, telemetry.kernel_ms,
               telemetry.backend_overhead_ms, telemetry.upload_ms,
               telemetry.h2d_bytes, telemetry.d2h_bytes,
               telemetry.host_syncs);
    }
    result = 0;
cleanup:
    free(logits);
    q38_token_batch_free(&seed);
    if (session_ready) q38_session_destroy(&session);
    q38_runtime_destroy(&runtime);
    return result;
}

int main(int argc, char **argv) {
    q2_options options;
    if (!parse_options(argc, argv, &options)) {
        usage(stderr);
        return 2;
    }
    return run(&options);
}
