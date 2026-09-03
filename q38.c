/* q38.c — q38 inspection CLI plus the minimal CUDA generation smoke path.
 *
 * A single executable that implements --platform, --inspect, --list-tensors,
 * and --memory-plan. The --generate path is only a thin wrapper around the
 * validated decode/backend APIs; all output is human-readable or JSON.
 */

#include "q38.h"
#include "q38_cuda.h"
#include "q38_gguf.h"
#include "q38_memory.h"
#include "q38_platform.h"
#include "q38_decode.h"
#include "q38_forward_cuda.h"
#include "q38_tokenizer.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(FILE *fp) {
    fprintf(fp,
        "usage: q38 <mode> [options]\n"
        "\n"
        "modes:\n"
        "  --platform                 Probe the platform (CUDA + host memory)\n"
        "  --platform-json            Platform probe in JSON format\n"
        "  --inspect <model.gguf>     Print GGUF metadata and tensor summary\n"
        "  --list-tensors <model.gguf> List individual tensors\n"
        "  --memory-plan <model.gguf> Dry-run memory plan (no allocation)\n"
        "  --generate <model.gguf>    CUDA greedy text-generation smoke path\n"
        "\n"
        "options:\n"
        "  --tokenizer <model-dir>    Native tokenizer assets (for --generate)\n"
        "  --prompt <text>            Prompt (for --generate)\n"
        "  --max-tokens <n>           Generated tokens, 1..32 (default: 16)\n"
        "  --json                     Machine-readable output\n"
        "  --verbose                  Extra diagnostics\n");
}

static void print_bytes_json(uint64_t b) { printf("%" PRIu64, b); }

static void print_platform_human(const q38_platform_info *p) {
    printf("cuda devices:       %d\n", p->cuda_device_count);
    printf("cuda device:        %d\n", p->cuda_device);
    printf("device name:        %s\n", p->device_name);
    printf("compute capability: sm_%d%d\n", p->cc_major, p->cc_minor);
    printf("driver version:     %s\n", p->driver_version[0] ? p->driver_version : "n/a");
    printf("runtime version:    %s\n", p->runtime_version[0] ? p->runtime_version : "n/a");
    printf("cuda total:         ");
    print_bytes_json(p->cuda_total_bytes);
    printf(" bytes\n");
    printf("cuda free:          ");
    print_bytes_json(p->cuda_free_bytes);
    printf(" bytes\n");
    printf("host mem total:     ");
    print_bytes_json(p->mem_total_bytes);
    printf(" bytes\n");
    printf("host mem available: ");
    print_bytes_json(p->mem_available_bytes);
    printf(" bytes\n");
}

static void print_platform_json(const q38_platform_info *p) {
    printf("{\"cuda_device_count\":%d,\"cuda_device\":%d,"
           "\"cc_major\":%d,\"cc_minor\":%d,"
           "\"cuda_total_bytes\":%" PRIu64 ",\"cuda_free_bytes\":%" PRIu64
           ",\"mem_total_bytes\":%" PRIu64 ",\"mem_available_bytes\":%" PRIu64
           ",\"device_name\":\"%s\",\"driver_version\":\"%s\","
           "\"runtime_version\":\"%s\"}\n",
           p->cuda_device_count, p->cuda_device,
           p->cc_major, p->cc_minor,
           p->cuda_total_bytes, p->cuda_free_bytes,
           p->mem_total_bytes, p->mem_available_bytes,
           p->device_name, p->driver_version, p->runtime_version);
}

static int cmd_platform(const q38_options *opt) {
    q38_platform_info p;
    char reason[256];
    if (q38_platform_probe(&p, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "q38: unsupported platform: %s\n", reason);
        return 1;
    }
    if (opt->json) {
        print_platform_json(&p);
    } else {
        print_platform_human(&p);
    }
    return 0;
}

static void model_summary(const q38_gguf *m, uint64_t *tensor_bytes,
                          uint64_t *params) {
    *tensor_bytes = 0;
    *params = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        *tensor_bytes += m->tensors[i].bytes;
        *params += m->tensors[i].elements;
    }
}

static int cmd_inspect(const q38_options *opt) {
    char err[256];
    q38_gguf *m = q38_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q38: %s\n", err);
        return 1;
    }

    q38_str name = {0}, arch = {0};
    q38_gguf_get_string(m, "general.name", &name);
    q38_gguf_get_string(m, "general.architecture", &arch);

    uint64_t tensor_bytes = 0, params = 0;
    model_summary(m, &tensor_bytes, &params);

    if (opt->json) {
        printf("{\"name\":\"%.*s\",\"architecture\":\"%.*s\","
               "\"version\":%u,\"metadata_keys\":%" PRIu64
               ",\"tensors\":%" PRIu64
               ",\"file_bytes\":%" PRIu64
               ",\"tensor_bytes\":%" PRIu64
               ",\"logical_parameters\":%" PRIu64 "}\n",
               (int)name.len, name.ptr ? name.ptr : "",
               (int)arch.len, arch.ptr ? arch.ptr : "",
               m->version, m->n_kv, m->n_tensors,
               m->size, tensor_bytes, params);
    } else {
        printf("model:     %.*s\n", (int)name.len, name.ptr ? name.ptr : "");
        printf("arch:      %.*s\n", (int)arch.len, arch.ptr ? arch.ptr : "");
        printf("gguf:      v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
               m->version, m->n_kv, m->n_tensors);
        printf("file size: %" PRIu64 " bytes\n", m->size);
        printf("tensor bytes: %" PRIu64 "\n", tensor_bytes);
        printf("logical parameters: %" PRIu64 "\n", params);

        printf("tensor types:\n");
        for (uint32_t type = 0; type < 64; type++) {
            uint64_t count = 0, bytes = 0;
            for (uint64_t i = 0; i < m->n_tensors; i++) {
                if (m->tensors[i].type == type) {
                    count++;
                    bytes += m->tensors[i].bytes;
                }
            }
            if (count != 0) {
                printf("  %-8s %5" PRIu64 " tensors, %" PRIu64 " bytes\n",
                       q38_gguf_type_name(type), count, bytes);
            }
        }
    }

    q38_gguf_close(m);
    return 0;
}

static int cmd_list_tensors(const q38_options *opt) {
    char err[256];
    q38_gguf *m = q38_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q38: %s\n", err);
        return 1;
    }

    if (opt->json) {
        printf("{\"tensors\":[");
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const q38_tensor *t = &m->tensors[i];
            printf("%s{\"name\":\"%.*s\",\"type\":\"%s\",\"ndim\":%u,"
                   "\"elements\":%" PRIu64 ",\"bytes\":%" PRIu64 "}",
                   i ? "," : "",
                   (int)t->name.len, t->name.ptr,
                   q38_gguf_type_name(t->type),
                   t->ndim, t->elements, t->bytes);
        }
        printf("]}\n");
    } else {
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const q38_tensor *t = &m->tensors[i];
            printf("%-48.*s %-8s %" PRIu64 " elems %" PRIu64 " bytes\n",
                   (int)t->name.len, t->name.ptr,
                   q38_gguf_type_name(t->type),
                   t->elements, t->bytes);
        }
    }

    q38_gguf_close(m);
    return 0;
}

static int cmd_memory_plan(const q38_options *opt) {
    char err[256];
    q38_gguf *m = q38_gguf_open(opt->model_path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "q38: %s\n", err);
        return 1;
    }

    uint64_t tensor_bytes = 0, params = 0;
    model_summary(m, &tensor_bytes, &params);

    /* Dry run: no cudaMalloc, no host-registration. The mapping is already in
     * place; snapshot RSS + host available + CUDA free/total. */
    q38_memory_tracker tracker;
    q38_memory_tracker_init(&tracker);

    q38_platform_info p;
    char reason[256];
    uint64_t cuda_total = 0, cuda_free = 0;
    if (q38_platform_probe(&p, reason, sizeof(reason)) == 0) {
        cuda_total = p.cuda_total_bytes;
        cuda_free = p.cuda_free_bytes;
    }

    q38_memory_snapshot snap;
    q38_memory_capture(&tracker, "gguf_mapped",
                       m->size, m->size, 0, &snap);
    snap.cuda_total_bytes = cuda_total;
    snap.cuda_free_bytes = cuda_free;

    if (opt->json) {
        char buf[1024];
        q38_memory_snapshot_json(&snap, buf, sizeof(buf));
        printf("%s\n", buf);
    } else {
        printf("model file:        %" PRIu64 " bytes\n", snap.model_file_bytes);
        printf("model mapped:      %" PRIu64 " bytes\n", snap.model_mapped_bytes);
        printf("rss:               %" PRIu64 " bytes\n", snap.rss_bytes);
        printf("mem available:     %" PRIu64 " bytes\n", snap.mem_available_bytes);
        printf("cuda free:         %" PRIu64 " bytes\n", snap.cuda_free_bytes);
        printf("cuda total:        %" PRIu64 " bytes\n", snap.cuda_total_bytes);
        printf("tensor bytes:      %" PRIu64 "\n", tensor_bytes);
        printf("peak internal:     %" PRIu64 " bytes\n", snap.peak_internal_bytes);
    }

    q38_gguf_close(m);
    return 0;
}

typedef struct {
    double started_ms;
    double last_generated_ms;
    double first_token_ms;
    double per_token_ms[32];
    size_t generated_seen;
    size_t prompt_seen;
    size_t nan_count;
    size_t inf_count;
    bool fallback;
    uint64_t backend_rows;
    uint64_t scalar_rows;
    uint64_t backend_declines;
    uint64_t initial_cuda_free;
    uint64_t min_cuda_free;
    uint64_t cuda_total;
    uint64_t peak_rss;
    uint64_t model_bytes;
    q38_memory_tracker memory;
} q38_generate_evidence;

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void sample_generate_memory(q38_generate_evidence *evidence,
                                   uint64_t model_bytes) {
    q38_platform_info platform;
    char reason[256];
    if (q38_platform_probe(&platform, reason, sizeof(reason)) != 0) return;
    if (!evidence->initial_cuda_free)
        evidence->initial_cuda_free = platform.cuda_free_bytes;
    if (!evidence->min_cuda_free ||
        platform.cuda_free_bytes < evidence->min_cuda_free)
        evidence->min_cuda_free = platform.cuda_free_bytes;
    evidence->cuda_total = platform.cuda_total_bytes;
    q38_memory_snapshot snapshot;
    uint64_t allocated = evidence->initial_cuda_free >
        platform.cuda_free_bytes
        ? evidence->initial_cuda_free - platform.cuda_free_bytes : 0;
    q38_memory_capture(&evidence->memory, "generate", model_bytes, model_bytes,
                       allocated, &snapshot);
    if (snapshot.rss_bytes > evidence->peak_rss)
        evidence->peak_rss = snapshot.rss_bytes;
}

static bool generate_stage_trace(const q38_forward_stage_usage *usage,
                                 void *opaque, char *error, size_t error_len) {
    q38_generate_evidence *evidence = opaque;
    if (!evidence || !usage) {
        if (error && error_len) snprintf(error, error_len,
                                         "invalid generate stage evidence");
        return false;
    }
    evidence->backend_rows += usage->backend_rows;
    evidence->scalar_rows += usage->scalar_rows;
    evidence->backend_declines += usage->backend_declines;
    evidence->fallback |= usage->scalar_rows != 0 ||
                          usage->backend_declines != 0;
    return true;
}

static bool generate_trace(const q38_decode_step *step, void *opaque,
                           char *error, size_t error_len) {
    q38_generate_evidence *evidence = opaque;
    if (!evidence || !step) {
        if (error && error_len) snprintf(error, error_len,
                                         "invalid generate trace evidence");
        return false;
    }
    evidence->prompt_seen += step->generated ? 0 : 1;
    if (step->generated) {
        const double now = monotonic_ms();
        const double elapsed = now - evidence->started_ms;
        if (!evidence->generated_seen) evidence->first_token_ms = elapsed;
        if (evidence->generated_seen < 32)
            evidence->per_token_ms[evidence->generated_seen] =
                evidence->generated_seen ? now - evidence->last_generated_ms
                                         : elapsed;
        evidence->last_generated_ms = now;
        evidence->generated_seen++;
    }
    if (!step->logits_finite) evidence->inf_count++;
    evidence->nan_count += step->gdn_state_stats.nan_count +
                           step->conv_history_stats.nan_count +
                           step->ple_history_stats.nan_count;
    evidence->inf_count += step->gdn_state_stats.inf_count +
                           step->conv_history_stats.inf_count +
                           step->ple_history_stats.inf_count;
    sample_generate_memory(evidence, evidence->model_bytes);
    return step->finite && step->logits_finite;
}

static void json_string(const char *text) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)(text ? text : "");
         *p; ++p) {
        switch (*p) {
        case '"': fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\b': fputs("\\b", stdout); break;
        case '\f': fputs("\\f", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (*p < 0x20) printf("\\u%04x", *p);
            else putchar(*p);
        }
    }
    putchar('"');
}

static void print_ids_json(const uint32_t *ids, size_t count) {
    putchar('[');
    for (size_t i = 0; i < count; ++i)
        printf("%s%u", i ? "," : "", ids[i]);
    putchar(']');
}

static int cmd_generate(const q38_options *opt) {
    if (!opt->model_path || !opt->tokenizer_path || !opt->prompt ||
        !opt->prompt[0] || opt->max_tokens < 1 || opt->max_tokens > 32) {
        fprintf(stderr, "q38: --generate requires --tokenizer, --prompt, and "
                        "--max-tokens in 1..32\n");
        return 2;
    }

    char error[256] = {0};
    q38_platform_info platform;
    char reason[256];
    if (q38_platform_probe(&platform, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "q38: CUDA runtime unavailable: %s\n", reason);
        return 1;
    }
    q38_gguf *model = q38_gguf_open(opt->model_path, error, sizeof(error));
    if (!model) {
        fprintf(stderr, "q38: %s\n", error);
        return 1;
    }
    q38_tokenizer tokenizer;
    memset(&tokenizer, 0, sizeof(tokenizer));
    q38_token_batch prompt = {0};
    q38_weights weights;
    memset(&weights, 0, sizeof(weights));
    q38_forward_state state;
    memset(&state, 0, sizeof(state));
    q38_forward_cuda_context *cuda = NULL;
    uint32_t *generated = NULL;
    float *logits = NULL;
    char *generated_text = NULL;
    size_t generated_text_len = 0;
    int rc = 1;

    if (!q38_tokenizer_init(&tokenizer, opt->tokenizer_path, NULL, error,
                            sizeof(error)) ||
        !q38_tokenizer_encode(&tokenizer, opt->prompt, false, &prompt, error,
                               sizeof(error)) ||
        !prompt.token_count) {
        if (!error[0]) snprintf(error, sizeof(error), "prompt encoded to zero tokens");
        fprintf(stderr, "q38: tokenizer: %s\n", error);
        goto cleanup;
    }
    if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error)) ||
        !q38_forward_state_init(&state, &weights, tokenizer.eos_id, error,
                                sizeof(error))) {
        fprintf(stderr, "q38: model state: %s\n", error);
        goto cleanup;
    }
    cuda = q38_forward_cuda_context_create(error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "q38: CUDA backend: %s\n", error);
        goto cleanup;
    }
    generated = calloc(opt->max_tokens, sizeof(*generated));
    logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*logits));
    if (!generated || !logits) {
        fprintf(stderr, "q38: generation buffers allocation failed\n");
        goto cleanup;
    }

    q38_generate_evidence evidence;
    memset(&evidence, 0, sizeof(evidence));
    evidence.started_ms = monotonic_ms();
    evidence.initial_cuda_free = platform.cuda_free_bytes;
    evidence.min_cuda_free = platform.cuda_free_bytes;
    evidence.cuda_total = platform.cuda_total_bytes;
    evidence.model_bytes = model->size;
    q38_memory_tracker_init(&evidence.memory);
    sample_generate_memory(&evidence, model->size);
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.stage_trace = generate_stage_trace;
    diagnostics.trace_user = &evidence;
    if (!q38_decode_stream_with_matrix_backend(
            model, &weights, &state, prompt.tokens, prompt.token_count,
            generated, opt->max_tokens, logits, Q38_DECODE_VOCAB_SIZE,
            &diagnostics, q38_forward_cuda_matvec_backend,
            q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend,
            cuda, generate_trace, &evidence, error, sizeof(error))) {
        fprintf(stderr, "q38: CUDA decode: %s\n", error);
        goto cleanup;
    }
    sample_generate_memory(&evidence, model->size);
    if (!q38_tokenizer_decode(&tokenizer, generated, opt->max_tokens,
                              &generated_text, &generated_text_len, error,
                              sizeof(error))) {
        fprintf(stderr, "q38: generated decode: %s\n", error);
        goto cleanup;
    }

    const uint64_t peak_cuda_allocated =
        evidence.initial_cuda_free > evidence.min_cuda_free
        ? evidence.initial_cuda_free - evidence.min_cuda_free : 0;
    const bool nan_inf = evidence.nan_count != 0 || evidence.inf_count != 0;
    if (opt->json) {
        printf("{\"format\":\"q38-cuda-cli-smoke-v1\",\"prompt\":");
        json_string(opt->prompt);
        printf(",\"prompt_ids\":");
        print_ids_json(prompt.tokens, prompt.token_count);
        printf(",\"generated_ids\":");
        print_ids_json(generated, opt->max_tokens);
        printf(",\"generated_text\":");
        json_string(generated_text);
        printf(",\"timing_ms\":{\"first_token\":%.6f,\"per_token\":[",
               evidence.first_token_ms);
        for (size_t i = 0; i < opt->max_tokens; ++i)
            printf("%s%.6f", i ? "," : "", evidence.per_token_ms[i]);
        printf("]},\"memory\":{\"cuda_total_bytes\":%" PRIu64
               ",\"cuda_free_initial_bytes\":%" PRIu64
               ",\"cuda_free_min_bytes\":%" PRIu64
               ",\"peak_cuda_allocated_bytes\":%" PRIu64
               ",\"peak_rss_bytes\":%" PRIu64
               ",\"peak_internal_bytes\":%" PRIu64
               "},\"nan_inf\":{\"present\":%s,\"nan_count\":%zu,"
               "\"inf_count\":%zu},\"fallback\":{\"used\":%s,"
               "\"backend_rows\":%" PRIu64 ",\"scalar_rows\":%" PRIu64
               ",\"backend_declines\":%" PRIu64 "}}\n",
               evidence.cuda_total, evidence.initial_cuda_free,
               evidence.min_cuda_free, peak_cuda_allocated,
               evidence.peak_rss, evidence.memory.peak_internal_bytes,
               nan_inf ? "true" : "false", evidence.nan_count,
               evidence.inf_count, evidence.fallback ? "true" : "false",
               evidence.backend_rows, evidence.scalar_rows,
               evidence.backend_declines);
    } else {
        printf("prompt ids: ");
        for (size_t i = 0; i < prompt.token_count; ++i)
            printf("%s%u", i ? " " : "", prompt.tokens[i]);
        printf("\ngenerated ids: ");
        for (size_t i = 0; i < opt->max_tokens; ++i)
            printf("%s%u", i ? " " : "", generated[i]);
        printf("\ngenerated text: %s\nfirst token: %.3f ms\n",
               generated_text, evidence.first_token_ms);
        printf("per-token ms:");
        for (size_t i = 0; i < opt->max_tokens; ++i)
            printf(" %.3f", evidence.per_token_ms[i]);
        printf("\npeak CUDA allocated: %" PRIu64 " bytes\n"
               "NaN/Inf: %s (nan=%zu inf=%zu)\n"
               "fallback: %s (backend_rows=%" PRIu64 ", scalar_rows=%" PRIu64
               ", declines=%" PRIu64 ")\n",
               peak_cuda_allocated, nan_inf ? "present" : "none",
               evidence.nan_count, evidence.inf_count,
               evidence.fallback ? "used" : "none", evidence.backend_rows,
               evidence.scalar_rows, evidence.backend_declines);
    }
    (void)generated_text_len;
    rc = 0;

cleanup:
    free(generated_text);
    free(generated);
    free(logits);
    if (cuda) q38_forward_cuda_context_destroy(cuda);
    if (state.initialized) q38_forward_state_destroy(&state);
    q38_token_batch_free(&prompt);
    q38_tokenizer_destroy(&tokenizer);
    q38_gguf_close(model);
    return rc;
}

int main(int argc, char **argv) {
    q38_options opt;
    memset(&opt, 0, sizeof(opt));

    q38_mode mode = Q38_MODE_NONE;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--platform") == 0 || strcmp(a, "--platform-json") == 0) {
            mode = Q38_MODE_PLATFORM;
            opt.platform = true;
            if (strcmp(a, "--platform-json") == 0) opt.json = true;
        } else if (strcmp(a, "--inspect") == 0) {
            mode = Q38_MODE_INSPECT;
            opt.inspect = true;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--list-tensors") == 0) {
            mode = Q38_MODE_LIST_TENSORS;
            opt.list_tensors = true;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--memory-plan") == 0) {
            mode = Q38_MODE_MEMORY_PLAN;
            opt.memory_plan = true;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--generate") == 0) {
            mode = Q38_MODE_GENERATE;
            if (i + 1 < argc) opt.model_path = argv[++i];
        } else if (strcmp(a, "--tokenizer") == 0) {
            if (i + 1 < argc) opt.tokenizer_path = argv[++i];
        } else if (strcmp(a, "--prompt") == 0) {
            if (i + 1 < argc) opt.prompt = argv[++i];
        } else if (strcmp(a, "--max-tokens") == 0) {
            if (i + 1 < argc) opt.max_tokens = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(a, "--json") == 0) {
            opt.json = true;
        } else if (strcmp(a, "--verbose") == 0) {
            opt.verbose = true;
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "q38: unknown argument '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (mode == Q38_MODE_NONE) {
        usage(stderr);
        return 2;
    }
    if (mode == Q38_MODE_GENERATE && opt.max_tokens == 0)
        opt.max_tokens = 16;

    int rc;
    switch (mode) {
    case Q38_MODE_PLATFORM:
        rc = cmd_platform(&opt);
        break;
    case Q38_MODE_INSPECT:
        rc = cmd_inspect(&opt);
        break;
    case Q38_MODE_LIST_TENSORS:
        rc = cmd_list_tensors(&opt);
        break;
    case Q38_MODE_MEMORY_PLAN:
        rc = cmd_memory_plan(&opt);
        break;
    case Q38_MODE_GENERATE:
        rc = cmd_generate(&opt);
        break;
    default:
        rc = 2;
        break;
    }

    q38_cuda_cleanup();
    return rc;
}
