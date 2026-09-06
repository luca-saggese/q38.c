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
#include "q38_directional_steering.h"
#include "q38_forward_cuda.h"
#include "q38_session.h"
#include "q38_tokenizer.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <math.h>
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
        "  --generate <model.gguf>    CUDA greedy session generation\n"
        "\n"
        "options:\n"
        "  --tokenizer <model-dir>    Native tokenizer assets (for --generate)\n"
        "  --prompt <text>            Prompt (for --generate)\n"
        "  --ctx <n>                  Session context capacity (default: 8192)\n"
        "  --prefill-chunk <n>        CUDA prefill chunk size (default: 128)\n"
        "  --prefill-reference        Use serial prefill oracle\n"
        "  --trace-state              Enable full semantic state snapshots\n"
        "  --max-tokens <n>           Maximum generated tokens (default: 256)\n"
        "  --disable-ple              Omit PLE output while retaining PLE state\n"
        "  --dir-steering-file FILE   Load a Q38 48x2560 f32 direction\n"
        "  --dir-steering-ffn F       Apply steering after FFN outputs\n"
        "  --dir-steering-attn F      Apply steering after attention outputs\n"
        "  --dump-steering-dir DIR    Dump per-layer activation rows\n"
        "  --dump-steering-component C ffn_out or attn_out\n"
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
    char name[64];
    uint32_t layer;
    uint64_t calls;
    uint64_t backend_rows;
    uint64_t scalar_rows;
    uint64_t backend_declines;
    double elapsed_ms;
} q38_stage_account;

enum { Q38_STAGE_ACCOUNT_CAPACITY = 2048 };

typedef struct {
    double started_ms;
    double prefill_ms;
    double last_generated_ms;
    double first_token_ms;
    double *per_token_ms;
    double *per_token_forward_ms;
    double *per_token_argmax_ms;
    double *per_token_bookkeeping_ms;
    double *per_token_ple_stall_ms;
    size_t per_token_capacity;
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
    size_t prompt_count;
    size_t forward_index;
    size_t target_forward_index;
    uint32_t target_input_token;
    uint64_t target_position;
    uint64_t target_committed_tokens;
    uint32_t prompt_final_argmax;
    size_t generated_emit_seen;
    size_t generated_consume_seen;
    uint64_t non_ple_upload_bytes;
    uint64_t non_ple_residency_misses;
    q38_stage_account stages[Q38_STAGE_ACCOUNT_CAPACITY];
    size_t stage_count;
    double stage_accounted_ms;
    double ple_stage_elapsed_ms;
    q38_ple_scheduler_stats ple_stats;
    bool ple_stats_valid;
    uint64_t telemetry_callbacks;
    uint64_t telemetry_allocations;
    uint64_t telemetry_syncs;
    uint64_t telemetry_h2d_bytes;
    uint64_t telemetry_d2h_bytes;
    uint64_t telemetry_upload_bytes;
    double telemetry_kernel_ms;
    double telemetry_backend_overhead_ms;
    double telemetry_upload_ms;
    double telemetry_host_wait_gpu_ms;
    uint64_t telemetry_dispatches;
    uint32_t first_consume_input;
    uint32_t first_consume_argmax;
    uint64_t first_consume_committed_tokens;
    uint64_t prefix4_logits_hash;
    uint64_t prompt_final_logits_hash;
    uint32_t prompt_top_ids[20];
    float prompt_top_values[20];
    bool prompt_top_valid;
    uint32_t first_prompt_top_ids[20];
    float first_prompt_top_values[20];
    bool first_prompt_top_valid;
    q38_decode_stats first_hidden_before_ple;
    q38_decode_stats first_ple_contribution;
    q38_decode_stats first_hidden_after_ple;
    q38_decode_stats first_final_hidden;
    bool first_hidden_before_ple_valid;
    bool first_ple_contribution_valid;
    bool first_hidden_after_ple_valid;
    bool first_final_hidden_valid;
    uint64_t first_ple_history_hash;
    bool first_ple_history_valid;
    bool target_trace_captured;
    q38_decode_stats latest_hidden_before_ple;
    q38_decode_stats latest_ple_contribution;
    q38_decode_stats latest_hidden_after_ple;
    q38_decode_stats latest_final_hidden;
    bool latest_hidden_before_ple_valid;
    bool latest_ple_contribution_valid;
    bool latest_hidden_after_ple_valid;
    bool latest_final_hidden_valid;
    bool diagnostic_state_valid;
    uint64_t diagnostic_logits_hash;
    uint64_t diagnostic_gdn_state_hash;
    uint64_t diagnostic_conv_history_hash;
    uint64_t diagnostic_ple_history_hash;
    double diagnostic_trace_ms;
    const char *steering_dump_dir;
    const char *steering_dump_component;
    uint64_t steering_dump_layers;
    q38_memory_tracker memory;
} q38_generate_evidence;

static q38_stage_account *generate_stage_account(
    q38_generate_evidence *evidence, const char *name, uint32_t layer) {
    if (!evidence || !name) return NULL;
    for (size_t i = 0; i < evidence->stage_count; ++i)
        if (evidence->stages[i].layer == layer &&
            strcmp(evidence->stages[i].name, name) == 0)
            return &evidence->stages[i];
    if (evidence->stage_count >= Q38_STAGE_ACCOUNT_CAPACITY) return NULL;
    q38_stage_account *account = &evidence->stages[evidence->stage_count++];
    memset(account, 0, sizeof(*account));
    snprintf(account->name, sizeof(account->name), "%s", name);
    account->layer = layer;
    return account;
}

static void generate_cuda_telemetry(
    const q38_forward_cuda_telemetry *telemetry, void *opaque) {
    q38_generate_evidence *evidence = opaque;
    if (!evidence || !telemetry) return;
    evidence->telemetry_callbacks++;
    evidence->telemetry_dispatches++;
    evidence->telemetry_allocations += telemetry->allocation_count;
    evidence->telemetry_syncs += telemetry->sync_count;
    evidence->telemetry_upload_bytes += telemetry->upload_bytes;
    evidence->telemetry_h2d_bytes +=
        telemetry->upload_bytes + telemetry->activation_read_bytes;
    evidence->telemetry_d2h_bytes += telemetry->d2h_bytes;
    evidence->telemetry_kernel_ms += telemetry->kernel_ms;
    evidence->telemetry_backend_overhead_ms +=
        telemetry->backend_overhead_ms;
    evidence->telemetry_upload_ms += telemetry->upload_ms;
    evidence->telemetry_host_wait_gpu_ms += telemetry->backend_overhead_ms;
    if (telemetry->non_ple_residency_miss &&
        !telemetry->ple_file_backed_access)
        evidence->non_ple_residency_misses++;
    if (!telemetry->ple_file_backed_access)
        evidence->non_ple_upload_bytes += telemetry->upload_bytes;
}

static uint64_t cli_hash_bytes(const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static q38_decode_stats cli_stats_floats(const float *values, size_t count) {
    q38_decode_stats result;
    memset(&result, 0, sizeof(result));
    result.min = INFINITY;
    result.max = -INFINITY;
    result.checksum = cli_hash_bytes(values, count * sizeof(*values));
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
        result.min = 0.0f;
        result.max = 0.0f;
        result.mean = 0.0f;
        result.rms = 0.0f;
    }
    return result;
}

static void print_decode_stats_json(const q38_decode_stats *stats) {
    printf("{\"min\":%.9g,\"max\":%.9g,\"mean\":%.9g,\"rms\":%.9g,"
           "\"max_abs\":%.9g,\"finite_count\":%zu,\"nan_count\":%zu,"
           "\"inf_count\":%zu,\"checksum\":\"%016" PRIx64 "\"}",
           stats->min, stats->max, stats->mean, stats->rms, stats->max_abs,
           stats->finite_count, stats->nan_count, stats->inf_count,
           stats->checksum);
}

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
    q38_stage_account *account = generate_stage_account(
        evidence, usage->logical_stage ? usage->logical_stage : "unknown",
        usage->layer);
    if (account) {
        account->calls++;
        account->backend_rows += usage->backend_rows;
        account->scalar_rows += usage->scalar_rows;
        account->backend_declines += usage->backend_declines;
        account->elapsed_ms += usage->elapsed_ms;
        if (strstr(account->name, "ple") != NULL)
            evidence->ple_stage_elapsed_ms += usage->elapsed_ms;
        else
            evidence->stage_accounted_ms += usage->elapsed_ms;
    }
    return true;
}

static double generate_observed_forward_ms(
    const q38_generate_evidence *evidence) {
    if (!evidence) return 0.0;
    if (!evidence->generated_seen) return evidence->prefill_ms;
    double total = evidence->first_token_ms;
    for (size_t i = 1; i < evidence->generated_seen &&
                       i < evidence->per_token_capacity; ++i)
        total += evidence->per_token_ms[i];
    return total;
}

static void print_generate_instrumentation_json(
    const q38_generate_evidence *evidence) {
    if (!evidence) {
        printf("\"instrumentation\":null");
        return;
    }
    const double observed = generate_observed_forward_ms(evidence);
    const double critical_accounted = evidence->stage_accounted_ms +
        (evidence->ple_stats_valid ? evidence->ple_stats.wait_ms : 0.0);
    const double unattributed = observed > critical_accounted
        ? observed - critical_accounted : 0.0;
    const double cpu_stage_ms = evidence->stage_accounted_ms >
        evidence->telemetry_kernel_ms + evidence->telemetry_host_wait_gpu_ms
        ? evidence->stage_accounted_ms - evidence->telemetry_kernel_ms -
          evidence->telemetry_host_wait_gpu_ms : 0.0;
    printf("\"instrumentation\":{\"observed_forward_ms\":%.6f,"
           "\"stage_accounted_ms\":%.6f,\"ple_stage_elapsed_ms\":%.6f,"
           "\"ple_elapsed_ms\":%.6f,\"ple_overlap_ms\":%.6f,"
           "\"ple_critical_stall_ms\":%.6f,\"unattributed_ms\":%.6f,"
           "\"gpu_busy_ms\":%.6f,"
           "\"cpu_waiting_on_gpu_estimate_ms\":%.6f,"
           "\"cpu_stage_orchestration_estimate_ms\":%.6f,"
           "\"cuda\":{\"telemetry_callbacks\":%" PRIu64
           ",\"backend_dispatches\":%" PRIu64 ",\"kernel_ms\":%.6f,"
           "\"backend_overhead_ms\":%.6f,\"upload_ms\":%.6f,"
           "\"h2d_bytes\":%" PRIu64 ",\"d2h_bytes\":%" PRIu64
           ",\"syncs\":%" PRIu64 ",\"allocations\":%" PRIu64
           ",\"weight_upload_bytes\":%" PRIu64 "},\"stages\":[",
           observed, evidence->stage_accounted_ms,
           evidence->ple_stage_elapsed_ms,
           evidence->ple_stats_valid ? evidence->ple_stats.elapsed_ms : 0.0,
           evidence->ple_stats_valid ? evidence->ple_stats.overlap_ms : 0.0,
           evidence->ple_stats_valid ? evidence->ple_stats.wait_ms : 0.0,
           unattributed,
           evidence->telemetry_kernel_ms,
           evidence->telemetry_host_wait_gpu_ms, cpu_stage_ms,
           evidence->telemetry_callbacks, evidence->telemetry_dispatches,
           evidence->telemetry_kernel_ms,
           evidence->telemetry_backend_overhead_ms,
           evidence->telemetry_upload_ms, evidence->telemetry_h2d_bytes,
           evidence->telemetry_d2h_bytes, evidence->telemetry_syncs,
           evidence->telemetry_allocations,
           evidence->telemetry_upload_bytes);
    for (size_t i = 0; i < evidence->stage_count; ++i) {
        const q38_stage_account *stage = &evidence->stages[i];
        printf("%s{\"layer\":%u,\"name\":\"%s\",\"calls\":%" PRIu64
               ",\"elapsed_ms\":%.6f,\"backend_rows\":%" PRIu64
               ",\"scalar_rows\":%" PRIu64 ",\"backend_declines\":%" PRIu64
               "}", i ? "," : "", stage->layer, stage->name, stage->calls,
               stage->elapsed_ms, stage->backend_rows, stage->scalar_rows,
               stage->backend_declines);
    }
    printf("]}");
}

static bool generate_trace(const q38_decode_step *step, void *opaque,
                           char *error, size_t error_len) {
    q38_generate_evidence *evidence = opaque;
    if (!evidence || !step) {
        if (error && error_len) snprintf(error, error_len,
                                         "invalid generate trace evidence");
        return false;
    }
    evidence->prompt_seen +=
        step->kind == Q38_DECODE_TRACE_PROMPT_PREDICTION ? 1 : 0;
    if (step->kind == Q38_DECODE_TRACE_PROMPT_PREDICTION) {
        if (evidence->prompt_seen + 2 == evidence->prompt_count)
            evidence->prefix4_logits_hash = step->logits_hash;
        if (evidence->prompt_seen + 1 == evidence->prompt_count)
            evidence->prompt_final_logits_hash = step->logits_hash;
        if (!evidence->first_prompt_top_valid) {
            memcpy(evidence->first_prompt_top_ids, step->top_ids,
                   sizeof(evidence->first_prompt_top_ids));
            memcpy(evidence->first_prompt_top_values, step->top_values,
                   sizeof(evidence->first_prompt_top_values));
            evidence->first_prompt_top_valid = true;
            evidence->first_ple_history_hash = step->ple_history_hash;
            evidence->first_ple_history_valid = true;
        }
        memcpy(evidence->prompt_top_ids, step->top_ids,
               sizeof(evidence->prompt_top_ids));
        memcpy(evidence->prompt_top_values, step->top_values,
               sizeof(evidence->prompt_top_values));
        evidence->prompt_top_valid = true;
    }
    if (step->kind == Q38_DECODE_TRACE_GENERATED_EMIT ||
        step->kind == Q38_DECODE_TRACE_GENERATED_CONSUME) {
        const double now = monotonic_ms();
        const double elapsed = now - evidence->started_ms;
        if (!evidence->generated_seen) evidence->first_token_ms = elapsed;
        if (evidence->generated_seen < evidence->per_token_capacity)
            evidence->per_token_ms[evidence->generated_seen] =
                evidence->generated_seen ? now - evidence->last_generated_ms
                                         : elapsed;
        evidence->last_generated_ms = now;
        evidence->generated_seen++;
    }
    if (step->kind == Q38_DECODE_TRACE_GENERATED_CONSUME &&
        !evidence->target_trace_captured) {
        memcpy(evidence->first_prompt_top_ids, step->top_ids,
               sizeof(evidence->first_prompt_top_ids));
        memcpy(evidence->first_prompt_top_values, step->top_values,
               sizeof(evidence->first_prompt_top_values));
        evidence->first_ple_history_hash = step->ple_history_hash;
        evidence->target_input_token = step->consumed_token;
        evidence->target_position = evidence->prompt_count;
        evidence->target_committed_tokens = step->committed_tokens;
        evidence->first_consume_input = step->consumed_token;
        evidence->first_consume_argmax = step->next_token;
        evidence->first_consume_committed_tokens = step->committed_tokens;
        evidence->first_hidden_before_ple = evidence->latest_hidden_before_ple;
        evidence->first_ple_contribution = evidence->latest_ple_contribution;
        evidence->first_hidden_after_ple = evidence->latest_hidden_after_ple;
        evidence->first_final_hidden = evidence->latest_final_hidden;
        evidence->first_hidden_before_ple_valid =
            evidence->latest_hidden_before_ple_valid;
        evidence->first_ple_contribution_valid =
            evidence->latest_ple_contribution_valid;
        evidence->first_hidden_after_ple_valid =
            evidence->latest_hidden_after_ple_valid;
        evidence->first_final_hidden_valid =
            evidence->latest_final_hidden_valid;
        evidence->target_trace_captured = true;
    }
    if (step->kind == Q38_DECODE_TRACE_GENERATED_EMIT)
        evidence->generated_emit_seen++;
    else if (step->kind == Q38_DECODE_TRACE_GENERATED_CONSUME)
        evidence->generated_consume_seen++;
    if (step->kind == Q38_DECODE_TRACE_PROMPT_PREDICTION)
        evidence->prompt_final_argmax = step->next_token;
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

static bool generate_diagnostic_trace(const q38_decode_step *step,
                                      void *opaque, char *error,
                                      size_t error_len) {
    q38_generate_evidence *evidence = opaque;
    if (!evidence || !step) {
        if (error && error_len)
            snprintf(error, error_len, "invalid diagnostic state trace");
        return false;
    }
    evidence->diagnostic_state_valid = true;
    evidence->diagnostic_logits_hash = step->logits_hash;
    evidence->diagnostic_gdn_state_hash = step->gdn_state_hash;
    evidence->diagnostic_conv_history_hash = step->conv_history_hash;
    evidence->diagnostic_ple_history_hash = step->ple_history_hash;
    return step->finite && step->logits_finite;
}

static bool generate_boundary_trace(uint32_t layer, const char *boundary,
                                    const float *values, size_t token_count,
                                    size_t width, void *opaque, char *error,
                                    size_t error_len) {
    q38_generate_evidence *evidence = opaque;
    if (!evidence || !boundary || !values) {
        if (error && error_len)
            snprintf(error, error_len, "invalid generate boundary evidence");
        return false;
    }
    /*
     * Boundary callbacks run before generate_trace. Select the forward that
     * consumes the first emitted token, after all prompt tokens have run.
     */
    if (token_count != 1) return true;
    if (layer < Q38_MODEL_LAYERS && evidence->steering_dump_dir &&
        evidence->steering_dump_component &&
        ((strcmp(evidence->steering_dump_component, "ffn_out") == 0 &&
          strcmp(boundary, "ffn_output") == 0) ||
         (strcmp(evidence->steering_dump_component, "attn_out") == 0 &&
          strcmp(boundary, "gdn_qsa_output") == 0)) &&
        !(evidence->steering_dump_layers & (UINT64_C(1) << layer))) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s-%u-pos0.bin",
                 evidence->steering_dump_dir,
                 evidence->steering_dump_component, layer);
        FILE *file = fopen(path, "wb");
        bool ok = false;
        if (file) {
            ok = fwrite(values, sizeof(float), width, file) == width;
            if (fclose(file) != 0) ok = false;
        }
        if (!ok) {
            if (error && error_len)
                snprintf(error, error_len,
                         "failed to write Q38 steering activation %s", path);
            return false;
        }
        evidence->steering_dump_layers |= UINT64_C(1) << layer;
    }
    q38_decode_stats stats = cli_stats_floats(values, token_count * width);
    if (layer == 1 && strcmp(boundary, "hidden_before_ple") == 0) {
        evidence->latest_hidden_before_ple = stats;
        evidence->latest_hidden_before_ple_valid = true;
    } else if (layer == 1 && strcmp(boundary, "ple_contribution") == 0) {
        evidence->latest_ple_contribution = stats;
        evidence->latest_ple_contribution_valid = true;
    } else if (layer == 1 && strcmp(boundary, "hidden_after_ple") == 0) {
        evidence->latest_hidden_after_ple = stats;
        evidence->latest_hidden_after_ple_valid = true;
    } else if (layer == UINT32_MAX &&
               strcmp(boundary, "final_hidden") == 0) {
        evidence->latest_final_hidden = stats;
        evidence->latest_final_hidden_valid = true;
        if (evidence->target_trace_captured) {
            evidence->first_final_hidden = stats;
            evidence->first_final_hidden_valid = true;
        }
    }
    return true;
}

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static double percentile_sorted(const double *values, size_t count,
                                double percentile) {
    if (!values || !count) return 0.0;
    if (count == 1) return values[0];
    const double index = percentile * (double)(count - 1);
    const size_t lower = (size_t)index;
    const size_t upper = lower + (lower + 1 < count ? 1 : 0);
    return values[lower] +
           (values[upper] - values[lower]) * (index - (double)lower);
}

static double decode_median_ms(const q38_generate_evidence *evidence) {
    if (!evidence || evidence->generated_seen <= 1 ||
        evidence->generated_seen > evidence->per_token_capacity)
        return 0.0;
    const size_t count = evidence->generated_seen - 1;
    double *copy = (double *)malloc(count * sizeof(*copy));
    if (!copy) return 0.0;
    memcpy(copy, evidence->per_token_ms + 1, count * sizeof(*copy));
    qsort(copy, count, sizeof(*copy), compare_double);
    const double result = percentile_sorted(copy, count, 0.5);
    free(copy);
    return result;
}

static double decode_p95_ms(const q38_generate_evidence *evidence) {
    if (!evidence || evidence->generated_seen <= 1 ||
        evidence->generated_seen > evidence->per_token_capacity)
        return 0.0;
    const size_t count = evidence->generated_seen - 1;
    double *copy = (double *)malloc(count * sizeof(*copy));
    if (!copy) return 0.0;
    memcpy(copy, evidence->per_token_ms + 1, count * sizeof(*copy));
    qsort(copy, count, sizeof(*copy), compare_double);
    const double result = percentile_sorted(copy, count, 0.95);
    free(copy);
    return result;
}

static void stream_piece(uint32_t token, const char *piece, size_t len,
                         void *userdata) {
    (void)token;
    (void)userdata;
    if (piece && len) {
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
    }
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

static int cmd_generate_legacy(const q38_options *opt) {
    if (!opt->model_path || !opt->tokenizer_path || !opt->prompt ||
        !opt->prompt[0] || !opt->max_tokens) {
        fprintf(stderr, "q38: --generate requires --tokenizer, --prompt, and "
                        "a positive --max-tokens value\n");
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
    if (!q38_forward_cuda_prepare_lm_head(cuda, model, weights.output, error,
                                          sizeof(error))) {
        fprintf(stderr, "q38: LM-head residency: %s\n", error);
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
    evidence.prompt_count = prompt.token_count;
    evidence.target_forward_index = prompt.token_count;
    q38_memory_tracker_init(&evidence.memory);
    sample_generate_memory(&evidence, model->size);
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.stage_trace = generate_stage_trace;
    diagnostics.boundary_trace = generate_boundary_trace;
    diagnostics.trace_user = &evidence;
    diagnostics.disable_ple = opt->disable_ple;
    if (!q38_decode_stream_with_matrix_backend(
            model, &weights, &state, prompt.tokens, prompt.token_count,
            generated, opt->max_tokens, logits, Q38_DECODE_VOCAB_SIZE,
            &diagnostics, q38_forward_cuda_matvec_backend,
            q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend,
            cuda, generate_trace, &evidence, error, sizeof(error))) {
        fprintf(stderr, "q38: CUDA decode: %s\n", error);
        goto cleanup;
    }
    if (prompt.token_count == 5 && prompt.tokens[0] == 17 &&
        prompt.tokens[1] == 478 && prompt.tokens[2] == 220 &&
        prompt.tokens[3] == 17 && prompt.tokens[4] == 283 &&
        opt->max_tokens == 2) {
        const uint32_t expected_next = opt->disable_ple ? 19u : 20u;
        if (evidence.prompt_seen != 5 ||
            evidence.prompt_final_argmax != 220 ||
            evidence.generated_emit_seen != 1 ||
            evidence.generated_consume_seen != 1 ||
            generated[0] != 220 ||
            evidence.first_consume_input != 220 ||
            evidence.first_consume_argmax != expected_next ||
            evidence.first_consume_committed_tokens != 6) {
            fprintf(stderr, "q38: canonical decode protocol assertion failed\n");
            goto cleanup;
        }
        if (evidence.prefix4_logits_hash == evidence.prompt_final_logits_hash) {
            fprintf(stderr, "q38: canonical prompt-prefix logits assertion failed\n");
            goto cleanup;
        }
    }
    evidence.ple_stats_valid =
        q38_forward_state_get_ple_prefetch_stats(&state, &evidence.ple_stats);
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
        printf(",\"prompt_final_top20\":[");
        for (size_t i = 0; i < 20; ++i)
            printf("%s{\"id\":%u,\"value\":%.9g}", i ? "," : "",
                   evidence.prompt_top_ids[i], evidence.prompt_top_values[i]);
        printf("],\"prompt_final_margin\":%.9g",
               evidence.prompt_top_values[0] - evidence.prompt_top_values[1]);
        printf(",\"disable_ple\":%s,\"first_step_evidence\":",
               opt->disable_ple ? "true" : "false");
        if (!evidence.first_prompt_top_valid) {
            printf("null");
        } else {
            printf("{\"target\":\"first_generated_consume\","
                   "\"token_index\":%zu,\"position\":%" PRIu64
                   ",\"input_token\":%u,\"committed_tokens\":%" PRIu64
                   ",\"top20\":[",
                   evidence.target_forward_index, evidence.target_position,
                   evidence.target_input_token,
                   evidence.target_committed_tokens);
            for (size_t i = 0; i < 20; ++i)
                printf("%s{\"id\":%u,\"value\":%.9g}", i ? "," : "",
                       evidence.first_prompt_top_ids[i],
                       evidence.first_prompt_top_values[i]);
            printf("],\"top1_top2_margin\":%.9g,"
                   "\"hidden_before_ple\":",
                   evidence.first_prompt_top_values[0] -
                   evidence.first_prompt_top_values[1]);
            if (evidence.first_hidden_before_ple_valid)
                print_decode_stats_json(&evidence.first_hidden_before_ple);
            else
                printf("null");
            printf(",\"ple_contribution\":");
            if (evidence.first_ple_contribution_valid)
                print_decode_stats_json(&evidence.first_ple_contribution);
            else
                printf("null");
            printf(",\"hidden_after_ple\":");
            if (evidence.first_hidden_after_ple_valid)
                print_decode_stats_json(&evidence.first_hidden_after_ple);
            else
                printf("null");
            printf(",\"final_hidden\":");
            if (evidence.first_final_hidden_valid)
                print_decode_stats_json(&evidence.first_final_hidden);
            else
                printf("null");
            printf(",\"first_8_generated_ids\":[");
            const size_t first_count = opt->max_tokens < 8
                ? opt->max_tokens : 8;
            for (size_t i = 0; i < first_count; ++i)
                printf("%s%u", i ? "," : "", generated[i]);
            printf("],\"ple_history_hash\":\"%016" PRIx64 "\"}",
                   evidence.first_ple_history_hash);
        }
        if (prompt.token_count == 5 && prompt.tokens[0] == 17 &&
            prompt.tokens[1] == 478 && prompt.tokens[2] == 220 &&
            prompt.tokens[3] == 17 && prompt.tokens[4] == 283 &&
            opt->max_tokens == 2)
            printf(",\"protocol_assertions\":{\"prompt_consumed_exact\":true,"
                   "\"prompt_final_argmax\":220,\"generated0\":220,"
                   "\"generated0_forward\":false,\"first_consume_input\":220,"
                   "\"first_consume_position\":5,\"committed_before\":5,"
                   "\"committed_after\":%" PRIu64 ",\"first_consume_argmax\":%u,"
                   "\"prefix4_logits_hash\":\"%016" PRIx64 "\","
                   "\"prompt_final_logits_hash\":\"%016" PRIx64 "\"}",
                   evidence.first_consume_committed_tokens,
                   evidence.first_consume_argmax,
                   evidence.prefix4_logits_hash,
                   evidence.prompt_final_logits_hash);
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
               ",\"backend_declines\":%" PRIu64 "},",
               evidence.cuda_total, evidence.initial_cuda_free,
               evidence.min_cuda_free, peak_cuda_allocated,
               evidence.peak_rss, evidence.memory.peak_internal_bytes,
               nan_inf ? "true" : "false", evidence.nan_count,
               evidence.inf_count, evidence.fallback ? "true" : "false",
               evidence.backend_rows, evidence.scalar_rows,
               evidence.backend_declines);
        print_generate_instrumentation_json(&evidence);
        puts("}");
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
    q38_weights_release(&weights);
    q38_token_batch_free(&prompt);
    q38_tokenizer_destroy(&tokenizer);
    q38_gguf_close(model);
    return rc;
}

static int cmd_generate(const q38_options *opt) {
    if (!opt->model_path || !opt->tokenizer_path || !opt->prompt ||
        !opt->prompt[0] || !opt->max_tokens || !opt->ctx_size) {
        fprintf(stderr, "q38: --generate requires --tokenizer, --prompt, "
                        "--ctx, and --max-tokens\n");
        return 2;
    }

    char error[256] = {0};
    q38_platform_info platform;
    char reason[256];
    if (q38_platform_probe(&platform, reason, sizeof(reason)) != 0) {
        fprintf(stderr, "q38: CUDA runtime unavailable: %s\n", reason);
        return 1;
    }

    q38_runtime runtime = {0};
    q38_session session = {0};
    q38_token_batch prompt = {0};
    q38_generate_evidence evidence = {0};
    bool session_initialized = false;
    uint32_t *generated = NULL;
    float *logits = NULL;
    char *generated_text = NULL;
    size_t generated_text_len = 0;
    size_t generated_count = 0;
    int rc = 1;
    const bool trace_state = opt->trace_state ||
        (getenv("Q38_DIAGNOSTIC_STATE_TRACE") &&
         strcmp(getenv("Q38_DIAGNOSTIC_STATE_TRACE"), "0") != 0);

    if (!q38_runtime_init(&runtime, opt->model_path, opt->tokenizer_path,
                          error, sizeof(error)) ||
        !q38_tokenizer_encode(&runtime.tokenizer, opt->prompt, false, &prompt,
                              error, sizeof(error)) ||
        !prompt.token_count) {
        if (!error[0])
            snprintf(error, sizeof(error), "prompt encoded to zero tokens");
        fprintf(stderr, "q38: runtime/tokenizer: %s\n", error);
        goto cleanup;
    }
    if (opt->directional_steering_file) {
        if (!q38_runtime_load_directional_steering(
                &runtime, opt->directional_steering_file,
                opt->directional_steering_ffn,
                opt->directional_steering_attn, error, sizeof(error))) {
            fprintf(stderr, "q38: steering: %s\n", error);
            goto cleanup;
        }
    } else if (opt->directional_steering_ffn != 0.0f ||
               opt->directional_steering_attn != 0.0f) {
        fprintf(stderr, "q38: steering scales require --dir-steering-file\n");
        goto cleanup;
    }
    if (!q38_session_create(&session, &runtime, opt->ctx_size, error,
                            sizeof(error))) {
        fprintf(stderr, "q38: session: %s\n", error);
        goto cleanup;
    }
    session_initialized = true;
    if (prompt.token_count > opt->ctx_size) {
        fprintf(stderr, "q38: prompt has %u tokens but --ctx is %u\n",
                prompt.token_count, opt->ctx_size);
        goto cleanup;
    }

    const size_t remaining = (size_t)opt->ctx_size - prompt.token_count;
    const size_t generation_limit = opt->max_tokens < remaining
        ? opt->max_tokens : remaining;
    generated = calloc(generation_limit ? generation_limit : 1,
                       sizeof(*generated));
    logits = calloc(Q38_DECODE_VOCAB_SIZE, sizeof(*logits));
    evidence.per_token_capacity = generation_limit;
    evidence.per_token_ms = generation_limit
        ? calloc(generation_limit, sizeof(*evidence.per_token_ms)) : NULL;
    evidence.per_token_forward_ms = generation_limit
        ? calloc(generation_limit, sizeof(*evidence.per_token_forward_ms)) : NULL;
    evidence.per_token_argmax_ms = generation_limit
        ? calloc(generation_limit, sizeof(*evidence.per_token_argmax_ms)) : NULL;
    evidence.per_token_bookkeeping_ms = generation_limit
        ? calloc(generation_limit,
                 sizeof(*evidence.per_token_bookkeeping_ms)) : NULL;
    evidence.per_token_ple_stall_ms = generation_limit
        ? calloc(generation_limit,
                 sizeof(*evidence.per_token_ple_stall_ms)) : NULL;
    evidence.steering_dump_dir = opt->steering_dump_dir;
    evidence.steering_dump_component = opt->steering_dump_component;
    if (!generated || !logits ||
        (generation_limit && (!evidence.per_token_ms ||
                              !evidence.per_token_forward_ms ||
                              !evidence.per_token_argmax_ms ||
                              !evidence.per_token_bookkeeping_ms ||
                              !evidence.per_token_ple_stall_ms))) {
        fprintf(stderr, "q38: generation/timing buffer allocation failed\n");
        goto cleanup;
    }

    evidence.started_ms = monotonic_ms();
    evidence.initial_cuda_free = platform.cuda_free_bytes;
    evidence.min_cuda_free = platform.cuda_free_bytes;
    evidence.cuda_total = platform.cuda_total_bytes;
    evidence.model_bytes = runtime.model->size;
    evidence.prompt_count = prompt.token_count;
    evidence.target_forward_index = prompt.token_count;
    q38_memory_tracker_init(&evidence.memory);
    sample_generate_memory(&evidence, runtime.model->size);
    q38_forward_cuda_set_telemetry_observer(
        runtime.cuda, generate_cuda_telemetry, &evidence);

    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.stage_trace = generate_stage_trace;
    diagnostics.boundary_trace =
        (trace_state || opt->steering_dump_dir) ? generate_boundary_trace : NULL;
    diagnostics.trace_user = &evidence;
    diagnostics.disable_ple = opt->disable_ple;
    size_t step_index = 0;
    uint32_t next_token = runtime.tokenizer.eos_id;
    const double prefill_started = monotonic_ms();
    q38_decode_trace state_trace = trace_state ? generate_trace : NULL;
    void *state_trace_user = trace_state ? &evidence : NULL;
    const bool prefill_ok = opt->prefill_reference
        ? q38_session_prefill_reference(
              &session, prompt.tokens, prompt.token_count, logits,
              Q38_DECODE_VOCAB_SIZE, &next_token, &diagnostics,
              state_trace, state_trace_user, &step_index, error, sizeof(error))
        : q38_session_prefill_chunked(
              &session, prompt.tokens, prompt.token_count, opt->prefill_chunk,
              logits, Q38_DECODE_VOCAB_SIZE, &next_token, &diagnostics,
              state_trace, state_trace_user, &step_index, error, sizeof(error));
    if (!prefill_ok) {
        fprintf(stderr, "q38: prefill: %s\n", error);
        goto cleanup;
    }
    evidence.prefill_ms = monotonic_ms() - prefill_started;
    evidence.first_token_ms = evidence.prefill_ms;

    if (!opt->json && generation_limit) {
        fputs("stream: ", stdout);
        fflush(stdout);
    }
    if (generation_limit) {
        generated[0] = next_token;
        generated_count = 1;
        if (!q38_session_emit(&session, logits, next_token, state_trace,
                              state_trace_user, &step_index, error,
                              sizeof(error)) ||
            (!opt->json &&
             !q38_session_stream_token(&session, next_token, stream_piece,
                                        NULL, error, sizeof(error)))) {
            fprintf(stderr, "q38: first token emission: %s\n", error);
            goto cleanup;
        }
        while (generated_count < generation_limit &&
               generated[generated_count - 1] !=
                   q38_session_eos_token(&session)) {
            const uint32_t input = generated[generated_count - 1];
            q38_decode_timing step_timing = {0};
            q38_ple_scheduler_stats step_ple = {0};
            const double eval_started = monotonic_ms();
            if (!q38_session_eval_timed(
                    &session, input, logits, Q38_DECODE_VOCAB_SIZE,
                    &next_token, &diagnostics,
                    Q38_DECODE_TRACE_GENERATED_CONSUME, next_token, input,
                    state_trace, state_trace_user, &step_index, &step_timing,
                    &step_ple, error, sizeof(error))) {
                fprintf(stderr, "q38: decode: %s\n", error);
                goto cleanup;
            }
            const double eval_wall = monotonic_ms() - eval_started;
            if (generated_count < evidence.per_token_capacity) {
                evidence.per_token_ms[generated_count] = eval_wall;
                evidence.per_token_forward_ms[generated_count] =
                    step_timing.forward_core_ms;
                evidence.per_token_argmax_ms[generated_count] =
                    step_timing.argmax_ms;
                evidence.per_token_bookkeeping_ms[generated_count] =
                    eval_wall - step_timing.total_ms;
                evidence.per_token_ple_stall_ms[generated_count] =
                    step_ple.wait_ms;
            }
            generated[generated_count++] = next_token;
            if (!opt->json &&
                !q38_session_stream_token(&session, next_token, stream_piece,
                                           NULL, error, sizeof(error))) {
                fprintf(stderr, "q38: token streaming: %s\n", error);
                goto cleanup;
            }
        }
        if (!opt->json) putchar('\n');
    }
    evidence.generated_seen = generated_count;
    if (generated_count) {
        const double diagnostic_started = monotonic_ms();
        if (!q38_session_emit(
                &session, logits, generated[generated_count - 1],
                generate_diagnostic_trace, &evidence, &step_index, error,
                sizeof(error)))
            goto cleanup;
        evidence.diagnostic_trace_ms = monotonic_ms() - diagnostic_started;
    }

    if (trace_state && prompt.token_count == 5 && prompt.tokens[0] == 17 &&
        prompt.tokens[1] == 478 && prompt.tokens[2] == 220 &&
        prompt.tokens[3] == 17 && prompt.tokens[4] == 283 &&
        opt->max_tokens == 2 && generated_count == 2) {
        const uint32_t expected_next = opt->disable_ple ? 19u : 20u;
        if (evidence.prompt_seen != 5 ||
            evidence.prompt_final_argmax != 220 ||
            evidence.generated_emit_seen != 1 ||
            evidence.generated_consume_seen != 1 ||
            generated[0] != 220 ||
            evidence.first_consume_input != 220 ||
            evidence.first_consume_argmax != expected_next ||
            evidence.first_consume_committed_tokens != 6 ||
            evidence.prefix4_logits_hash == evidence.prompt_final_logits_hash) {
            fprintf(stderr, "q38: canonical decode protocol assertion failed\n");
            goto cleanup;
        }
    }

    evidence.ple_stats_valid =
        q38_forward_state_get_ple_prefetch_stats(
            &session.state, &evidence.ple_stats);
    sample_generate_memory(&evidence, runtime.model->size);
    if (generated_count) {
        if (!q38_tokenizer_decode(&runtime.tokenizer, generated,
                                  generated_count, &generated_text,
                                  &generated_text_len, error, sizeof(error))) {
            fprintf(stderr, "q38: generated decode: %s\n", error);
            goto cleanup;
        }
    } else {
        generated_text = strdup("");
        if (!generated_text) {
            fprintf(stderr, "q38: generated text allocation failed\n");
            goto cleanup;
        }
    }

    const uint64_t peak_cuda_allocated =
        evidence.initial_cuda_free > evidence.min_cuda_free
        ? evidence.initial_cuda_free - evidence.min_cuda_free : 0;
    const bool nan_inf = evidence.nan_count != 0 || evidence.inf_count != 0;
    const double prefill_tps = evidence.prefill_ms > 0.0
        ? (double)prompt.token_count * 1000.0 / evidence.prefill_ms : 0.0;
    const double decode_median = decode_median_ms(&evidence);
    const double decode_p95 = decode_p95_ms(&evidence);
    const double generation_tps = decode_median > 0.0
        ? 1000.0 / decode_median : 0.0;

    if (opt->json) {
        printf("{\"format\":\"q38-functional-runtime-v1\",\"prefill_path\":%s,"
               "\"prefill_chunk\":%zu,\"prompt\":",
               opt->prefill_reference ? "\"reference\"" : "\"chunked\"",
               opt->prefill_chunk);
        json_string(opt->prompt);
        printf(",\"ctx_size\":%u,\"prompt_ids\":", opt->ctx_size);
        print_ids_json(prompt.tokens, prompt.token_count);
        printf(",\"generated_ids\":");
        print_ids_json(generated, generated_count);
        printf(",\"prompt_tokens\":%u,\"generated_tokens\":%zu,"
               "\"prefill_ms\":%.6f,\"prefill_tps\":%.6f,"
               "\"ttft_ms\":%.6f,\"decode_median_ms\":%.6f,"
               "\"decode_p95_ms\":%.6f,\"generation_tps\":%.6f,"
               "\"ple_wait_at_injection_ms\":%.6f,"
               "\"non_ple_upload_bytes\":%" PRIu64
               ",\"non_ple_residency_misses\":%" PRIu64,
               prompt.token_count, generated_count, evidence.prefill_ms,
               prefill_tps, evidence.first_token_ms, decode_median,
               decode_p95, generation_tps,
               session.ple_wait_at_injection_ms,
               evidence.non_ple_upload_bytes,
               evidence.non_ple_residency_misses);
        printf(",\"generated_text\":");
        json_string(generated_text);
        printf(",\"timing_ms\":{\"first_token\":%.6f,\"per_token\":[",
               evidence.first_token_ms);
        for (size_t i = 0; i < generated_count; ++i)
            printf("%s%.6f", i ? "," : "", evidence.per_token_ms[i]);
        printf("],\"forward_core\":[");
        for (size_t i = 0; i < generated_count; ++i)
            printf("%s%.6f", i ? "," : "", evidence.per_token_forward_ms[i]);
        printf("],\"argmax\":[");
        for (size_t i = 0; i < generated_count; ++i)
            printf("%s%.6f", i ? "," : "", evidence.per_token_argmax_ms[i]);
        printf("],\"bookkeeping\":[");
        for (size_t i = 0; i < generated_count; ++i)
            printf("%s%.6f", i ? "," : "",
                   evidence.per_token_bookkeeping_ms[i]);
        printf("],\"ple_critical_stall\":[");
        for (size_t i = 0; i < generated_count; ++i)
            printf("%s%.6f", i ? "," : "",
                   evidence.per_token_ple_stall_ms[i]);
        printf("]},\"memory\":{\"cuda_total_bytes\":%" PRIu64
               ",\"cuda_free_initial_bytes\":%" PRIu64
               ",\"cuda_free_min_bytes\":%" PRIu64
               ",\"peak_cuda_allocated_bytes\":%" PRIu64
               ",\"peak_rss_bytes\":%" PRIu64
               ",\"peak_internal_bytes\":%" PRIu64
               "},\"nan_inf\":{\"present\":%s,\"nan_count\":%zu,"
               "\"inf_count\":%zu},\"fallback\":{\"used\":%s,"
               "\"backend_rows\":%" PRIu64 ",\"scalar_rows\":%" PRIu64
               ",\"backend_declines\":%" PRIu64 "},",
               evidence.cuda_total, evidence.initial_cuda_free,
               evidence.min_cuda_free, peak_cuda_allocated,
               evidence.peak_rss, evidence.memory.peak_internal_bytes,
               nan_inf ? "true" : "false", evidence.nan_count,
               evidence.inf_count, evidence.fallback ? "true" : "false",
               evidence.backend_rows, evidence.scalar_rows,
               evidence.backend_declines);
        print_generate_instrumentation_json(&evidence);
        printf(",\"trace_state_enabled\":%s,\"diagnostic_state\":",
               trace_state ? "true" : "false");
        if (!evidence.diagnostic_state_valid) {
            puts("null}");
        } else {
            printf("{\"trace_ms\":%.6f,\"logits_hash\":\"%016" PRIx64
                   "\",\"gdn_state_hash\":\"%016" PRIx64
                   "\",\"conv_history_hash\":\"%016" PRIx64
                   "\",\"ple_history_hash\":\"%016" PRIx64 "\"}}\n",
                   evidence.diagnostic_trace_ms,
                   evidence.diagnostic_logits_hash,
                   evidence.diagnostic_gdn_state_hash,
                   evidence.diagnostic_conv_history_hash,
                   evidence.diagnostic_ple_history_hash);
        }
    } else {
        printf("prompt tokens:       %u\n", prompt.token_count);
        printf("generated tokens:    %zu\n", generated_count);
        printf("prompt ids: ");
        for (size_t i = 0; i < prompt.token_count; ++i)
            printf("%s%u", i ? " " : "", prompt.tokens[i]);
        printf("\ngenerated ids: ");
        for (size_t i = 0; i < generated_count; ++i)
            printf("%s%u", i ? " " : "", generated[i]);
        printf("\ngenerated text: %s\n"
               "prefill:            %.3f ms\n"
               "prefill speed:      %.3f tok/s\n"
               "TTFT:               %.3f ms\n"
               "decode median:      %.3f ms/token\n"
               "decode p95:         %.3f ms/token\n"
               "generation speed:   %.3f tok/s\n"
               "trace-state:        %s\n"
               "diagnostic trace:   %.3f ms (outside timed region)\n"
               "PLE wait-at-injection: %.3f ms\n"
               "non-PLE upload:     %" PRIu64 " bytes\n"
               "non-PLE misses:     %" PRIu64 "\n"
               "peak CUDA allocated: %" PRIu64 " bytes\n"
               "NaN/Inf:            %s (nan=%zu inf=%zu)\n"
               "fallback:           %s (backend_rows=%" PRIu64
               ", scalar_rows=%" PRIu64 ", declines=%" PRIu64 ")\n",
               generated_text, evidence.prefill_ms, prefill_tps,
               evidence.first_token_ms, decode_median, decode_p95,
               generation_tps, trace_state ? "enabled" : "disabled",
               evidence.diagnostic_trace_ms,
               session.ple_wait_at_injection_ms,
               evidence.non_ple_upload_bytes,
               evidence.non_ple_residency_misses, peak_cuda_allocated,
               nan_inf ? "present" : "none", evidence.nan_count,
               evidence.inf_count, evidence.fallback ? "used" : "none",
               evidence.backend_rows, evidence.scalar_rows,
               evidence.backend_declines);
    }
    rc = 0;

cleanup:
    free(generated_text);
    free(evidence.per_token_ms);
    free(evidence.per_token_forward_ms);
    free(evidence.per_token_argmax_ms);
    free(evidence.per_token_bookkeeping_ms);
    free(evidence.per_token_ple_stall_ms);
    free(generated);
    free(logits);
    q38_token_batch_free(&prompt);
    if (session_initialized) q38_session_destroy(&session);
    q38_runtime_destroy(&runtime);
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
        } else if (strcmp(a, "--ctx") == 0) {
            if (i + 1 < argc)
                opt.ctx_size = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(a, "--prefill-chunk") == 0) {
            if (i + 1 < argc)
                opt.prefill_chunk = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(a, "--prefill-reference") == 0) {
            opt.prefill_reference = true;
        } else if (strcmp(a, "--trace-state") == 0) {
            opt.trace_state = true;
        } else if (strcmp(a, "--disable-ple") == 0) {
            opt.disable_ple = true;
        } else if (strcmp(a, "--dir-steering-file") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            opt.directional_steering_file = argv[++i];
        } else if (strcmp(a, "--dir-steering-ffn") == 0 ||
                   strcmp(a, "--dir-steering-attn") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            char *end = NULL;
            const float value = strtof(argv[++i], &end);
            if (!end || *end || !isfinite(value) ||
                value < -100.0f || value > 100.0f) {
                fprintf(stderr, "q38: invalid steering scale\n");
                return 2;
            }
            opt.directional_steering_scale_set = true;
            if (strcmp(a, "--dir-steering-ffn") == 0)
                opt.directional_steering_ffn = value;
            else
                opt.directional_steering_attn = value;
        } else if (strcmp(a, "--dump-steering-dir") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            opt.steering_dump_dir = argv[++i];
        } else if (strcmp(a, "--dump-steering-component") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            opt.steering_dump_component = argv[++i];
            if (strcmp(opt.steering_dump_component, "ffn_out") != 0 &&
                strcmp(opt.steering_dump_component, "attn_out") != 0) {
                fprintf(stderr, "q38: invalid steering dump component\n");
                return 2;
            }
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
        opt.max_tokens = 256;
    if (mode == Q38_MODE_GENERATE && opt.ctx_size == 0)
        opt.ctx_size = 8192;
    if (mode == Q38_MODE_GENERATE && opt.prefill_chunk == 0)
        opt.prefill_chunk = 128;
    if (opt.directional_steering_file &&
        !opt.directional_steering_scale_set)
        opt.directional_steering_ffn = 1.0f;

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
