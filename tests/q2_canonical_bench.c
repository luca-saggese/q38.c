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
#include <unistd.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <cuda_runtime_api.h>

enum {
    VOCAB_SIZE = Q38_DECODE_VOCAB_SIZE,
    MAX_GENERATED = 4096,
    MAX_PREFILL_CASES = 8,
    REFERENCE_WARMUPS = 1,
    REFERENCE_RUNS = 10
};

enum {
    Q2_MAX_TIMING_EVENTS = 1024,
    Q2_OWNER_COUNT = 10
};

#ifndef Q38_QUICK_RUNS
#define Q38_QUICK_RUNS 2
#endif

typedef enum {
    Q2_OWNER_QSA = 0,
    Q2_OWNER_GDN,
    Q2_OWNER_MOE,
    Q2_OWNER_GR,
    Q2_OWNER_PLE,
    Q2_OWNER_LM_HEAD,
    Q2_OWNER_NORM,
    Q2_OWNER_ROUTER,
    Q2_OWNER_OTHER_LAYER,
    Q2_OWNER_UNKNOWN
} q2_owner;

typedef struct {
    uint64_t calls;
    double callback_wall_ms;
    double kernel_ms;
    double host_wait_ms;
    double host_overhead_ms;
    uint64_t bytes_read;
} q2_owner_stat;

typedef struct {
    char name[48];
    char parent[48];
    char owner[24];
    uint32_t layer;
    uint64_t calls;
    double elapsed_ms;
} q2_timing_event;

typedef struct {
    double wall_ms;
    double forward_ms;
    double argmax_ms;
    double bookkeeping_ms;
    double ple_critical_stall_ms;
    double ple_elapsed_ms;
    double ple_overlap_ms;
    double ple_request_build_ms;
    double ple_history_ngram_ms;
    double ple_index_lookup_ms;
    double ple_file_io_ms;
    double ple_decode_dequant_ms;
    double ple_accumulation_ms;
    double ple_async_submit_ms;
    double ple_worker_exec_ms;
    double ple_worker_cpu_ms;
    double ple_result_publish_ms;
    double ple_injection_ms;
    double ple_wait_at_injection_ms;
    uint64_t ple_file_read_ops;
    uint64_t ple_file_read_min_bytes;
    uint64_t ple_file_read_max_bytes;
    uint64_t ple_request_id;
    uint64_t ple_token_position;
    uint64_t ple_submit_position;
    uint64_t ple_injection_position;
    double ple_t0_ms;
    double ple_t1_ms;
    double ple_t2_ms;
    double ple_t3_ms;
    double ple_t4_ms;
    double ple_t5_ms;
    double ple_t6_ms;
    double ple_t7_ms;
    double ple_t8_ms;
    double ple_t9_ms;
    double ple_t10_ms;
    double ple_t11_ms;
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
    double telemetry_callback_wall_ms;
    double backend_host_work_ms;
    double memcpy_ms;
    double other_ms;
    uint64_t telemetry_callbacks;
    uint64_t kernel_launches;
    uint64_t host_syncs;
    uint64_t real_cuda_sync_count;
    double host_blocked_on_cuda_ms;
    uint64_t sync_reason_count[Q38_CUDA_SYNC_REASON_COUNT];
    double sync_reason_ms[Q38_CUDA_SYNC_REASON_COUNT];
    double sync_reason_max_ms[Q38_CUDA_SYNC_REASON_COUNT];
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint64_t d2d_bytes;
    uint64_t non_ple_upload_bytes;
    uint64_t non_ple_residency_misses;
    double timing_embedding_ms;
    double timing_ple_async_window_ms;
    double timing_qsa_ms;
    double timing_gdn_ms;
    double timing_moe_ms;
    double timing_gr_ms;
    double timing_lm_head_ms;
    double timing_norms_residual_glue_ms;
    double timing_other_layer_ms;
    double timing_unexplained_ms;
    q2_owner_stat owners[Q2_OWNER_COUNT];
} q2_sample;

typedef struct {
    q2_sample sample;
    q38_forward_qsa_timing qsa_timing;
    q38_forward_cuda_sync_stats sync_before;
    q2_timing_event timing_events[Q2_MAX_TIMING_EVENTS];
    size_t timing_event_count;
} q2_capture;

typedef struct {
    q2_sample summary;
    q2_sample *samples;
    uint32_t *generated;
    uint64_t final_hash;
    bool finite;
} q2_decode_run;

typedef struct {
    size_t token_count;
    uint32_t *tokens;
    q2_sample measured[REFERENCE_RUNS];
    uint32_t next_tokens[REFERENCE_RUNS];
    uint64_t logits_hashes[REFERENCE_RUNS];
    bool finite[REFERENCE_RUNS];
} q2_prefill_reference;

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
    double callback_wall_ms;
    double host_wait_ms;
    q2_owner_stat owners[Q2_OWNER_COUNT];
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
    uint64_t cuda_total_bytes;
    uint64_t min_cuda_free_bytes;
    uint64_t peak_unified_rss_bytes;
} q2_memory;

typedef struct {
    char device_name[128];
    char cpu_arch[64];
    int cuda_driver_version;
    int cuda_runtime_version;
    uint64_t cuda_total_bytes;
    uint64_t unified_memory_bytes;
} q2_hardware;

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

static void observe_memory(q2_memory *memory) {
    struct rusage usage;
    size_t rss_bytes = 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!memory) return;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        if (!memory->cuda_total_bytes)
            memory->cuda_total_bytes = (uint64_t)total_bytes;
        if (!memory->min_cuda_free_bytes ||
            free_bytes < memory->min_cuda_free_bytes)
            memory->min_cuda_free_bytes = (uint64_t)free_bytes;
    }
    memset(&usage, 0, sizeof(usage));
    if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss > 0)
        rss_bytes = (size_t)usage.ru_maxrss * 1024u;
    if (rss_bytes > memory->peak_unified_rss_bytes)
        memory->peak_unified_rss_bytes = rss_bytes;
}

static void query_hardware(q2_hardware *hardware) {
    struct cudaDeviceProp properties;
    struct utsname system_info;
    int device = 0;
    int driver = 0;
    int runtime = 0;
    long pages;
    long page_size;
    if (!hardware) return;
    memset(hardware, 0, sizeof(*hardware));
    if (uname(&system_info) == 0)
        snprintf(hardware->cpu_arch, sizeof(hardware->cpu_arch), "%.63s",
                 system_info.machine);
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        snprintf(hardware->device_name, sizeof(hardware->device_name), "%.127s",
                 properties.name);
        hardware->cuda_total_bytes = properties.totalGlobalMem;
    }
    (void)cudaDriverGetVersion(&driver);
    (void)cudaRuntimeGetVersion(&runtime);
    hardware->cuda_driver_version = driver;
    hardware->cuda_runtime_version = runtime;
    pages = sysconf(_SC_PHYS_PAGES);
    page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        hardware->unified_memory_bytes = (uint64_t)pages * (uint64_t)page_size;
}

static void print_hardware(const q2_hardware *hardware) {
    if (!hardware) return;
    printf("\"hardware\":{\"device\":");
    json_string(hardware->device_name);
    printf(",\"gpu\":\"DGX Spark / GB10\",\"cpu_arch\":");
    json_string(hardware->cpu_arch);
    printf(",\"cuda_driver_version\":%d,\"cuda_runtime_version\":%d,"
           "\"cuda_total_bytes\":%" PRIu64
           ",\"unified_memory_bytes\":%" PRIu64 "}",
           hardware->cuda_driver_version, hardware->cuda_runtime_version,
           hardware->cuda_total_bytes, hardware->unified_memory_bytes);
}

static void print_memory(const q2_memory *memory) {
    uint64_t peak_cuda = 0;
    if (memory && memory->cuda_total_bytes >= memory->min_cuda_free_bytes)
        peak_cuda = memory->cuda_total_bytes - memory->min_cuda_free_bytes;
    printf("\"memory\":{\"peak_cuda_bytes\":%" PRIu64
           ",\"peak_unified_rss_bytes\":%" PRIu64 "}",
           peak_cuda, memory ? memory->peak_unified_rss_bytes : 0);
}

static void zero_sample(q2_sample *sample) {
    if (sample) memset(sample, 0, sizeof(*sample));
}

static q2_owner owner_for_record(const q38_forward_cuda_telemetry *record) {
    const char *stage = record ? record->logical_stage : NULL;
    const char *subsystem = record ? record->subsystem : NULL;
    if ((stage && strstr(stage, "router")) ||
        (record && record->operation && strstr(record->operation, "router")))
        return Q2_OWNER_ROUTER;
    if ((subsystem && !strcmp(subsystem, "qsa")) ||
        (stage && strstr(stage, "qsa")))
        return Q2_OWNER_QSA;
    if ((subsystem && !strcmp(subsystem, "gdn")) ||
        (stage && strstr(stage, "gdn")))
        return Q2_OWNER_GDN;
    if ((subsystem && !strcmp(subsystem, "moe")) ||
        (stage && (strstr(stage, "moe") || strstr(stage, "expert"))))
        return Q2_OWNER_MOE;
    if ((subsystem && !strcmp(subsystem, "gr")) ||
        (stage && strstr(stage, "gr_")))
        return Q2_OWNER_GR;
    if (subsystem && !strcmp(subsystem, "ple"))
        return Q2_OWNER_PLE;
    if (record && record->ple_file_backed_access)
        return Q2_OWNER_PLE;
    if (subsystem && !strcmp(subsystem, "lm_head"))
        return Q2_OWNER_LM_HEAD;
    /*
     * The remaining backend callbacks are scalar/runtime glue (for example
     * row_matvec) that has no finer logical stage.  Keep them attributed to
     * the forward runtime rather than leaving a matrix call owner-less.
     */
    return Q2_OWNER_OTHER_LAYER;
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
    telemetry->callback_wall_ms += record->callback_wall_ms;
    telemetry->host_wait_ms += record->host_wait_ms;
    q2_owner_stat *owner = &telemetry->owners[owner_for_record(record)];
    owner->calls++;
    owner->callback_wall_ms += record->callback_wall_ms;
    owner->kernel_ms += record->kernel_ms;
    owner->host_wait_ms += record->host_wait_ms;
    owner->host_overhead_ms +=
        fmax(0.0f, record->backend_overhead_ms - record->host_wait_ms);
    owner->bytes_read += record->weight_bytes;
    if (!record->ple_file_backed_access) {
        telemetry->non_ple_upload_bytes += record->upload_bytes;
        if (record->non_ple_residency_miss)
            telemetry->non_ple_residency_misses++;
    }
}

static bool timing_trace(const q38_forward_timing_usage *usage, void *opaque,
                         char *error, size_t error_len) {
    q2_capture *capture = (q2_capture *)opaque;
    if (!capture || !usage || !usage->name || !usage->owner) {
        if (error && error_len)
            snprintf(error, error_len, "invalid forward timing span");
        return false;
    }
    for (size_t i = 0; i < capture->timing_event_count; ++i) {
        q2_timing_event *event = &capture->timing_events[i];
        if (event->layer == usage->layer &&
            !strcmp(event->name, usage->name) &&
            !strcmp(event->parent ? event->parent : "",
                    usage->parent ? usage->parent : "")) {
            event->calls++;
            event->elapsed_ms += usage->elapsed_ms;
            return true;
        }
    }
    if (capture->timing_event_count >= Q2_MAX_TIMING_EVENTS) {
        if (error && error_len)
            snprintf(error, error_len, "forward timing tree capacity exceeded");
        return false;
    }
    q2_timing_event *event =
        &capture->timing_events[capture->timing_event_count++];
    memset(event, 0, sizeof(*event));
    snprintf(event->name, sizeof(event->name), "%s", usage->name);
    snprintf(event->parent, sizeof(event->parent), "%s",
             usage->parent ? usage->parent : "");
    snprintf(event->owner, sizeof(event->owner), "%s", usage->owner);
    event->layer = usage->layer;
    event->calls = 1;
    event->elapsed_ms = usage->elapsed_ms;
    return true;
}

static double timing_child_ms(const q2_capture *capture,
                              const q2_timing_event *parent) {
    double child_ms = 0.0;
    if (!capture || !parent) return 0.0;
    for (size_t i = 0; i < capture->timing_event_count; ++i) {
        const q2_timing_event *event = &capture->timing_events[i];
        if (event->layer == parent->layer &&
            !strcmp(event->parent, parent->name))
            child_ms += event->elapsed_ms;
    }
    return child_ms;
}

static void add_timing_owner(q2_sample *sample, const char *name,
                             const char *owner, double elapsed_ms) {
    if (!sample || !owner || elapsed_ms <= 0.0) return;
    if (!strcmp(owner, "EMBEDDING")) sample->timing_embedding_ms += elapsed_ms;
    else if (!strcmp(owner, "PLE"))
        sample->timing_ple_async_window_ms += elapsed_ms;
    else if (!strcmp(owner, "QSA")) sample->timing_qsa_ms += elapsed_ms;
    else if (!strcmp(owner, "GDN")) sample->timing_gdn_ms += elapsed_ms;
    else if (!strcmp(owner, "MoE")) sample->timing_moe_ms += elapsed_ms;
    else if (!strcmp(owner, "GR")) sample->timing_gr_ms += elapsed_ms;
    else if (!strcmp(owner, "LM_HEAD"))
        sample->timing_lm_head_ms += elapsed_ms;
    else if (!strcmp(owner, "OTHER_LAYER")) {
        sample->timing_other_layer_ms += elapsed_ms;
        if (name && strstr(name, "residual"))
            sample->timing_norms_residual_glue_ms += elapsed_ms;
    } else if (!strcmp(owner, "NORM"))
        sample->timing_norms_residual_glue_ms += elapsed_ms;
}

static void finalize_timing_tree(q2_sample *sample,
                                const q2_capture *capture) {
    double accounted = 0.0;
    if (!sample || !capture) return;
    for (size_t i = 0; i < capture->timing_event_count; ++i) {
        const q2_timing_event *event = &capture->timing_events[i];
        double exclusive = event->elapsed_ms - timing_child_ms(capture, event);
        if (exclusive < 0.0) exclusive = 0.0;
        add_timing_owner(sample, event->name, event->owner, exclusive);
        if (strcmp(event->owner, "PLE") != 0)
            accounted += exclusive;
    }
    for (size_t i = 0; i < capture->timing_event_count; ++i) {
        const q2_timing_event *event = &capture->timing_events[i];
        if (strstr(event->name, "residual"))
            sample->timing_norms_residual_glue_ms += event->elapsed_ms;
    }
    const double total = sample->forward_ms > 0.0
        ? sample->forward_ms : sample->wall_ms;
    accounted += sample->ple_injection_ms + sample->ple_critical_stall_ms;
    sample->timing_unexplained_ms = total > accounted ? total - accounted : 0.0;
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
    ADD(ple_request_build_ms);
    ADD(ple_history_ngram_ms);
    ADD(ple_index_lookup_ms);
    ADD(ple_file_io_ms);
    ADD(ple_decode_dequant_ms);
    ADD(ple_accumulation_ms);
    ADD(ple_async_submit_ms);
    ADD(ple_worker_exec_ms);
    ADD(ple_worker_cpu_ms);
    ADD(ple_result_publish_ms);
    ADD(ple_injection_ms);
    ADD(ple_wait_at_injection_ms);
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
    ADD(telemetry_callback_wall_ms);
    ADD(backend_host_work_ms);
    ADD(memcpy_ms);
    ADD(other_ms);
    sum->telemetry_callbacks += sample->telemetry_callbacks;
    sum->kernel_launches += sample->kernel_launches;
    sum->host_syncs += sample->host_syncs;
    sum->real_cuda_sync_count += sample->real_cuda_sync_count;
    sum->host_blocked_on_cuda_ms += sample->host_blocked_on_cuda_ms;
    for (size_t i = 0; i < Q38_CUDA_SYNC_REASON_COUNT; ++i) {
        sum->sync_reason_count[i] += sample->sync_reason_count[i];
        sum->sync_reason_ms[i] += sample->sync_reason_ms[i];
        if (sample->sync_reason_max_ms[i] >
            sum->sync_reason_max_ms[i])
            sum->sync_reason_max_ms[i] = sample->sync_reason_max_ms[i];
    }
    sum->h2d_bytes += sample->h2d_bytes;
    sum->d2h_bytes += sample->d2h_bytes;
    sum->d2d_bytes += sample->d2d_bytes;
    sum->non_ple_upload_bytes += sample->non_ple_upload_bytes;
    sum->non_ple_residency_misses += sample->non_ple_residency_misses;
    ADD(timing_embedding_ms);
    ADD(timing_ple_async_window_ms);
    ADD(timing_qsa_ms);
    ADD(timing_gdn_ms);
    ADD(timing_moe_ms);
    ADD(timing_gr_ms);
    ADD(timing_lm_head_ms);
    ADD(timing_norms_residual_glue_ms);
    ADD(timing_other_layer_ms);
    ADD(timing_unexplained_ms);
    sum->ple_file_read_ops += sample->ple_file_read_ops;
    if (sample->ple_file_read_min_bytes != 0 &&
        (sum->ple_file_read_min_bytes == 0 ||
         sample->ple_file_read_min_bytes < sum->ple_file_read_min_bytes))
        sum->ple_file_read_min_bytes = sample->ple_file_read_min_bytes;
    if (sample->ple_file_read_max_bytes > sum->ple_file_read_max_bytes)
        sum->ple_file_read_max_bytes = sample->ple_file_read_max_bytes;
    for (size_t i = 0; i < Q2_OWNER_COUNT; ++i) {
        sum->owners[i].calls += sample->owners[i].calls;
        sum->owners[i].callback_wall_ms += sample->owners[i].callback_wall_ms;
        sum->owners[i].kernel_ms += sample->owners[i].kernel_ms;
        sum->owners[i].host_wait_ms += sample->owners[i].host_wait_ms;
        sum->owners[i].host_overhead_ms += sample->owners[i].host_overhead_ms;
        sum->owners[i].bytes_read += sample->owners[i].bytes_read;
    }
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
    DIV(ple_request_build_ms);
    DIV(ple_history_ngram_ms);
    DIV(ple_index_lookup_ms);
    DIV(ple_file_io_ms);
    DIV(ple_decode_dequant_ms);
    DIV(ple_accumulation_ms);
    DIV(ple_async_submit_ms);
    DIV(ple_worker_exec_ms);
    DIV(ple_worker_cpu_ms);
    DIV(ple_result_publish_ms);
    DIV(ple_injection_ms);
    DIV(ple_wait_at_injection_ms);
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
    DIV(telemetry_callback_wall_ms);
    DIV(backend_host_work_ms);
    DIV(memcpy_ms);
    DIV(other_ms);
    DIV(timing_embedding_ms);
    DIV(timing_ple_async_window_ms);
    DIV(timing_qsa_ms);
    DIV(timing_gdn_ms);
    DIV(timing_moe_ms);
    DIV(timing_gr_ms);
    DIV(timing_lm_head_ms);
    DIV(timing_norms_residual_glue_ms);
    DIV(timing_other_layer_ms);
    DIV(timing_unexplained_ms);
#undef DIV
    sample->telemetry_callbacks = (uint64_t)(
        (double)sample->telemetry_callbacks / divisor);
    sample->kernel_launches = (uint64_t)(
        (double)sample->kernel_launches / divisor);
    sample->host_syncs = (uint64_t)(
        (double)sample->host_syncs / divisor);
    sample->real_cuda_sync_count = (uint64_t)(
        (double)sample->real_cuda_sync_count / divisor);
    sample->host_blocked_on_cuda_ms /= divisor;
    for (size_t i = 0; i < Q38_CUDA_SYNC_REASON_COUNT; ++i) {
        sample->sync_reason_count[i] = (uint64_t)(
            (double)sample->sync_reason_count[i] / divisor);
        sample->sync_reason_ms[i] /= divisor;
    }
    sample->h2d_bytes = (uint64_t)((double)sample->h2d_bytes / divisor);
    sample->d2h_bytes = (uint64_t)((double)sample->d2h_bytes / divisor);
    sample->d2d_bytes = (uint64_t)((double)sample->d2d_bytes / divisor);
    sample->non_ple_upload_bytes = (uint64_t)(
        (double)sample->non_ple_upload_bytes / divisor);
    sample->non_ple_residency_misses = (uint64_t)(
        (double)sample->non_ple_residency_misses / divisor);
    sample->ple_file_read_ops = (uint64_t)(
        (double)sample->ple_file_read_ops / divisor);
    sample->ple_file_read_min_bytes = (uint64_t)(
        (double)sample->ple_file_read_min_bytes / divisor);
    sample->ple_file_read_max_bytes = (uint64_t)(
        (double)sample->ple_file_read_max_bytes / divisor);
    for (size_t i = 0; i < Q2_OWNER_COUNT; ++i) {
        sample->owners[i].calls = (uint64_t)(
            (double)sample->owners[i].calls / divisor);
        sample->owners[i].callback_wall_ms /= divisor;
        sample->owners[i].kernel_ms /= divisor;
        sample->owners[i].host_wait_ms /= divisor;
        sample->owners[i].host_overhead_ms /= divisor;
        sample->owners[i].bytes_read = (uint64_t)(
            (double)sample->owners[i].bytes_read / divisor);
    }
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
            "usage: q2_canonical_bench --mode decode|prefill|reference0|quick "
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
                             !strcmp(options->mode, "prefill") ||
                             !strcmp(options->mode, "reference0") ||
                             !strcmp(options->mode, "quick")) &&
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
    for (size_t i = 0; i < Q2_OWNER_COUNT; ++i) {
        sample->owners[i].calls =
            after->owners[i].calls - before->owners[i].calls;
        sample->owners[i].callback_wall_ms =
            after->owners[i].callback_wall_ms -
            before->owners[i].callback_wall_ms;
        sample->owners[i].kernel_ms =
            after->owners[i].kernel_ms - before->owners[i].kernel_ms;
        sample->owners[i].host_wait_ms =
            after->owners[i].host_wait_ms - before->owners[i].host_wait_ms;
        sample->owners[i].host_overhead_ms =
            after->owners[i].host_overhead_ms -
            before->owners[i].host_overhead_ms;
        sample->owners[i].bytes_read =
            after->owners[i].bytes_read - before->owners[i].bytes_read;
    }
}

static void add_sync_delta(
    q2_sample *sample, const q38_forward_cuda_sync_stats *before,
    const q38_forward_cuda_sync_stats *after, double backend_overhead_ms) {
    if (!sample || !before || !after) return;
    sample->real_cuda_sync_count =
        after->real_cuda_sync_count - before->real_cuda_sync_count;
    sample->host_blocked_on_cuda_ms =
        after->host_blocked_on_cuda_ms - before->host_blocked_on_cuda_ms;
    sample->cuda_sync_wait_ms = sample->host_blocked_on_cuda_ms;
    sample->telemetry_callback_wall_ms =
        after->telemetry_callback_wall_ms -
        before->telemetry_callback_wall_ms;
    sample->backend_host_work_ms = backend_overhead_ms >
        sample->host_blocked_on_cuda_ms +
        sample->telemetry_callback_wall_ms
        ? backend_overhead_ms - sample->host_blocked_on_cuda_ms -
          sample->telemetry_callback_wall_ms
        : 0.0;
    for (size_t i = 0; i < Q38_CUDA_SYNC_REASON_COUNT; ++i) {
        sample->sync_reason_count[i] =
            after->reason_count[i] - before->reason_count[i];
        sample->sync_reason_ms[i] =
            after->reason_ms[i] - before->reason_ms[i];
        sample->sync_reason_max_ms[i] = after->reason_max_ms[i];
    }
}

static void finalize_other(q2_sample *sample) {
    double accounted;
    double total;
    if (!sample) return;
    accounted = sample->qsa_ms + sample->moe_ms + sample->gdn_ms +
                sample->gr_ms + sample->norms_residual_glue_ms +
                sample->lm_head_ms;
    total = sample->forward_ms > 0.0 ? sample->forward_ms : sample->wall_ms;
    sample->other_ms = total > accounted ? total - accounted : 0.0;
}

static bool run_decode(q38_session *session, const q2_options *options,
                       const q38_token_batch *prompt, float *logits,
                       q2_telemetry *telemetry, q2_sample *summary,
                       q2_sample **samples_out, uint32_t **generated_out,
                       uint64_t *final_hash, bool *final_finite, char *error,
                       size_t error_len) {
    q38_forward_diagnostics diagnostics;
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
    diagnostics.timing_trace = timing_trace;
    memset(&prefill_capture, 0, sizeof(prefill_capture));
    diagnostics.trace_user = &prefill_capture;
    diagnostics.qsa_timing = &prefill_capture.qsa_timing;
    q38_forward_cuda_reset_sync_stats(session->runtime->cuda);
    q38_forward_cuda_get_sync_stats(session->runtime->cuda,
                                    &prefill_capture.sync_before);
    if (!q38_session_prefill_chunked(
            session, prompt->tokens, prompt->token_count,
            options->prefill_chunk, logits, VOCAB_SIZE, &next_token,
            &diagnostics, NULL, NULL, &step_index,
            error, error_len))
        goto fail;
    generated[0] = next_token;
    if (!q38_session_emit(session, logits, next_token, NULL, NULL,
                          &step_index, error, error_len))
        goto fail;
    q38_forward_cuda_set_telemetry_observer(
        session->runtime->cuda, telemetry_observer, telemetry);
    for (size_t index = 1; index < options->generated_count; ++index) {
        q2_capture capture;
        q38_decode_timing timing = {0};
        q38_ple_scheduler_stats ple = {0};
        const double started = now_ms();
        memset(&capture, 0, sizeof(capture));
        diagnostics.timing_trace = timing_trace;
        q38_forward_cuda_reset_sync_stats(session->runtime->cuda);
        q38_forward_cuda_get_sync_stats(session->runtime->cuda,
                                        &capture.sync_before);
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
        capture.sample.ple_request_build_ms = ple.request_build_ms;
        capture.sample.ple_history_ngram_ms = ple.history_ngram_ms;
        capture.sample.ple_index_lookup_ms = ple.index_lookup_ms;
        capture.sample.ple_file_io_ms = ple.file_io_ms;
        capture.sample.ple_decode_dequant_ms = ple.decode_dequant_ms;
        capture.sample.ple_accumulation_ms = ple.accumulation_ms;
        capture.sample.ple_async_submit_ms = ple.async_submit_ms;
        capture.sample.ple_worker_exec_ms = ple.elapsed_ms;
        capture.sample.ple_worker_cpu_ms = ple.worker_cpu_ms;
        capture.sample.ple_result_publish_ms = ple.result_publish_ms;
        capture.sample.ple_injection_ms = ple.injection_ms;
        capture.sample.ple_wait_at_injection_ms = ple.wait_at_injection_ms;
        capture.sample.ple_file_read_ops = ple.file_read_ops;
        capture.sample.ple_file_read_min_bytes = ple.file_read_min_bytes;
        capture.sample.ple_file_read_max_bytes = ple.file_read_max_bytes;
        capture.sample.ple_request_id = ple.request_id;
        capture.sample.ple_token_position = ple.token_position;
        capture.sample.ple_submit_position = ple.submit_position;
        capture.sample.ple_injection_position = ple.injection_position;
        capture.sample.ple_t0_ms = ple.t0_token_forward_begin_ms;
        capture.sample.ple_t1_ms = ple.t1_ple_request_submit_ms;
        capture.sample.ple_t2_ms = ple.t2_layer0_begin_ms;
        capture.sample.ple_t3_ms = ple.t3_layer0_end_ms;
        capture.sample.ple_t4_ms = ple.t4_layer1_begin_ms;
        capture.sample.ple_t5_ms = ple.t5_layer1_end_ms;
        capture.sample.ple_t6_ms = ple.t6_ple_injection_arrival_ms;
        capture.sample.ple_t7_ms = ple.t7_ple_wait_begin_ms;
        capture.sample.ple_t8_ms = ple.t8_ple_wait_end_ms;
        capture.sample.ple_t9_ms = ple.t9_ple_injection_begin_ms;
        capture.sample.ple_t10_ms = ple.t10_ple_injection_end_ms;
        capture.sample.ple_t11_ms = ple.t11_token_forward_end_ms;
        apply_qsa_timing(&capture.sample, &capture.qsa_timing);
        finalize_timing_tree(&capture.sample, &capture);
        add_telemetry_delta(&capture.sample, &telemetry_before, telemetry);
        q38_forward_cuda_sync_stats sync_after;
        q38_forward_cuda_get_sync_stats(session->runtime->cuda,
                                        &sync_after);
        add_sync_delta(&capture.sample, &capture.sync_before, &sync_after,
                       capture.sample.cuda_dispatch_ms);
        finalize_other(&capture.sample);
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
    diagnostics.timing_trace = timing_trace;
    diagnostics.trace_user = &capture;
    diagnostics.qsa_timing = &qsa_timing;
    q38_forward_cuda_reset_sync_stats(session->runtime->cuda);
    q38_forward_cuda_get_sync_stats(session->runtime->cuda,
                                    &capture.sync_before);
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
    finalize_timing_tree(&capture.sample, &capture);
    sample->timing_embedding_ms = capture.sample.timing_embedding_ms;
    sample->timing_ple_async_window_ms =
        capture.sample.timing_ple_async_window_ms;
    sample->timing_qsa_ms = capture.sample.timing_qsa_ms;
    sample->timing_gdn_ms = capture.sample.timing_gdn_ms;
    sample->timing_moe_ms = capture.sample.timing_moe_ms;
    sample->timing_gr_ms = capture.sample.timing_gr_ms;
    sample->timing_lm_head_ms = capture.sample.timing_lm_head_ms;
    sample->timing_norms_residual_glue_ms =
        capture.sample.timing_norms_residual_glue_ms;
    sample->timing_other_layer_ms = capture.sample.timing_other_layer_ms;
    sample->timing_unexplained_ms = capture.sample.timing_unexplained_ms;
    add_telemetry_delta(sample, &before, telemetry);
    q38_forward_cuda_sync_stats sync_after;
    q38_forward_cuda_get_sync_stats(session->runtime->cuda, &sync_after);
    add_sync_delta(sample, &capture.sync_before, &sync_after,
                   sample->cuda_dispatch_ms);
    finalize_other(sample);
    *logits_hash = hash_bytes(logits, VOCAB_SIZE * sizeof(*logits));
    *finite = true;
    for (size_t i = 0; i < VOCAB_SIZE; ++i)
        if (!isfinite(logits[i])) *finite = false;
    return true;
}

static void print_sample(const q2_sample *sample);

static void print_decode_run(const q2_decode_run *run,
                             const q2_options *options) {
    if (!run || !options) return;
    printf("\"summary\":");
    print_sample(&run->summary);
    printf(",\"generated_ids\":");
    print_ids(run->generated, options->generated_count);
    printf(",\"correctness\":{\"final_logits_hash\":\"%016" PRIx64
           "\",\"argmax\":%u,\"nan_inf\":%s},\"samples\":[",
           run->final_hash, run->generated[options->generated_count - 1],
           run->finite ? "false" : "true");
    for (size_t i = options->measure_first; i <= options->measure_last; ++i) {
        if (i != options->measure_first) putchar(',');
        print_sample(&run->samples[i]);
    }
    printf("]}");
}

static void print_prefill_reference(const q2_prefill_reference *reference,
                                    const q2_options *options) {
    if (!reference || !options) return;
    printf("{\"token_count\":%zu,\"tokens\":", reference->token_count);
    print_ids(reference->tokens, reference->token_count);
    printf(",\"measured_runs\":[");
    for (size_t i = 0; i < REFERENCE_RUNS; ++i) {
        if (i) putchar(',');
        printf("{\"run\":%zu,\"next_token\":%u,"
               "\"logits_hash\":\"%016" PRIx64
               "\",\"nan_inf\":%s,\"sample\":",
               i + 1, reference->next_tokens[i],
               reference->logits_hashes[i],
               reference->finite[i] ? "false" : "true");
        print_sample(&reference->measured[i]);
        putchar('}');
    }
    printf("]}");
}

static bool reference_correct_decode(const q2_decode_run *runs,
                                     size_t count,
                                     const q2_decode_run *warmup,
                                     const q2_options *options) {
    if (!runs || !count || !warmup || !options || !warmup->generated)
        return false;
    for (size_t run = 0; run < count; ++run) {
        if (!runs[run].finite || runs[run].final_hash != runs[0].final_hash)
            return false;
        for (size_t i = 0; i < options->generated_count; ++i)
            if (runs[run].generated[i] != runs[0].generated[i])
                return false;
    }
    if (!warmup->finite || warmup->final_hash != runs[0].final_hash)
        return false;
    for (size_t i = 0; i < options->generated_count; ++i)
        if (warmup->generated[i] != runs[0].generated[i]) return false;
    return true;
}

static bool reference_correct_prefill(
    const q2_prefill_reference *reference, size_t count, uint32_t warm_next,
    uint64_t warm_hash, bool warm_finite) {
    if (!reference || !count || !warm_finite) return false;
    for (size_t i = 0; i < count; ++i)
        if (!reference->finite[i] ||
            reference->next_tokens[i] != reference->next_tokens[0] ||
            reference->logits_hashes[i] != reference->logits_hashes[0])
            return false;
    return warm_finite && warm_next == reference->next_tokens[0] &&
           warm_hash == reference->logits_hashes[0];
}

static int compare_double_values(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static double decode_sample_percentile(const q2_decode_run *run,
                                       const q2_options *options,
                                       double percentile) {
    const size_t count = options->measure_last - options->measure_first + 1;
    double *values = calloc(count, sizeof(*values));
    if (!values) return 0.0;
    size_t index = 0;
    for (size_t i = options->measure_first;
         i <= options->measure_last; ++i)
        values[index++] = run->samples[i].wall_ms;
    qsort(values, count, sizeof(*values), compare_double_values);
    size_t rank = (size_t)ceil(percentile * (double)count);
    if (!rank) rank = 1;
    if (rank > count) rank = count;
    const double result = values[rank - 1];
    free(values);
    return result;
}

static double quick_percentile(const q2_decode_run *runs, size_t run_count,
                               const q2_options *options, double percentile) {
    const size_t per_run =
        options->measure_last - options->measure_first + 1;
    const size_t count = run_count * per_run;
    double *values = calloc(count, sizeof(*values));
    if (!values) return 0.0;
    size_t index = 0;
    for (size_t run = 0; run < run_count; ++run)
        for (size_t i = options->measure_first;
             i <= options->measure_last; ++i)
            values[index++] = runs[run].samples[i].wall_ms;
    qsort(values, count, sizeof(*values), compare_double_values);
    size_t rank = (size_t)ceil(percentile * (double)count);
    if (!rank) rank = 1;
    if (rank > count) rank = count;
    const double result = values[rank - 1];
    free(values);
    return result;
}

static void free_decode_run(q2_decode_run *run);

static void print_sync_stats_json(const q38_forward_cuda_sync_stats *stats) {
    printf("{\"real_cuda_sync_count\":%" PRIu64
           ",\"host_blocked_on_cuda_ms\":%.6f,"
           "\"telemetry_callback_wall_ms\":%.6f,\"reasons\":[",
           stats ? stats->real_cuda_sync_count : 0,
           stats ? stats->host_blocked_on_cuda_ms : 0.0,
           stats ? stats->telemetry_callback_wall_ms : 0.0);
    for (size_t i = 0; i < Q38_CUDA_SYNC_REASON_COUNT; ++i) {
        if (i) putchar(',');
        printf("{\"reason\":");
        json_string(q38_forward_cuda_sync_reason_name(
            (q38_forward_cuda_sync_reason)i));
        printf(",\"count\":%" PRIu64 ",\"total_ms\":%.6f,"
               "\"mean_us\":%.6f,\"max_us\":%.6f}",
               stats ? stats->reason_count[i] : 0,
               stats ? stats->reason_ms[i] : 0.0,
               stats && stats->reason_count[i]
                   ? stats->reason_ms[i] * 1000.0 /
                         (double)stats->reason_count[i]
                   : 0.0,
               stats ? stats->reason_max_ms[i] * 1000.0 : 0.0);
    }
    printf("]}");
}

static bool run_quick(
    q38_session *session, const q2_options *options,
    const q38_token_batch *seed, float *logits,
    const q38_forward_cuda_sync_stats *init_sync,
    char *error, size_t error_len) {
    enum { QUICK_RUNS = Q38_QUICK_RUNS };
    q2_telemetry telemetry = {0};
    q2_memory memory = {0};
    q2_decode_run warmup = {0};
    q2_decode_run runs[QUICK_RUNS] = {{0}};
    q2_sample aggregate = {0};
    q38_forward_cuda_residency_stats residency = {0};
    q2_hardware hardware;
    bool generated_ids_identical = false;
    bool final_hash_identical = false;
    bool all_finite = false;
    bool green = false;
    query_hardware(&hardware);
    observe_memory(&memory);
    q38_session_reset(session);
    if (!run_decode(session, options, seed, logits, &telemetry,
                    &warmup.summary, &warmup.samples, &warmup.generated,
                    &warmup.final_hash, &warmup.finite, error, error_len))
        goto cleanup;
    for (size_t run = 0; run < QUICK_RUNS; ++run) {
        q38_session_reset(session);
        if (!run_decode(session, options, seed, logits, &telemetry,
                        &runs[run].summary, &runs[run].samples,
                        &runs[run].generated, &runs[run].final_hash,
                        &runs[run].finite, error, error_len))
            goto cleanup;
        add_sample(&aggregate, &runs[run].summary);
    }
    divide_sample(&aggregate, QUICK_RUNS);
    double min_wall = runs[0].samples[options->measure_first].wall_ms;
    double max_wall = min_wall;
    for (size_t run = 0; run < QUICK_RUNS; ++run)
        for (size_t i = options->measure_first;
             i <= options->measure_last; ++i) {
            min_wall = fmin(min_wall, runs[run].samples[i].wall_ms);
            max_wall = fmax(max_wall, runs[run].samples[i].wall_ms);
        }
    if (QUICK_RUNS == 1) {
        generated_ids_identical = true;
        final_hash_identical = true;
        all_finite = warmup.finite && runs[0].finite;
    } else {
        generated_ids_identical = reference_correct_decode(
            runs, QUICK_RUNS, &warmup, options);
        final_hash_identical = runs[0].final_hash == runs[1].final_hash;
        all_finite = warmup.finite && runs[0].finite && runs[1].finite;
    }
    q38_forward_cuda_get_residency_stats(session->runtime->cuda, &residency);
    const bool fallback_zero = residency.q2_gate_up_fallback_calls == 0;
    const bool upload_zero = telemetry.non_ple_upload_bytes == 0;
    const bool miss_zero = telemetry.non_ple_residency_misses == 0;
    const bool ple_stall_zero = aggregate.ple_critical_stall_ms == 0.0;
    green = generated_ids_identical && final_hash_identical && all_finite &&
            fallback_zero && upload_zero && miss_zero && ple_stall_zero &&
            residency.all_non_ple_resident;
    printf("{\"format\":\"q2-forward-exclusive-attribution-raw-v1\","
           "\"PERF_SCHEMA\":\"PERF_SCHEMA_V2\","
           "\"REF_ID\":\"Q2_FORWARD_EXCLUSIVE_ATTRIBUTION_V1\","
           "\"baseline_reference\":\"S4A_CUDA_WAIT_ATTRIBUTION_V1\","
           "\"benchmark_kind\":\"quick_forward_exclusive_attribution\","
           "\"prompt\":");
    json_string(options->prompt);
    printf(",\"prompt_ids\":");
    print_ids(seed->tokens, seed->token_count);
    printf(",\"context_size\":%zu,\"generated_count\":%zu,"
           "\"measure_first\":%zu,\"measure_last\":%zu,"
           "\"warmup_runs\":1,\"measured_runs\":%d,"
           "\"init_sync\":",
           options->context_size, options->generated_count,
           options->measure_first, options->measure_last, QUICK_RUNS);
    print_sync_stats_json(init_sync);
    printf(",\"runs\":[");
    for (size_t run = 0; run < QUICK_RUNS; ++run) {
        double run_min = runs[run].samples[options->measure_first].wall_ms;
        double run_max = run_min;
        if (run) putchar(',');
        for (size_t i = options->measure_first + 1;
             i <= options->measure_last; ++i) {
            run_min = fmin(run_min, runs[run].samples[i].wall_ms);
            run_max = fmax(run_max, runs[run].samples[i].wall_ms);
        }
        printf("{\"run\":%zu,\"p95_wall_ms\":%.6f,"
               "\"min_wall_ms\":%.6f,\"max_wall_ms\":%.6f,",
               run + 1,
               decode_sample_percentile(&runs[run], options, 0.95),
               run_min, run_max);
        print_decode_run(&runs[run], options);
    }
    printf("],\"aggregate\":");
    print_sample(&aggregate);
    printf(",\"throughput_tok_s\":%.9f,\"p95_wall_ms\":%.6f,"
           "\"min_wall_ms\":%.6f,\"max_wall_ms\":%.6f,"
           "\"correctness\":{\"green\":%s,"
           "\"generated_ids_identical\":%s,\"final_logits_hash_identical\":%s,"
           "\"argmax_final_identical\":%s,\"nan_inf\":%s,"
           "\"fallback\":%s,\"non_ple_upload_bytes\":%" PRIu64
           ",\"non_ple_residency_misses\":%" PRIu64
           ",\"ple_critical_stall_ms\":%.6f},"
           "\"accounting\":{\"wall_ms\":%.6f,"
           "\"embedding_ms\":%.6f,\"ple_async_window_ms\":%.6f,"
           "\"QSA_ms\":%.6f,"
           "\"GDN_ms\":%.6f,\"MoE_ms\":%.6f,\"GR_ms\":%.6f,"
           "\"LM_head_ms\":%.6f,\"norms_residual_glue_ms\":%.6f,"
           "\"other_layer_ms\":%.6f,\"host_blocked_on_cuda_ms\":%.6f,"
           "\"diagnostic_overhead_ms\":%.6f,\"unexplained_ms\":%.6f,"
           "\"unexplained_fraction\":%.9f},"
           "\"prefill\":{\"status\":\"not_run\"},"
           "\"residency\":{\"all_non_ple_resident\":%s,"
           "\"persistent_resident_bytes\":%zu,"
           "\"persistent_resident_tensors\":%" PRIu64
           ",\"persistent_ple_entries\":%" PRIu64
           ",\"non_ple_upload_bytes\":%" PRIu64
           ",\"non_ple_residency_misses\":%" PRIu64
           ",\"q2_gate_up_fallback_calls\":%" PRIu64 "},",
           aggregate.wall_ms > 0.0 ? 1000.0 / aggregate.wall_ms : 0.0,
           quick_percentile(runs, QUICK_RUNS, options, 0.95),
           min_wall, max_wall,
           green ? "true" : "false",
           generated_ids_identical ? "true" : "false",
           final_hash_identical ? "true" : "false",
           generated_ids_identical ? "true" : "false",
           all_finite ? "false" : "true", fallback_zero ? "false" : "true",
           telemetry.non_ple_upload_bytes,
           telemetry.non_ple_residency_misses,
           aggregate.ple_critical_stall_ms,
           aggregate.forward_ms > 0.0 ? aggregate.forward_ms : aggregate.wall_ms,
           aggregate.timing_embedding_ms,
           aggregate.timing_ple_async_window_ms,
           aggregate.timing_qsa_ms, aggregate.timing_gdn_ms,
           aggregate.timing_moe_ms, aggregate.timing_gr_ms,
           aggregate.timing_lm_head_ms,
           aggregate.timing_norms_residual_glue_ms,
           aggregate.timing_other_layer_ms,
           aggregate.host_blocked_on_cuda_ms,
           aggregate.telemetry_callback_wall_ms,
           aggregate.timing_unexplained_ms,
           (aggregate.forward_ms > 0.0 ? aggregate.forward_ms
                                        : aggregate.wall_ms) > 0.0
               ? aggregate.timing_unexplained_ms /
                     (aggregate.forward_ms > 0.0 ? aggregate.forward_ms
                                                  : aggregate.wall_ms)
               : 0.0,
           residency.all_non_ple_resident ? "true" : "false",
           residency.persistent_resident_bytes,
           residency.persistent_resident_tensors,
           residency.persistent_ple_entries,
           telemetry.non_ple_upload_bytes,
           telemetry.non_ple_residency_misses,
           residency.q2_gate_up_fallback_calls);
    print_hardware(&hardware);
    putchar(',');
    print_memory(&memory);
    printf("}\n");
cleanup:
    free_decode_run(&warmup);
    for (size_t run = 0; run < QUICK_RUNS; ++run)
        free_decode_run(&runs[run]);
    return green;
}

static void print_sync_attribution(const q2_sample *sample) {
    printf("\"sync_attribution\":{\"real_cuda_sync_count\":%" PRIu64
           ",\"host_blocked_on_cuda_ms\":%.6f,"
           "\"telemetry_callback_wall_ms\":%.6f,"
           "\"backend_host_work_ms\":%.6f,\"reasons\":[",
           sample->real_cuda_sync_count, sample->host_blocked_on_cuda_ms,
           sample->telemetry_callback_wall_ms,
           sample->backend_host_work_ms);
    for (size_t i = 0; i < Q38_CUDA_SYNC_REASON_COUNT; ++i) {
        if (i) putchar(',');
        printf("{\"reason\":");
        json_string(q38_forward_cuda_sync_reason_name(
            (q38_forward_cuda_sync_reason)i));
        printf(",\"count\":%" PRIu64 ",\"total_ms\":%.6f,"
               "\"mean_us\":%.6f,\"max_us\":%.6f}",
               sample->sync_reason_count[i],
               sample->sync_reason_ms[i],
               sample->sync_reason_count[i]
                   ? sample->sync_reason_ms[i] * 1000.0 /
                         (double)sample->sync_reason_count[i]
                   : 0.0,
               sample->sync_reason_max_ms[i] * 1000.0);
    }
    printf("]}");
}

static const char *owner_name(q2_owner owner) {
    static const char *const names[Q2_OWNER_COUNT] = {
        "QSA", "GDN", "MoE", "GR", "PLE", "LM_HEAD", "NORM",
        "ROUTER", "OTHER_LAYER", "UNKNOWN"
    };
    return owner < Q2_OWNER_COUNT ? names[owner] : "UNKNOWN";
}

static void print_owner_attribution(const q2_sample *sample) {
    printf("\"matrix_owner_attribution\":[");
    for (size_t i = 0; i < Q2_OWNER_COUNT; ++i) {
        if (i) putchar(',');
        const q2_owner_stat *owner = &sample->owners[i];
        printf("{\"owner\":");
        json_string(owner_name((q2_owner)i));
        printf(",\"calls\":%" PRIu64
               ",\"callback_cpu_wall_ms\":%.6f,\"kernel_enqueue_ms\":%.6f,"
               "\"host_pre_post_ms\":%.6f,\"host_wait_ms\":%.6f,"
               "\"bytes_read\":%" PRIu64 "}",
               owner->calls, owner->callback_wall_ms, owner->kernel_ms,
               owner->host_overhead_ms, owner->host_wait_ms,
               owner->bytes_read);
    }
    printf("]");
}

static void print_sample(const q2_sample *sample) {
    printf("{\"wall_ms\":%.6f,\"forward_core_ms\":%.6f,"
           "\"argmax_ms\":%.6f,\"bookkeeping_ms\":%.6f,"
           "\"ple_critical_stall_ms\":%.6f,\"ple_elapsed_ms\":%.6f,"
           "\"ple_overlap_ms\":%.6f,\"ple_attribution\":{"
           "\"request_build_ms\":%.6f,\"history_ngram_ms\":%.6f,"
           "\"index_lookup_ms\":%.6f,\"file_io_ms\":%.6f,"
           "\"decode_dequant_ms\":%.6f,\"accumulation_ms\":%.6f,"
           "\"async_submit_ms\":%.6f,\"worker_elapsed_ms\":%.6f,"
           "\"worker_cpu_ms\":%.6f,\"result_publish_ms\":%.6f,"
           "\"ple_injection_main_thread_ms\":%.6f,"
           "\"wait_at_injection_ms\":%.6f,\"file_read_ops\":%" PRIu64
           ",\"file_read_min_bytes\":%" PRIu64
           ",\"file_read_max_bytes\":%" PRIu64
           "},\"timeline\":{\"request_id\":%" PRIu64
           ",\"token_position\":%" PRIu64
           ",\"submit_position\":%" PRIu64
           ",\"injection_position\":%" PRIu64
           ",\"T0\":%.6f,\"T1\":%.6f,\"T2\":%.6f,\"T3\":%.6f"
           ",\"T4\":%.6f,\"T5\":%.6f,\"T6\":%.6f,\"T7\":%.6f"
           ",\"T8\":%.6f,\"T9\":%.6f,\"T10\":%.6f,\"T11\":%.6f"
           ",\"true_ple_overlap_window_ms\":%.6f"
           ",\"wait_at_injection_ms\":%.6f"
           ",\"synchronous_injection_ms\":%.6f},\"categories\":{"
           "\"QSA\":{\"ms\":%.6f,\"qkv_ms\":%.6f,"
           "\"output_projection_ms\":%.6f,\"attention_ms\":%.6f,"
           "\"index_compress_ms\":%.6f,\"state_glue_ms\":%.6f},"
           "\"MoE\":{\"ms\":%.6f},\"GDN\":{\"ms\":%.6f},"
           "\"GR\":{\"ms\":%.6f},\"norms_residual_glue\":{\"ms\":%.6f},"
           "\"LM_head\":{\"ms\":%.6f},\"host_scalar\":{\"ms\":%.6f},"
           "\"cuda_dispatch\":{\"ms\":%.6f},"
           "\"cuda_sync_wait\":{\"ms\":%.6f},"
           "\"memcpy\":{\"ms\":%.6f},"
           "\"PLE_critical_stall\":{\"ms\":%.6f},"
           "\"other\":{\"ms\":%.6f}},"
           "\"traffic\":{\"kernel_launches\":null,\"host_syncs\":null,"
           "\"legacy_kernel_launches\":%" PRIu64
           ",\"legacy_host_syncs\":%" PRIu64
           ",\"telemetry_callbacks\":%" PRIu64 ",\"h2d_bytes\":%" PRIu64
           ",\"d2h_bytes\":%" PRIu64 ",\"d2d_bytes\":%" PRIu64 "}",
           sample->wall_ms, sample->forward_ms, sample->argmax_ms,
           sample->bookkeeping_ms, sample->ple_critical_stall_ms,
           sample->ple_elapsed_ms, sample->ple_overlap_ms,
           sample->ple_request_build_ms, sample->ple_history_ngram_ms,
           sample->ple_index_lookup_ms, sample->ple_file_io_ms,
           sample->ple_decode_dequant_ms, sample->ple_accumulation_ms,
           sample->ple_async_submit_ms, sample->ple_worker_exec_ms,
           sample->ple_worker_cpu_ms, sample->ple_result_publish_ms,
           sample->ple_injection_ms,
           sample->ple_wait_at_injection_ms, sample->ple_file_read_ops,
           sample->ple_file_read_min_bytes, sample->ple_file_read_max_bytes,
           sample->ple_request_id, sample->ple_token_position,
           sample->ple_submit_position, sample->ple_injection_position,
           sample->ple_t0_ms, sample->ple_t1_ms, sample->ple_t2_ms,
           sample->ple_t3_ms, sample->ple_t4_ms, sample->ple_t5_ms,
           sample->ple_t6_ms, sample->ple_t7_ms, sample->ple_t8_ms,
           sample->ple_t9_ms, sample->ple_t10_ms, sample->ple_t11_ms,
           sample->ple_t6_ms - sample->ple_t1_ms,
           sample->ple_t8_ms - sample->ple_t7_ms,
           sample->ple_t10_ms - sample->ple_t9_ms,
           sample->qsa_ms,
           sample->qsa_qkv_ms, sample->qsa_output_projection_ms,
           sample->qsa_attention_ms, sample->qsa_index_compress_ms,
           sample->qsa_state_glue_ms, sample->moe_ms, sample->gdn_ms,
           sample->gr_ms, sample->norms_residual_glue_ms,
           sample->lm_head_ms, sample->host_scalar_ms,
           sample->cuda_dispatch_ms, sample->cuda_sync_wait_ms,
           sample->memcpy_ms, sample->ple_critical_stall_ms,
           sample->other_ms,
           sample->kernel_launches, sample->host_syncs,
           sample->telemetry_callbacks, sample->h2d_bytes,
           sample->d2h_bytes, sample->d2d_bytes);
    putchar(',');
    printf("\"exclusive_forward_timing\":{\"embedding_ms\":%.6f,"
    "\"ple_async_window_ms\":%.6f,\"QSA_ms\":%.6f,\"GDN_ms\":%.6f,"
           "\"MoE_ms\":%.6f,\"GR_ms\":%.6f,\"LM_head_ms\":%.6f,"
           "\"norms_residual_glue_ms\":%.6f,\"other_layer_ms\":%.6f,"
           "\"unexplained_ms\":%.6f},",
           sample->timing_embedding_ms, sample->timing_ple_async_window_ms,
           sample->timing_qsa_ms, sample->timing_gdn_ms,
           sample->timing_moe_ms, sample->timing_gr_ms,
           sample->timing_lm_head_ms,
           sample->timing_norms_residual_glue_ms,
           sample->timing_other_layer_ms, sample->timing_unexplained_ms);
    print_owner_attribution(sample);
    putchar(',');
    print_sync_attribution(sample);
    putchar('}');
}

static void free_decode_run(q2_decode_run *run) {
    if (!run) return;
    free(run->samples);
    free(run->generated);
    memset(run, 0, sizeof(*run));
}

static bool run_reference0(q38_session *session, const q2_options *options,
                           const q38_token_batch *seed, float *logits,
                           q38_forward_cuda_residency_stats *residency,
                           char *error, size_t error_len) {
    q2_telemetry telemetry = {0};
    q2_hardware hardware;
    q2_memory memory = {0};
    q2_decode_run warmup = {0};
    q2_decode_run decode_runs[REFERENCE_RUNS] = {0};
    q2_prefill_reference prefill[MAX_PREFILL_CASES];
    q2_sample warm_sample = {0};
    uint32_t warm_next = 0;
    uint64_t warm_hash = 0;
    bool warm_finite = false;
    bool decode_ok = false;
    bool prefill_ok = true;
    bool all_finite = true;
    memset(prefill, 0, sizeof(prefill));
    query_hardware(&hardware);
    observe_memory(&memory);

    q38_session_reset(session);
    if (!run_decode(session, options, seed, logits, &telemetry,
                    &warmup.summary, &warmup.samples, &warmup.generated,
                    &warmup.final_hash, &warmup.finite, error, error_len))
        goto cleanup;
    observe_memory(&memory);
    for (size_t run = 0; run < REFERENCE_RUNS; ++run) {
        q38_session_reset(session);
        if (!run_decode(session, options, seed, logits, &telemetry,
                        &decode_runs[run].summary,
                        &decode_runs[run].samples,
                        &decode_runs[run].generated,
                        &decode_runs[run].final_hash,
                        &decode_runs[run].finite, error, error_len))
            goto cleanup;
        observe_memory(&memory);
    }
    decode_ok = reference_correct_decode(
        decode_runs, REFERENCE_RUNS, &warmup, options);
    all_finite = warmup.finite;
    for (size_t run = 0; run < REFERENCE_RUNS; ++run)
        all_finite = all_finite && decode_runs[run].finite;

    for (size_t c = 0; c < options->prefill_count; ++c) {
        q2_prefill_reference *reference = &prefill[c];
        reference->token_count = options->prefill_sizes[c];
        if (!make_repeated_tokens(seed, reference->token_count,
                                  &reference->tokens))
            goto cleanup;
        q38_session_reset(session);
        if (!run_prefill_case(
                session, options, reference->tokens, reference->token_count,
                logits, &telemetry, &warm_sample, &warm_next, &warm_hash,
                &warm_finite, error, error_len))
            goto cleanup;
        observe_memory(&memory);
        for (size_t run = 0; run < REFERENCE_RUNS; ++run) {
            q38_session_reset(session);
            if (!run_prefill_case(
                    session, options, reference->tokens,
                    reference->token_count, logits, &telemetry,
                    &reference->measured[run],
                    &reference->next_tokens[run],
                    &reference->logits_hashes[run],
                    &reference->finite[run], error, error_len))
                goto cleanup;
            observe_memory(&memory);
        }
        prefill_ok = prefill_ok &&
            reference_correct_prefill(
                reference, REFERENCE_RUNS, warm_next, warm_hash,
                warm_finite);
        all_finite = all_finite && warm_finite;
        for (size_t run = 0; run < REFERENCE_RUNS; ++run)
            all_finite = all_finite && reference->finite[run];
    }
    q38_forward_cuda_get_residency_stats(session->runtime->cuda, residency);

    printf("{\"format\":\"q2-canonical-reference0-raw-v1\","
           "\"reference_id\":\"Q2_DECODE_REFERENCE_0\","
           "\"prompt\":");
    json_string(options->prompt);
    printf(",\"prompt_ids\":");
    print_ids(seed->tokens, seed->token_count);
    printf(",\"context_size\":%zu,\"generated_count\":%zu,"
           "\"measure_first\":%zu,\"measure_last\":%zu,"
           "\"prefill_chunk\":%zu,\"warmup_runs\":%d,"
           "\"measured_runs\":%d,",
           options->context_size, options->generated_count,
           options->measure_first, options->measure_last,
           options->prefill_chunk, REFERENCE_WARMUPS, REFERENCE_RUNS);
    print_hardware(&hardware);
    putchar(',');
    print_memory(&memory);
    printf(",\"decode\":{\"runs\":[");
    for (size_t run = 0; run < REFERENCE_RUNS; ++run) {
        if (run) putchar(',');
        printf("{\"run\":%zu,", run + 1);
        print_decode_run(&decode_runs[run], options);
    }
    printf("]},\"prefill\":{\"cases\":[");
    for (size_t c = 0; c < options->prefill_count; ++c) {
        if (c) putchar(',');
        print_prefill_reference(&prefill[c], options);
    }
    printf("]},\"correctness\":{\"decode\":%s,\"prefill\":%s,"
           "\"all_finite\":%s},"
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
           decode_ok ? "true" : "false", prefill_ok ? "true" : "false",
           all_finite ? "true" : "false",
           residency->all_non_ple_resident ? "true" : "false",
           residency->persistent_resident_bytes,
           residency->persistent_resident_tensors,
           residency->persistent_ple_entries,
           telemetry.non_ple_upload_bytes,
           telemetry.non_ple_residency_misses, telemetry.callbacks,
           telemetry.kernel_ms, telemetry.backend_overhead_ms,
           telemetry.upload_ms, telemetry.h2d_bytes, telemetry.d2h_bytes,
           telemetry.host_syncs);
    for (size_t run = 0; run < REFERENCE_RUNS; ++run)
        free_decode_run(&decode_runs[run]);
    free_decode_run(&warmup);
    for (size_t c = 0; c < options->prefill_count; ++c)
        free(prefill[c].tokens);
    return decode_ok && prefill_ok && all_finite;
cleanup:
    for (size_t run = 0; run < REFERENCE_RUNS; ++run)
        free_decode_run(&decode_runs[run]);
    free_decode_run(&warmup);
    for (size_t c = 0; c < options->prefill_count; ++c)
        free(prefill[c].tokens);
    return false;
}

static int run(const q2_options *options) {
    char error[256] = {0};
    q38_runtime runtime = {0};
    q38_session session = {0};
    q38_token_batch seed = {0};
    float *logits = NULL;
    q2_telemetry telemetry = {0};
    q38_forward_cuda_sync_stats init_sync = {0};
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
    q38_forward_cuda_reset_sync_stats(runtime.cuda);
    if (!q38_session_create(&session, &runtime, (uint32_t)options->context_size,
                            error, sizeof(error))) {
        fprintf(stderr, "canonical benchmark session failed: %s\n", error);
        goto cleanup;
    }
    q38_forward_cuda_get_sync_stats(runtime.cuda, &init_sync);
    session_ready = true;
    logits = calloc(VOCAB_SIZE, sizeof(*logits));
    if (!logits) goto cleanup;
    if (!strcmp(options->mode, "reference0")) {
        q38_forward_cuda_residency_stats residency = {0};
        if (!run_reference0(&session, options, &seed, logits, &residency,
                            error, sizeof(error))) {
            fprintf(stderr, "canonical reference 0 failed: %s\n", error);
            goto cleanup;
        }
    } else if (!strcmp(options->mode, "quick")) {
        if (!run_quick(&session, options, &seed, logits, &init_sync,
                       error, sizeof(error))) {
            fprintf(stderr, "canonical CUDA wait attribution failed: %s\n",
                    error);
            goto cleanup;
        }
    } else if (!strcmp(options->mode, "decode")) {
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
