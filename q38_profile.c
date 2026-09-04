#include "q38_profile.h"
#include "q38_forward_cuda.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static const char *const names[Q38_PROFILE_SUBSYSTEM_COUNT] = {
    "gdn", "qsa", "moe", "ple", "lm_head"
};

static q38_profile_subsystem classify(const char *name) {
    if (name) {
        if (strstr(name, "qsa") || strstr(name, "attention"))
            return Q38_PROFILE_QSA;
        if (strstr(name, "gdn")) return Q38_PROFILE_GDN;
        if (strstr(name, "moe") || strstr(name, "expert") ||
            strstr(name, "router"))
            return Q38_PROFILE_MOE;
        if (strstr(name, "ple")) return Q38_PROFILE_PLE;
        if (strstr(name, "lm_head"))
            return Q38_PROFILE_LM_HEAD;
    }

    return (q38_profile_subsystem)-1;
}

bool q38_profile_qsa_trace(uint32_t layer, const uint32_t *selected,
                           size_t count, void *user, char *error,
                           size_t error_len) {
    (void)layer; (void)selected; (void)error; (void)error_len;
    q38_profile *profile = (q38_profile *)user;
    q38_profile_record *record = q38_profile_get(profile, Q38_PROFILE_QSA);
    if (!record || (!selected && count)) return false;
    record->callback_count++;
    record->kernel_launches++;
    return true;
}

void q38_profile_init(q38_profile *profile) {
    if (!profile) return;
    memset(profile, 0, sizeof(*profile));
    for (size_t i = 0; i < Q38_PROFILE_SUBSYSTEM_COUNT; ++i)
        profile->subsystem[i].name = names[i];
}

void q38_profile_destroy(q38_profile *profile) {
    if (!profile) return;
    q38_profile_cuda_destroy(profile);
    memset(profile, 0, sizeof(*profile));
}

q38_profile_record *q38_profile_get(q38_profile *profile,
                                    q38_profile_subsystem subsystem) {
    if (!profile || subsystem < 0 ||
        subsystem >= Q38_PROFILE_SUBSYSTEM_COUNT)
        return NULL;
    return &profile->subsystem[subsystem];
}

void q38_profile_record_launch(q38_profile *profile,
                               q38_profile_subsystem subsystem) {
    q38_profile_record *record = q38_profile_get(profile, subsystem);
    if (record) record->kernel_launches++;
}

void q38_profile_record_sync(q38_profile *profile,
                             q38_profile_subsystem subsystem) {
    q38_profile_record *record = q38_profile_get(profile, subsystem);
    if (record) record->synchronizations++;
}

void q38_profile_record_allocation(q38_profile *profile,
                                   q38_profile_subsystem subsystem,
                                   size_t bytes) {
    q38_profile_record *record = q38_profile_get(profile, subsystem);
    if (!record) return;
    record->allocation_count++;
    record->allocation_bytes += bytes;
}

void q38_profile_record_runtime_allocation(q38_profile *profile, size_t bytes) {
    if (!profile) return;
    profile->allocation_count++;
    profile->allocation_bytes += bytes;
}

static void update_bandwidth_rates(q38_profile *profile) {
    if (!profile) return;
    profile->effective_weight_gbps =
        profile->bandwidth.kernel_ms > 0.0
            ? (double)profile->bandwidth.weight_bytes /
              (profile->bandwidth.kernel_ms * 1e6)
            : 0.0;
    const double tokens = profile->token_count ? (double)profile->token_count : 1.0;
    profile->weight_bytes_per_token =
        (double)profile->bandwidth.weight_bytes / tokens;
    profile->kernel_ms_per_token = profile->bandwidth.kernel_ms / tokens;
    profile->host_syncs_per_token =
        (double)profile->bandwidth.host_syncs / tokens;
    profile->d2h_bytes_per_token =
        (double)profile->bandwidth.d2h_bytes / tokens;
}

void q38_profile_set_token_count(q38_profile *profile, size_t token_count) {
    if (profile) {
        profile->token_count = token_count;
        update_bandwidth_rates(profile);
    }
}

void q38_profile_record_cuda_telemetry(
    q38_profile *profile, const q38_forward_cuda_telemetry *telemetry) {
    if (!profile || !telemetry || profile->telemetry_count >=
        sizeof(profile->telemetry) / sizeof(profile->telemetry[0]))
        return;
    q38_profile_telemetry_record *out =
        &profile->telemetry[profile->telemetry_count++];
    memset(out, 0, sizeof(*out));
    snprintf(out->subsystem, sizeof(out->subsystem), "%s",
             telemetry->subsystem ? telemetry->subsystem : "unknown");
    snprintf(out->logical_stage, sizeof(out->logical_stage), "%s",
             telemetry->logical_stage ? telemetry->logical_stage : "");
    snprintf(out->tensor_name, sizeof(out->tensor_name), "%s",
             telemetry->tensor_name ? telemetry->tensor_name : "");
    out->layer = telemetry->layer;
    out->qtype = telemetry->qtype;
    out->rows = telemetry->rows;
    out->cols = telemetry->cols;
    out->bytes = telemetry->bytes;
    out->resident_hit = telemetry->resident_hit;
    out->resident_miss = telemetry->resident_miss;
    out->upload_bytes = telemetry->upload_bytes;
    out->weight_bytes = telemetry->weight_bytes;
    out->activation_read_bytes = telemetry->activation_read_bytes;
    out->activation_write_bytes = telemetry->activation_write_bytes;
    out->d2h_bytes = telemetry->d2h_bytes;
    out->upload_ms = telemetry->upload_ms;
    out->kernel_ms = telemetry->kernel_ms;
    out->backend_overhead_ms = telemetry->backend_overhead_ms;
    out->allocation_count = telemetry->allocation_count;
    out->sync_count = telemetry->sync_count;
    out->host_syncs = telemetry->host_syncs;
    profile->bandwidth.weight_bytes += telemetry->weight_bytes;
    profile->bandwidth.activation_read_bytes += telemetry->activation_read_bytes;
    profile->bandwidth.activation_write_bytes += telemetry->activation_write_bytes;
    profile->bandwidth.d2h_bytes += telemetry->d2h_bytes;
    profile->bandwidth.host_syncs += telemetry->host_syncs;
    profile->bandwidth.kernel_ms += telemetry->kernel_ms;
    q38_profile_record *aggregate =
        q38_profile_get(profile, classify(telemetry->subsystem));
    if (aggregate) {
        aggregate->resident_hits += telemetry->resident_hit ? 1 : 0;
        aggregate->resident_misses += telemetry->resident_miss ? 1 : 0;
        aggregate->upload_bytes += telemetry->upload_bytes;
        aggregate->upload_ms += telemetry->upload_ms;
        aggregate->kernel_ms += telemetry->kernel_ms;
        aggregate->backend_overhead_ms += telemetry->backend_overhead_ms;
        aggregate->bandwidth.weight_bytes += telemetry->weight_bytes;
        aggregate->bandwidth.activation_read_bytes += telemetry->activation_read_bytes;
        aggregate->bandwidth.activation_write_bytes += telemetry->activation_write_bytes;
        aggregate->bandwidth.d2h_bytes += telemetry->d2h_bytes;
        aggregate->bandwidth.host_syncs += telemetry->host_syncs;
        aggregate->bandwidth.kernel_ms += telemetry->kernel_ms;
        aggregate->effective_weight_gbps =
            aggregate->bandwidth.kernel_ms > 0.0
                ? (double)aggregate->bandwidth.weight_bytes /
                  (aggregate->bandwidth.kernel_ms * 1e6)
                : 0.0;
    }
    update_bandwidth_rates(profile);
}

bool q38_profile_json(const q38_profile *profile, char *buffer,
                      size_t buffer_len) {
    if (!profile || !buffer || !buffer_len) return false;
    size_t used = 0;
#define APPEND(...) do { \
        int n__ = snprintf(buffer + used, buffer_len - used, __VA_ARGS__); \
        if (n__ < 0 || (size_t)n__ >= buffer_len - used) return false; \
        used += (size_t)n__; \
    } while (0)
    APPEND("{\"version\":2,\"token_count\":%zu,\"weight_bytes\":%llu,\"activation_read_bytes\":%llu,\"activation_write_bytes\":%llu,\"kernel_ms\":%.9g,\"effective_weight_gbps\":%.9g,\"effective_GBps\":%.9g,\"host_syncs\":%llu,\"d2h_bytes\":%llu,\"weight_bytes_per_token\":%.9g,\"kernel_ms_per_token\":%.9g,\"host_syncs_per_token\":%.9g,\"d2h_bytes_per_token\":%.9g,\"cuda_elapsed_ms\":%.9g,\"cuda_synchronizations\":%llu,\"allocation_count\":%llu,\"allocation_bytes\":%llu,\"subsystems\":[",
           profile->token_count,
           (unsigned long long)profile->bandwidth.weight_bytes,
           (unsigned long long)profile->bandwidth.activation_read_bytes,
           (unsigned long long)profile->bandwidth.activation_write_bytes,
           profile->bandwidth.kernel_ms, profile->effective_weight_gbps,
           profile->effective_weight_gbps,
           (unsigned long long)profile->bandwidth.host_syncs,
           (unsigned long long)profile->bandwidth.d2h_bytes,
           profile->weight_bytes_per_token, profile->kernel_ms_per_token,
           profile->host_syncs_per_token, profile->d2h_bytes_per_token,
           profile->cuda_elapsed_ms, (unsigned long long)profile->cuda_synchronizations,
           (unsigned long long)profile->allocation_count, (unsigned long long)profile->allocation_bytes);
    for (size_t i = 0; i < Q38_PROFILE_SUBSYSTEM_COUNT; ++i) {
        const q38_profile_record *r = &profile->subsystem[i];
        APPEND("%s{\"name\":\"%s\",\"callbacks\":%llu,\"kernel_launches\":%llu,\"synchronizations\":%llu,\"allocation_count\":%llu,\"allocation_bytes\":%llu,\"elapsed_ms\":%.9g,\"resident_hits\":%llu,\"resident_misses\":%llu,\"upload_bytes\":%llu,\"upload_ms\":%.9g,\"kernel_ms\":%.9g,\"backend_overhead_ms\":%.9g,\"weight_bytes\":%llu,\"activation_read_bytes\":%llu,\"activation_write_bytes\":%llu,\"d2h_bytes\":%llu,\"host_syncs\":%llu,\"effective_weight_gbps\":%.9g}",
               i ? "," : "", r->name, (unsigned long long)r->callback_count,
               (unsigned long long)r->kernel_launches, (unsigned long long)r->synchronizations,
               (unsigned long long)r->allocation_count, (unsigned long long)r->allocation_bytes,
               r->elapsed_ms, (unsigned long long)r->resident_hits,
               (unsigned long long)r->resident_misses, (unsigned long long)r->upload_bytes,
               r->upload_ms, r->kernel_ms, r->backend_overhead_ms,
               (unsigned long long)r->bandwidth.weight_bytes,
               (unsigned long long)r->bandwidth.activation_read_bytes,
               (unsigned long long)r->bandwidth.activation_write_bytes,
               (unsigned long long)r->bandwidth.d2h_bytes,
               (unsigned long long)r->bandwidth.host_syncs,
               r->effective_weight_gbps);
    }
    APPEND("],\"records\":[");
    for (size_t i = 0; i < profile->telemetry_count; ++i) {
        const q38_profile_telemetry_record *r = &profile->telemetry[i];
        APPEND("%s{\"subsystem\":\"%s\",\"layer\":%u,\"logical_stage\":\"%s\",\"tensor_name\":\"%s\",\"qtype\":%u,\"rows\":%zu,\"cols\":%zu,\"bytes\":%zu,\"resident_hit\":%s,\"resident_miss\":%s,\"upload_bytes\":%zu,\"weight_bytes\":%zu,\"activation_read_bytes\":%zu,\"activation_write_bytes\":%zu,\"d2h_bytes\":%zu,\"upload_ms\":%.9g,\"kernel_ms\":%.9g,\"backend_overhead_ms\":%.9g,\"allocation_count\":%llu,\"sync_count\":%llu,\"host_syncs\":%llu}",
               i ? "," : "", r->subsystem, r->layer, r->logical_stage,
               r->tensor_name, r->qtype, r->rows, r->cols, r->bytes,
               r->resident_hit ? "true" : "false", r->resident_miss ? "true" : "false",
               r->upload_bytes, r->weight_bytes, r->activation_read_bytes,
               r->activation_write_bytes, r->d2h_bytes, r->upload_ms,
               r->kernel_ms, r->backend_overhead_ms,
               (unsigned long long)r->allocation_count,
               (unsigned long long)r->sync_count,
               (unsigned long long)r->host_syncs);
    }
    APPEND("]}");
#undef APPEND
    return true;
}

bool q38_profile_boundary_trace(uint32_t layer, const char *boundary,
                                const float *values, size_t token_count,
                                size_t width, void *user, char *error,
                                size_t error_len) {
    (void)layer; (void)values; (void)token_count; (void)width;
    (void)error; (void)error_len;
    q38_profile *profile = (q38_profile *)user;
    q38_profile_record *record = q38_profile_get(profile, classify(boundary));
    if (!record) return true;
    record->callback_count++;
    return true;
}

bool q38_profile_stage_trace(const q38_forward_stage_usage *usage, void *user,
                             char *error, size_t error_len) {
    (void)error; (void)error_len;
    q38_profile *profile = (q38_profile *)user;
    q38_profile_record *record = q38_profile_get(profile,
                                                 classify(usage ? usage->name : NULL));
    if (!usage) return false;
    if (!record) return true;
    record->callback_count++;
    record->kernel_launches += usage->matrix_calls;
    record->elapsed_ms += usage->elapsed_ms;
    if (profile->telemetry_count <
        sizeof(profile->telemetry) / sizeof(profile->telemetry[0])) {
        q38_profile_telemetry_record *event =
            &profile->telemetry[profile->telemetry_count++];
        memset(event, 0, sizeof(*event));
        snprintf(event->subsystem, sizeof(event->subsystem), "%s",
                 record->name ? record->name : "unknown");
        snprintf(event->logical_stage, sizeof(event->logical_stage), "%s",
                 usage->logical_stage ? usage->logical_stage :
                 (usage->name ? usage->name : ""));
        event->layer = usage->layer;
        event->backend_overhead_ms = usage->elapsed_ms;
        event->allocation_count = record->allocation_count;
        event->sync_count = record->synchronizations;
    }
    return true;
}
