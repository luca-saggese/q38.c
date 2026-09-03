#include "q38_profile.h"

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
        if (strstr(name, "lm_head") || strstr(name, "output"))
            return Q38_PROFILE_LM_HEAD;
    }

    return Q38_PROFILE_LM_HEAD;
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

bool q38_profile_json(const q38_profile *profile, char *buffer,
                      size_t buffer_len) {
    if (!profile || !buffer || !buffer_len) return false;
    int written = snprintf(
        buffer, buffer_len,
        "{\"version\":1,\"cuda_elapsed_ms\":%.6f,\"cuda_synchronizations\":%" PRIu64 ",\"allocation_count\":%" PRIu64 ",\"allocation_bytes\":%" PRIu64 ",\"subsystems\":["
        "{\"name\":\"%s\",\"callbacks\":%" PRIu64 ",\"kernel_launches\":%" PRIu64 ",\"synchronizations\":%" PRIu64 ",\"allocation_count\":%" PRIu64 ",\"allocation_bytes\":%" PRIu64 ",\"elapsed_ms\":%.6f},"
        "{\"name\":\"%s\",\"callbacks\":%" PRIu64 ",\"kernel_launches\":%" PRIu64 ",\"synchronizations\":%" PRIu64 ",\"allocation_count\":%" PRIu64 ",\"allocation_bytes\":%" PRIu64 ",\"elapsed_ms\":%.6f},"
        "{\"name\":\"%s\",\"callbacks\":%" PRIu64 ",\"kernel_launches\":%" PRIu64 ",\"synchronizations\":%" PRIu64 ",\"allocation_count\":%" PRIu64 ",\"allocation_bytes\":%" PRIu64 ",\"elapsed_ms\":%.6f},"
        "{\"name\":\"%s\",\"callbacks\":%" PRIu64 ",\"kernel_launches\":%" PRIu64 ",\"synchronizations\":%" PRIu64 ",\"allocation_count\":%" PRIu64 ",\"allocation_bytes\":%" PRIu64 ",\"elapsed_ms\":%.6f},"
        "{\"name\":\"%s\",\"callbacks\":%" PRIu64 ",\"kernel_launches\":%" PRIu64 ",\"synchronizations\":%" PRIu64 ",\"allocation_count\":%" PRIu64 ",\"allocation_bytes\":%" PRIu64 ",\"elapsed_ms\":%.6f}]}",
        profile->cuda_elapsed_ms, profile->cuda_synchronizations,
        profile->allocation_count, profile->allocation_bytes,
        profile->subsystem[0].name, profile->subsystem[0].callback_count, profile->subsystem[0].kernel_launches, profile->subsystem[0].synchronizations, profile->subsystem[0].allocation_count, profile->subsystem[0].allocation_bytes, profile->subsystem[0].elapsed_ms,
        profile->subsystem[1].name, profile->subsystem[1].callback_count, profile->subsystem[1].kernel_launches, profile->subsystem[1].synchronizations, profile->subsystem[1].allocation_count, profile->subsystem[1].allocation_bytes, profile->subsystem[1].elapsed_ms,
        profile->subsystem[2].name, profile->subsystem[2].callback_count, profile->subsystem[2].kernel_launches, profile->subsystem[2].synchronizations, profile->subsystem[2].allocation_count, profile->subsystem[2].allocation_bytes, profile->subsystem[2].elapsed_ms,
        profile->subsystem[3].name, profile->subsystem[3].callback_count, profile->subsystem[3].kernel_launches, profile->subsystem[3].synchronizations, profile->subsystem[3].allocation_count, profile->subsystem[3].allocation_bytes, profile->subsystem[3].elapsed_ms,
        profile->subsystem[4].name, profile->subsystem[4].callback_count, profile->subsystem[4].kernel_launches, profile->subsystem[4].synchronizations, profile->subsystem[4].allocation_count, profile->subsystem[4].allocation_bytes, profile->subsystem[4].elapsed_ms);
    return written >= 0 && (size_t)written < buffer_len;
}

bool q38_profile_boundary_trace(uint32_t layer, const char *boundary,
                                const float *values, size_t token_count,
                                size_t width, void *user, char *error,
                                size_t error_len) {
    (void)layer; (void)values; (void)token_count; (void)width;
    (void)error; (void)error_len;
    q38_profile *profile = (q38_profile *)user;
    q38_profile_record *record = q38_profile_get(profile, classify(boundary));
    if (!record) return false;
    record->callback_count++;
    return true;
}

bool q38_profile_stage_trace(const q38_forward_stage_usage *usage, void *user,
                             char *error, size_t error_len) {
    (void)error; (void)error_len;
    q38_profile *profile = (q38_profile *)user;
    q38_profile_record *record = q38_profile_get(profile,
                                                 classify(usage ? usage->name : NULL));
    if (!record || !usage) return false;
    record->callback_count++;
    record->kernel_launches += usage->matrix_calls;
    record->elapsed_ms += usage->elapsed_ms;
    return true;
}
