#ifndef Q38_PROFILE_H
#define Q38_PROFILE_H

#include "q38_forward.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Q38_PROFILE_GDN = 0,
    Q38_PROFILE_QSA,
    Q38_PROFILE_MOE,
    Q38_PROFILE_PLE,
    Q38_PROFILE_LM_HEAD,
    Q38_PROFILE_SUBSYSTEM_COUNT
} q38_profile_subsystem;

typedef struct {
    const char *name;
    uint64_t callback_count;
    uint64_t kernel_launches;
    uint64_t synchronizations;
    uint64_t allocation_count;
    uint64_t allocation_bytes;
    double elapsed_ms;
} q38_profile_record;

typedef struct {
    q38_profile_record subsystem[Q38_PROFILE_SUBSYSTEM_COUNT];
    void *cuda_state;
    double cuda_elapsed_ms;
    uint64_t cuda_synchronizations;
    uint64_t allocation_count;
    uint64_t allocation_bytes;
} q38_profile;

void q38_profile_init(q38_profile *profile);
void q38_profile_destroy(q38_profile *profile);
q38_profile_record *q38_profile_get(q38_profile *profile,
                                    q38_profile_subsystem subsystem);
void q38_profile_record_launch(q38_profile *profile,
                               q38_profile_subsystem subsystem);
void q38_profile_record_sync(q38_profile *profile,
                             q38_profile_subsystem subsystem);
void q38_profile_record_allocation(q38_profile *profile,
                                  q38_profile_subsystem subsystem,
                                  size_t bytes);
void q38_profile_record_runtime_allocation(q38_profile *profile, size_t bytes);
bool q38_profile_json(const q38_profile *profile, char *buffer,
                      size_t buffer_len);

/* Classify existing observational callback names without changing execution. */
bool q38_profile_boundary_trace(uint32_t layer, const char *boundary,
                                const float *values, size_t token_count,
                                size_t width, void *user, char *error,
                                size_t error_len);
bool q38_profile_stage_trace(const q38_forward_stage_usage *usage, void *user,
                             char *error, size_t error_len);

/* CUDA-event timing is optional at runtime; all functions return CUDA status. */
int q38_profile_cuda_init(q38_profile *profile);
void q38_profile_cuda_destroy(q38_profile *profile);
int q38_profile_cuda_begin(q38_profile *profile, q38_profile_subsystem subsystem,
                           void *stream);
int q38_profile_cuda_end(q38_profile *profile, q38_profile_subsystem subsystem,
                         void *stream);

/* Optional NVTX guards. They are no-ops unless Q38_ENABLE_NVTX is defined. */
void q38_profile_nvtx_push(const char *name);
void q38_profile_nvtx_pop(void);

#ifdef __cplusplus
}
#endif

#endif /* Q38_PROFILE_H */
