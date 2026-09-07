#ifndef Q38_RESIDENCY_PLAN_H
#define Q38_RESIDENCY_PLAN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "q38_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*q38_residency_plan_is_ple_fn)(const q38_tensor *tensor,
                                              void *user);

typedef struct {
    uint32_t tensor_index;
    uint64_t file_offset;
    uint64_t bytes;
    uint32_t ownership_class;
} q38_residency_plan_entry;

typedef struct {
    uint64_t file_offset;
    uint64_t bytes;
    size_t first_entry;
    size_t entry_count;
} q38_residency_plan_span;

typedef struct {
    q38_residency_plan_entry *entries;
    size_t entry_count;
    q38_residency_plan_span *spans;
    size_t span_count;
    uint64_t resident_bytes;
    uint64_t excluded_ple_bytes;
    uint64_t excluded_ple_tensors;
} q38_residency_plan;

void q38_residency_plan_init(q38_residency_plan *plan);
void q38_residency_plan_destroy(q38_residency_plan *plan);

/*
 * Build a deterministic file-offset ordered plan. Spans may include small
 * non-tensor gaps, but never cross an excluded PLE range or a large gap.
 */
bool q38_residency_plan_build(
    const q38_gguf *model,
    q38_residency_plan_is_ple_fn is_ple,
    void *is_ple_user,
    uint64_t max_gap_bytes,
    uint64_t max_span_bytes,
    q38_residency_plan *plan,
    char *error,
    size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
