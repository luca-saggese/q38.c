#ifndef Q38_MEMORY_H
#define Q38_MEMORY_H

#include <stdint.h>

#include "q38.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q38_memory — memory telemetry (spec §8).
 *
 * Tracks RSS, MemAvailable, mmap bytes, CUDA free/total, internal
 * allocations, and peak counters. A snapshot is JSON-serializable with
 * deterministic keys.
 * ========================================================================= */

typedef struct {
    uint64_t internal_allocated_bytes; /* live internal allocations */
    uint64_t peak_internal_bytes;      /* high-water mark (monotonic) */
} q38_memory_tracker;

void q38_memory_tracker_init(q38_memory_tracker *t);

/* Record an internal allocation/release. Keeps peak monotonic. */
void q38_memory_track_alloc(q38_memory_tracker *t, uint64_t bytes);
void q38_memory_track_free(q38_memory_tracker *t, uint64_t bytes);

/* Fill a snapshot for the given phase. model_file_bytes/model_mapped_bytes
 * and cuda_* are supplied by the caller; RSS/MemAvailable/internal/peak are
 * read live. */
void q38_memory_capture(q38_memory_tracker *t,
                        const char *phase,
                        uint64_t model_file_bytes,
                        uint64_t model_mapped_bytes,
                        uint64_t cuda_allocated_bytes,
                        q38_memory_snapshot *out);

/* Serialize a snapshot as a single-line JSON object. Returns bytes written. */
int q38_memory_snapshot_json(const q38_memory_snapshot *s,
                             char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* Q38_MEMORY_H */
