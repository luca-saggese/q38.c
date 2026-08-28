#ifndef Q38_PLATFORM_H
#define Q38_PLATFORM_H

#include <stdbool.h>
#include <stdint.h>

#include "q38.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q38_platform — host platform guard and memory telemetry helpers.
 *
 * M0 refuses any platform that is not GB10 / SM 12.1. No fallback, no
 * silent degradation.
 * ========================================================================= */

/* Host memory figures from /proc/meminfo (Linux). Returns 0 on success. */
int q38_platform_host_memory(uint64_t *total_bytes, uint64_t *available_bytes);

/* Current process RSS in bytes. Returns 0 on success. */
int q38_platform_rss_bytes(uint64_t *rss_bytes);

/* Combine CUDA + host into a full platform probe. Returns 0 on success and
 * fills `out`. On unsupported platform returns non-zero and a reason. */
int q38_platform_probe(q38_platform_info *out, char *reason, size_t reason_len);

#ifdef __cplusplus
}
#endif

#endif /* Q38_PLATFORM_H */
