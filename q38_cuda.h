#ifndef Q38_CUDA_H
#define Q38_CUDA_H

#include <stdbool.h>
#include <stdint.h>

#include "q38.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q38_cuda — narrow CUDA primitive surface for M0.
 *
 * Only platform interrogation and device lifecycle. No tensor allocators, no
 * attention/cache semantics, no MoE or host-registration primitives: those
 * belong to M1+ and are introduced behind an explicit, measured API.
 * ========================================================================= */

/* Interrogate the CUDA device. Returns 0 on success, non-zero on failure.
 * On failure the platform is considered unsupported and must be refused. */
int q38_cuda_probe(q38_platform_info *out);

/* Minimal device lifecycle. Returns 0 on success. */
int q38_cuda_init(void);
void q38_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* Q38_CUDA_H */
