#ifndef Q38_H
#define Q38_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * q38 — Qwen3.8-Flash-Next / DGX Spark prototype runtime skeleton.
 *
 * The CLI exposes platform probe, GGUF inspection, memory planning, and a
 * minimal CUDA generation smoke wrapper. The only supported target is GB10 /
 * SM 12.1 CUDA on Linux aarch64. Anything else is refused explicitly, never
 * silently degraded.
 * ========================================================================= */

/* --- Command mode ------------------------------------------------ */
typedef enum {
    Q38_MODE_NONE = 0,
    Q38_MODE_PLATFORM,     /* --platform           */
    Q38_MODE_INSPECT,      /* --inspect model.gguf */
    Q38_MODE_LIST_TENSORS, /* --list-tensors model.gguf */
    Q38_MODE_MEMORY_PLAN,  /* --memory-plan model.gguf */
    Q38_MODE_GENERATE,     /* --generate model.gguf */
} q38_mode;

/* --- Narrowed engine options -------------------------------------- */
typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *prompt;
    size_t max_tokens;
    int context_hint;     /* KV context size hint (bytes/none, informational) */
    int prefill_hint;     /* prefill chunk hint (informational)               */
    bool inspect;         /* --inspect */
    bool list_tensors;    /* --list-tensors */
    bool memory_plan;     /* --memory-plan */
    bool platform;        /* --platform */
    bool json;            /* --json */
    bool verbose;         /* --verbose */
} q38_options;

/* --- Platform probe (spec §7) ------------------------------------- */
typedef struct {
    int cuda_device_count;
    int cuda_device;
    int cc_major;
    int cc_minor;
    uint64_t cuda_total_bytes;
    uint64_t cuda_free_bytes;
    uint64_t mem_total_bytes;
    uint64_t mem_available_bytes;
    char device_name[128];
    char driver_version[32];
    char runtime_version[32];
} q38_platform_info;

/* --- Memory telemetry snapshot (spec §8) -------------------------- */
typedef struct {
    const char *phase;             /* e.g. "gguf_mapped" */
    uint64_t rss_bytes;
    uint64_t mem_available_bytes;
    uint64_t cuda_free_bytes;
    uint64_t cuda_total_bytes;
    uint64_t model_file_bytes;
    uint64_t model_mapped_bytes;
    uint64_t cuda_allocated_bytes;
    uint64_t peak_internal_bytes;
} q38_memory_snapshot;

#ifdef __cplusplus
}
#endif

#endif /* Q38_H */
