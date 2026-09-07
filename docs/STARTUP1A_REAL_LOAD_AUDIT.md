# STARTUP-1A Real Model Load Timeline Audit

## Scope

This audit corrects the STARTUP-1 startup decomposition without changing the
tokenizer, tensor layout, forward path, direct I/O policy, or residency
strategy. The residency pipeline is instrumented with
`CLOCK_MONOTONIC_RAW`; no page cache is dropped.

## Existing timer boundaries

`q38_runtime_init()` in `q38_session.c:80-213` currently has these boundaries:

| Label | Start | End | Contents | Status |
|---|---|---|---|---|
| `gguf_open_ms` | `q38_session.c:88` | `q38_session.c:96` | `q38_gguf_open()` | valid GGUF open/parse interval |
| `tokenizer_ms` | `q38_session.c:100-102` | `q38_session.c:110-112` | `q38_tokenizer_init()` only | valid tokenizer interval; does not contain CUDA residency |
| `binding_ms` | `q38_session.c:113-115` | `q38_session.c:122-124` | `q38_weights_bind_subset()` | valid metadata/binding interval |
| `cuda_prepare_ms` | `q38_session.c:127-129` | `q38_session.c:141-142` | CUDA context creation, non-PLE residency, LM-head preparation | broad parent; must be decomposed |
| `runtime_init_ms` | `q38_session.c:88` | `q38_session.c:171` | all initialization phases above | valid total init interval |

The earlier `tokenizer_ms` label is not contaminated by residency planning,
file paging, staging, CUDA upload, fingerprinting, or weight binding. The
manual observation of residency activity during the approximately 50-second
interval corresponds to the old broad `cuda_prepare_ms` interval, not
`tokenizer_ms`.

The call chain is:

```text
q38_runtime_init
├─ q38_gguf_open                         q38_gguf.c:345
│  ├─ open/fstat
│  ├─ mmap(MAP_PRIVATE|PROT_READ)        q38_gguf.c:355-388
│  ├─ parse_metadata
│  └─ parse_tensors
├─ q38_tokenizer_init                    q38_tokenizer.c:313
│  ├─ read tokenizer.json
│  ├─ parse vocabulary / merges
│  └─ verify special tokens
├─ q38_weights_bind_subset                q38_weights.c:357
│  ├─ metadata validation
│  ├─ exact tensor binding
│  └─ layer tensor classification
└─ q38_forward_cuda_context_create       cuda/q38_forward_cuda.cu:1063
   ├─ q38_forward_cuda_enable_all_non_ple_residency
   │  ├─ residency planner
   │  ├─ one CUDA allocation per resident tensor
   │  ├─ pinned staging + device transfer allocation
   │  ├─ mmap -> staging memcpy per span
   │  ├─ H2D enqueue per span
   │  ├─ D2D enqueue per resident tensor
   │  └─ one final stream wait
   └─ q38_forward_cuda_prepare_lm_head  cuda/q38_forward_cuda.cu:1551
```

## Residency timing semantics

The residency instrumentation is inside
`q38_forward_cuda_enable_all_non_ple_residency()`:

| Bucket | Boundary | Meaning |
|---|---|---|
| `residency_plan_ms` | `cuda/q38_forward_cuda.cu:1174-1183` | catalog sorting, PLE exclusion, and span construction |
| `residency_device_alloc_ms` | `cuda/q38_forward_cuda.cu:1240-1321` | resident tensor `cudaMalloc`, pinned staging allocation, and transfer-buffer allocation |
| `residency_source_copy_ms` | `cuda/q38_forward_cuda.cu:1327-1344` | CPU `memcpy` from the GGUF mmap into pinned staging |
| `residency_h2d_enqueue_ms` | immediately around each H2D enqueue | host CPU time submitting the span H2D operation |
| `residency_d2d_enqueue_ms` | around each destination D2D enqueue loop | host CPU time submitting tensor-slice copies |
| `residency_final_wait_ms` | `cuda/q38_forward_cuda.cu:1380-1395` | final `cudaStreamSynchronize` completion wait |

These are sequential main-thread intervals and are not additive with
`cuda_prepare_ms` twice. `residency_other_ms` is the remaining part of
`cuda_prepare_ms`, primarily CUDA context setup and LM-head preparation; it is
not a residual masking bucket for the full token wall.

For every coalesced span, diagnostics record:

- source file offset;
- span bytes;
- mmap-to-staging copy wall time;
- H2D enqueue wall time.

The final completion is associated with the single final stream wait, which
covers all earlier H2D and D2D submissions on the residency stream.

## Byte and memory accounting

The diagnostics report:

- planned resident payload bytes;
- mmap-to-staging bytes;
- submitted H2D bytes;
- planned span count and H2D transfer count;
- PLE upload bytes, which must remain zero;
- CUDA allocation count and total allocated bytes;
- minimum/maximum allocation-independent page residency via `mincore`;
- process minor and major page-fault counters before and after residency.

Coalescing may make staged/H2D bytes larger than resident payload bytes because
small gaps inside a span are copied intentionally. The difference is padding
and is reported by comparing `residency_staged_bytes` with
`residency_planned_bytes`.

## Staging reuse

STARTUP-1 uses one pinned staging buffer and one device transfer buffer. It is
not double-buffered. The sequence is therefore:

```text
fill staging -> enqueue H2D -> enqueue D2D slices
fill same staging -> enqueue next H2D -> enqueue next D2D slices
...
final stream wait
```

There is no explicit per-span fence before staging reuse. CUDA stream ordering
does not by itself serialize a host overwrite of a pinned buffer against a
previous asynchronous H2D operation. Therefore the current implementation has
no true double buffering and no explicit reuse fence; this is an unresolved
correctness/design risk, not evidence of overlap. The current design should be
treated as serialized staging and audited before any optimization.

## Synthetic validation

The no-model fixture remains `tests/bench_residency_startup.cu`. It verifies
destination hashes, excludes the synthetic PLE tensor, and compares the
per-tensor uploader with C1. The fixture is not used to claim storage
throughput: its source is an mmap-backed synthetic file.

## Real-load output

The diagnostic binary emits a single `startup_timing` JSON object after
successful runtime initialization. The required artifact is:

`artifacts/perf/current/startup1a_real_load_attribution_v1.json`

No decode measurement is part of this audit.

## Real sample

The single permitted runtime sample produced:

| Component | Time |
|---|---:|
| Total `runtime_init_ms` | 33,813.716 ms |
| GGUF open + tokenizer + binding | 27,291.339 ms |
| Residency planning | 26.162 ms |
| Device allocations | 1,956.967 ms |
| mmap -> pinned staging | 3,382.299 ms |
| H2D enqueue | 2.273 ms |
| D2D enqueue | 7.341 ms |
| Final CUDA completion wait | 12.718 ms |
| Other CUDA preparation | 1,135.609 ms |

The runtime sample planned and staged `49,500,850,200` bytes (46.101 GiB)
across 148 spans and submitted the same number of bytes to H2D. PLE submitted
bytes were zero. There were 1,168 CUDA/pinned allocations totalling
`52,043,647,000` bytes; the largest staging/transfer allocation was
`1,271,398,400` bytes.

The source copy rate was 14.635 GB/s for mmap-to-staging CPU copies. The
reported H2D number in the artifact is explicitly only
`bytes / final cudaStreamSynchronize blocked time`; it is not a physical DMA
throughput measurement because C1 does not record CUDA event elapsed time.
The sample had zero major faults and the process minor-fault count increased
from 278,195 to 1,350,115. `mincore` reported the same resident-page count
before and after the residency phase.

The old `cuda_prepare_ms` value (7,030.452 ms) is therefore invalid as a
single residency cost: it contains allocations, source copies, enqueue work,
the final wait, CUDA context work, and LM-head preparation. The new artifact
keeps those intervals separate.
