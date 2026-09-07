# DS4 Main Memory Ownership Audit

## Scope

This document classifies memory ownership from static source inspection. The previously observed q38 values of approximately 46.1 GiB persistent resident and 120.5 GiB peak CUDA are treated as external observations; this audit explains the ownership model that can produce them but does not claim a new measurement.

## Ownership table

| Category | DS4 main | q38 current | Consequence |
|---|---|---|---|
| Raw model | GGUF mmap remains the source map. It may be left unpinned when complete expert replacement artifacts are active (`ds4_cuda.cu:3750-3757`). | Private read-only GGUF mmap remains open for the runtime (`q38_gguf.c:345-395`). | Both designs retain a raw source, but DS4 can stop pinning replaced expert ranges. |
| Packed/aligned model | Derived Q8/IQ2/Q2K artifacts are separately allocated and indexed in `g_derived_ranges` (`ds4_cuda.cu:4354-4473`). | No general aligned derived artifact tier in the current q38 residency path. | DS4 trades startup work for decode-friendly layouts. |
| Workspace | Each DS4 repack worker has pinned host stage, device scratch, and a CUDA stream (`ds4_repack.cu:566-629`). Forward kernels also use persistent or pool scratch. | q38 allocates persistent device activation/state buffers through `ensure_buffer()` and per-context fields (`q38_forward_cuda.cu:985-1090,2047-2202`). | Worker scratch is transient in DS4; q38 forward workspace is long-lived after first use. |
| State | DS4 owns graph/cache/session state in CUDA-side structures. | q38 owns recurrent GDN state and convolution history in host session state, with device mirrors (`q38_forward_cuda.cu:1038-1090`). | q38 can simultaneously hold host state and device state. |
| KV/cache | DS4 attention kernels and session graph structures own device-side attention/KV buffers in the donor implementation. | q38 session state and QSA backends own their state; exact KV capacity is determined by q38 state layout and is separate from model residency. | KV is not part of the non-PLE tensor sum and must be reported separately. |
| PLE | DS4 audit has no equivalent q38 PLE embedding contract in the donor path. | Large PLE embedding shards are excluded from resident CUDA copies by `is_ple_embedding_table()`; dense PLE projection weights are resident. | q38 intentionally retains a file-backed PLE subset. |
| Duplicate | DS4 Q8 artifacts are explicitly additive; IQ2/Q2K are replacement candidates only when complete coverage is proven. | q38 normally has raw mmap plus one resident CUDA copy for each non-PLE tensor (`q38_forward_cuda.cu:1096-1229`). | Raw + resident coexistence explains a large part of peak memory. |
| Temporary | DS4 repack stage/scratch are per-worker and freed after artifact construction; optional hash readbacks add temporary activity. | q38 has shared device input/output/weight/aux buffers and stage-specific GR/GDN/MoE/QSA workspaces; `ensure_buffer()` grows them and retains them until context destruction. | Peak CUDA includes the resident model plus simultaneously allocated workspaces. |

## DS4 lifetime model

```text
raw mmap
  -> catalog records
  -> worker stage buffers + device scratch
  -> persistent aligned artifacts
  -> derived-range ownership map
  -> optional raw expert replacement
  -> raw mmap remains only as source/fallback, not necessarily pinned
```

The critical ownership switch is `g_derived_replaces_complete`. `ds4_gpu_model_range_replaced()` returns true only for covered IQ2/Q2K source ranges after complete artifact publication (`ds4_cuda.cu:4432-4489`). Q8 is documented as additive, so the source raw span remains valid for consumers that do not use the aligned Q8 dispatch.

## q38 lifetime model

```text
private GGUF mmap
  -> q38 tensor metadata and host pointers
  -> one cudaMalloc per non-PLE tensor
  -> one cudaMemcpyAsync per tensor
  -> persistent exec tensor index
  -> stage workspaces allocated lazily by ensure_buffer()
  -> device mirrors for GDN state and history
  -> final logits/token readback
```

`q38_runtime_init()` opens the mmap, binds weights, creates the CUDA context, enables all non-PLE residency, and prepares the LM head (`q38_session.c:74-119`). The residency loop keeps the host tensor pointer and stores a separate device allocation in `persistent_tensor`; the execution descriptor points to that device allocation (`q38_forward_cuda.cu:1145-1183`).

## Explaining the observed q38 memory delta

The approximate 46.1 GiB persistent resident value is consistent with the sum of all non-PLE GGUF tensor payloads admitted to `q38_forward_cuda_enable_all_non_ple_residency()`, excluding the large PLE embedding shards. It is a **resident execution sum**, not a peak allocation sum.

The approximate 120.5 GiB peak CUDA value can exceed that resident sum because peak accounting can include:

1. the persistent non-PLE model copies;
2. LM-head storage and any separately prepared dense weights;
3. GDN, GR, QSA, MoE, input/output, route, accumulation, and recurrent device workspaces;
4. temporary allocation growth during the first forward;
5. CUDA allocator/driver bookkeeping and reclaimable unified-memory pages;
6. any simultaneous source/derived or fallback allocations during startup.

The source does not justify assigning an exact byte total to each subcategory without the existing allocation artifact. In particular, q38's `cudaMemGetInfo()` comments explicitly note that unified-memory free-page accounting does not equal reclaimable host page-cache accounting (`q38_forward_cuda.cu:1110-1118`).

## Main ownership advantage

DS4 separates:

* raw source ranges;
* additive derived artifacts;
* replacement artifacts;
* selective cache ranges;
* temporary worker staging.

q38 currently collapses most non-PLE tensors into a single “resident” class and retains the mmap as a live fallback source. That is simple and correct, but it makes it difficult to reclaim raw expert residency or batch ownership transitions.

## Recommended ownership contract for a future q38 port

Every tensor/range should have an explicit state:

```text
FILE_BACKED
RESIDENT_RAW
DERIVED_ADDITIVE
DERIVED_REPLACEMENT
WORKSPACE
STATE
```

The execution lookup must reject a `DERIVED_REPLACEMENT` miss rather than silently falling back to a stale raw pointer, while a `DERIVED_ADDITIVE` miss may use the raw source. The large PLE embedding table must remain `FILE_BACKED`; this audit does not recommend changing that invariant.

## Validation required before any ownership port

* Build a static byte ledger from GGUF tensor metadata.
* Record every CUDA allocation and free by ownership class.
* Verify that replacement artifacts cover complete expert tensors before raw ranges are released or left unpinned.
* Verify first-token correctness with raw, additive, and replacement paths.
* Track peak and persistent memory separately; do not compare `cudaMemGetInfo()` snapshots directly to tensor-byte sums.

