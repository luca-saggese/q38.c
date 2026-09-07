# PLE Projection Audit

## Scope and baseline

This audit isolates the two projection calls inside `full_ple()` without
loading the runtime model or running a full-chain benchmark. The fixture uses
three production-captured cases (`early`, `middle`, and `late`) from
`artifacts/m4/ple_injection_vectors.json`, the corresponding 16 PLE rows, and
the source-model BF16 projection tensors extracted from the two safetensor
shards. The isolated benchmark is `tests/bench_ple_projection` and is frozen
as `PLE_PROJ_BASELINE_0`.

The runtime invariant remains unchanged: the PLE embedding table is
permanently file-backed. Only the two small dense projection weights are
materialized in the fixture.

## Static call graph

For each token, the production path is:

```text
q38_forward_full()                         q38_forward.c:2145-2480
  -> full_ple()                            q38_forward.c:1659-1860
       -> q38_ple_ngram_ids_ref()          q38_forward.c:1731-1735
       -> q38_ple_store_read_row()         q38_forward.c:1740-1744
       -> row decode (BF16/Q8_0)            q38_forward.c:1745-1760
       -> full_matvec(... key_proj ...)    q38_forward.c:1761-1764
            -> full_tensor_data()          q38_forward.c:817-821
            -> full_row_dot() x 10240      q38_forward.c:845-851
                 -> full_backend()         q38_forward.c:727-750
                      -> q38_forward_cuda_matvec_backend()
       -> full_matvec(... value_proj ..)  q38_forward.c:1764-1768
            -> full_row_dot() x 2560       q38_forward.c:845-851
                 -> full_backend()
                      -> q38_forward_cuda_matvec_backend()
       -> PLE gate/convolution/accumulation
```

`full_matvec()` deliberately does not call `full_matrix_backend()` for a
tensor whose name is identified by `full_is_file_backed_ple()`
(`q38_forward.c:694-701`, `q38_forward.c:823-841`). Consequently the
projection does not use a grouped matrix operation: it enters the row loop and
dispatches one row matvec callback per output row.

The CUDA row backend (`cuda/q38_forward_cuda.cu:1622-1749`) allocates/reuses
three work buffers, uploads one weight row and one input vector, launches one
row operation, copies one scalar back, and synchronizes the stream for every
row. For non-resident PLE tensors it uses the GGUF-mapped host row directly;
the projection weights are not placed in the persistent non-PLE residency set.

## Contract and operation counts

The validated tensor shapes are defined in
`q38_weights.c:147-177`:

| Operation | Input | Output | Weight | Runtime dtype | Runtime quantization | Calls/token |
|---|---:|---:|---:|---|---|---:|
| key projection | 2560 F32 | 10240 F32 | `[10240,2560]` | F32 activation / F32 output | Q8_0 row in current GGUF | 1 |
| value projection | 2560 F32 | 2560 F32 | `[2560,2560]` | F32 activation / F32 output | Q8_0 row in current GGUF | 1 |

The source-model tensors used to build the fixture are BF16. The current
runtime GGUF telemetry identifies 2720 bytes per projection row, which is the
Q8_0 row size (`2560 / 32 * 34`), so the runtime production path is Q8_0
despite the source tensors being BF16.

Therefore, per token:

```text
projection matrix calls = 2
row matvec calls        = 10240 + 2560 = 12800
matrix backend calls    = 0 for PLE projection
scalar fallback rows    = 0 in the CUDA diagnostic path
scalar dot products     = 12800 rows x 2560 elements
```

The CPU fallback has the same 12,800 row loop and performs the dequant/dot
itself for Q8_0. `split_ngram_parts = 128` is validated in
`q38_model_config.c:30` and `q38_model_config.c:99`, but it is not read by
`full_ple()`, `full_matvec()`, or the PLE projection loop. It affects the
embedding-table layout/shard inventory, not the projection operation count.

## Mystery counter

The old `UNKNOWN` count is not a row-lookup count. The current telemetry
artifact reports, per token:

```text
PLE owner callbacks       12800
PLE weight bytes           34816000
bytes per callback             2720
```

`12800 * 2720 = 34816000`, exactly matching 12,800 Q8_0 projection rows.
The 16 PLE embedding rows are separate: their logical row traffic is
`16 * 170 = 2720` bytes/token. Thus the old 12,800 events and 34,816,000
bytes/token originate from PLE projection row matvecs, not row lookup.
The current diagnostic owner mapping already classifies these callbacks as
`PLE` through the logical stage/subsystem (`tests/q2_canonical_bench.c:379-428`).

## Residency, transfers, and synchronization

For each current runtime Q8_0 row callback:

* the row is read from the GGUF mmap-backed host mapping;
* one row is uploaded H2D (`2720` bytes);
* the 2560-element F32 input is uploaded H2D (`10240` bytes);
* one scalar F32 result is copied D2H (`4` bytes);
* the stream is synchronized before returning.

The CUDA backend reuses its temporary allocations, but the work remains
12,800 independent enqueue/upload/execute/D2H/synchronize sequences per
token. The fixture-only benchmark does not invoke CUDA, so its CUDA timing
fields are explicitly `null`, with zero transfers and waits rather than
invented measurements.

## Fixture and independent oracle

`artifacts/m4/ple_projection_fixture/fixture.json` records hashes and shapes
for:

* 3 cases and 48 decoded PLE rows (`rows_f32.bin`);
* 3 captured hidden vectors (`hidden_f32.bin`);
* BF16 raw source projection weights;
* expected key/value projection outputs.

`tests/bench_ple_projection` keeps allocations resident, warms the two
matmuls, runs 100 iterations per case, checks finite outputs, and validates a
small deterministic sample with an independent scalar BF16-to-F32 row-dot
oracle. No model loader, GGUF runtime, or full forward is used.

The baseline measurements are recorded in
`artifacts/perf/current/ple_projection_baseline_v0.json`. They are a CPU
fixture baseline for the dense projection contract, not a replacement for the
runtime Q8_0 CUDA cost.

## Backend audit

The current design is many tiny generic row matvec calls, not one grouped
projection:

* two `full_matvec()` calls are made per token;
* the PLE name filter disables the matrix backend;
* 12,800 row callbacks are issued;
* each callback performs a separate H2D/D2H/synchronization sequence;
* runtime Q8_0 dequantization occurs in the row operation;
* no per-call allocation is required after buffers are grown, but the
  per-row transfer and synchronization remain;
* `token_count == 1` selects the non-batched path.

## Candidate ladder (not implemented)

1. **Grouped/batched CUDA PLE projection**: replace the 12,800 row callbacks
   with one grouped operation per projection while preserving file-backed
   staging semantics.
2. **Persistent device projection weights/workspace**: retain only the small
   dense projection weights and reusable workspaces on device; never retain
   the PLE embedding table.
3. **Fuse Q8_0 dequantization, projection, and accumulation**: remove
   intermediate row transfers and scalar result round trips.
4. **Specialized Qwen3.8 PLE projection kernel**: combine the two projections
   and exploit the fixed 2560/10240 geometry after the first three candidates
   are measured.

No candidate is implemented by this checkpoint.
