# q38.c — Post-M5 / Pre-M6 Integration Hardening

Target branch: `qwen38-spark-proto`

Target hardware:
- NVIDIA DGX Spark
- GB10 / SM121
- 128 GB unified coherent memory
- CUDA only

Purpose:
This document inserts a mandatory integration/hardening gate **after M5 and before M6**.

The objective is to incorporate the most important implementation lessons discovered on Qwen3.8-Flash-Next running on DGX Spark before starting the full 48-layer MoE/end-to-end path.

This gate does **not** replace M5 acceptance.
It runs only after M5 is green.

---

# 1. Why this gate exists

At the end of M5 we should have individually validated:

- tokenizer;
- Gated Residual;
- Gated DeltaNet;
- PLE semantics;
- PLE file-backed access baseline;
- QSA semantics;
- QSA CUDA reference/naive path;
- chunk invariance;
- QSA state growth.

Before M6 adds:

- 512 routed experts;
- top-10 routing;
- shared expert;
- full 48-layer integration;
- first real decode loop;

we need to prove that the already-built subsystems behave correctly **when combined**, and that our GB10-specific design has not inherited known classes of errors seen in other Qwen3.8 implementations.

---

# 2. Definition of Done

The post-M5 gate passes only when:

- a real integrated forward probe exists for the first complete architectural block;
- PLE injection is validated on actual hidden states;
- QSA top-k is deterministic and exact;
- QSA partial/pending compression state is explicitly modeled;
- PLE mmap/file-backed baseline is retained as the semantic reference;
- PLE dedup/staging optimization is implemented only if output-equivalent;
- GB10 shared-memory limits are measured at runtime;
- CUDA kernels pass tail-size and warp/block stress tests;
- all M0–M5 correctness tests still pass.

Only then may M6 begin.

---

# 3. Integrated forward probe — mandatory

## 3.1 Target

Build an integrated probe over the first complete architecture cycle:

```text
embedding
   ↓
layer 0
   ↓
layer 1
   ↓
layer 2
   ↓
layer 3
```

where the actual GDN/QSA/PLE placement must come from the frozen Qwen3.8 configuration and current q38 artifact mapping.

Do not rely on the schematic numbering in this document if the frozen model maps PLE differently.

## 3.2 Dump points

For each layer, where applicable, dump:

```text
input hidden
norm output
GR read output
GDN/QSA projections
GDN/QSA output
PLE row IDs
PLE contribution
hidden before PLE
hidden after PLE
GR write output
layer final hidden
```

When MoE is not yet integrated, stop the probe at the exact pre-MoE or post-core boundary and document that boundary precisely.

## 3.3 External golden requirement

The expected values must come from an independent reference:

Priority:

1. official Qwen/Transformers implementation;
2. llama.cpp Qwen4Exp semantic implementation;
3. a minimal custom reference using original checkpoint tensors.

Do **not** generate expected values from q38 itself.

## 3.4 Acceptance

For every stage:

```text
external reference
    ==
q38 scalar/reference
    ==
q38 CUDA
```

within the pre-declared numeric tolerance.

---

# 4. PLE hidden-state injection gate

This must be complete before M6.

Required golden fields:

```json
{
  "tokens": [],
  "ngram_ids": [],
  "row_ids": [],
  "hidden_before_ple": [],
  "ple_contribution": [],
  "hidden_after_ple": [],
  "model_revision": "",
  "reference_revision": "",
  "checksums": {}
}
```

Required tests:

- single short sequence;
- repeated n-gram sequence;
- chunked vs unchunked;
- cold mmap vs warm mmap;
- baseline mmap vs optimized staging path.

Hard gate:

```text
same token stream
→ same PLE contribution
→ same hidden_after_ple
```

independent of chunking and cache state.

---

# 5. PLE file-backed optimization — minimal pre-M6 version

The current mmap/page-cache path remains the reference.

Add only a minimal optimized path before M6.

## 5.1 Suggested pipeline

```text
ngram row IDs
   ↓
deduplicate row IDs
   ↓
map to physical row/block addresses
   ↓
sort by physical offset
   ↓
coalesce adjacent reads where possible
   ↓
bounded pinned staging buffer
   ↓
cudaMemcpyAsync
   ↓
GPU expansion using inverse-index mapping
```

## 5.2 Important semantic rule

Deduplication may reduce storage reads.

It must **not** alter mathematical accumulation order unless the resulting numerical difference has been explicitly validated.

## 5.3 Cache policy

Do not introduce a large LRU by default.

First measure:

```text
unique rows/token
reuse distance
hit rate
bytes saved
cache bytes
lookup latency
```

Start with:

```text
0 GiB cache
1 GiB
2 GiB
4 GiB
```

Only keep cache configurations with a measurable benefit.

## 5.4 No-go designs

Do not use:

- whole-PLE pinned mirror;
- whole-PLE CUDA allocation;
- whole-PLE managed-memory materialization;
- persistent full dequantized table.

---

# 6. QSA exact top-k — mandatory default

Known Qwen3.8 implementations on GB10 have encountered sparse top-k paths that were fast but non-deterministic or candidate-dropping.

q38 policy:

```text
QSA exact top-k = default
fast/approx top-k = experimental only
```

## 6.1 Required invariants

For every test:

```text
score vector:
    floating tolerance allowed

selected IDs:
    exact match required
```

unless a formally verified exact tie exists.

## 6.2 Required stress cases

Test:

```text
context 1,2,3,4,5,6,7,8
127,128,129
511,512,513
2047,2048,2049
4095,4096,4097
random lengths
```

Test tie cases intentionally.

## 6.3 Chunk invariance

Same committed token stream must produce:

```text
same compressed/index state
same selected IDs
same gathered positions
same downstream output
```

across different prefill partitions.

---

# 7. QSA pending/raw compression state

Do not model QSA state only as:

```text
main K/V
compressed index
```

The frozen QSA semantics must explicitly answer:

```text
What happens before a compression group of 4 is complete?
Where are raw/pending keys stored?
When is a compressed group committed?
How many pending tokens can exist?
How does chunked prefill affect the partial group?
How is pending state reset/restored?
```

## 7.1 Proposed logical state

Exact layout must follow the verified reference, but conceptually expect something equivalent to:

```c
typedef struct {
    q38_kv_store main_kv;
    q38_qsa_index_store compressed;

    q38_qsa_pending_ring pending;
    uint32_t pending_count;
    uint32_t pending_pos;

    uint64_t committed_tokens;
} q38_qsa_state;
```

Do not implement this shape by guessing; use it as a checklist against upstream semantics.

## 7.2 MTP future-proofing

Even though MTP belongs to M9, QSA state introduced now must not make MTP impossible later.

Document how pending capacity would be affected by multi-token speculative proposals.

Do not yet optimize specifically for MTP.

---

# 8. GB10 shared-memory specialization

Do not import shared-memory assumptions from H100/B200.

At startup, record:

```text
cudaDevAttrMaxSharedMemoryPerBlock
cudaDevAttrMaxSharedMemoryPerBlockOptin
warp size
SM count
compute capability
```

## 8.1 Kernel policy

Every GDN/QSA custom kernel that uses dynamic shared memory must have:

```text
requested shared bytes
selected tile
selected block size
selected warp count
```

recorded in debug/profiling builds.

## 8.2 Required benchmark variants

At least:

```text
small tile
large tile within GB10 opt-in shared-memory limit
```

for the critical QSA/GDN kernels.

Choose based on measured GB10 performance.

---

# 9. CUDA race and tail-size stress

Qwen3.8 implementations on Blackwell have exposed failures that only appear at specific:

- warp counts;
- block/tile sizes;
- ubatch tails;
- partial groups.

Before M6, run a dedicated stress suite.

## 9.1 Ubatch sizes

At minimum:

```text
1..64
127,128,129
255,256,257
507,508,509
511,512,513
1015,1016,1017
random up to 2048
```

## 9.2 QSA group tails

At minimum:

```text
1,2,3,4,5,6,7,8
all lengths mod 4
```

## 9.3 Kernel launch variants

Where applicable:

```text
different warp counts
different block sizes
different shared-memory tile sizes
```

Correctness must not depend on scheduler details.

---

# 10. Tokenizer final pre-M6 parity

Before M6, native tokenizer parity should include:

```text
ASCII
Unicode
NFC/NFD
emoji
CJK
RTL
special tokens
chat template
multi-turn
structured/tool-like content
image/video placeholder markers
```

Hard gate:

```text
native q38 token IDs
==
frozen official reference token IDs
```

100%.

Special token IDs that are compile-time constants must also be validated at load time against the frozen tokenizer metadata.

---

# 11. Binder semantic validation

Do not rely primarily on aggregate tensor counts.

For every layer:

```text
if GDN:
    require exact GDN tensor set

if QSA:
    require exact QSA tensor set

if PLE injection layer:
    require exact PLE-specific tensor set

always:
    require GR tensor set
    require MoE-related expected tensors when M6 begins
```

Global tensor count remains only a sanity check.

---

# 12. Post-M5/pre-M6 acceptance suite

Suggested gate:

```text
make post-m5-integration
```

It should include:

- M0 acceptance;
- M1 acceptance;
- M2 acceptance;
- M3 acceptance;
- M4 acceptance;
- M5 acceptance;
- PLE hidden-state golden validation;
- first integrated forward probe;
- exact QSA top-k tests;
- QSA pending-state tests;
- QSA chunk invariance;
- PLE mmap vs optimized-path equivalence;
- GB10 shared-memory probe;
- CUDA tail/warp stress;
- tokenizer extended parity;
- `git diff --check`.

---

# 13. Required artifacts

```text
artifacts/post_m5/
  platform_sm121.json
  shared_memory_limits.json
  integrated_forward_goldens.json
  ple_hidden_injection.json
  ple_dedup_profile.json
  ple_staging_profile.json
  qsa_exact_topk.json
  qsa_pending_state.json
  qsa_chunk_invariance.json
  cuda_tail_stress.json
  tokenizer_extended_parity.json
  binder_semantic_audit.json
  acceptance.txt
```

---

# 14. Exit criterion

M6 may begin only when:

```text
M0-M5 green
+
PLE injection hidden-state validated
+
first integrated forward probe green
+
exact QSA top-k green
+
QSA pending state verified
+
GB10 kernel/tail stress green
```

The purpose of this gate is to ensure M6 debugging is about MoE/end-to-end integration, not unresolved PLE/QSA/state correctness.
