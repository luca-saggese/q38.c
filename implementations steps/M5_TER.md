# q38.c — Supplemental Pre-M6 Hardening

Target branch: `qwen38-spark-proto`

Placement in roadmap:

```text
M5
  ↓
POST_M5_PRE_M6_INTEGRATION_HARDENING
  ↓
THIS DOCUMENT — Supplemental Pre-M6 Hardening
  ↓
M6
```

This document contains only the remaining implementation details not fully explicit in the main Post-M5/Pre-M6 hardening document.

It must be completed **before M6 starts**.

---

# 1. Definition of Done

This supplemental gate passes only when:

- the PLE gather path has an explicit small-request fast path;
- the threshold between direct and parallel/coalesced gather is measured on GB10 rather than guessed;
- the PLE pinned staging buffers are persistent, bounded and reused;
- no per-token/per-chunk pinned allocation occurs;
- prefix/session cache restore tests prove that GDN and QSA semantic state are restored identically to a cache miss/recompute path;
- the new tests are integrated into the existing Post-M5 acceptance command.

---

# 2. PLE small-gather fast path

## 2.1 Motivation

For sparse PLE lookup, especially during decode, the number of unique rows requested can be small.

For small gathers, the overhead of:

- thread-pool scheduling;
- parallel work partitioning;
- synchronization;
- generic coalescing machinery;

can cost more than simply reading the rows directly.

Therefore q38 should have two PLE gather strategies:

```text
small gather
    -> direct/simple path

large gather
    -> parallel/coalesced path
```

Do not assume the crossover point.

---

## 2.2 Proposed API

Conceptually:

```c
typedef enum {
    Q38_PLE_GATHER_DIRECT,
    Q38_PLE_GATHER_PARALLEL
} q38_ple_gather_mode;

q38_ple_gather_mode q38_ple_choose_gather_mode(
    uint32_t unique_rows,
    const q38_ple_tuning *tuning);
```

The actual API can differ.

The important requirement is that the threshold is:

- explicit;
- measurable;
- recorded in the platform/tuning report;
- not hidden in unrelated code.

---

## 2.3 Benchmark threshold on GB10

Benchmark candidate unique-row counts:

```text
1
2
4
8
16
32
64
128
256
384
512
768
1024
2048
4096
```

For each size measure:

```text
direct gather latency
parallel gather latency
bytes read
CPU time
wall time
p50
p95
```

Use both:

```text
cold-ish page-cache condition
warm page-cache condition
```

where operationally feasible.

Do not hardcode `512` simply because another implementation found that useful.

---

## 2.4 Decode and prefill may use different thresholds

It is acceptable to have:

```text
decode_direct_threshold
prefill_direct_threshold
```

if benchmarking shows a stable difference.

Decode optimization should prioritize latency.

Prefill optimization should prioritize throughput.

---

## 2.5 Correctness gate

Both paths must produce:

```text
same selected rows
same quantized row bytes
same decoded row values
same PLE contribution
same hidden_after_ple
```

within the existing quant/numeric tolerance.

Path selection must never change model semantics.

---

# 3. Persistent bounded pinned staging pool

## 3.1 Motivation

The optimized PLE path may stage sparse quantized rows before asynchronous H2D transfer.

Do not allocate/free pinned host memory for every:

- token;
- lookup;
- chunk;
- prefill batch.

On GB10 this can introduce:

- allocator overhead;
- transient memory pressure;
- fragmentation;
- synchronization;
- unstable latency.

---

## 3.2 Required design

Allocate a bounded staging pool once during runtime/model initialization.

Conceptually:

```c
typedef struct {
    void   *host_ptr;
    size_t  capacity;
    size_t  used;
    bool    in_flight;
    cudaEvent_t ready;
} q38_ple_stage_buffer;

typedef struct {
    q38_ple_stage_buffer *buffers;
    uint32_t count;
    size_t bytes_per_buffer;
} q38_ple_stage_pool;
```

The exact implementation may differ.

---

## 3.3 Lifecycle

Expected lifecycle:

```text
startup/model init:
    allocate bounded pinned staging pool

runtime:
    acquire free staging buffer
    fill quantized rows
    cudaMemcpyAsync
    mark buffer in-flight
    reuse only after completion event

shutdown:
    release staging pool
```

No steady-state `cudaHostAlloc` / `cudaFreeHost` calls should occur in the hot path.

---

## 3.4 Suggested initial configurations

Benchmark:

```text
1 buffer
2 buffers
3 buffers

1 MiB each
2 MiB each
4 MiB each
8 MiB each
```

Do not assume the largest pool is best.

Selection criteria:

```text
lookup latency
H2D overlap
peak memory
prefill throughput
decode latency
```

---

## 3.5 Memory telemetry

Add:

```text
ple_stage_pool_bytes
ple_stage_high_watermark
ple_stage_wait_count
ple_stage_wait_us
ple_stage_h2d_bytes
ple_stage_h2d_us
```

These fields must appear in M7 profiling later as well.

---

## 3.6 Safety

Every staging request must validate:

```text
required_bytes <= buffer_capacity
```

For oversized requests:

- split into bounded batches;
- or use the verified fallback path.

Never silently overflow or dynamically allocate an unbounded temporary pinned buffer.

---

# 4. Prefix/session cache hit-vs-miss semantic equivalence

## 4.1 Motivation

Qwen3.8 combines:

- recurrent GDN state;
- convolution history;
- QSA K/V;
- QSA compressed/index state;
- QSA pending/raw state.

A cache hit that restores only part of this state can produce plausible but incorrect continuation.

The cache system must therefore be validated as a **semantic state restore mechanism**, not merely as a KV optimization.

---

# 5. Required cache state domains

A reusable prefix checkpoint/cache entry must account for all semantic state required at the cached position.

At minimum:

```text
committed token position
token/ngram history

for each GDN layer:
    recurrent state
    conv history

for each QSA layer:
    main K state
    main V state
    compressed/index state
    pending/raw partial-group state
    pending count/position

other semantic counters required by the frozen implementation
```

Do not include reconstructible performance caches such as:

```text
PLE hot-row cache
expert hot cache
CUDA workspace
GR transient activation
```

---

# 6. Prefix cache equivalence test

For a deterministic prompt split into:

```text
prefix P
suffix S
```

perform two runs.

## 6.1 MISS path

```text
fresh session
    ↓
process P normally
    ↓
process S
```

Save:

```text
state after P
state after P+S
next-token logits
greedy tokens
```

## 6.2 HIT path

```text
fresh session
    ↓
restore cached checkpoint/state after P
    ↓
process S
```

Compare against MISS.

---

# 7. Required comparisons after prefix restore

## 7.1 GDN

For every GDN layer:

```text
recurrent state HIT == MISS
conv history HIT == MISS
```

Use exact comparison where serialization is byte-preserving, otherwise the established numeric tolerance.

---

## 7.2 QSA

For every QSA layer:

```text
main K logical content HIT == MISS
main V logical content HIT == MISS
compressed/index state HIT == MISS
pending/raw state HIT == MISS
position/counters HIT == MISS
```

Selected QSA IDs for the next suffix tokens must match exactly.

---

## 7.3 Model output

Require:

```text
next-token logits HIT ~= MISS
greedy next token HIT == MISS
full deterministic suffix continuation HIT == MISS
```

The token sequence must match exactly for greedy decoding.

---

# 8. Cache-boundary stress matrix

Test cache checkpoints at positions deliberately chosen around recurrent/QSA boundaries.

At minimum:

```text
1
2
3
4
5
7
8
15
16
31
32
127
128
511
512
2047
2048
4095
4096
```

Additionally test positions:

```text
mod 4 = 0
mod 4 = 1
mod 4 = 2
mod 4 = 3
```

because QSA compression ratio is 4 and the pending-group state must be exercised.

---

# 9. Chunking + cache interaction

Test:

```text
same prefix token stream
```

constructed with different prefill chunk partitions.

Then cache at the same committed token position.

Require the resulting cached semantic state to be equivalent.

Example:

```text
Path A:
    prefix 1024 in one chunk

Path B:
    256 + 256 + 256 + 256

Path C:
    random partitions

cache after token 1024
```

Then continue with the same suffix.

Required:

```text
same restored state
same selected QSA IDs
same logits
same greedy continuation
```

---

# 10. Future MTP compatibility

This pre-M6 gate does not implement MTP.

However the prefix/checkpoint state representation introduced now must preserve enough QSA pending/raw information that M9 MTP can extend it.

Do not define a checkpoint format that assumes:

```text
maximum pending QSA tokens = 1
```

unless the frozen semantics explicitly guarantees this and the format is versioned for later extension.

Prefer versioned state metadata.

---

# 11. Suggested implementation commits

```text
PM5S-C00
docs: freeze supplemental pre-M6 hardening gate

PM5S-C01
perf: add PLE direct-vs-parallel gather modes

PM5S-C02
bench: determine GB10 PLE gather crossover threshold

PM5S-C03
perf: add persistent bounded pinned PLE staging pool

PM5S-C04
test: verify staging path equivalence with mmap reference

PM5S-C05
test: add prefix cache HIT-vs-MISS semantic state comparison

PM5S-C06
test: add QSA pending-boundary cache matrix

PM5S-C07
test: add chunked-prefix cache equivalence

PM5S-C08
integration: add supplemental gate to post-M5 acceptance
```

Keep these commits separate until the gate passes.

---

# 12. Test matrix

| ID | Test | PASS |
|---|---|---|
| PM5S-T01 | PLE direct gather correctness | Same rows/contribution as reference |
| PM5S-T02 | PLE parallel gather correctness | Same rows/contribution as reference |
| PM5S-T03 | Gather crossover benchmark | Threshold measured on GB10 |
| PM5S-T04 | Persistent staging reuse | No hot-path pinned allocation |
| PM5S-T05 | Staging memory bound | Never exceeds configured pool |
| PM5S-T06 | mmap vs staged path | Same hidden/logits |
| PM5S-T07 | Prefix MISS baseline | Reference captured |
| PM5S-T08 | Prefix HIT GDN state | HIT == MISS |
| PM5S-T09 | Prefix HIT QSA state | HIT == MISS |
| PM5S-T10 | QSA mod-4 cache boundaries | All pass |
| PM5S-T11 | Chunked-prefix cache state | Equivalent |
| PM5S-T12 | Greedy continuation HIT/MISS | Exact token sequence |
| PM5S-T13 | Memory telemetry | Complete staging/cache accounting |
| PM5S-T14 | Full M0-M5 regression | Green |

---

# 13. Required artifacts

```text
artifacts/post_m5_supplement/
  ple_gather_benchmark.json
  ple_gather_threshold.json
  ple_stage_pool_config.json
  ple_stage_pool_profile.json
  ple_mmap_vs_stage_equivalence.json
  prefix_cache_miss_state.json
  prefix_cache_hit_state.json
  prefix_cache_state_diff.json
  qsa_cache_boundary_matrix.json
  chunked_prefix_cache_equivalence.json
  greedy_hit_miss.json
  memory.json
  acceptance.txt
```

---

# 14. Acceptance command

Suggested:

```bash
make post-m5-integration
make post-m5-supplement
make m0-m5-regression
git diff --check
```

Exact Make target names may differ.

---

# 15. Exit criterion

M6 may begin only when both:

```text
POST_M5_PRE_M6_INTEGRATION_HARDENING
```

and:

```text
Supplemental Pre-M6 Hardening
```

are green.

The expected roadmap is therefore:

```text
M5
 ↓
Post-M5 Integration Hardening
 ↓
Supplemental Pre-M6 Hardening
 ↓
M6 — MoE + first complete end-to-end Q2 inference
```

The purpose of this supplemental gate is to ensure that M6 starts with:

- a latency-aware PLE gather path;
- bounded reusable pinned staging;
- proven semantic prefix-state restoration;

rather than discovering these state/memory issues during the first complete model integration.
