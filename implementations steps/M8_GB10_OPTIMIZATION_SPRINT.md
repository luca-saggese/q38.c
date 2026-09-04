# q38.c — Post-M8 GB10 Optimization Sprint

**Placement:** execute immediately after **M8 COMPLETE** and before resuming feature-heavy M9 work.

**Target runtime:** Qwen3.8-Flash-Next / `qwen4_exp` only  
**Target hardware:** NVIDIA DGX Spark / GB10 / SM121 / CUDA / 128 GiB unified coherent memory  
**Persistent storage policy:** **PLE remains permanently file-backed**. No full PLE host/device mirror is permitted in this sprint.

---

# 0. Purpose

M6 established correct end-to-end inference. M7 removed large runtime/staging inefficiencies and established resident non-PLE execution. M8 establishes the final/selective Q4 recipe and its quality/memory/performance acceptance.

This post-M8 sprint applies the most transferable optimization lessons from `antirez/ds4` to q38, without copying ds4 architecture blindly. q38 remains a specialized runtime for Qwen3.8 and GB10.

The sprint has five goals:

1. reduce per-token **weight traffic and runtime dispatch**;
2. resolve all hot weight pointers before decode;
3. fuse high-ROI projection groups in GDN/QSA/GR without changing semantics;
4. establish a **physical bandwidth ceiling** for vanilla decode;
5. prepare M9 MTP/speculative decoding so it is lossless relative to vanilla greedy.

The sprint must stop once vanilla decode is demonstrably close to the GB10 bandwidth floor. Beyond that point, further throughput gains should come from **fewer bytes per emitted token** and/or **multiple accepted tokens per target evaluation**, not endless micro-fusion.

---

# 1. Source lessons being imported

## 1.1 GB10 decode eventually becomes bandwidth-bound

ds4 reports a saturating-read bandwidth around 231–234 GB/s on GB10 and observes ordinary decode at roughly 85–90% of that physical limit once its CUDA path is mature.

Implication for q38:

```text
if effective_weight_GBps approaches measured_GB10_read_GBps:
    stop micro-optimizing vanilla decode
    move effort to:
        lower bytes/token
        MTP/speculation
        batching
```

Reference: https://github.com/antirez/ds4/issues/773

## 1.2 Resident weights beat repeated mapped/staged access

ds4 added HBM/resident model support and selected-span loading for hot model ranges.

Implication for q38:
- keep all accepted non-PLE compute weights resident when the M8 recipe fits;
- resolve a stable execution pointer once;
- keep mmap/file metadata out of the decode inner loop;
- retain the file mapping only as backing/storage, not as a repeated hot-path lookup mechanism.

Relevant ds4 lineage:
- HBM-resident model work around commit `15f42aa`
- selected model span loading around `4624e15`
- model-map span API around `caa60f2`

## 1.3 Q4 startup can suffer dangerous transient duplication

ds4 has a GB10 report where Q4-class loading transiently approaches ~1.8x the mapped tensor footprint.

Implication for q38:
- never allow source + converted temporary + final resident copy for huge banks at the same time;
- load/convert progressively;
- write directly into final packed resident storage where possible;
- reclaim temporary/source physical pages aggressively.

Reference: https://github.com/antirez/ds4/issues/721

## 1.4 Whole-GGUF host registration is unsafe

A ds4 CUDA worker issue showed that registering the entire GGUF instead of the needed model slice can OOM a 128 GiB Spark.

Implication for q38:
- preserve the existing prohibition on whole-file `cudaHostRegister`;
- PLE remains file-backed with bounded staging only;
- model residency is tensor/span-based.

Reference: https://github.com/antirez/ds4/issues/293

## 1.5 CUDA Graphs are useful only after the data path is stable

ds4's CUDA Graph audit shows graph capture is a modest optimization once the runtime is already near the hardware floor. It also highlights the need for stable pointers and device-side per-token parameters.

Implication for q38:
- graph capture is **late** in this sprint;
- no graph work until resident pointers, workspace, and launch topology are stable;
- target per-layer or segmented graphs first.

Reference: https://github.com/antirez/ds4/issues/534

## 1.6 Speculative verification can silently break greedy identity

ds4 documented a real case where a batch verification path produced numerically different persistent attention/frontier state from the single-token path; the drift eventually flipped greedy tokens.

Implication for q38 M9:
- accepted speculative tokens must leave GDN/QSA/PLE semantic state equivalent to vanilla committed decode;
- target logits matching alone is insufficient;
- when necessary, rollback + replay accepted tokens through the canonical single-token state-update path.

Reference: https://github.com/antirez/ds4/issues/658

---

# 2. Non-negotiable q38 invariants

Throughout this sprint:

```text
PLE:
    always file-backed
    never full resident
    never full dequant mirror

non-PLE:
    resident in final packed representation when memory permits

routed experts:
    no persistent BF16/FP32 mirror
    dequantize in-register / in-kernel

semantic state:
    GDN/QSA/token/PLE history correctness always wins over throughput

QSA top-k:
    exact deterministic selection remains default

M6/M8 correctness:
    every accepted optimization must pass the existing regression/golden gates
```

---

# 3. Phase P8O-0 — Freeze the post-M8 baseline

Before optimization, freeze one immutable baseline.

Required metadata:

```json
{
  "commit_sha": "...",
  "binary_sha256": "...",
  "model_sha256": "...",
  "quant_manifest_sha256": "...",
  "weight_abi": "qwen38-original-layout-v1",
  "device": "GB10",
  "sm": 121,
  "resident_non_ple_bytes": 0,
  "ple_policy": "file-backed",
  "warm_median_ms_per_token": 0,
  "warm_p95_ms_per_token": 0,
  "cold_ms": 0
}
```

Run:

```text
1 cold
5 warm
same process
semantic state reset between runs
resident weights/workspace preserved
```

Record:
- wall time;
- kernel time;
- H2D/upload;
- backend overhead;
- subsystem times;
- resident hit/miss;
- allocation/sync counts;
- RSS;
- MemAvailable;
- CUDA/unified peak;
- generated-token identity.

Suggested artifact:

```text
artifacts/post_m8_opt/baseline.json
```

---

# 4. Phase P8O-1 — Pre-resolve execution tensors

## 4.1 Problem

The decode loop must not repeatedly perform:

```text
GGUF name lookup
-> tensor metadata lookup
-> qtype dispatch lookup
-> residency lookup
-> range/span lookup
-> pointer resolution
-> launch
```

All immutable resolution should happen at model initialization.

## 4.2 Proposed execution descriptor

```c
typedef enum {
    Q38_STORAGE_RESIDENT,
    Q38_STORAGE_FILE_BACKED_PLE,
} q38_storage_class;

typedef struct {
    const void *ptr;          /* stable execution pointer */
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t qtype;
    uint32_t tensor_id;
    q38_storage_class storage;

    /* Debug-only metadata. */
    uint64_t gguf_offset;
    const char *name;
} q38_exec_tensor;
```

The production forward path receives `q38_exec_tensor *`, not a generic GGUF tensor object.

## 4.3 Initialization

```c
static int q38_exec_tensor_bind(
        q38_exec_tensor *dst,
        const q38_weight_tensor *src,
        const q38_resident_map *resident)
{
    memset(dst, 0, sizeof(*dst));

    dst->bytes     = src->bytes;
    dst->rows      = src->rows;
    dst->cols      = src->cols;
    dst->qtype     = src->qtype;
    dst->tensor_id = src->id;
    dst->name      = src->name;

    if (q38_tensor_is_ple(src)) {
        dst->storage = Q38_STORAGE_FILE_BACKED_PLE;
        dst->ptr = NULL;
        dst->gguf_offset = src->file_offset;
        return 0;
    }

    const void *p = q38_resident_lookup(resident, src->id);
    if (!p) return -1;

    dst->storage = Q38_STORAGE_RESIDENT;
    dst->ptr = p;
    return 0;
}
```

## 4.4 Strict production gate

Add:

```text
Q38_EXEC_STRICT=1
```

In strict mode, any non-PLE hot tensor without a resident execution pointer is fatal.

Counters:

```text
resident_lookup_in_decode = 0
gguf_name_lookup_in_decode = 0
non_ple_residency_miss = 0
non_ple_upload_bytes_per_token = 0
```

---

# 5. Phase P8O-2 — Residency arenas / selected spans

Even if all non-PLE weights fit, do not make the implementation depend on one monolithic allocation.

Recommended grouping:

```text
arena_core:
    embeddings
    GR
    GDN
    QSA
    router
    shared expert
    final output

arena_experts_0..N:
    routed expert banks grouped by qtype / contiguous storage

PLE:
    never in these arenas
```

## 5.1 Data structures

```c
typedef struct {
    void *base;
    uint64_t capacity;
    uint64_t used;
    uint32_t tensor_count;
} q38_resident_arena;

typedef struct {
    q38_resident_arena core;
    q38_resident_arena *expert_banks;
    uint32_t expert_bank_count;
} q38_model_residency;
```

## 5.2 Alignment

```c
#define Q38_RESIDENT_ALIGNMENT 256u

static inline uint64_t q38_align_up_u64(uint64_t x, uint64_t a) {
    return (x + a - 1) & ~(a - 1);
}
```

## 5.3 Progressive loading

Do not allocate conversion scratch proportional to the entire model.

```text
for each arena/span:
    allocate final resident destination
    for each tensor:
        copy/convert directly into final packed destination
        validate checksum/shape/qtype
        release transient scratch immediately
    record memory telemetry
```

Hard gate:

```text
peak_transient_extra_bytes <= configured transient budget
```

Do not accept a loader with a large unexplained transient multiplier.

---

# 6. Phase P8O-3 — Dedicated GPU greedy argmax

If q38 still copies the full vocabulary logits to the CPU for greedy decode, replace that with a device reduction.

The LM head already writes the full logit vector on device. For greedy decode we only need:

```text
token_id
optionally top-1 logit
```

## 6.1 Device result

```c
typedef struct {
    float value;
    int32_t id;
} q38_argmax_pair;
```

## 6.2 Kernel sketch

This is q38-specific implementation guidance, not copied ds4 code:

```cuda
__device__ __forceinline__
q38_argmax_pair q38_argmax_choose(q38_argmax_pair a,
                                  q38_argmax_pair b)
{
    if (b.value > a.value) return b;
    if (b.value < a.value) return a;
    return (b.id < a.id) ? b : a; /* deterministic tie rule */
}

__global__ void q38_argmax_kernel(
        const float *logits,
        int n,
        q38_argmax_pair *block_out)
{
    q38_argmax_pair best = { -INFINITY, INT_MAX };

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        q38_argmax_pair x = { logits[i], i };
        best = q38_argmax_choose(best, x);
    }

    /* Warp/block reduction uses q38_argmax_choose() at each level.
       One tiny second-stage reduction chooses the global winner. */
}
```

Acceptance:

```text
GPU argmax ID == CPU/reference argmax ID
```

for:
- random vectors;
- exact ties;
- real M6/M8 logits.

Do not replace exact greedy semantics with a non-deterministic reduction.

---

# 7. Phase P8O-4 — Physical bytes/token accounting

This is required before deeper fusion.

For every resident matrix operation, accumulate:

```text
logical_weight_bytes_touched
activation_bytes_read
activation_bytes_written
kernel_ms
```

```c
typedef struct {
    uint64_t weight_bytes;
    uint64_t activation_read_bytes;
    uint64_t activation_write_bytes;
    double kernel_ms;
} q38_bandwidth_account;
```

Compute:

```text
effective_weight_GBps = weight_bytes / kernel_seconds / 1e9
```

Also run a simple GB10 bandwidth probe to establish the local machine's measured read ceiling rather than relying only on nominal 273 GB/s.

## 7.1 GB10 read probe sketch

```cuda
__global__ void q38_read_bw_probe(
        const float4 *src,
        float *sink,
        size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;

    for (; i < n; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }

    if (acc == -1234567.0f) *sink = acc;
}
```

Measure on a multi-GiB resident allocation with CUDA events.

## 7.2 Stop rule

When vanilla decode repeatedly reaches approximately:

```text
>= 85% of measured saturating read bandwidth
```

and the profile shows true GPU execution rather than upload/sync gaps:

**stop trying to obtain multi-x vanilla decode speedups through micro-fusion.**

That is the decision point for MTP/speculation.

---

# 8. Phase P8O-5 — Multi-projection fusion

Do this only after P8O-4 shows that launch/intermediate traffic still matters.

Qwen3.8 repeatedly projects the same normalized hidden vector into multiple outputs.

General principle:

```text
one normalized input load
+ one dispatch
-> multiple projection outputs
```

Do not fuse merely to reduce launch count if the kernel is already bandwidth-saturated.

---

# 9. GDN projection fusion candidate

The exact Qwen3.8 GDN tensors must follow the already-frozen M3 semantics.

Instead of conceptually:

```text
norm(hidden)
q = Wq*x
k = Wk*x
v = Wv*x
z = Wz*x
a = Wa*x
b = Wb*x
```

evaluate a grouped interface:

```c
typedef struct {
    float *q;
    float *k;
    float *v;
    float *z;
    float *a;
    float *b;
} q38_gdn_projection_out;

int q38_cuda_gdn_project_grouped(
    const q38_gdn_weights *w,
    const float *normalized_hidden,
    q38_gdn_projection_out *out,
    cudaStream_t stream);
```

Possible decomposition:

```text
kernel A:
    RMSNorm + normalized hidden staging

kernel B:
    grouped projection family
    outputs q/k/v/z/a/b

kernel C:
    recurrence + gating + output projection
```

Do **not** force a single mega-kernel initially.

Acceptance per candidate:

```text
q/k/v/z/a/b error within M3/M6 tolerance
GDN recurrent state equivalent
conv history equivalent
final token parity
```

---

# 10. QSA projection/indexer fusion candidate

QSA is especially sensitive because selected position IDs must remain exact.

Candidate decomposition:

```text
normalized hidden
    ↓
grouped Q/K/V projection
    ↓
indexer projection/compression
    ↓
exact score/top-k
    ↓
selected gather
    ↓
attention
```

Potential fused boundaries:

```text
A. Q/K/V projections together
B. indexer projection + compression-state write
C. score + exact top-k
D. gather + attention
```

Do not combine C/D until exact-ID tests remain fully stable.

```c
typedef struct {
    float *q;
    float *k;
    float *v;
    float *index;
} q38_qsa_projected;

int q38_cuda_qsa_project_grouped(
    const q38_qsa_weights *w,
    const float *x,
    q38_qsa_projected *dst,
    cudaStream_t stream);
```

Hard gate:

```text
selected QSA IDs EXACT
```

No tolerance.

---

# 11. Phase P8O-6 — GR / hyper-connection audit

Qwen3.8 uses four hyper-connection branches and a wide state of 10240.

Audit every GR kernel for:

```text
thread/lane serially scans all branches
repeated gamma/norm loads
temporary full-width copies
CPU-side scalar reads
```

Desired pattern:

```text
branch/lane parallelism
shared normalized input where profitable
coalesced 4-branch read/write
single epilogue reduction
```

Example logical indexing helper:

```c
__device__ __forceinline__
int q38_gr_index(int branch, int d) {
    return branch * 2560 + d;
}
```

Do not change the already-correct GR math while optimizing layout.

---

# 12. Phase P8O-7 — Eliminate repeated host synchronization

Audit for:

```text
cudaDeviceSynchronize()
cudaStreamSynchronize()
cudaEventSynchronize()
D2H scalar readbacks
```

inside token/layer loops.

Classify each sync:

```text
semantic dependency
debug-only
measurement-only
unnecessary
```

Production rule:

```text
host should normally synchronize only when:
    next token ID is required
    file-backed PLE needs CPU-side work
    explicit API boundary requires completion
```

Everything else should remain ordered by stream dependencies/events.

Add counters:

```text
host_syncs_per_token
device_to_host_bytes_per_token
```

---

# 13. Phase P8O-8 — Stable device-side decode parameters

Prepare the runtime for CUDA Graph without enabling it yet.

```c
typedef struct {
    uint32_t input_token;
    uint32_t position;
    uint32_t committed_tokens;
    uint32_t qsa_pending_count;
    uint32_t qsa_pending_pos;
    uint32_t flags;
} q38_decode_params;
```

Allocate once:

```c
cudaMalloc(&runtime->decode_params_dev, sizeof(q38_decode_params));
```

Per token:

```c
q38_decode_params p = {
    .input_token = token,
    .position = pos,
    .committed_tokens = committed,
    .qsa_pending_count = qsa->pending_count,
    .qsa_pending_pos = qsa->pending_pos,
    .flags = 0,
};

cudaMemcpyAsync(runtime->decode_params_dev,
                &p,
                sizeof(p),
                cudaMemcpyHostToDevice,
                runtime->stream);
```

Hard requirement:
- pointer to `decode_params_dev` is stable;
- state arrays have stable bases;
- per-token offsets are derived on device.

---

# 14. Phase P8O-9 — Segmented / per-layer CUDA Graph experiment

Only begin when:

```text
non-PLE weight pointers stable
workspace pointers stable
hot-path allocations = 0
launch topology stable
device-side parameter block implemented
```

Do not graph SSD/NVMe file I/O.

Recommended first experiment:

```text
PLE CPU/file-backed prepare
        ↓
device PLE contribution ready
        ↓
CUDA graph segment for compute
        ↓
GPU argmax
```

If PLE injection only occurs at its configured layer, a per-layer/per-island graph may be simpler than whole-token capture.

## 14.1 Acceptance

```text
eager greedy tokens == graph greedy tokens
```

and state checksums for:
- GDN;
- QSA;
- PLE history;
- final logits.

## 14.2 Performance rule

If graph replay gives only a few percent because decode is already bandwidth-bound, keep the simpler implementation unless server/MTP integration benefits from graph stability.

---

# 15. Phase P8O-10 — Prepare M9 MTP correctly

This is the most important correctness lesson imported from ds4.

Speculative verification may evaluate multiple tokens in a batched kernel. Even if mathematically equivalent, floating-point accumulation order may differ from single-token decode.

For q38, persistent semantic state includes at least:

```text
GDN recurrent states for all GDN layers
GDN convolution history
QSA K/V state
QSA compressed/index state
QSA pending/raw ring
token history
PLE n-gram history
```

A speculative accept path must not silently commit numerically different state if M9 promises lossless greedy identity.

---

# 16. Canonical state after speculative acceptance

Define:

```text
canonical committed state
=
state produced by ordinary one-token-at-a-time q38 decode
for the same committed token sequence
```

MTP batch verification may use a faster temporary state for deciding acceptance, but committed state must satisfy the M9 contract.

Recommended safe initial implementation:

```text
snapshot pre-verify semantic state

batch verify proposed tokens
determine accepted prefix

restore pre-verify semantic state

for each accepted token:
    replay canonical single-token commit path

continue decode
```

```c
typedef struct {
    q38_session_snapshot before_verify;
    uint32_t proposed[Q38_MTP_MAX_DRAFT];
    uint32_t accepted;
} q38_mtp_cycle;

int q38_mtp_commit_canonical(
    q38_session *s,
    const q38_mtp_cycle *cycle)
{
    if (q38_session_restore(s, &cycle->before_verify) != 0)
        return -1;

    for (uint32_t i = 0; i < cycle->accepted; i++) {
        if (q38_decode_commit_one(s, cycle->proposed[i]) != 0)
            return -1;
    }

    return 0;
}
```

This is intentionally conservative.

---

# 17. MTP state-equivalence gate

For every speculative fixture:

```text
Path A:
vanilla greedy token-by-token

Path B:
MTP proposal
batch verify
canonical commit/replay
```

Require:

```text
tokens A == tokens B exact
```

Then compare:

```text
GDN recurrent state
GDN conv history
QSA KV logical contents
QSA index/compressed contents
QSA pending/raw state
PLE history
position/counters
final logits
```

A late divergence after dozens of tokens is still a failure.

Run at least:

```text
32 tokens
128 tokens
400 tokens
800 tokens
```

on deterministic prompts.

---

# 18. MTP economics / scheduler instrumentation

Once correctness is proven, measure whether speculation is profitable.

```c
typedef struct {
    uint32_t proposed;
    uint32_t accepted;
    double proposal_ms;
    double verify_ms;
    double canonical_commit_ms;
    double vanilla_target_ms;
} q38_mtp_stats;
```

Report:

```text
mean proposal length
mean accepted length
acceptance histogram
verify ms
saved vanilla target evaluations
effective tokens/sec
```

Scheduler decision should be empirical.

Do not enable MTP by default if:

```text
proposal + verify + commit >= vanilla decode
```

for the current workload.

---

# 19. Optional M9 session batching

This is throughput optimization, not single-session latency optimization.

```c
typedef struct {
    q38_session **sessions;
    uint32_t count;
} q38_session_batch;

int q38_sessions_decode_one(
    q38_runtime *rt,
    q38_session_batch *batch,
    int32_t *next_tokens);
```

Requirements:
- batch size 1 is semantically identical to ordinary decode;
- each session has independent GDN/QSA/PLE semantic state;
- shared model weights remain read-only/stable;
- no cross-session state aliasing.

Do not use aggregate tok/s to claim single-session latency.

---

# 20. What NOT to import from ds4

## 20.1 No full dequant cache

Do not create a persistent BF16/FP16 mirror of Q4 routed experts merely because dequantization costs time.

Default:

```text
packed Q4 resident
-> decode in registers
-> FMA
```

Only reconsider after a measured experiment with strong ROI and safe memory.

## 20.2 No whole-file host registration

Never:

```c
cudaHostRegister(whole_gguf_mapping, whole_file_bytes, ...);
```

## 20.3 No architecture-general abstraction tax

Prefer frozen Qwen3.8 facts where they materially reduce dispatch:
- 48 layers;
- 36 GDN + 12 QSA;
- 4 GR branches;
- hidden size 2560;
- 512 routed experts;
- top-10 routing;
- GB10-only CUDA target.

---

# 21. Benchmark suite

Introduce:

```text
make post-m8-opt-micro
make post-m8-opt-warm
make post-m8-opt-correctness
make post-m8-opt-bandwidth
```

## 21.1 Micro

Targets:
- LM-head;
- Q4 expert gate/up;
- Q4 expert down;
- grouped GDN projections;
- QSA projections;
- exact top-k;
- GR read/write;
- GPU argmax.

Each microbenchmark reports:

```text
kernel_ms
logical bytes read
effective GB/s
output checksum
correctness
```

## 21.2 Warm end-to-end

Always:

```text
1 cold
5 warm
same process
semantic state reset
weights/workspace resident
PLE file-backed
```

Report median + p95.

---

# 22. Acceptance thresholds

## Correctness

```text
M0-M8 regressions            PASS
greedy token fixtures        exact
QSA selected IDs             exact
MoE selected experts         exact
NaN/Inf                      0
fallback                     0
```

## Residency

```text
non-PLE warm uploads         0
non-PLE warm misses          0
stable execution pointers    true
PLE full residency           false
full dequant mirrors         0
```

## Runtime

```text
hot-path weight allocations  0
host sync count              explained/minimized
full-vocab D2H for greedy    0 if GPU argmax enabled
```

## Bandwidth decision

The final report states one of:

```text
A. vanilla decode is still well below GB10 read ceiling:
   continue kernel/data-path optimization

B. vanilla decode is close to measured bandwidth ceiling:
   freeze vanilla path and move throughput work to MTP
```

---

# 23. Suggested commit sequence

```text
P8O-C00 docs: freeze post-M8 GB10 optimization plan
P8O-C01 runtime: add resolved execution tensor descriptors
P8O-C02 runtime: add resident arenas and strict resident lookup
P8O-C03 loader: eliminate large transient Q4 duplication
P8O-C04 perf: add physical bytes-per-token and GB10 bandwidth probe
P8O-C05 cuda: add deterministic device greedy argmax
P8O-C06 cuda: benchmark grouped GDN projection path
P8O-C07 cuda: benchmark grouped QSA projection path
P8O-C08 cuda: parallelize/fuse GR epilogue where beneficial
P8O-C09 runtime: eliminate redundant host synchronizations
P8O-C10 runtime: add persistent device-side decode parameters
P8O-C11 cuda: evaluate segmented/per-layer graph capture
P8O-C12 perf: freeze vanilla decode bandwidth ceiling decision
P8O-C13 m9-prep: add canonical speculative state-equivalence fixture
P8O-C14 docs: final post-M8 optimization acceptance
```

Do not merge a commit solely because it reduces launches. It must improve measured end-to-end behavior or establish infrastructure required by M9.

---

# 24. Debug acceleration

Preserve the fast replay harness introduced during M7.

For each grouped/fused kernel, record a fixture containing only:

```text
input hidden
relevant resident tensor IDs
semantic state before
reference output/state after
```

Then run:

```text
old canonical kernel
vs
candidate kernel
```

without a 48-layer full forward.

Only after micro fixture passes:

```text
single full layer
-> 4-layer GDN/GDN/GDN/QSA cycle
-> full token
-> long deterministic decode
```

---

# 25. Performance accounting example

Final reports should look like:

```text
WARM TOKEN
wall_ms                     <measured>
weight_bytes_read           <measured>
effective_weight_GBps       <measured>

MoE
  weight bytes              ...
  kernel ms                 ...
  upload ms                 0
  backend overhead ms       ...

GDN
  weight bytes              ...
  kernel ms                 ...

QSA
  weight bytes              ...
  kernel ms                 ...

GR
  bytes                     ...
  kernel ms                 ...

PLE
  file bytes read           ...
  staging ms                ...
  injection kernel ms       ...

LM head
  matrix bytes              ...
  kernel ms                 ...

argmax
  kernel ms                 ...
  D2H bytes                 4
```

Never hardcode hypothetical values as measured results.

---

# 26. Decision tree after the sprint

```text
                Post-M8 optimized vanilla decode
                           |
                 measure effective GB/s
                           |
             +-------------+-------------+
             |                           |
       well below ceiling          near GB10 ceiling
             |                           |
      optimize dominant              freeze vanilla
      kernel/data path                    |
             |                       resume M9
             |                           |
        remeasure                 MTP/speculative
                                         |
                                 lossless-state gate
                                         |
                                 acceptance economics
```

---

# 27. Relationship to M9

This sprint is not a replacement for M9.

It should be launched after M8 because it makes M9 materially easier:
- stable resident pointers simplify server workers;
- device-side parameters simplify graph capture;
- physical bandwidth measurement determines whether MTP is necessary;
- speculative state-equivalence fixtures prevent subtle MTP corruption;
- existing session reset/clone/checkpoint/replay support becomes the foundation for safe rollback/replay.

M9 still owns:
- HTTP single-worker server;
- request isolation/cancellation;
- CUDA-fatal supervisor/restart;
- soak tests;
- optional MTP production path.

---

# 28. CUDA fatal policy reminder for M9

A CUDA illegal memory access can poison the process context. Do not attempt to keep serving from a poisoned CUDA context.

Server policy:

```text
worker owns CUDA context
fatal CUDA error
    ↓
worker exits
    ↓
supervisor restarts worker
```

Reference: https://github.com/antirez/ds4/issues/759

---

# 29. Final target

The point of this sprint is not to promise a specific tok/s number.

The required outcome is to know quantitatively which of the following is true:

```text
1. q38 vanilla decode is still software-limited
```

or:

```text
2. q38 vanilla decode is bandwidth-limited on GB10
```

If (1), continue improving the dominant measured data path.

If (2), stop spending time on low-ROI micro-optimizations and move to M9 MTP/speculative decoding, because the main remaining throughput lever is reducing **target-model weight traffic per emitted token**.

That is the optimization boundary this document is designed to establish.
