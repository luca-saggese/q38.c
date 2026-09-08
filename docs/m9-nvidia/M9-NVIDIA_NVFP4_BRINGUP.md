# M9-NVIDIA — Native NVIDIA NVFP4 Bring-up on DGX Spark / GB10

**Project:** q38.c  
**Target branch:** `qwen38-spark-proto`  
**Milestone:** `M9-NVIDIA`  
**Primary hardware:** NVIDIA DGX Spark / GB10 / SM121 / ARM64 / CUDA  
**Primary model:** `nvidia/Qwen3.8-Flash-Next-NVFP4`  
**Date:** 2026-09-08  
**Status:** implementation plan / bring-up specification

---

## 0. Executive decision

M9-NVIDIA introduces a second high-quality model arm for q38: the official NVIDIA ModelOpt NVFP4 checkpoint for Qwen3.8-Flash-Next.

This is **not** a generic NVFP4 backend and **not** a generic Hugging Face runtime. The implementation is specialized for:

- Qwen3.8-Flash-Next / `qwen4_exp`;
- NVIDIA GB10 / SM121;
- the exact mixed-precision layout of `nvidia/Qwen3.8-Flash-Next-NVFP4`;
- q38's existing device-resident decoder chain;
- q38's permanent file-backed PLE design;
- a strict 118 GiB unified-memory hard ceiling.

The first goal is not maximum speed. The first goal is:

1. understand the exact NVIDIA checkpoint ABI;
2. ingest it without silently changing quantization semantics;
3. keep PLE file-backed;
4. keep the non-PLE model resident;
5. execute routed experts with faithful NVFP4 semantics;
6. obtain a correct text-generation run;
7. only then optimize the NVFP4 MoE path.

The current Q2 production arm remains frozen as the rollback/reference path during M9-NVIDIA.

---

# 1. Current q38 reference before M9-NVIDIA

The production path immediately before this milestone is the device-resident decoder chain.

Frozen observed decode reference:

```text
Q2_DEVICE_CHAIN_V1

OLD host-boundary path:
    110.393 ms/token
    9.059 tok/s

DEVICE decoder chain:
     76.282 ms/token
    13.109 tok/s

DEVICE improvement:
    +44.71% throughput
    -30.90% latency

generated token sequence equal:
    yes
```

Important runtime invariants already established:

```text
hardware:
    DGX Spark / GB10 / SM121 / ARM64 / CUDA

memory:
    soft watermark = 114 GiB
    hard ceiling   = 118 GiB

PLE:
    permanently file-backed
    never promoted to full resident RAM/GPU storage
    async overlap remains enabled

decoder:
    device-resident layer chain is production

routing:
    router remains precision-sensitive
    single-token routing uses the cooperative fast path

QSA:
    persistent device chain state
    geometric capacity growth only when required > capacity

diagnostics:
    q38 production and q38-diag are logically separate build modes
```

Do not overwrite this Q2 reference while M9-NVIDIA is under construction.

---

# 2. Confirmed facts about the official NVIDIA checkpoint

Repository:

```text
nvidia/Qwen3.8-Flash-Next-NVFP4
```

The public repository is approximately **133 GB decimal** and currently contains:

```text
10 main safetensors shards

model-00001-of-00010.safetensors
...
model-00010-of-00010.safetensors

plus:
model-fp8-mtp-ple.safetensors
```

The large `model-fp8-mtp-ple.safetensors` sidecar is approximately **53.7 GB decimal**.

The main shards are approximately **78.9 GB decimal total**.

This split is extremely favorable for q38 because PLE is already a natural separate storage domain.

NVIDIA describes the checkpoint as mixed precision:

```text
main-language-model routed MoE expert linear layers:
    W4A4 NVFP4
    MSE-calibrated weight scales

main-model attention:
    BF16

main-model shared experts:
    BF16

other main-model layers:
    BF16

MTP routed experts:
    128x128 block-scaled FP8

PLE n-gram embedding:
    per-tensor FP8
```

The NVIDIA model card states that the complete MTP module and PLE n-gram embedding are taken from the official Qwen FP8 checkpoint, while the main-model routed experts are the NVFP4 component.

NVIDIA Model Optimizer version associated with the checkpoint:

```text
ModelOpt 0.46.0
```

The model card identifies vLLM as the supported runtime and provides a specific minimum upstream compatibility point for serving. For q38, vLLM/ModelOpt are therefore useful **semantic references**, not implementation dependencies.

The model retains:

```text
native context:
    262,144

extended context:
    up to 1,000,000 using YaRN

architecture:
    qwen4_exp
```

M9-NVIDIA does **not** enable 1M context yet; it merely preserves a memory layout that makes that later milestone feasible.

---

# 3. NVFP4 numerical contract

NVFP4 must not be treated as "generic INT4".

The NVIDIA NVFP4 representation uses:

```text
value:
    FP4 E2M1

block scale:
    FP8 E4M3-class scale

global/tensor scale:
    FP32

nominal block:
    16 consecutive values
```

Conceptually:

```text
x ≈ fp4_e2m1_value
    * block_scale
    * global_scale
```

The actual exported ModelOpt checkpoint may also carry activation/input scale metadata.

The exact safetensors tensor shapes, transpositions, suffixes and scale layout **must be discovered from the actual NVIDIA checkpoint** before writing production loader assumptions.

Expected ModelOpt concepts include tensors corresponding to:

```text
weight
weight_scale
weight_scale_2
input_scale
```

but M9-NVIDIA must not hard-code suffix/shape assumptions based only on documentation.

The first milestone is an exact **format inventory**.

---

# 4. Critical warning: do not assume the scale layout

NVIDIA's current ModelOpt exporter supports multiple NVFP4 scale/layout forms, including transformations intended for cuBLAS block-scaled GEMM.

Therefore:

```text
DO NOT assume:
    weight_scale is already cuBLASLt-swizzled

DO NOT assume:
    weight_scale is flat 1D

DO NOT assume:
    all projections share the same scale layout

DO NOT assume:
    weight is stored in q38's preferred matrix orientation

DO NOT assume:
    all input_scale tensors are scalar or identical

DO NOT assume:
    a safetensors shape equals the logical [out, in] shape after packing
```

Derive the ABI from:

1. actual safetensors headers;
2. `config.json`;
3. `hf_quant_config.json`;
4. `model.safetensors.index.json`;
5. ModelOpt 0.46 export code;
6. a tiny independent dequantization oracle.

No production CUDA work starts before this contract is frozen.

---

# 5. Why M9-NVIDIA is plausible on one Spark

Approximate disk-size split:

```text
main shards:
    ~78.9 GB decimal
    ~73.5 GiB

MTP + PLE sidecar:
    ~53.7 GB decimal
    ~50.0 GiB
```

The PLE portion remains file-backed and is not counted as fully resident memory.

This means the core main model is in the range where a Spark can plausibly keep it resident while leaving substantial headroom under:

```text
Q38_MEMORY_HARD_LIMIT = 118 GiB
```

The actual runtime footprint must be measured, not inferred from file size.

The target is:

```text
main model + CUDA context + persistent workspaces + active state
    comfortably below 114 GiB

hard failure before:
    118 GiB
```

For the first M9-NVIDIA text bring-up, MTP may remain unloaded.

This is deliberate: first prove the main NVFP4 model, then add the draft head.

---

# 6. M9-NVIDIA scope

## Included

M9-NVIDIA includes:

```text
official NVIDIA checkpoint inspection
ModelOpt NVFP4 tensor ABI discovery
native q38 checkpoint ingestion
BF16 non-expert tensor ingestion
NVFP4 routed expert ingestion
FP8 PLE file-backed ingestion
memory-safe resident loading
independent NVFP4 oracle
single-expert fixture
single-layer routed-MoE fixture
device-chain integration
text-only full-model smoke
memory measurement
performance baseline
```

## Explicitly excluded from the first bring-up

Do not mix these into the initial implementation:

```text
MTP speculative decode
1M YaRN
FP8 QSA KV cache
SSD QSA KV tier
SSD expert paging
multimodal / vision
server work
new concurrency work
new Q2 optimization
generic safetensors runtime
generic ModelOpt support
support for other NVFP4 models
```

Those become follow-up stages only after the base NVIDIA arm is correct.

---

# 7. Architecture decision: import once, keep runtime minimal

q38 should **not** become a generic Hugging Face/safetensors engine.

Preferred architecture:

```text
Official NVIDIA HF checkpoint
        |
        v
offline inspect/import tool
        |
        v
q38-native tensor manifest / pack
        |
        +---- main BF16 tensors
        |
        +---- raw NVFP4 expert payloads + scales
        |
        +---- PLE external file descriptor/offset metadata
        |
        v
minimal q38 runtime loader
```

Recommended tools:

```text
tools/q38_nvfp4_inspect.py
tools/q38_nvfp4_pack.py
```

The Python tools are development/import utilities, not production runtime dependencies.

The q38 executable remains C/CUDA.

---

# 8. Native pack requirements

The pack/import stage must preserve exact quantized bytes whenever possible.

For routed experts, preserve:

```text
packed FP4 nibbles
per-block scale bytes
per-tensor/global scale
input-scale metadata
logical matrix shape
stored matrix shape
orientation/transposition metadata
layer
expert ID
projection type
```

Do not dequantize expert weights during import.

Do not requantize NVIDIA weights into q38 Q4.

The entire point of M9-NVIDIA is to preserve NVIDIA's calibrated NVFP4 checkpoint.

For BF16 tensors, direct copy/reindex is acceptable.

For PLE:

```text
do not copy 50 GiB into another resident representation
```

Prefer recording the exact backing safetensors file and tensor data offset, or creating a dedicated file-backed PLE pack without ever materializing the full table in memory.

---

# 9. PLE invariant for M9-NVIDIA

PLE is permanently file-backed.

This is non-negotiable.

The NVIDIA checkpoint carries the PLE table in FP8. M9-NVIDIA must support this precision without loading the full table into unified memory.

Required architecture:

```text
PLE backing file on NVMe
        |
        v
bounded async read / mmap access
        |
        v
small reusable staging buffers
        |
        v
decode selected PLE rows
        |
        v
existing early async PLE overlap
        |
        v
layer-2 injection
```

Do not:

```text
cudaMalloc the entire PLE tensor
malloc the entire PLE tensor
pin the entire PLE tensor
copy the entire PLE tensor into a temporary conversion file at runtime
```

Any offline conversion must itself be streaming and bounded.

The current successful invariant:

```text
PLE critical wait at injection ≈ 0 on warm decode
```

must remain a target after the NVFP4 arm is functional.

---

# 10. MTP handling during initial bring-up

The official NVIDIA sidecar also contains MTP tensors.

Initial M9-NVIDIA policy:

```text
parse/index MTP tensors:
    yes

validate their metadata:
    yes

load them resident:
    no, unless needed by a loader dependency

execute MTP:
    no
```

Reason:

```text
main-model correctness must be established first
```

The NVIDIA model card states that MTP routed experts use a different quantization regime:

```text
128x128 block-scaled FP8
```

That is a separate kernel/format contract and should not contaminate the first NVFP4 MoE bring-up.

---

# 11. Memory architecture

## 11.1 No transient model duplication

The loader must never require:

```text
packed model resident
+
dequantized model resident
```

at the same time.

Likewise, no whole-shard read buffers.

Use bounded streaming:

```text
file
 -> bounded staging
 -> final resident allocation
```

Recommended staging ceiling:

```text
64–128 MiB
```

Do not use `largest_tensor` as the staging allocation size.

The previously established memory-budget layer must reserve bytes before allocation.

## 11.2 Suggested bring-up memory thresholds

Before enabling long context:

```text
soft runtime memory ceiling:
    114 GiB

hard runtime memory ceiling:
    118 GiB
```

For the initial NVFP4 model load, a healthy target is substantially below the soft ceiling.

Do not define success as "it technically reaches 117.9 GiB".

We want headroom for:

```text
QSA state
GDN state
CUDA context
workspaces
future FP8 KV
MTP
server sessions
```

## 11.3 Memory accounting categories

At minimum report:

```text
resident main-model BF16 bytes
resident routed-expert FP4 payload bytes
resident expert scale bytes
persistent CUDA workspace bytes
QSA state bytes
GDN state bytes
GR workspace bytes
MoE workspace bytes
PLE staging bytes
MTP resident bytes
total accounted bytes
peak accounted bytes
RSS
```

PLE backing-file size is reported separately as:

```text
file_backed_ple_bytes
```

It is not equated to resident RAM.

---

# 12. Loader implementation sequence

## M9-NVIDIA-N0 — Freeze current Q2 arm

No new benchmark required.

Record:

```text
Q2_DEVICE_CHAIN_V1
76.282 ms/token
13.109 tok/s
```

Tag or document the exact commit.

Do not modify Q2 kernels while the NVIDIA arm is being brought up.

---

## M9-NVIDIA-N1 — Metadata-only checkpoint audit

Download/inspect only metadata first.

Required files:

```text
README.md
config.json
hf_quant_config.json
model.safetensors.index.json
```

Do not start by downloading or parsing 133 GB of tensor payload.

Produce:

```text
docs/m9-nvidia/NVFP4_CHECKPOINT_CONTRACT.md
artifacts/m9-nvidia/nvfp4_tensor_inventory.json
```

Inventory every logical tensor with:

```text
tensor name
source shard
dtype
stored shape
logical shape if inferable
component class
layer
expert ID
projection
precision class
resident/file-backed/skipped
associated scale tensor(s)
```

Summaries required:

```text
total tensors
BF16 tensors
NVFP4 weight tensors
NVFP4 block-scale tensors
NVFP4 global-scale tensors
activation/input-scale tensors
FP8 PLE tensors
FP8 MTP tensors
other tensors
unknown tensors
```

Acceptance:

```text
unknown tensor class = 0
```

before production import code is written.

---

## M9-NVIDIA-N2 — Inspect actual safetensors headers

Inspect at least one real NVFP4 expert tensor group from an early layer.

Obtain actual:

```text
weight dtype
weight stored shape
weight byte count

weight_scale dtype
weight_scale shape
weight_scale byte count

weight_scale_2 dtype
weight_scale_2 shape

input_scale dtype
input_scale shape
```

Repeat on:

```text
early layer
middle layer
late layer

gate projection
up projection
down projection
```

Do not assume all three projections have identical exported scale shapes.

Produce a frozen table in:

```text
docs/m9-nvidia/NVFP4_CHECKPOINT_CONTRACT.md
```

---

## M9-NVIDIA-N3 — Independent CPU/Python NVFP4 oracle

Create a small test-only decoder independent of q38 CUDA.

Recommended location:

```text
tests/reference/nvfp4_modelopt_reference.py
```

It must implement or call a trusted implementation of:

```text
FP4 E2M1 unpack
nibble ordering
FP8 scale decode
global scale multiplication
logical matrix reconstruction
activation quantization semantics
```

First gate:

```text
dequantized selected weight values
```

must match ModelOpt's own dequantized tensor for a small sample.

This test is about format semantics, not performance.

Freeze three real fixtures:

```text
early expert/projection
middle expert/projection
late expert/projection
```

Each fixture contains only the small slices needed to validate the ABI.

Do not build fixtures by loading the complete model into RAM.

---

# 13. NVFP4 expert execution strategy

The main model routed experts are:

```text
512 experts/layer
top-10 routed experts/token
expert FFN = 640
hidden = 2560
48 decoder layers
```

The q38 production decoder is single-token optimized.

Therefore a generic large-GEMM implementation is not automatically the best decode implementation.

M9-NVIDIA uses two stages.

---

## 13.1 Correctness kernel first

Implement one q38 CUDA primitive for the routed experts that can consume the exact NVIDIA packed representation directly.

Conceptual interface:

```c
q38_moe_cuda_nvfp4_decode(...)
```

Inputs include:

```text
device hidden BF16/FP32 as required
selected expert IDs
route weights

packed gate/up/down NVFP4 weights
block scales
global scales
input scale metadata
```

The first implementation may prioritize correctness over speed.

Requirements:

```text
no persistent BF16 dequantized expert copy
no per-token cudaMalloc
no host repacking
no per-token expert weight copies
resident packed weights
device selected IDs
deterministic weighted accumulation
```

The existing successful MOE_OPT_V1 organization should be reused conceptually:

```text
grouped/indexed routed execution
parallel down/weighted accumulation
persistent workspaces
device routing
```

Do not fall back to one host loop per expert.

---

## 13.2 Preserve W4A4 semantics

The official checkpoint is W4A4, not merely W4A16.

Therefore the first semantically authoritative path must model the activation quantization correctly.

Do not silently run:

```text
BF16 activation × dequantized W4 weight
```

and call it "the NVIDIA checkpoint".

That may be useful as a diagnostic oracle, but it is not the final quantized execution contract.

Freeze activation-quantization behavior using ModelOpt reference fixtures.

---

# 14. Production-kernel direction on GB10

GB10 is Blackwell SM121 and supports FP4 instructions.

Current CUDA/cuBLASLt documentation supports:

```text
FP4 E2M1
16-element block scaling
E4M3-class block scales
FP32 compute accumulation
Blackwell compute capability >= 10
```

However, q38 decode has atypical MoE shapes:

```text
single token
10 selected experts
hidden 2560
expert width 640
```

Therefore do not assume a conventional cuBLASLt GEMM will win for decode.

Candidate ladder after correctness:

```text
NV-C1:
    direct packed NVFP4 cooperative decode kernel
    extend the existing q38 grouped-indexed MoE structure

NV-C2:
    fuse gate + up read/dequant/compute where practical

NV-C3:
    fuse activation quantization with FC1 input preparation

NV-C4:
    native Blackwell FP4 MMA/tensor-core path for decode shapes

NV-C5:
    cuBLASLt FP4 path where matrix geometry is favorable,
    especially prefill/batched execution
```

A cuBLASLt prototype is valuable as a performance reference.

Do not make TensorRT a q38 dependency merely to obtain an FP4 kernel.

Public NVIDIA/TensorRT work shows that SM121 needs GB10-appropriate kernel geometry and that cuBLASLt FP4 has worked on GB10 even where B200-oriented CUTLASS tiles exceeded GB10 shared-memory limits.

That is an important implementation clue:

```text
do not reuse B200 tile geometry blindly on SM121
```

---

# 15. Exact numerical gates

## 15.1 Raw format gate

For sampled weights:

```text
q38 unpacked E2M1 values
    ==
reference unpacked E2M1 values

q38 block scales
    ==
reference block scales

q38 global scale
    ==
reference global scale
```

Nibble-level packing must be exact.

## 15.2 Dequantization gate

For selected slices:

```text
q38 dequantized weights
    match ModelOpt dequantized weights
```

No "looks close" acceptance.

Any discrepancy must be explained by an explicitly documented dtype conversion.

## 15.3 Activation quantization gate

For fixed real activation fixtures:

```text
packed activation bytes
block scales
global/input scale semantics
```

must match the chosen authoritative ModelOpt reference.

## 15.4 Expert output gate

Compare:

```text
gate projection
up projection
SiLU/multiply
down projection
route-weight application
```

against the independent reference.

The tolerance must be defined **before** looking at the candidate result.

Preferred rule:

```text
packed quantized inputs/scales:
    exact

routing IDs:
    exact

BF16 output:
    within the expected BF16/FP32 accumulation rounding envelope
```

Record:

```text
max_abs
RMSE
NaN count
Inf count
```

---

# 16. M9-NVIDIA-N4 — Native pack/importer

After the contract is frozen, implement:

```text
tools/q38_nvfp4_pack.py
```

Responsibilities:

```text
read HF safetensors metadata
stream tensor payloads
classify tensors against the frozen contract
preserve raw NVFP4 data
copy BF16 tensors without precision conversion
record PLE source location
optionally skip MTP payload for the first arm
write q38-native manifest
write model fingerprint
```

The importer must be deterministic.

Two imports of the same source checkpoint should produce the same manifest and payload hashes.

Suggested metadata:

```text
source_repo
source_revision
source_file_sha256
model_type
model_fingerprint
quantization = NVIDIA_MODELOPT_NVFP4
group_size = 16
PLE precision
MTP precision
tensor inventory hash
```

---

# 17. M9-NVIDIA-N5 — Load-only residency milestone

Before inference, prove that the native pack can be loaded.

This stage performs:

```text
runtime init
main-model residency
PLE backing-source open
workspace initialization
runtime destroy
```

No generation required.

Acceptance:

```text
all expected non-PLE tensors bound
all NVFP4 expert tensors bound
all scale tensors bound
PLE source valid
MTP intentionally skipped
unknown tensor = 0
missing required tensor = 0

peak memory < 118 GiB
steady memory materially below 114 GiB
process exits cleanly
```

The loader must not create a second dense representation of the expert weights.

---

# 18. M9-NVIDIA-N6 — Single-expert real fixture

Use one real routed expert from the NVIDIA checkpoint.

Test:

```text
input hidden fixture
selected expert
gate
up
SiLU/mul
down
```

Compare q38 CUDA vs independent ModelOpt reference.

Run for:

```text
early layer
middle layer
late layer
```

No full model.

This is the first production CUDA correctness gate.

---

# 19. M9-NVIDIA-N7 — Complete routed-MoE fixture

Use a real layer fixture with:

```text
router output
top-10 expert IDs
route weights
10 selected NVFP4 experts
shared expert BF16
deterministic reduction
```

Validate:

```text
expert IDs exact
route weights equivalent
routed output
shared output
combined MoE output
```

The router remains BF16 / current sensitive precision.

Do not quantize the router merely because the experts are NVFP4.

---

# 20. M9-NVIDIA-N8 — Device decoder-chain integration

Only after N7 is green.

Replace the routed-expert execution inside the current production device chain with:

```text
q38_moe_cuda_nvfp4_decode
```

Do not create another full decoder orchestration path.

Production call chain remains conceptually:

```text
q38
 -> q38_runtime
 -> q38_session
 -> q38_forward_token
 -> decoder_layer_chain
 -> existing GR/GDN/QSA
 -> NVIDIA NVFP4 routed MoE
```

The NVIDIA arm should differ in tensor format/kernel selection, not in the entire forward architecture.

This prevents a repeat of the earlier duplicated-QSA-path bug class.

---

# 21. First full-model run

Only after:

```text
checkpoint contract frozen
importer complete
load-only green
single-expert fixture green
full MoE fixture green
device-chain integration compiled
```

Run one normal text generation.

Use:

```text
MTP off
vision off
native context only
PLE file-backed
device decoder chain on
greedy decoding
```

Record:

```text
load seconds
steady memory GiB
peak memory GiB
prompt token count
generated token IDs
ms/token
tok/s
PLE critical stall
```

This run is not yet the final performance benchmark.

Its purpose is:

```text
does the official NVIDIA arm work end-to-end?
```

---

# 22. Semantic reference for the first full-model gate

Because the NVIDIA checkpoint is officially distributed for ModelOpt/vLLM, the preferred semantic hierarchy is:

```text
1. actual NVIDIA checkpoint bytes
2. ModelOpt 0.46 tensor/dequant semantics
3. compatible vLLM execution as full-model reference
4. q38 implementation
```

Do not use llama.cpp as the ABI oracle for this arm.

For deterministic comparison, use:

```text
same tokenizer assets
same input token IDs
greedy decoding
MTP disabled
same RoPE mode
same native context
same chat formatting or raw token input
```

The strongest test is raw-token continuation rather than prompt-template comparison.

---

# 23. Performance policy

Do not optimize before correctness.

Once the NVIDIA arm is correct, freeze:

```text
NVFP4_REFERENCE_0
```

with:

```text
load time
memory
decode ms/token
tok/s
```

Then profile only the NVFP4 production path.

The current Q2 result:

```text
13.109 tok/s
```

is a useful engineering comparison but **not** a correctness or promotion requirement.

NVFP4 is a higher-quality model representation.

A small speed loss may still be acceptable if quality and memory objectives improve.

The engineering target, however, should be to recover or exceed Q2 throughput by exploiting native Blackwell FP4.

---

# 24. Why we should not immediately add SSD expert paging

The NVFP4 main-model body is small enough that full resident experts are likely viable.

Therefore M9-NVIDIA first tries:

```text
all main-model routed experts resident
PLE only on SSD
```

This gives:

```text
predictable latency
no expert-page misses
simpler CUDA scheduling
best opportunity for native FP4 throughput
```

Only if measured memory pressure later prevents 1M context/MTP should we evaluate:

```text
hot expert resident set
cold expert NVMe tier
```

That is a separate milestone and must be driven by measured router hit-rate distributions.

---

# 25. Preparing for 1M context without implementing it yet

M9-NVIDIA should preserve enough memory headroom for a later compact QSA cache.

Qwen's architecture is favorable:

```text
36 GDN layers:
    recurrent state, not linear full KV growth

12 QSA layers:
    context-dependent state
```

The future direction is:

```text
QSA index resident
recent/hot QSA KV resident
compact BF16 or FP8 KV representation
optional SSD cold tier only if necessary
```

Do not add YaRN or SSD KV in the initial M9-NVIDIA implementation.

---

# 26. Follow-up milestone after NVFP4 base bring-up

Once `NVFP4_REFERENCE_0` exists:

```text
M9-NVIDIA-A:
    NVFP4 resident main model + file-backed PLE

M9-NVIDIA-B:
    native SM121 FP4 MoE optimization

M9-NVIDIA-C:
    FP8 QSA KV cache

M9-NVIDIA-D:
    1M YaRN context

M9-NVIDIA-E:
    MTP FP8 draft head

M9-NVIDIA-F:
    only if needed: SSD QSA KV cold tier

M9-NVIDIA-G:
    only if needed: SSD cold-expert tier
```

The sequence can change based on measured memory and performance after A/B.

---

# 27. Source-tree discipline

Do not proliferate files or duplicated backends.

Preferred additions:

```text
tools/q38_nvfp4_inspect.py
tools/q38_nvfp4_pack.py

tests/reference/nvfp4_modelopt_reference.py
tests/nvfp4/

cuda/q38_moe_nvfp4_cuda.cu
cuda/q38_moe_nvfp4_cuda.h

docs/m9-nvidia/
```

If the existing MoE CUDA file is the natural place for the new kernel, prefer consolidation rather than creating many tiny translation units.

Do not create:

```text
q38_forward_nvfp4.c
q38_forward_nvfp4_diag.c
q38_forward_nvfp4_old.c
```

There remains one forward architecture.

---

# 28. Build policy

Production:

```text
q38
```

must use the same maximum-performance production build policy already established.

Diagnostic:

```text
q38-diag
```

may report NVFP4 tensor format and memory information, but this milestone should not add per-stage timing overhead merely for bring-up.

The import/reference Python scripts are not linked into q38.

---

# 29. Failure modes to explicitly guard against

## Format

```text
wrong FP4 nibble order
wrong E2M1 lookup
treating FP4 as signed INT4
wrong E4M3 scale decode
missing weight_scale_2/global scale
wrong input_scale semantics
wrong block grouping
wrong logical transpose
wrong scale tensor layout
using a cuBLAS-swizzled assumption on an unswizzled export
```

## Memory

```text
loading PLE resident
whole-shard temporary buffers
persistent dense dequantized experts
duplicated resident weights
unbounded pinned staging
runtime conversion of the full checkpoint
memory accounting bypass
```

## Execution

```text
W4 weights but BF16 activations while claiming W4A4
host expert repacking per token
per-token cudaMalloc/free
one kernel launch per expert when a grouped path is available
loss of deterministic route reduction
router quantization
shared expert accidentally treated as NVFP4
MTP expert quant format confused with main-model expert quant format
```

## Architecture

```text
second full decoder path
generic backend abstraction
safetensors parser spread throughout runtime
new CPU fallback
new non-GB10 backend
```

---

# 30. Artifact requirements

During M9-NVIDIA create:

```text
docs/m9-nvidia/M9-NVIDIA_NVFP4_BRINGUP.md
docs/m9-nvidia/NVFP4_CHECKPOINT_CONTRACT.md
docs/m9-nvidia/NVFP4_MEMORY_BUDGET.md

artifacts/m9-nvidia/nvfp4_tensor_inventory.json
artifacts/m9-nvidia/nvfp4_load_only.json
artifacts/m9-nvidia/nvfp4_fixture_early.json
artifacts/m9-nvidia/nvfp4_fixture_middle.json
artifacts/m9-nvidia/nvfp4_fixture_late.json
artifacts/m9-nvidia/nvfp4_reference_0.json
```

Do not generate performance JSON before there is a real meaningful measurement.

---

# 31. Acceptance criteria for M9-NVIDIA base milestone

M9-NVIDIA-A is complete only when all of the following are true:

```text
[ ] official NVIDIA checkpoint ABI inventoried
[ ] unknown tensor classes = 0
[ ] NVFP4 raw bytes/scales understood
[ ] independent ModelOpt reference exists
[ ] q38 exact unpack/dequant fixture passes
[ ] activation-quantization fixture passes
[ ] routed expert fixture passes
[ ] top-10 MoE layer fixture passes
[ ] main model loads without dense expert expansion
[ ] PLE remains file-backed
[ ] MTP remains disabled for first arm
[ ] resident/peak memory stays under hard limit
[ ] device decoder chain remains the single production orchestration
[ ] end-to-end text generation succeeds
[ ] no NaN/Inf
[ ] first NVFP4 reference performance is recorded
```

No long-context work begins before these boxes are satisfied.

---

# 32. Exact instructions for the first work session

The first session is **inspection only**.

Do not write CUDA yet.

Do not modify the decoder chain yet.

Do not download all model payloads unless they are already local.

Do:

```text
1. create docs/m9-nvidia/

2. freeze current q38 commit and Q2_DEVICE_CHAIN_V1 reference

3. obtain NVIDIA metadata:
   README.md
   config.json
   hf_quant_config.json
   model.safetensors.index.json

4. parse the index and classify every tensor name

5. identify:
   - all main-model routed-expert tensors
   - all expert weight_scale tensors
   - all expert weight_scale_2 tensors
   - all expert input_scale tensors
   - all BF16 non-expert tensors
   - PLE tensors
   - MTP tensors
   - vision tensors

6. compute byte totals by category

7. inspect real safetensors headers for:
   early/middle/late expert tensors

8. write NVFP4_CHECKPOINT_CONTRACT.md

9. stop
```

Required report before proceeding:

```text
main-shard total bytes
PLE/MTP-sidecar total bytes

main NVFP4 weight bytes
main NVFP4 scale bytes
main BF16 bytes

number of expert weight tensors
number of weight_scale tensors
number of weight_scale_2 tensors
number of input_scale tensors

exact stored shape/dtype for:
    gate
    up
    down

exact PLE tensor name/dtype/shape

exact MTP quantization classes

unknown/unclassified tensors
```

If `unknown/unclassified tensors != 0`, do not start the importer.

---

# 33. Suggested metadata download commands

One possible metadata-only approach with `huggingface_hub`:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/Qwen3.8-Flash-Next-NVFP4",
    local_dir="models/Qwen3.8-Flash-Next-NVFP4",
    allow_patterns=[
        "README.md",
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
    ],
)
PY
```

Do not add `*.safetensors` to the first metadata-only download.

Once the tensor contract is frozen, fetch the required payloads.

For the eventual exact NVIDIA arm, keep:

```text
model-fp8-mtp-ple.safetensors
```

on NVMe because it contains the official FP8 PLE data.

---

# 34. Suggested first commit sequence

Keep commits independently reviewable:

```text
M9N-01 Inspect NVIDIA NVFP4 checkpoint metadata

M9N-02 Add independent ModelOpt NVFP4 reference fixtures

M9N-03 Add q38 native NVFP4 pack importer

M9N-04 Load NVIDIA mixed-precision model without execution

M9N-05 Add correct NVFP4 routed-expert CUDA path

M9N-06 Integrate NVFP4 MoE into device decoder chain

M9N-07 Freeze NVIDIA NVFP4 Reference 0
```

Do not combine all of M9-NVIDIA into one giant commit.

---

# 35. Promotion rules

At every stage:

```text
format correctness
    before
kernel correctness
    before
whole-layer correctness
    before
full-model correctness
    before
performance optimization
```

Performance candidates are promoted only if they preserve the frozen NVFP4 fixture semantics.

The Q2 path is not modified to make the NVIDIA path look better.

---

# 36. Long-term target

The intended final Spark configuration is:

```text
Qwen3.8-Flash-Next

main routed experts:
    NVIDIA NVFP4

router:
    BF16

attention / GDN / GR / shared expert:
    NVIDIA checkpoint precision, initially BF16

PLE:
    official FP8
    permanently NVMe/file-backed
    async overlapped

QSA state:
    compact resident representation
    later FP8 if validated

context:
    native 262K first
    1M YaRN later

MTP:
    official FP8 block-scaled draft head later

decoder:
    q38 device-resident chain

memory:
    soft 114 GiB
    hard 118 GiB
```

This configuration optimizes for **model quality first**, then uses GB10-specific execution and the memory hierarchy to recover throughput.

---

# 37. External technical references

Official NVIDIA checkpoint:

https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4

Official NVIDIA model card / mixed-precision description:

https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/main/README.md

Official Qwen FP8 model / architecture / 1M YaRN guidance:

https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8

NVIDIA Transformer Engine NVFP4 format:

https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html

NVIDIA cuBLAS block-scaled FP4 documentation:

https://docs.nvidia.com/cuda/cublas/index.html

NVIDIA Model Optimizer exporter:

https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/export/unified_export_hf.py

NVIDIA ModelOpt NVFP4 tensor implementation:

https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py

Relevant SM121 / GB10 FP4 implementation evidence:

https://github.com/NVIDIA/TensorRT-LLM/issues/11368

---

# 38. Final engineering rule

M9-NVIDIA is successful only if q38 consumes the NVIDIA checkpoint **as the NVIDIA checkpoint actually is**.

We are not trying to turn NVFP4 into q38 Q4.

We are not trying to make a universal quantization runtime.

We are trying to build the smallest possible GB10-native path for:

```text
Qwen3.8-Flash-Next
+
NVIDIA ModelOpt NVFP4 routed experts
+
BF16 sensitive core
+
FP8 file-backed PLE
+
device-resident decoder
```

Once this base arm is stable, it becomes the foundation for the real next objective:

```text
maximum-quality Qwen3.8-Flash-Next
on one DGX Spark
with 1M context
and MTP
without crossing 118 GiB.
```
