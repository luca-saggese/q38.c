# q38.c — Corrective Code Review

Branch reviewed: `qwen38-spark-proto`

Scope of this review:
- current implementation through early M3;
- Qwen3.8-Flash-Next only;
- DGX Spark / GB10 only;
- CUDA runtime;
- q38 GGUF intentionally generated from the original Qwen3.8 checkpoint tensor layout, according to the project specification.

This review is intended as a corrective gate before continuing deeper into M3.

---

## Executive summary

The project direction is sound, but several issues should be corrected before continuing with optimized GDN/CUDA work.

The most important findings are:

| Priority | Finding | Required action |
|---|---|---|
| P0 | Gated Residual math does not match verified Qwen4Exp semantics | Fix C reference and CUDA implementation |
| P0 | Current GR tests can pass while GR math is wrong | Add independent non-zero golden tests |
| P0 | GDN recurrent/session state is effectively sized for one GDN layer | Allocate independent state for all 36 GDN layers |
| P0/P1 | Gated Residual is modeled as persistent session state | Move GR to per-forward/per-token activation workspace |
| P1 | Tokenizer is currently a Python/reference compatibility bridge rather than a native runtime tokenizer | Keep it as oracle for now, but implement native tokenizer before final end-to-end runtime |
| P2 | PLE layer-count/binding condition appears off by one | Fix and add boundary tests |
| P2 | Quant artifact naming should be tied more strongly to the quant manifest | Add manifest hash / recipe metadata |

**Do not treat the GGUF layout as an issue.**

The q38 GGUF is intentionally generated from the original Qwen3.8 tensors and is a q38-specific runtime artifact. `llama.cpp` is used as a semantic/reference implementation, not as the binary tensor-layout ABI.

---

# 1. P0 — Gated Residual math is inconsistent with the verified Qwen4Exp semantics

## Files to inspect

At minimum:

- `q38_gr_ref.c`
- `q38_gr.cu`
- `tests/test_m3_gr_ref.c`
- `tests/test_m3_gr_cuda.cu`
- `docs/qwen_gdn_semantics.md`

## Problem

The verified Qwen4Exp Gated Residual semantics use a low-rank gated mixing path with:

1. division by `hc_count` before the activation;
2. SiLU activation;
3. sigmoid gating;
4. a write/injection gate using `2 * sigmoid(x / hc_count)`.

For Qwen3.8:

```text
hc_count = 4
```

The current reference implementation appears to use an ELU-like expression:

```c
value > 0.0f ? value : value * expf(value)
```

This is **not SiLU**.

SiLU is:

```c
silu(x) = x * sigmoid(x)
        = x / (1 + exp(-x))
```

The current implementation also appears to omit the required division by `hc_count` before:
- the read-side activation;
- the write/injection sigmoid.

If the implementation currently resembles:

```c
bottleneck[rank] =
    value > 0.0f ? value : value * expf(value);
```

and:

```c
scale = 2.0f * sigmoid(value);
```

then both operations need correction according to the frozen Qwen4Exp semantics.

## Expected conceptual form

The exact variable naming can differ, but the math should be equivalent to:

```c
float x = low_rank_value / (float) Q38_HC_COUNT;
float activated = x / (1.0f + expf(-x));
```

and for injection/write gating:

```c
float scale =
    2.0f / (1.0f + expf(-(inject_value / (float) Q38_HC_COUNT)));
```

The implementation must follow the equations frozen in `docs/qwen_gdn_semantics.md`, not this review text if the frozen reference is more precise.

## Why this is critical

A wrong GR transform affects every layer and therefore contaminates:

- GDN inputs;
- QSA inputs;
- MoE inputs;
- hidden-state trajectories;
- all future golden vectors;
- eventual generated logits.

This is a high-amplification architectural error.

## Required fix

- Correct the scalar/reference implementation first.
- Correct CUDA only after the scalar path has independent golden coverage.
- Do not update expected golden outputs merely to make the new implementation pass.
- Regenerate goldens only from an external verified reference.

## Required tests

Add at least:

### GR-T01 — zero sanity

Useful only as a smoke test.

### GR-T02 — deterministic non-zero matrices

Use:
- non-zero `down`;
- non-zero `up`;
- non-zero `inject`;
- deterministic hidden input;
- both positive and negative intermediate activations.

Expected result must come from the frozen upstream/reference implementation, not from `q38_gr_ref.c`.

### GR-T03 — negative SiLU coverage

Construct inputs where the low-rank preactivation contains negative values.

This test must distinguish SiLU from ELU-like behavior.

### GR-T04 — hc scaling coverage

Construct a case where omitting `/4` produces a clearly different result.

### GR-T05 — CUDA vs external golden

CUDA should be checked against the external golden, not only against q38 scalar code.

---

# 2. P0 — Current GR tests are not independent enough

## Problem

Current tests appear to initialize important GR matrices such as:

- `down`
- `up`
- `inject`

to zero or near-trivial values.

When the matrices are zero:

```text
low-rank output = 0
sigmoid(0) = 0.5
injection gate = neutral/simple
```

This means radically different nonlinearities can produce the same test result.

As a result:

```text
wrong scalar implementation
==
wrong CUDA implementation
```

can still produce a green test suite.

## Required action

Keep the current zero test as a smoke test, but add external-golden non-zero tests as described in Finding #1.

## Acceptance criterion

The GR gate is considered reliable only when:

```text
external reference
    ==
q38 scalar reference
    ==
q38 CUDA
```

within the defined numeric tolerance.

CUDA-vs-q38-scalar alone is not sufficient.

---

# 3. P0 — GDN recurrent state must exist independently for every GDN layer

## Files to inspect

At minimum:

- `q38_state.h`
- `q38_state.c`
- `tests/test_m3_state.c`
- future `q38_forward.*`

## Problem

The current session/state structures appear to represent one recurrent GDN state plus one convolution history:

```c
q38_gdn_state_desc recurrent;
q38_conv_history_desc conv_history;
```

and storage similar to:

```c
float *recurrent_state;
float *conv_history;
```

This is sufficient for a **single-layer GDN prototype**, but not for the complete model.

Qwen3.8-Flash-Next has:

```text
48 total decoder layers
36 GDN layers
12 QSA/full-attention layers
```

Every GDN layer needs an independent:

- recurrent matrix state;
- convolution history.

State cannot be shared between GDN layers.

## Expected architecture

Prefer an explicit model-layer to GDN-state-slot mapping.

For example:

```c
#define Q38_N_LAYER      48
#define Q38_N_GDN_LAYER  36

typedef struct {
    float *recurrent;
    float *conv_history;
} q38_gdn_layer_state;

typedef struct {
    q38_gdn_layer_state gdn[Q38_N_GDN_LAYER];
    ...
} q38_session_state;
```

And a fixed table:

```c
static const int8_t q38_layer_to_gdn_slot[Q38_N_LAYER] = {
    0, 1, 2, -1,
    3, 4, 5, -1,
    ...
};
```

Do not rely on implicit arithmetic later if a compile-time mapping table is clearer and testable.

## Memory accounting

The session-memory report must account for all 36 states.

Do not leave tests asserting a total persistent GDN allocation corresponding to only one layer.

## Required tests

### STATE-T01 — unique state per layer

Modify layer A state and verify layer B state remains byte-identical.

### STATE-T02 — layer mapping

Verify every model layer maps to:
- a unique GDN slot for GDN layers;
- `-1` for QSA layers.

### STATE-T03 — reset

Reset must zero/reset all 36 GDN states and all convolution histories.

### STATE-T04 — memory accounting

Measured persistent state bytes must equal the sum of all per-layer GDN state allocations.

### STATE-T05 — sequential isolation

Run the same synthetic update through two different GDN layer slots and verify their histories evolve independently.

---

# 4. P0/P1 — Gated Residual should not be persistent session state

## Files to inspect

At minimum:

- `q38_state.h`
- `q38_state.c`
- future forward graph / workspace allocator
- checkpoint/session code when introduced

## Problem

The current state model appears to treat GR branch values as session-owned persistent state.

Conceptually, Qwen4Exp Gated Residual is part of the **current forward activation stream**.

It is not analogous to:
- GDN recurrent state;
- convolution history;
- QSA K/V cache;
- QSA index state.

At the start of a forward sequence/token block, the wide HC residual is derived from the current input activations and then flows through the layers.

It should not survive as semantic recurrent state between independently committed forward calls.

## Required architecture change

Move GR from:

```text
persistent session state
```

to:

```text
activation/workspace state
```

For example:

```c
typedef struct {
    float *hc_residual;
    ...
} q38_forward_workspace;
```

The exact shape depends on decode vs prefill:

```text
decode:
    [hc_count][hidden]

prefill:
    [tokens][hc_count][hidden]
```

or the chosen physical layout.

## Session persistence rule

Future M9 checkpointing must include:

```text
YES:
- committed position
- token history
- GDN recurrent state
- GDN conv history
- QSA state

NO:
- GR transient activation
- PLE cache
- expert cache
- CUDA workspace
```

## Required tests

### GRSTATE-T01

Fresh forward from the same committed semantic state must reconstruct the same GR activation without requiring previous GR buffers.

### GRSTATE-T02

Clearing GR workspace between tokens must not alter semantic results.

### GRSTATE-T03

Session checkpoint serialization must not contain GR activation buffers.

---

# 5. P1 — Tokenizer is currently a compatibility/reference bridge, not the final native runtime tokenizer

## Files to inspect

- `q38_tokenizer.c`
- `q38_tokenizer.h`
- tokenizer tests/reference scripts

## Current situation

The historical implementation invoked Python / Hugging Face tooling, e.g. through:

```text
fork
exec python3
Transformers/tokenizers
```

That bridge remains available only as a **test oracle** for frozen golden
vectors. The runtime implementation now loads the tokenizer files and performs
encoding, decoding, special-token handling, multimodal recognition, and the
frozen chat rendering natively in C.

It is useful because it guarantees token IDs from the frozen official tokenizer.

The native path is the only production/runtime path.

## Why it matters later

Leaving Python/tokenizers in the production path introduces:

- process-spawn overhead;
- Python environment dependency;
- deployment fragility;
- server latency;
- more complex error handling;
- a runtime that is no longer a compact native executable.

## Required action

The corrective audit gate passes only when the native parity suite and
`tokenizer-runtime-gate` pass. Keep the native tokenizer as the only runtime
path through final end-to-end acceptance and server hardening.

## Native acceptance criterion

For the frozen corpus:

```text
native q38 token IDs
==
frozen Python/HF token IDs
```

100%, including:

- ASCII;
- UTF-8 / Unicode;
- multilingual text;
- special tokens;
- BOS/EOS behavior;
- chat template;
- markup;
- structured/tool-like content used by the project.

---

# 6. P2 — PLE layer-count/binding boundary appears off by one

## File to inspect

Likely:

- `q38_weights.c`

## Problem

A condition equivalent to:

```c
max_layer >= 1
```

appears to be used to determine whether PLE tensors should be included/bound, while the frozen architecture places PLE at model layer:

```text
layer 2
```

with zero-based indexing.

If `max_layer` means maximum included model-layer index, expected behavior should be:

```text
max_layer = 0 -> no PLE
max_layer = 1 -> no PLE
max_layer = 2 -> include PLE
max_layer >= 2 -> include PLE
```

## Required action

Verify the exact semantics of `max_layer`.

If zero-based maximum included layer, change the condition accordingly.

## Required tests

```text
max_layer=0
max_layer=1
max_layer=2
max_layer=3
```

and assert exact expected tensor counts.

Do not fix only the arithmetic constant; test actual binding behavior.

---

# 7. P2 — Quant artifact naming / identity should be tied to the manifest

## Problem

Human-readable names such as:

```text
Q2Experts-BF16Core-BF16PLE
```

are useful but are not sufficient as authoritative artifact identity.

As the project evolves, recipes will become more heterogeneous:

- gate/up may differ from down;
- PLE may be Q2 or Q4;
- output may be Q8/Q6;
- shared expert may differ;
- experiments in M8 will create many nearby recipes.

## Required change

Store authoritative metadata such as:

```text
q38.quant_recipe_name
q38.quant_manifest_sha256
q38.converter_revision
q38.source_model_revision
q38.weight_abi_version
```

inside the artifact metadata or an inseparable sidecar.

Prefer artifact names containing a short manifest hash, e.g.:

```text
q38-q2bootstrap-a17c23ef.gguf
```

rather than relying only on a descriptive name.

## Acceptance

`q38 --quant-audit` should print the manifest hash and fail if expected recipe metadata is absent or inconsistent.

---

# 8. Explicit non-finding — q38 GGUF tensor layout

This is **not a defect**.

The project intentionally generates its GGUF from the original Qwen3.8 tensors according to the q38 specification.

Therefore:

```text
original Qwen3.8 checkpoint layout
        ->
q38 converter
        ->
q38-specific GGUF
        ->
q38 kernels
```

is the intended ABI.

`llama.cpp` is used to validate model semantics and equations, but the q38 runtime is **not required to consume llama.cpp-converted tensor layouts**.

## Recommendation only

Document the distinction explicitly:

```text
q38 GGUF ABI:
- source layout: original Qwen3.8 checkpoint tensors
- transforms: only transforms defined by q38 converter
- llama.cpp GGUF transformed layout: not assumed
- llama.cpp usage: semantic/reference implementation
```

Optionally add:

```text
q38.weight_abi = "qwen38-original-layout-v1"
```

to the metadata.

This is documentation/robustness work, not a blocker.

---

# 9. Corrective gate before continuing M3

Before moving deeper into optimized GDN work, complete a corrective sub-gate.

Suggested name:

```text
M3-AUDIT
```

## Required checklist

- [ ] GR scalar math corrected to verified Qwen4Exp equations.
- [ ] GR CUDA math corrected.
- [ ] Non-zero independent GR goldens added.
- [ ] GR negative-SiLU test added.
- [ ] `/hc_count` scaling test added.
- [ ] Session GDN state expanded to all 36 GDN layers.
- [ ] Layer→GDN-slot mapping added and tested.
- [ ] Persistent memory accounting updated.
- [ ] GR removed from persistent semantic session state.
- [ ] PLE layer boundary/off-by-one checked and fixed if confirmed.
- [ ] Current tokenizer explicitly labeled as reference bridge if still Python-backed.
- [ ] `git diff --check` clean.
- [ ] Full M2 regression green.
- [ ] Existing M3 gates re-run after corrections.

## Suggested acceptance command

Exact target names may differ, but the final corrective gate should be equivalent to:

```bash
make m2-acceptance
make m3-gr-ref
make m3-gr-cuda
make m3-state
make m3-audit
git diff --check
```

---

# 10. Rules for correcting the findings

1. **Do not move expected outputs to match q38 implementation unless the new expected value comes from an external verified reference.**
2. Fix scalar/reference code before CUDA.
3. Every corrected semantic equation needs at least one non-trivial golden test.
4. Do not continue into performance optimization while semantic tests are self-referential.
5. Do not skip previous milestone regressions after changing session-state or GR math.
6. Preserve the frozen Qwen3.8 original-tensor GGUF ABI.
7. Keep `llama.cpp` and official Qwen/Transformers code as semantic references, not implicit binary-layout requirements.
8. If a required architecture detail is not established by the frozen references, stop that specific implementation point and investigate upstream instead of guessing.

---

# 11. Recommended commit sequence

Keep corrective changes small and auditable.

Suggested sequence:

```text
AUDIT-C01
fix: align gated residual scalar math with verified Qwen4Exp semantics

AUDIT-C02
test: add independent non-zero gated residual goldens

AUDIT-C03
fix: align CUDA gated residual with verified scalar/reference semantics

AUDIT-C04
refactor: allocate independent recurrent state for all GDN layers

AUDIT-C05
refactor: move gated residual from session state to forward workspace

AUDIT-C06
fix/test: correct PLE layer binding boundary if confirmed

AUDIT-C07
docs: mark Python tokenizer as reference bridge and document q38 weight ABI

AUDIT-C08
test: rerun M2/M3 corrective acceptance
```

Do not squash these until the audit has passed; separate commits make first-divergence debugging much easier.

---

# 12. Exit criterion

After `M3-AUDIT` is green, resume the original M3 sequence from the first not-yet-completed semantic/CUDA GDN step.

The project should **not** advance to M4 merely because a partial M3 graph runs.

The expected state before resuming is:

```text
M0   PASS
M1   PASS
M2   PASS
M3 semantic freeze   PASS
M3 corrective audit PASS

then:
M3 GDN implementation / CUDA / chunk invariance / acceptance
```
