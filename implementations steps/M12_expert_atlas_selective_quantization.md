# q38.c — M12 Expert Atlas & Data-Driven Selective Quantization

Target branch: `qwen38-spark-proto`

Target runtime:
- Qwen3.8-Flash-Next only
- NVIDIA DGX Spark / GB10 / SM121
- CUDA-only
- q38 original-layout GGUF ABI
- selective expert quantization
- PLE as an independent storage/quantization domain

Placement in roadmap:

```text
M0–M10
  ↓
M11 long-context / FP8 KV / 500k→1M
  ↓
M12 — Expert Atlas & Data-Driven Selective Quantization
```

M12 may reuse the selective-Q4 machinery introduced in M8, but replaces heuristic or globally uniform expert recipes with a measured, domain-aware, sensitivity-driven assignment.

---

# 1. Objective

Build a reproducible system that answers, for every routed expert instance:

```text
Where is this expert used?
How strongly is it used?
For which domains/languages?
At which layers?
How sensitive is model quality to lowering its precision?
```

Then use those measurements to derive a per-expert quantization recipe.

Primary rule:

```text
quantize based on measured sensitivity,
not on assumed semantic labels
```

Domain/language specialization is metadata and a coverage constraint, not by itself a reason to lower precision.

---

# 2. Scope

Qwen3.8 has:

```text
48 decoder layers
512 routed experts/layer
top-10 routing
```

Therefore M12 covers up to:

```text
48 × 512 = 24,576 routed expert instances
```

Each `(layer, expert_id)` is treated as an independent unit unless later clustering proves a grouped treatment is both safe and beneficial.

Shared experts, routers, norms, GDN recurrent state, QSA state and PLE are **not** routed-expert candidates in this milestone unless explicitly included in a separate ablation.

---

# 3. Core principles

## 3.1 No semantic guessing

Do not assume:

```text
expert 17 = Chinese
expert 81 = coding
expert 203 = narrative
```

Specialization must be inferred from observed routing statistics.

## 3.2 Usage does not equal importance

An expert can be:

```text
rarely selected
but high-impact when selected
```

Therefore routing frequency alone must never determine quantization.

## 3.3 General-purpose quality is the target

Coding is one domain among many, not the optimization center.

Required coverage includes:
- narrative;
- science;
- mathematics;
- reasoning;
- humanities;
- multilingual;
- translation;
- general knowledge;
- analysis/summarization;
- conversation;
- structured output;
- coding;
- long-context.

## 3.4 Independent evaluation

A candidate recipe must be evaluated on prompts/tasks not used to derive its expert importance scores.

---

# 4. M12 structure

```text
M12A — Corpus and routing instrumentation
M12B — Expert Atlas
M12C — Per-expert sensitivity experiments
M12D — Quant recipe solver
M12E — Domain quality acceptance
M12F — Runtime integration and final recipe
```

---

# 5. M12A — General-purpose calibration corpus

## 5.1 Corpus requirements

Create a frozen calibration/evaluation corpus with explicit domain labels.

At minimum:

```text
narrative
creative_writing
literary_analysis
science
physics
chemistry
biology
astronomy
mathematics
formal_reasoning
general_reasoning
history
philosophy
economics
linguistics
general_knowledge
summarization
document_analysis
conversation
translation
multilingual
structured_output
json
coding
long_context
```

## 5.2 Narrative coverage

Required prompt families:

```text
dialogue
descriptive prose
story continuation
character consistency
style transformation
long-form narrative
literary analysis
poetry/prose variation
scene reconstruction
plot reasoning
```

## 5.3 Science coverage

Include:

```text
physics explanation
physics reasoning
chemistry concepts
chemical reasoning
biology
astronomy
geology
scientific summarization
causal scientific reasoning
technical explanation
```

## 5.4 Mathematics/reasoning

Include:

```text
arithmetic
algebra
geometry
probability
calculus
symbolic reasoning
multi-step reasoning
logic
word problems
```

## 5.5 Humanities

Include:

```text
history
philosophy
economics
literature
linguistics
art/culture
comparative analysis
argumentation
```

## 5.6 Multilingual

At minimum include strong coverage for:

```text
English
Italian
Chinese
Japanese
Korean
Spanish
French
German
Portuguese
Arabic
```

Test both monolingual tasks, translation pairs, code-switching, and mixed-script text.

## 5.7 Coding

Coding remains important but not dominant.

Include:

```text
code generation
debugging
code explanation
algorithm design
API usage
systems programming
Python
C/C++
CUDA
structured data manipulation
```

## 5.8 Long-context

Use long documents and retrieval tasks to detect experts that become important only at large context lengths.

Suggested buckets:

```text
4k
16k
64k
262k
500k if M11 is available
```

---

# 6. Corpus balance

Do not let one category dominate token count.

Store both prompt count and token count per domain.

The routing profiler must report:

```text
raw token-weighted metrics
+
domain-normalized metrics
```

so that long-context samples do not overwhelm all other domains.

---

# 7. Dataset split

Freeze three disjoint sets:

```text
CALIBRATION
VALIDATION
HOLDOUT
```

Purpose:

```text
CALIBRATION:
    routing census
    sensitivity estimation

VALIDATION:
    recipe search

HOLDOUT:
    final quality gate
```

Do not optimize directly against HOLDOUT.

---

# 8. M12A — Router instrumentation

For every token and routed layer, collect at least:

```text
layer
token position
token ID
prompt/domain ID
language label
router logits summary
top-10 expert IDs
top-10 routing weights
rank of each selected expert
```

Optional detailed traces may contain all 512 router logits for selected diagnostic subsets.

---

# 9. Expert metrics

For every `(layer, expert)` compute:

```text
selection_count
selection_rate
top1_count
top1_rate
mean_selected_rank
mean_router_weight
median_router_weight
max_router_weight
weighted_usage = sum(router_weight)
```

Also compute normalized versions per domain/language.

---

# 10. Domain specialization metrics

For each expert build:

```text
P(domain | expert selected)
P(language | expert selected)
```

and derive:

```text
domain_entropy
language_entropy
domain_concentration
language_concentration
```

Low entropy suggests specialization. High entropy suggests general-purpose use.

Do not interpret low entropy as proof of semantic function.

---

# 11. Layer-aware analysis

An expert ID has no guaranteed semantic identity across layers.

Treat `(layer 3, expert 17)` as distinct from `(layer 27, expert 17)`.

All atlas records must therefore use the compound key:

```text
layer_id + expert_id
```

---

# 12. Expert contribution metrics

Routing weight alone is insufficient.

Where computationally feasible, estimate:

```text
||expert_output||
router_weight * ||expert_output||
||weighted_expert_output||
relative contribution to routed_sum
```

Aggregate mean, p95 and max contribution.

---

# 13. M12B — Expert Atlas

Produce one persistent atlas record for each routed expert instance.

Example:

```json
{
  "layer": 22,
  "expert": 137,
  "usage": {
    "selection_rate": 0.0047,
    "top1_rate": 0.0009,
    "weighted_usage": 0.0069,
    "mean_rank": 5.2
  },
  "domains": {
    "narrative": 0.34,
    "science": 0.09,
    "math": 0.04,
    "coding": 0.02,
    "multilingual": 0.41
  },
  "languages": {
    "en": 0.32,
    "it": 0.08,
    "zh": 0.38
  },
  "entropy": {
    "domain": 1.52,
    "language": 1.11
  },
  "contribution": {
    "mean_weighted_norm": 0.037,
    "p95_weighted_norm": 0.119
  },
  "sensitivity": {
    "q4": null,
    "q3": null,
    "q2": null
  }
}
```

---

# 14. Atlas outputs

Required artifacts:

```text
artifacts/m12/atlas/
  expert_atlas.jsonl
  expert_usage_by_layer.csv
  expert_usage_by_domain.csv
  expert_usage_by_language.csv
  expert_entropy.csv
  expert_contribution.csv
```

---

# 15. M12C — Sensitivity experiments

Purpose:

```text
what happens if one expert,
or a controlled group of experts,
is quantized more aggressively?
```

---

# 16. Baseline precision

Start from the best validated recipe available after M8/M11.

Conceptual baseline:

```text
routed experts      Q4
router              high precision
shared expert       high precision / validated recipe
core                validated precision
PLE                 independent recipe
```

Freeze the exact baseline before sensitivity testing.

---

# 17. Candidate expert precisions

Depending on q38 format support, test:

```text
Q4
Q3
Q2
```

Optionally Q5/Q6/Q8 for highly sensitive experts.

---

# 18. Single-expert perturbation

For selected experts:

```text
baseline recipe
↓
lower precision of exactly one expert
↓
rerun calibration subset
↓
measure delta
```

Metrics:

```text
logit max_abs error
logit RMS error
KL divergence
next-token disagreement rate
greedy sequence disagreement
task score delta
domain score delta
```

Also record downstream routing changes.

---

# 19. Batched sensitivity

Testing all 24,576 expert instances one-by-one across the entire corpus may be too expensive.

Use a staged strategy.

## Stage 1 — cheap screening

Use:

```text
usage
weighted usage
contribution norm
domain entropy
```

## Stage 2 — targeted perturbation

Run individual or small-group tests on prioritized candidates.

## Stage 3 — recipe-level validation

Validate the combined candidate recipe on full validation/holdout sets.

Do not assume perturbation effects add linearly.

---

# 20. Sensitivity score

Define a composite score such as:

```text
S(E) =
    w1 * normalized_KL
  + w2 * token_disagreement
  + w3 * task_score_loss
  + w4 * domain_worst_case_loss
```

Exact weights must be frozen and documented.

Always store the raw metrics too.

---

# 21. Rare-expert protection

A low-usage expert must not be automatically assigned low precision.

Protect an expert if any of these hold:

```text
high p95 contribution
high max routing weight
high domain concentration in a protected domain
large perturbation effect
high downstream routing instability
```

---

# 22. Domain protection

All major domains are protected.

At minimum:

```text
narrative
science
mathematics
reasoning
humanities
multilingual
coding
general_chat
```

No recipe can compensate a large loss in one protected domain with gains elsewhere.

---

# 23. Chinese/CJK specialization policy

M12 may discover experts strongly associated with Chinese, Japanese, Korean, CJK scripts or translation.

This is useful atlas information.

But the precision rule must remain:

```text
if low sensitivity:
    candidate for lower precision

if high sensitivity:
    preserve precision
```

Never:

```text
CJK expert:
    lower precision
```

---

# 24. Optional specialized profiles

After GENERAL is stable, M12 may generate optional profiles:

```text
GENERAL
SCIENCE_HEAVY
NARRATIVE_HEAVY
CODING_HEAVY
MULTILINGUAL_HEAVY
```

These must be separate quant manifests.

The default release remains GENERAL.

---

# 25. M12D — Quantization recipe solver

Input:

```text
expert atlas
sensitivity measurements
memory target
quality constraints
supported qtypes
```

Output:

```text
per-expert qtype assignment
```

---

# 26. Optimization objective

Conceptually:

```text
minimize resident/model bytes
```

subject to:

```text
overall quality loss <= threshold
worst-domain loss <= threshold
memory <= target
protected experts constraints satisfied
runtime qtype support available
```

This is a constrained optimization problem, not a simple threshold rule.

---

# 27. Recipe identity

Every recipe must include:

```text
recipe_name
source_model_revision
q38_converter_revision
q38_runtime_revision
expert_atlas_revision
corpus_revision
solver_revision
quant_manifest_sha256
```

Example:

```text
GENERAL-Q4MIX-7f4c21ab
```

---

# 28. Quant manifest format

The manifest must allow per-expert assignment.

Example:

```json
{
  "layers": {
    "22": {
      "experts": {
        "0": "Q4",
        "1": "Q3",
        "2": "Q2",
        "137": "Q4"
      }
    }
  }
}
```

Use a compact runtime representation if required; the JSON remains the auditable source artifact.

---

# 29. Runtime lookup

The mixed expert-bank runtime should support:

```text
expert_id
→ qtype bank
→ local bank index
→ physical weight address
```

Precompute mappings during model load.

Do not introduce a slow hash/map lookup in the hot path.

---

# 30. Bank layout optimization

Experts of the same qtype should be physically grouped where possible.

Desired runtime path:

```text
selected expert ID
↓
bank ID
↓
local index
↓
base + stride
```

Do not reorder logical expert IDs. Maintain explicit logical→physical mapping.

---

# 31. M12E — Quality evaluation

Quality is a vector, not a single score.

Required dashboard:

```text
overall
narrative
science
math
reasoning
humanities
multilingual
coding
general_chat
long_context
```

---

# 32. Worst-domain gate

A recipe passes only if:

```text
max_domain_regression <= configured threshold
```

Even if aggregate score is excellent.

---

# 33. Narrative evaluation

Narrative quality should include:

```text
coherence
character consistency
instruction adherence
style fidelity
long-form continuity
repetition rate
lexical diversity
```

Use automated metrics and frozen judge procedures where appropriate. Record evaluator revision.

---

# 34. Science evaluation

Include:

```text
factual accuracy
causal reasoning
multi-step explanation
terminology precision
quantitative reasoning
```

Separate factual recall from reasoning where possible.

---

# 35. Multilingual evaluation

Measure per-language regressions separately.

Do not report only a multilingual average.

At minimum:

```text
en
it
zh
ja
ko
es
fr
de
pt
ar
```

---

# 36. Routing stability metrics

For baseline vs candidate recipe, measure:

```text
route top-10 agreement
route top-1 agreement
weighted route overlap
first routing divergence layer
```

---

# 37. Logit stability metrics

Measure:

```text
argmax agreement
top-k overlap
KL divergence
RMS logit error
max_abs logit error
```

---

# 38. Sequence-level metrics

For deterministic greedy runs:

```text
first disagreement token
total token disagreement rate
exact sequence match rate
```

A candidate may have small one-step KL but large autoregressive drift.

---

# 39. Long-context sensitivity

Repeat selected tests at:

```text
64k
262k
500k where M11 is available
```

Do not derive the final recipe exclusively from short prompts.

---

# 40. M12F — Final integration

Once a recipe is selected:

1. generate q38 artifact;
2. run full preflight;
3. run M6 end-to-end regression;
4. run M7 performance regression;
5. run M9 session/replay regression;
6. run M10 multimodal regression;
7. run M11 long-context regression where available;
8. run M12 holdout quality suite.

---

# 41. Performance acceptance

Selective expert qtypes can alter kernel dispatch and memory access patterns.

Measure:

```text
prefill tok/s
decode tok/s
expert matvec bandwidth
expert bank switch rate
cache behavior
memory footprint
```

A smaller model that becomes materially slower due to pathological bank fragmentation may not be preferable.

---

# 42. Memory accounting

Report:

```text
routed expert bytes
shared expert bytes
core bytes
PLE bytes
QSA KV/state
GDN state
workspace
OS headroom
```

Also report savings relative to uniform Q4/Q3/Q2 expert recipes where supported.

---

# 43. Reproducibility

All profiling and sensitivity runs must record:

```text
model revision
runtime commit
converter commit
corpus revision
prompt IDs
seed
decoding mode
context length
hardware
CUDA version
driver version
```

---

# 44. M12 commit outline

```text
M12-C00 docs: freeze expert-atlas methodology and corpus taxonomy
M12-C01 feat: add routed-expert tracing instrumentation
M12-C02 feat: build general-purpose calibration corpus manifest
M12-C03 feat: aggregate expert usage/domain/language statistics
M12-C04 feat: generate expert atlas
M12-C05 feat: add contribution-norm instrumentation
M12-C06 feat: implement single-expert perturbation harness
M12-C07 feat: implement batched sensitivity screening
M12-C08 feat: compute per-expert sensitivity profiles
M12-C09 feat: implement constrained quant-recipe solver
M12-C10 feat: generate GENERAL per-expert quant manifest
M12-C11 test: validate domain-level quality gates
M12-C12 test: validate multilingual per-language gates
M12-C13 test: validate long-context sensitivity
M12-C14 perf: validate expert-bank runtime performance
M12-C15 integration: regenerate final q38 artifact
M12-C16 acceptance: full holdout and regression suite
```

---

# 45. Required artifacts

```text
artifacts/m12/
  corpus/
    calibration_manifest.json
    validation_manifest.json
    holdout_manifest.json
    domain_balance.json
    language_balance.json

  routing/
    routing_summary.json
    routing_by_layer.json
    routing_by_domain.json
    routing_by_language.json

  atlas/
    expert_atlas.jsonl
    expert_usage_by_layer.csv
    expert_usage_by_domain.csv
    expert_usage_by_language.csv
    expert_entropy.csv
    expert_contribution.csv

  sensitivity/
    expert_sensitivity.jsonl
    screening_summary.json
    protected_experts.json
    perturbation_results.json

  recipes/
    baseline_manifest.json
    candidate_manifests/
    selected_manifest.json
    selected_manifest.sha256

  quality/
    overall.json
    narrative.json
    science.json
    mathematics.json
    reasoning.json
    humanities.json
    multilingual.json
    coding.json
    long_context.json
    routing_stability.json
    logits_stability.json
    sequence_stability.json

  performance/
    memory.json
    prefill.json
    decode.json
    expert_bank_profile.json

  acceptance.txt
```

---

# 46. Suggested expert classification labels

Descriptive only; they must not directly determine precision.

```text
GENERALIST
LANGUAGE_SPECIALIZED
DOMAIN_SPECIALIZED
RARE_HIGH_IMPACT
RARE_LOW_IMPACT
HIGH_USAGE_LOW_SENSITIVITY
HIGH_USAGE_HIGH_SENSITIVITY
PROTECTED
```

An expert may have multiple labels.

---

# 47. Example decision logic

Bad:

```text
if chinese_usage > 80%:
    qtype = Q2
```

Good:

```text
if sensitivity_q2 < threshold
and worst_domain_delta < threshold
and contribution_risk < threshold:
    qtype = Q2
else if sensitivity_q3 < threshold:
    qtype = Q3
else:
    qtype = Q4
```

Domain/language information is used to ensure coverage and explain decisions.

---

# 48. Stop conditions

Stop recipe optimization and investigate if:

- one domain degrades disproportionately;
- one language degrades disproportionately;
- routing instability explodes;
- candidate recipe passes average metrics but fails holdout;
- perturbation effects are strongly non-additive;
- bank fragmentation causes severe performance loss;
- long-context behavior differs materially from short-context estimates;
- quant recipe reproducibility fails.

---

# 49. Acceptance gate

M12 passes only when:

```text
expert atlas complete
+
routing census reproducible
+
sensitivity measurements available
+
GENERAL recipe generated
+
holdout domain gates pass
+
multilingual gates pass
+
long-context gate passes where available
+
runtime memory target achieved
+
performance regression acceptable
+
full q38 regression suite green
```

---

# 50. Final goal

M12 should make it possible to answer, with evidence:

```text
Why is expert (layer L, expert E) Q2/Q3/Q4?
```

The answer must come from:

```text
measured routing behavior
+
measured contribution
+
measured sensitivity
+
domain/language quality constraints
+
memory/performance optimization
```

not from intuition.

The desired result is a Qwen3.8 q38 build that remains general-purpose across narrative, science, reasoning, humanities, multilingual use and coding while spending precision only where the model demonstrably needs it.
