# Canonical Benchmark Protocol

This document is normative for performance benchmarks in this repository.
Historical measurements that do not satisfy this protocol are evidence only and
must not be used to claim a speedup or regression.

## Reference 0 identity

Reference 0 is immutable:

- Decode: `Q2_DECODE_REFERENCE_0`
- Prefill: `Q2_PREFILL_REFERENCE_0`
- Schema: `PERF_SCHEMA_V2`
- Decode artifact: `artifacts/perf/reference_0/q2_decode_reference_0.json`
- Prefill artifact: `artifacts/perf/reference_0/q2_prefill_reference_0.json`

Each immutable JSON artifact contains the matching uppercase `REF_ID` field;
that identifier must never be reused for a current or candidate result.

The canonical entry points are:

```sh
make bench-q2-decode
make bench-q2-prefill
```

These targets write non-immutable current artifacts:

```text
artifacts/perf/current/q2_decode_canonical.json
artifacts/perf/current/q2_prefill_canonical.json
```

Reference 0 is created once with `make bench-q2-reference-0`. That target
performs one model load and produces both immutable artifacts; subsequent
invocations must not overwrite them.

## MODEL

The Reference 0 model is exactly:

```text
artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf
SHA-256: 68b29ce24ab62bf2cbcd8bbdeff5e6d36941e5b00e2fe34df90e92f98d51f518
```

The quantization recipe is:

```text
tools/quant_manifest_q2.json
SHA-256: 739fffc7bc1ebaccfcdf6efb621e23a1cd6d68f2cc9f6839861c3a0d774410c7
```

The tokenizer root is `/home/lvx/q38model`. Its fingerprint is computed over
the canonical tokenizer files `chat_template.jinja`, `merges.txt`,
`tokenizer.json`, `tokenizer_config.json`, and `vocab.json`:

```text
SHA-256: 5f35fb517d3a016e4ca70d175e921ba5e9c92f7352663516a28f001b7225b93d
```

Every artifact records the exact tokenizer fingerprint, branch, commit, and
dirty-tree state used for that run. The protocol was established on branch
`qwen38-spark-proto`; the artifact metadata is authoritative for the exact
Reference 0 commit.

## BUILD

The default build is production CUDA build mode:

```text
CC: cc
CFLAGS: -O3 -g -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -I. -pthread
NVCC: /usr/local/cuda/bin/nvcc
NVCC flags: -O3 -g -lineinfo --use_fast_math -gencode arch=compute_121a,code=sm_121a
CUDA architecture: sm_121a
Build type: production -O3
PERF_SCHEMA: PERF_SCHEMA_V2
```

No full-state trace, semantic state snapshot, or invasive diagnostic probe may
be enabled in the timed region.

## HARDWARE

Reference 0 is measured on:

```text
Platform: DGX Spark / GB10 / NVIDIA GB10
CPU architecture: aarch64
Unified memory: approximately 128 GiB unified memory; exact value is recorded
in the artifact hardware metadata
CUDA driver: recorded in the artifact hardware metadata
CUDA runtime: recorded in the artifact hardware metadata
```

Hardware metadata is part of the artifact comparison key. A run on another
device, driver, runtime, or CPU architecture is not automatically comparable.

## RESIDENCY

The residency contract is mandatory:

- all non-PLE tensors are resident after initialization;
- PLE remains permanently file-backed;
- non-PLE upload after initialization is exactly `0` bytes;
- non-PLE residency misses are exactly `0`;
- persistent PLE resident entries are exactly `0`;
- PLE file-backed activity is reported separately from non-PLE residency;
- residency counters are recorded in every Reference 0 artifact.

## EXECUTION PATH

The only canonical production path is:

```text
q38_runtime
  -> q38_session
  -> q38_session_prefill
  -> q38_session_eval
```

The benchmark must not use a legacy forward runner, a manually reconstructed
backend configuration, or a one-off diagnostic/probe path. Backend selection
is installed by the runtime/session configuration and is not overridden by a
benchmark-specific copy.

## SAMPLING

- greedy deterministic sampling;
- temperature `0`;
- no MTP;
- no speculative decoding;
- no sampling randomness or external sampling service.

## DECODE INPUT

The exact canonical prompt is:

```text
Explain in simple terms why the sky appears blue during the day and red near sunset.
```

The artifact stores the exact prompt token IDs and every generated token ID.
The decode workload is:

- `max_new_tokens = 128`;
- measurement positions `16..127`, inclusive;
- the first 16 generated positions are discarded from the measured window;
- one semantic warmup run;
- ten measured warm runs;
- greedy decode with the same context size and reset semantics on every run.

## PREFILL INPUT

The prefill corpus is the canonical prompt token sequence repeated to exactly:

- `128` tokens;
- `512` tokens;
- `2048` tokens.

The artifact stores the full token corpus for every workload. The chunk size is
exactly `128` tokens. Each workload uses one semantic warmup run and ten
measured warm runs.

Prefill TTFT is the wall time until the final prefill logits are available from
the synchronous canonical prefill call. It is reported separately from
decode-token throughput.

## RESET POLICY

Every run begins from a semantic session reset. The model and runtime are not
reloaded between warm runs. The persistent worker is preferred and must remain
alive whenever the execution environment supports it.

The timed region contains production session evaluation only. Full-state
hashing, state copies, and state snapshots are forbidden in that region.
Correctness uses final logits hashes, generated IDs, finite-value checks, and
prefill state/output equivalence.

## RUN COUNT AND METRICS

Each canonical workload uses exactly:

- `1` warmup run;
- `10` measured warm runs.

Decode artifacts report:

- wall median, p95, minimum, maximum, and standard deviation;
- tokens per second;
- QSA;
- MoE;
- GDN;
- GR;
- norms/residual/glue;
- LM head;
- argmax;
- host/scalar work;
- CUDA sync/wait;
- CUDA dispatch;
- memcpy;
- PLE critical stall;
- other/unattributed residual;
- non-additive PLE elapsed and PLE overlap;
- non-PLE resident bytes, upload bytes, and residency misses;
- kernel launches, host synchronizations, and traffic counters;
- correctness hashes and exact generated IDs.

Prefill artifacts report these metrics for each of the three token counts:

- prefill wall;
- tokens per second;
- TTFT;
- chunk size;
- peak CUDA and unified-memory estimates;
- correctness and state/output equivalence;
- residency and PLE accounting.

`PLE_elapsed` and `PLE_overlap` are non-additive diagnostic values. Only
`PLE_critical_stall` contributes to critical-path accounting. Temporal spans
that overlap are never summed as if they were exclusive.

## Comparability rule

Two benchmarks are comparable **only if all** of these fields match:

- model fingerprint;
- quant recipe;
- tokenizer fingerprint;
- execution path;
- input token IDs;
- context and reset semantics;
- residency policy;
- build mode and flags;
- `PERF_SCHEMA` version;
- measurement window.

If any field differs, the result must be marked:

```text
NON-COMPARABLE
```

It must not be used to declare a speedup or regression. A different prompt
string with the same apparent intent is also non-comparable unless its exact
token IDs match.

## Optimization reporting rule

Every future performance commit must report one comparison row:

```text
REF0 wall
candidate wall
absolute ms saved/token
relative %
tok/s before
tok/s after
correctness
```

The primary metric is:

```text
ms_saved_per_token
```

An isolated kernel or microbenchmark speedup is supporting evidence only and
cannot replace the canonical end-to-end measurement.

## Model-load policy

Full model load is approximately three minutes. During debugging and
optimization:

- do not run repeated fresh-process benchmark loops;
- keep the persistent worker alive whenever possible;
- allow at most one fresh model load per optimization task unless explicitly
  justified;
- use code inspection, unit tests, fixtures, and microbenchmarks first;
- run the full canonical benchmark only after the candidate is ready.

## Immutability and protocol revisions

Never overwrite `artifacts/perf/reference_0/`. Future measurements belong in:

```text
artifacts/perf/current/
```

If the protocol changes incompatibly, create Reference 1 and a new schema
version, for example `PERF_SCHEMA_V3`. Reference 0 remains archived and
immutable forever.
