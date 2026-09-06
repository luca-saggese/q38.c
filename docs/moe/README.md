# MoE subsystem

S2 follows the same fixture-first workflow as GR:

```text
freeze contract
  -> identify production call chain
  -> capture compact early/middle/late fixtures
  -> run independent CPU oracle
  -> benchmark production-equivalent CUDA orchestration
  -> measure the dominant stage
  -> implement one candidate
  -> correctness and >=10% isolated gate
  -> integrate and freeze
```

The contract and production classification are in
[`tests/moe/MOE_CONTRACT.md`](../../tests/moe/MOE_CONTRACT.md). The standalone
benchmark is `make bench-moe`; it requires all three complete real fixture
directories and fails closed when any required tensor is missing. It does not
open or load the full GGUF.

The benchmark reports router/top-k, one expert gate/up, one expert down, one
complete expert, ten selected experts, the shared expert, and the complete
MoE layer. Each scope reports median/p95 wall time, launch and synchronization
counts, transfers, and bytes read. The layer correctness gate compares both
the independent CPU oracle and the captured expected output.

## S2 status

S2 is frozen as `MOE_OPT_V1`. The real baseline and promoted candidate are
recorded in:

- [`artifacts/perf/subsystems/moe_reference.json`](../../artifacts/perf/subsystems/moe_reference.json)
- [`artifacts/perf/subsystems/moe_opt_v1.json`](../../artifacts/perf/subsystems/moe_opt_v1.json)

`MOE-C2` groups the selected routed experts into one gate/up-plus-SiLU launch
and one down-plus-weighted-accumulation launch. The production Q2 callback
uses indexed resident expert tensors and persistent device workspaces; it does
not repack the ten selected experts through host memory.

Across the early, middle, and late real fixtures, the promoted path reduces
complete-layer median wall time from roughly 1.07--1.09 ms to 0.28--0.29 ms,
reduces launches from 34 to 6, keeps host synchronizations at 5, and preserves
the independent oracle and captured-output correctness gates. No full-model
load or full-chain benchmark is required to run this suite.
