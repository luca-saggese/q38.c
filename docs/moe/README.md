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
