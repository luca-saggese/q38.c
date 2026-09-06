# MoE optimization changelog

## 2026-09-06 - S2 suite foundation

- Added the frozen production MoE contract and implementation classification.
- Added an independent CPU oracle for BF16 router casting, top-k ordering,
  BF16 routing-weight normalization, Q2 routed experts, and the shared expert.
- Added zero-fixture/router-tie tests through `make test-moe`.
- Added the fixture-only CUDA benchmark through `make bench-moe`.
- The benchmark requires real early/middle/late compact captures and fails
  closed when the fixture pack is incomplete.
- No MoE candidate has been promoted and no full-model load or full-chain
  benchmark was run.

## 2026-09-06 - MOE-C2 grouped routed execution

- Captured and validated real early, middle, and late MoE fixtures without a
  persistent worker by using the single authorized model-load capture.
- Measured the production-equivalent Q2 baseline at approximately
  1.07--1.09 ms per complete MoE layer, with 34 kernel launches and 5 host
  synchronizations.
- Rejected the first grouped implementation because its complete-layer gain
  was below 1% despite reducing launch count.
- Promoted the corrected grouped implementation after replacing the serial
  cross-expert down loop with parallel per-expert down/weighted accumulation.
  The final isolated candidate measures approximately 0.28--0.29 ms, with
  6 launches, 5 host synchronizations, and 72--73% median wall reduction.
- Integrated indexed resident-tensor execution into the production Q2 MoE
  callback with persistent grouped workspaces.
- Frozen state: `MOE_OPT_V1`. No Reference 0 rerun, 128-token benchmark, or
  additional model load was performed during S2.
