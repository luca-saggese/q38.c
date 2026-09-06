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
