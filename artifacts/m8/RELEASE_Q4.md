# Q4 selective M8 release status

## Recipe

`quant_manifest_q4_selective.json` raises only routed expert `gate_up_proj` and
`down_proj` tensors from Q2_K to Q4_K. Router, shared expert, GDN, QSA, gated
residual, embedding/output, and PLE remain at the M7 precision policy. The
transposed down-projection layout is preserved.

## Verified

- Full inventory manifest validation and block-alignment checks pass.
- Full R1 GGUF conversion, inspection, and quant audit pass.
- Q4_K scalar/CUDA dequant and routed-expert CUDA/reference parity pass on GB10.
- Full R1 one-token and eight-token CUDA generation runs pass without NaN/Inf;
  paired frozen-corpus greedy comparison is exact.
- M0, M1 validation/block, M2, M4, independent M6, and M7 gates pass; see
  `full_regression.txt` for exact status.
- Memory solver calibration is inventory-backed and within 5% for M7 R0.
- PLE is permanently file-backed SSD/mmap with bounded cache/staging. PLE
  bytes are excluded from residency-fit accounting and no full PLE mirror is
  permitted.

## Evidence and blockers

The full R1 artifact is materialized at 132,212,799,456 bytes without deleting
preserved models. Its checksum is recorded in `R1_checksums.txt`; the binary is
retained in the workspace and intentionally not added to Git.

`tools/q38_eval.py` is now a deterministic paired evaluator for at least 32
records. It reports argmax agreement, top-k overlap, logit error, KL,
router-top-10 stability, and QSA selected-ID stability. The frozen 32-sequence
corpus is ready, but paired finite BF16-reference/full-R1 vectors are not
available, so no quality metrics are claimed.

Six real R1 CLI runs are recorded in `r1_bench_runs/` and aggregated in
`R1_memory_bench.json`. Cold, warm median/p95, RSS, and workspace CUDA
allocation are recorded. The existing CLI does not expose persistent non-PLE
residency, warm upload bytes/token, unified peak, MemAvailable, or residency
misses; those fields remain null. The same-process M7 harness could not be
rebuilt because unrelated CUDA files contain unresolved merge markers.

The report includes direct Q2 M7 versus Q4 R1 timing and limited greedy
comparison. BF16/reference quality drift is explicitly `not-computed` until
paired full-vector records exist; generated text is not used as a proxy.

The deleted layer-3 M5-C12 fixture is classified obsolete and replaced by a
direct full-R1 gate (`artifacts/m5/m5_c12_modern_gate.json`). M1/M3 no longer
recreate that fixture. M9 is suspended and was not started. M8 remains
conditional until paired quality vectors and required residency telemetry are
available.
