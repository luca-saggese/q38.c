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

## Explicit blockers

The full R1 artifact is now materialized at 132,212,799,456 bytes without
deleting preserved models. The available evaluator provides exact greedy
comparison only; perplexity/NLL/KL/layer-logit/task metrics remain uncomputed.
The 132 GB binary is retained in the workspace and recorded by
`R1_full_sha256.txt`; it is intentionally not added to Git because doing so
would duplicate the large artifact in repository storage.
The CLI also lacks long-context startup/prefill peak instrumentation. M5-C12
still fails its existing native/reference comparison, and M3/M1 full acceptance
requires the user-deleted obsolete layers0-3 fixture. M8 therefore remains
conditional rather than being marked complete.
