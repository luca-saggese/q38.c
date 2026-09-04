# Q4 selective M8 release status

## Recipe

`quant_manifest_q4_selective.json` raises only routed expert `gate_up_proj` and
`down_proj` tensors from Q2_K to Q4_K. Router, shared expert, GDN, QSA, gated
residual, embedding/output, and PLE remain at the M7 precision policy. The
transposed down-projection layout is preserved.

## Verified

- Full inventory manifest validation and block-alignment checks pass.
- Controlled layer-0 R1 GGUF conversion, inspection, and quant audit pass.
- Q4_K scalar/CUDA dequant and routed-expert CUDA/reference parity pass on GB10.
- M0 acceptance and available M2-M7 regression gates were run; see
  `full_regression.txt` for exact status.
- Memory solver calibration is inventory-backed and within 5% for M7 R0.

## Explicit blockers

A full R1 artifact is projected at 132,212,799,456 bytes. The filesystem had
approximately 79.6 GB free during the real conversion-plan check, so creating
it alongside preserved M7/R0 artifacts would require deleting user files and
was not attempted. The available 4,126,298,944-byte layer-0 artifact is a
controlled fixture, not a full-model quality or throughput candidate. Therefore
full R1 quality, startup/prefill peak, and throughput acceptance remain
unclaimed; no Spark or quality result is inferred.
