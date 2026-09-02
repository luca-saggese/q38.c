# M6 quant-matched reference

`tools/m6_quant_matched_reference.py` is an independent Qwen4Exp execution
path. It maps the q38 GGUF itself, decodes BF16/Q8_0/Q2_K rows, and evaluates
the Transformers Qwen4Exp layer equations without importing q38 or invoking
`q38_forward`. Routed gate/up and down rows are decoded only for the selected
experts.

M6-C12 keeps the original-checkpoint Transformers comparison as a diagnostic
artifact (`semantic_comparison.json`). The definitive numeric gate compares the
native trace with this GGUF reference in `quant_matched_comparison.json`.
Before the model-level comparison, `m6-dequant-fixtures` reads three real GGUF
rows (BF16 embedding, Q8_0 PLE projection, and Q2_K routed expert) through the
q38 decoder and an independent oracle. It fails closed on type, shape, endian,
row-stride, scale/min packing, decoded values, and raw-byte checksum mismatch;
the resulting checksums are recorded in
`artifacts/m6/gguf_dequant_fixtures.json`.
Routing hard gates begin at layer 3 so the known layer-2 boundary diagnostic
does not mask the first post-boundary selected-set divergence. Each routed
layer records the hidden input, all 512 effective router logits, rank 10/11
scores and margin, selected experts, weights by expert, and routed output.
Layer 1 and layer 9 additionally record progressive full-vector boundaries.
Layer 1 includes the applicable PLE contribution and grouped-normalization
substages. Layer 9 records the pre-router GR input, its RMSNorm output,
the GR output feeding the router, BF16 source-byte checksums and values for
ranking-relevant router rows, pre-cast matvecs, effective BF16 results, and
row-by-row matvec self-checks. It also records FP32 reduction versus
`F.linear` accumulation, BF16-input/BF16-weight matmul, the explicit
FP32-to-BF16 cast, and the corresponding BF16 bit patterns. These boundary
and rounding diagnostics do not alter routing or tolerance gates. The scalar
native path uses grouped 2560-wide RMS normalization for PLE streams; a
single 10240-wide norm is not equivalent.

Example:

```sh
PYTHONPATH=.venv-m6/lib/python3.12/site-packages \
  .venv-m6/bin/python tools/m6_quant_matched_reference.py \
  --gguf artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf \
  --trace artifacts/m6/real_forward_trace.json \
  --output artifacts/m6/quant_matched_reference.json
```

The reader requires only GGUF v3 and the q38 tensor types used by this
artifact; no GGUF regeneration or llama.cpp ABI is involved.

## Stateful cache oracle

`tools/m6_stateful_gguf_oracle.py` uses the official
`DynamicCache`, causal/recurrent mask builders, and
`Qwen4ExpTextDecoderLayer` directly. It constructs a meta-only text model,
materializes one GGUF-backed decoder layer at a time, and releases that layer
after its activation has been passed onward. GDN, QSA, and PLE state therefore
comes only from the official cache implementation; no cache equations are
duplicated in the oracle.

The first prompt token is a fresh-session step and is compared to the C12
one-shot artifact before continuation is allowed. The report records the
initial and final cache state, checkpoint evidence for layers 0, 3, 7, 15, 31,
and 47, final norm/logit statistics, and the greedy argmax. A missing
Transformers Qwen4Exp module, cache-shape error, non-finite result, or
one-shot mismatch is `blocked`/failure rather than a fabricated pass.
