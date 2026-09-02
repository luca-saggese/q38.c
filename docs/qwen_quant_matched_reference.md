# M6 quant-matched reference

`tools/m6_quant_matched_reference.py` is an independent Qwen4Exp execution
path. It maps the q38 GGUF itself, decodes BF16/Q8_0/Q2_K rows, and evaluates
the Transformers Qwen4Exp layer equations without importing q38 or invoking
`q38_forward`. Routed gate/up and down rows are decoded only for the selected
experts.

M6-C12 keeps the original-checkpoint Transformers comparison as a diagnostic
artifact (`semantic_comparison.json`). The definitive numeric gate compares the
native trace with this GGUF reference in `quant_matched_comparison.json`.
Routing hard gates begin at layer 3 so the known layer-2 boundary diagnostic
does not mask the first post-boundary selected-set divergence. Each routed
layer records the hidden input, all 512 effective router logits, rank 10/11
scores and margin, selected experts, weights by expert, and routed output.

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
