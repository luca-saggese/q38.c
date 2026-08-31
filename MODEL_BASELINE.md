# M1 model baseline

This file freezes the source reference used by the M1 inventory and conversion
work. Model artifacts stay outside this repository.

| Field | Value |
|---|---|
| Repository | `https://huggingface.co/Qwen/Qwen3.8-Flash-Next` |
| Revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Revision timestamp | `2026-08-27T05:03:36Z` |
| Local model directory | `/home/lvx/q38model` |
| Config SHA-256 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| Safetensors index SHA-256 | `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de` |
| Tokenizer SHA-256 | `0997f410c57a1f4e53b09e4be8f4a172d90edd956436fb0847030937229b9f3` |
| Converter revision | `UNKNOWN (converter not selected)` |
| llama.cpp revision | `UNKNOWN (compatibility audit pending)` |

The model config currently reports `qwen4_exp` with text model type
`qwen4_exp_text`, 48 layers, 512 experts, and 10 experts per token. The
`layer_types` array is read from the frozen config rather than inferred.
