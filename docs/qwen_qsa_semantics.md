# Qwen4Exp sparse-attention (QSA) semantics

This document freezes the text-only QSA contract for the local checkpoint.
The primary references are the official Qwen4Exp implementation in
`huggingface/transformers` (main as inspected on 2026-09-01) and the
`ggml-org/llama.cpp` implementation from PR #27742, merge commit
`eaf93765572e794b8e3754fe45adbe12d381e997`. The local checkpoint is
`/home/lvx/q38model`, `config.json` revision
`Qwen4ExpForConditionalGeneration`, with `transformers_version`
`5.8.0.dev0`.

## Layer schedule and tensors

`text_config.layer_types` is authoritative: layers 3, 7, 11, 15, 19, 23, 27,
31, 35, 39, 43, and 47 (zero-based) are full-attention/QSA layers. The other
36 layers use GDN. Every full-attention layer has these BF16 tensors:

| Role | Checkpoint suffix | Logical shape |
|---|---|---|
| query plus gate | `self_attn.q_proj.weight` | `[12288, 2560]` |
| key/value | `self_attn.k_proj.weight`, `v_proj.weight` | `[512, 2560]` |
| output | `self_attn.o_proj.weight` | `[2560, 6144]` |
| main Q/K RMS norms | `self_attn.q_norm.weight`, `k_norm.weight` | `[256]` |
| indexer Q/K projection | `self_attn.indexer.index_qk_proj.weight` | `[640, 2560]` |
| indexer Q/K RMS norms | `...q_layernorm.weight`, `...k_layernorm.weight` | `[128]` |

The projection is output-by-input. `q_proj` contains per-head query and gate
vectors interleaved as `[query(256), gate(256)]` for each of 24 heads.
K/V have two KV heads and head dimension 256. The attention gate is sigmoid,
applied after sparse attention and before `o_proj`.

## Position and RoPE

The local text configuration sets `rope_theta=10,000,000`,
`partial_rotary_factor=0.25`, `rope_type=default`,
`mrope_interleaved=true`, and `mrope_section=[11,11,10]`. The reference uses
`ggml_rope_multi` with these sections for both main Q/K and indexer Q/K.
Position IDs come from committed attention-cache positions. Rotary is applied
to Q/K after their RMS norms and before cache writes or attention. V is not
rotated on this path. The exact section/interleaving metadata must be carried
through the runtime; no YaRN or DeepSeek positional rule is substituted.

## Indexer and compression

For a full-attention layer, the indexer reads the same pre-attention stream as
main Q/K/V:

1. `index_k_proj` produces one raw 128-wide key per token; raw keys are written
   to a separate index cache without norm or RoPE.
2. For each query, the reference enumerates visible token cells in cache
   order. Complete groups are the first
   `floor(visible_count / compress_ratio)` contiguous groups. An incomplete
   final group is not scored; it is retained verbatim as the causal tail.
3. Each complete group key is the arithmetic mean of its member raw keys.
4. Group keys are RMS-normalized, then receive multi-section RoPE at the
   position of the group's first visible cell. Indexer Q is projected to four
   128-wide heads, RMS-normalized, and RoPE-transformed at the current query
   positions.
5. Each query/head dot product with each pooled group is rectified with ReLU,
   summed across the four indexer heads, and divided by
   `sqrt(indexer_head_dim)`.
6. The top `indexer_budget / compress_ratio` complete groups are selected with
   stable `(score descending, first-cell ID ascending)` ordering. Selected
   groups expand to all member token cells, in group order.
7. The incomplete tail is appended after selected complete groups. The
   resulting width is `min(visible_count, indexer_budget +
   compress_ratio - 1)`. The sparse attention path unmasks exactly these
   token-cell IDs.

The reference index cache tracks the main attention cache cell-for-cell and
uses per-stream cell mappings. For text-only single-sequence operation, the
runtime must preserve the same logical cell order. Selected IDs are semantic
outputs and must be deterministic; score floating-point comparisons may use a
documented tolerance, but IDs may not change except for a formally equivalent
tie. Top-k is over complete-group scores, not over expanded token scores.

## Main sparse attention

Main Q/K/V are projected from the same input stream. Q and K receive RMSNorm
and multi-section RoPE; V is reshaped but not rotated. Q/K/V are appended to
the main KV cache before the sparse read. Attention is ordinary dense GQA
restricted by a mask whose unmasked cells are exactly the selected indexer
IDs, with the normal causal mask retained. Attention output is multiplied by
the sigmoid gate extracted from `q_proj`, then projected by `o_proj`.

The indexer scores the pre-rotary path independently of main Q/K rotation.
Indexer state and main K/V state are separate allocations and accounting
domains, although both append one logical cell per committed token.

## State and chunking

Position is the number of committed tokens, not the size of a temporary
ubatch. Every prefill partition appends the same raw index key and main K/V
cell in the same order. Compression, causal-tail handling, selected IDs,
gathered positions, and output must therefore be invariant under chunking.
Cache growth preserves prior cells; reset clears main KV, indexer cache,
position, and committed-token count together.

The llama.cpp graph requires token counts divisible by the hyper-connection
stream count (4) for its multi-stream block layout. The scalar/runtime
reference path rejects unsupported multi-sequence sharing rather than silently
inventing predecessor or stream mappings. The Transformers single-sequence
text path uses the cache's causal visible-cell mask.

## Reference source locations

| Detail | Source |
|---|---|
| hyperparameters and layer schedule | `src/models/qwen4exp.cpp::load_arch_hparams` |
| tensor names and shapes | `src/models/qwen4exp.cpp::load_arch_tensors` |
| index cache input and cell mapping | `llm_graph_input_qsa::set_input` |
| block pooling, score, ReLU, expansion, top-k | `Qwen4ExpTextQSAIndexer.forward`, `graph::build_qsa_top_k` |
| RoPE and main Q/K/V order | `graph::build_layer_attn` |
| selected-cell mask and gather | `graph::build_attn_qsa` |

No QSA fusion, flash implementation, SSD movement, or alternate top-k
algorithm is semantic. Those are permitted only after the scalar probes and
the naive CUDA path match this contract. The reference graph and golden
generator load only tensors needed by a requested probe; they never mirror the
checkpoint and never invoke q38 to generate expected values.
