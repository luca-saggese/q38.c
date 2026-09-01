# Qwen3.8-Flash-Next PLE/n-gram semantics

This document freezes the semantic contract for the PLE (per-layer
embedding) n-gram path. It separates logical model semantics from checkpoint
and runtime physical layout. No hash, head mapping, or injection behavior
may be inferred from tensor dimensions alone.

## Reference identity

The executable reference is `ggml-org/llama.cpp` PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742),
read from `refs/pull/27742/head`:

- `src/models/qwen4exp.cpp`
  - `llama_model_qwen4exp::load_arch_hparams`
  - `llm_graph_input_ple::set_input`
  - `llama_model_qwen4exp::graph::build_ple`
- `src/models/qwen3next.cpp`
  - shared model and recurrent graph conventions
- `conversion/qwen4exp.py`
  - `set_gguf_parameters`
  - `modify_tensors`
  - `_place_ple_shard`

The local frozen configuration is `/home/lvx/q38model/config.json`, and the
local binding intentionally accepts the fixture's PLE tensor names under
`model.language_model.layers.1.ple`. The HF configuration expresses
`ple_layer_ids: [2]` as a one-based layer ID; the converter writes zero-based
GGUF layer metadata (`[1]`). These are the same semantic layer, not an
unresolved dimensional inference.

## Frozen configuration

| Parameter | Value | Source |
|---|---:|---|
| n-gram size | 3 | HF `ngram_size`; `load_arch_hparams` |
| heads per n-gram order | 8 | HF `heads_per_ngram` |
| total PLE heads | 16 | `(ngram_size - 1) * heads_per_ngram` |
| PLE embedding row width | 2560 per head | `embedding_length_per_layer_input`; `build_ple` |
| base vocabulary size | 20,000,000 | HF `ngram_vocab_size_base` |
| shard count | 128 | HF `split_ngram_parts`; `_place_ple_shard` |
| PLE convolution kernel | 4 | HF `ple_conv_kernel_size` |
| PLE layer | semantic layer 2 / zero-based layer 1 | HF `ple_layer_ids`; `set_gguf_parameters` |
| hidden size | 2560 | model config |
| hyper-connection stream count | 4 | HF `hc_count`; `build_ple` |

The active PLE head count is `(n - 1) * heads_per_ngram`; there are no
unigram heads. For the frozen model, 8 heads represent bigrams and 8 heads
represent trigrams.

## Token history and logical n-gram composition

For each current token `t[0]`, the logical context is:

```text
ctx[0] = current token
ctx[1] = immediately preceding token, or EOS
ctx[2] = two positions preceding, or EOS
```

The preceding tokens are obtained from the attention memory, including tokens
already stored from the current ubatch. A missing predecessor is treated as
EOS. If an EOS is encountered while walking backwards, all older positions in
that context are replaced by EOS. The EOS token itself does not remove its own
preceding context.

The state is session token history, not a PLE row cache. It must survive
prefill chunk boundaries and reset according to the same EOS rule.

For an n-gram order `n` in `{2, 3}`, the logical token tuple is
`(ctx[0], ..., ctx[n-1])`. There is no unigram lookup.

## Hash and row-index semantics

For order `n`, the reference computes a 64-bit unsigned mixed value:

```text
mixed_n = (uint64_t)ctx[0] * multiplier[0]
for j = 1 .. n-1:
    mixed_n ^= (uint64_t)ctx[j] * multiplier[j]
```

For head `g` within that order:

```text
head = (n - 2) * heads_per_ngram + g
row  = mixed_n % head_vocab_sizes[head] + head_offsets[head]
```

The multiplication is performed in unsigned 64-bit arithmetic and the XOR is
bitwise XOR on the resulting products. The modulo is applied before adding
the head's row offset. The resulting row index is a signed 32-bit gather ID
in the reference graph, after validating that the configured head range fits
the table.

The three multiplier values and the per-head offsets/vocabulary sizes are
int64 checkpoint constants. They must be loaded exactly; converting them
through float32 is forbidden because the multipliers can exceed the exact
integer range of float32.

If the input batch contains embeddings without token IDs (for example an
image batch), the reference uses `ple_image_token_id` when present, otherwise
the configured EOS ID as the stand-in token for hashing.

## Logical embedding table and head order

The shared table is logically a flat row table:

```text
per_layer_token_embd[global_row, head_component]
global_row: 0 .. total_rows - 1
head_component: 0 .. 2559
```

Each token position gathers 16 rows, one per PLE head. The gathered result is
flattened with the head dimension slowest in the graph:

```text
emb[position] =
    concat(row[head=0], row[head=1], ..., row[head=15])
shape per position: [16 * 2560]
```

The order groups are contiguous by `heads_per_ngram`, so the exact mapping is:

```text
bigram:  head = 0 .. 7
trigram: head = 8 .. 15
```

The reference's `ple_n_heads` is `(ngram_size - 1) * heads_per_ngram`.
The row-index loop in
`llm_graph_input_ple::set_input` is authoritative; implementations must use
that loop directly rather than infer ranges from tensor ratios.

The table is emitted by the converter as `per_layer_token_embd.weight`.
Checkpoint tensors named `.ngram_embedding.shard_<index>` are concatenated in
ascending shard-index order. Shard arrival order is not semantic. Every shard
must have the same row width; the converter emits one logical table and does
not keep the full table eagerly resident during conversion.

## PLE projections, gate, and convolution

For the configured PLE layer, the gathered embedding is projected as:

```text
key   = W_ple_key   * emb
value = W_ple_value * emb
query = W_ple_norm_query(hidden)
```

The reference uses grouped normalization over `hc_count` streams of hidden
size 2560. `ple_norm_key`, `ple_norm_query`, and `ple_norm_conv` are applied
after reshaping to `[hidden_size, hc_count, tokens]`, followed by the learned
weight. The converter adds one to the stored Gemma-centered norm weights.

The per-stream score and gate are:

```text
s    = sum(key * query) / sqrt(hidden_size)
mag  = sqrt(clamp(abs(s), 1e-6, +inf))
gate = sigmoid(sign(s) * mag)
```

The value is broadcast across all hyper-connection streams and gated:

```text
gated = repeat(value, hc_count) * gate
```

The gated value is normalized with `ple_norm_conv`, then passed through a
depthwise causal convolution. The convolution uses the `[kernel, channel]`
logical weights, kernel size 4, and dilation equal to the n-gram size (3):

```text
conv[t] = sum(k=0..3,
              ple_conv1d[k, channel] *
              normalized[t - (3 - k) * 3, channel])
conv_out = SiLU(conv)
```

The required convolution history is `(kernel - 1) * dilation = 9` prior
positions per sequence. It is persistent session state and must be prepended
before every chunk so single-shot and chunked prefill are equivalent.

## Injection point and aggregation

The reference returns the PLE result from `build_ple` as:

```text
ple_out = hidden + gated + conv_out
```

This is the exact injection point: the PLE branch is added to the hidden
stream at the configured PLE layer, after the PLE key/value projections,
grouped norms, signed-square-root gate, gated value normalization, causal
depthwise convolution, and SiLU. It is not routed-expert output, GR state, or
GDN recurrence state.

## Logical versus physical layout

The following are semantic and must remain invariant:

- token history order and EOS reset behavior;
- multiplier/XOR/modulo row-index formula;
- bigram/trigram head order;
- one gathered 2560-element row per active PLE head;
- grouped stream reshapes and gate equation;
- dilation-3 causal convolution and nine-token history;
- `hidden + gated + conv_out` injection.

The following are physical implementation choices and may change later:

- GGUF tensor dimension order and byte packing;
- shard file layout and mmap strategy;
- host versus CUDA execution of hash/index generation;
- row cache, staging buffers, deduplication, and prefetch;
- Q2/Q4 row storage and dequantization;
- GB10-specific physical layout.

No GB10 physical layout is frozen by this document. Optimizations may only be
introduced after the scalar reference path and golden vectors pass.

## Verified source locations

| Detail | Source |
|---|---|
| PLE metadata and exact int64 constants | `conversion/qwen4exp.py::set_gguf_parameters`, `modify_tensors` |
| Shard ordering/concatenation | `conversion/qwen4exp.py::_place_ple_shard` |
| Layer metadata conversion | `conversion/qwen4exp.py::set_gguf_parameters` |
| Hash, EOS reset, missing predecessor, multimodal stand-in | `src/models/qwen4exp.cpp::llm_graph_input_ple::set_input` |
| Gather flatten and head order | `src/models/qwen4exp.cpp::llama_model_qwen4exp::graph::build_ple` |
| Gate equation | `src/models/qwen4exp.cpp::...::build_ple` |
| Dilated causal convolution and injection | `src/models/qwen4exp.cpp::...::build_ple` |

## M4-C06 evidence boundary

The checked-in `tools/generate_m4_c06_goldens.py` is an offline oracle. It
reads the three int64 hash-constant tensors directly from the local
safetensors checkpoint and applies the frozen `llm_graph_input_ple::set_input`
loop; it does not import, link, or execute q38. The resulting
`artifacts/m4/ple_injection_golden.json` contains deterministic BOS, boundary,
repetition, EOS-reset, and chunk-history cases with token IDs and all 16 row
IDs, plus source/model revisions and SHA-256 checksums.

Checkpoint inspection also records the physical PLE row width as 160 BF16
values. Sixteen heads therefore form the logical 2560-wide gathered embedding.
The 2560-wide synthetic rows used by the earlier decoder tests are not a
claim about the production checkpoint layout.

This repository does not yet contain a complete q38 model-forward graph, and
the local Qwen checkpoint is approximately 336 GiB. No independent hidden
vectors are available without running that missing/unsuitable full forward.
Consequently the C probe validates the production hash/row-ID path and records
the exact reference injection boundary (`hidden + gated + conv_out`), while
`hidden_before_ple`, `hidden_after_ple`, and `ple_contribution_vector` remain
explicitly `null`. The probe rejects fabricated vectors; a future full-forward
oracle can populate these fields without changing the file format.
