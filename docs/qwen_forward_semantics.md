# Qwen4Exp text forward/decode ordering

The reference order is frozen from the official Transformers
`Qwen4ExpTextModel.forward` and `Qwen4ExpTextDecoderLayer.forward`, with the
QSA cache details cross-checked against llama.cpp PR #27742.

1. Gather the token embedding and keep the original token IDs for PLE.
2. Build full text position IDs from committed cache length. The text
   coordinate is the first of four mRoPE coordinates; text-only H/W
   coordinates equal the text coordinate.
3. Repeat the embedding into four hyper-connection streams.
4. For each layer, apply PLE when configured, then the attention gated
   residual read. Run GDN on linear-attention layers or QSA attention on
   full-attention layers. Write the block output through the gated residual.
5. Apply the second gated residual read, run the sparse MoE, add the separately
   sigmoided shared expert, and write the result back to the four streams.
6. Collapse the final streams through the hyper-connection mixer. This mixer
   is the output norm; there is no separate final RMSNorm. Apply `lm_head`.

The hyper-connection, QSA, and PLE RMSNorm weights use the Transformers
parameterization `normalized * (1 + weight)`. The GDN recurrent output norm is
the gated-norm variant and uses its direct multiplicative weight. These two
parameterizations must not be conflated in the native reference path.

Prefill and decode use the same equations. Decode is a one-token ubatch whose
position is the number of committed tokens before the token; cache writes are
committed before subsequent reads. The native reference graph keeps this
ordering and exposes a QSA layer-level probe first. It uses mmap-backed tensor
views where available and allocates only token activations, selected KV, and
state growth; it does not create a dequantized whole-model mirror.

No hidden state or logit is accepted as a golden unless it is computed from
the checkpoint by the independent reference generator. Missing full-model
coverage is reported explicitly rather than replaced with fabricated values.

## Frozen numeric and dtype contract

This contract was frozen on 2026-09-01 by comparing the local Transformers
5.16.1 source, the upstream
`transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, and the
independent checkpoint reader in `tools/m6_transformers_reference.py`.
`/home/lvx/q38model/config.json` declares `bfloat16`; the GGUF keeps the same
BF16 tensors and does not change the model ABI.

| Boundary | Transformers reference | q38 reference path | Contract/status |
| --- | --- | --- | --- |
| Token embedding and PLE embedding | BF16 | decoded to FP32 | checkpoint-derived; no fabricated rows |
| Four hyper-connection streams | BF16 | FP32 | same equations, wider native accumulator |
| Hyper-connection RMSNorm | variance in FP32, result cast to input BF16 | FP32 variance/result | equation and epsilon match |
| GDN/QSA projections and attention | BF16 linear outputs; reductions in FP32 where the source does so | FP32 activations, file weights decoded on read | same ordering; widening is explicit |
| Gated residual projections | BF16 tensors and BF16 output | FP32 | `/hc_count`, SiLU, sigmoid, and `2*sigmoid` boundaries match |
| Router projection (`router_logits`) | BF16 `F.linear` result | FP32 scalar dot plus explicit BF16 round | effective values expose the official output dtype |
| Router softmax | FP32 over effective BF16 logits | FP32 over effective logits | exact boundary; no BF16 softmax |
| Router top-k IDs | `torch.topk` over FP32 probabilities | existing deterministic pre-cast ordering | not switched until effective logits coincide |
| `router_top_value` before cast | FP32 selected probabilities, renormalized when configured | FP32 effective probabilities, renormalized over selected IDs | `selected_weights_pre_cast` |
| `router_top_value` after cast | cast to `router_logits.dtype` (BF16) | explicit BF16 round | `selected_weights_effective`, used for routed accumulation |
| Routed expert gate/up, SiLU, down, weighted sum | BF16 module activations and result | FP32 activations and accumulation | real file-backed computation; layer 2 is traced |
| Shared expert gate and MLP | sigmoid gate and BF16 result | sigmoid gate and FP32 result | separate shared output is added afterward |
| Mixer/output norm and LM head | BF16 model tensors | FP32 activations with decoded weights | same final graph; token-major FP32 output |

The layer-2 diagnostic records both the FP32 scalar dot and effective
BF16-rounded value for all 512 experts. It also records effective probability
ranks 1--15, the rank-10/rank-11 effective probability margin, selected weights
on both sides of the router-dtype cast, and routed expert output before the
shared expert is added. This is diagnostic data only: top-k selection is not
switched to effective logits until those logits have been shown to coincide
with the reference boundary.
