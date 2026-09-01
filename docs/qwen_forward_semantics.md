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
