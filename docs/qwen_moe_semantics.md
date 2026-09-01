# Qwen4Exp MoE semantics

This contract is taken from `Qwen4ExpTextTopKRouter`,
`Qwen4ExpTextExperts`, and `Qwen4ExpTextSparseMoeBlock` in the official
Transformers implementation inspected on 2026-09-01. The llama.cpp PR #27742
uses the same `build_moe_ffn` formulation.

* The router is a bias-free linear projection with weight shape
  `[num_experts, hidden_size]`.
* Router logits are `hidden @ weight.T`; with the checkpoint's BF16 model dtype
  the linear result is BF16. Softmax is evaluated in float32 after that
  effective BF16 result is produced.
* `topk` selects 10 experts by descending probability. Transformers delegates
  equal-value ordering to `torch.topk`; the independent q38 reference uses
  explicit `(probability descending, expert ID ascending)` ordering. q38 keeps
  that pre-cast selection until the layer-2 effective-logit diagnostic agrees.
* Selected probabilities are renormalized to sum to one
  (`norm_topk_prob` is true for this checkpoint), then converted back to the
  router output dtype (BF16).
* Routed expert weights are separate 3-D tensors:
  `gate_up_proj [512, 2*640, 2560]` and
  `down_proj [512, 2560, 640]`. For each selected expert:
  `silu(gate_proj(x)) * up_proj(x)`, followed by `down_proj`.
* Routed outputs are multiplied by their normalized routing probability and
  accumulated per token. The shared expert is not part of the routed expert
  array. It uses ordinary gate/up/down projections with intermediate width
  640, and its output is multiplied by
  `sigmoid(shared_expert_gate(x))` before being added to the routed result.
* There is no router bias, auxiliary routing term in the forward result, or
  implicit DeepSeek scaling.

The reference/runtime APIs keep router, routed experts, and shared expert
storage separate. Routed weights may remain file-backed and be materialized
only for selected expert slices; expected values must come from the reference
calculation, never from q38.

## Frozen source and checkpoint evidence

The primary source is
`transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`:
`Qwen4ExpTextTopKRouter.forward`,
`Qwen4ExpTextExperts.forward`, and
`Qwen4ExpTextSparseMoeBlock.forward`. The secondary implementation is
`llama.cpp` PR #27742, `src/models/qwen4exp.cpp::build_layer_ffn`.
`/home/lvx/q38model/config.json` confirms 512 experts, top-10 routing,
intermediate widths 640, hidden size 2560, and `norm_topk_prob` behavior is
verified from the reference code rather than inferred from config alone.
