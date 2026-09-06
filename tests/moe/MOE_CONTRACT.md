# MoE subsystem contract

This document freezes the MoE semantics and the production callback boundary
before any S2 optimization. The benchmark must reproduce this contract; old
per-expert wrappers and test-only kernels are not valid baselines.

## Fixed geometry

| Item | Value |
|---|---:|
| hidden size | 2560 |
| routed experts | 512 |
| selected experts | 10 |
| routed intermediate | 640 |
| router tensor | `[512, 2560]`, bias-free |
| routed gate/up tensor | `[512, 1280, 2560]`, Q2_K in the current production path |
| routed down tensor | logical `[512, 2560, 640]`; stored `[512, 640, 2560]`, Q2_K |
| shared gate/up tensors | `[640, 2560]`, BF16 |
| shared down tensor | `[2560, 640]`, BF16 |
| shared gate tensor | `[1, 2560]`, BF16 |

## Formula and dtype order

For one hidden vector `x`:

1. Compute router logits `l[e] = dot(router[e], x)` with FP32
   accumulation over BF16 router weights.
2. Keep the FP32 logits as `logits_pre_cast`, round each to BF16 and use that
   value as `logits_effective`.
3. Compute FP32 softmax probabilities for both vectors.
4. Select the ten experts by descending `probs_pre_cast`, breaking exact ties
   by ascending expert ID.
5. Take the selected values from `probs_effective`, renormalize their sum to
   one, and round each normalized weight to BF16.
6. For each selected expert, compute:

   `r_e = down_e(SiLU(gate_e(x)) * up_e(x))`

   using the Q2_K dequantized values and FP32 accumulation.
7. Compute the shared expert with BF16 weights and FP32 accumulation:

   `s = down_shared(SiLU(gate_shared(x)) * up_shared(x))`

   and multiply `s` by `sigmoid(shared_gate(x))`.
8. The layer output is:

   `y = sum_k route_weight[k] * r_selected[k] + s`.

No auxiliary routing loss, router bias, implicit scaling, or shared-expert
contribution is included in the routed expert array.

## Production call chain

The current runtime installs this callback chain:

```text
q38_forward_full_with_matrix_moe_layer_backend
  -> full_moe
     -> full_matvec_batch(router)
     -> host pre-cast/effective logits, softmax, top-k, renormalization
     -> q38_forward_cuda_moe_layer_q2_backend
        -> one H2D hidden upload
        -> for 10 selected experts:
           q38_moe_cuda_q2_gate_up
           q38_moe_cuda_q2_down
           q38_moe_cuda_accumulate_weighted
        -> one D2H output copy and one stream synchronization
     -> shared gate/up/down through the matrix backend
     -> host shared sigmoid and final addition
```

The production callback is wired in `q38_session.c` and in the worker path in
`tests/q38_dev_worker.c`. Its routed layer backend requires resident expert
weights and reuses persistent device workspaces.

## Classification

| Implementation | Classification | Reason |
|---|---|---|
| `full_moe` in `q38_forward.c` | production | owns router dtype, selection, weight normalization, and shared-expert semantics |
| `q38_forward_cuda_moe_layer_q2_backend` | production | installed layer-level routed-expert callback |
| `q38_moe_cuda_q2_gate_up` | production | called by the installed Q2 layer callback |
| `q38_moe_cuda_q2_down` | production | called by the installed Q2 layer callback |
| `q38_moe_cuda_accumulate_weighted` | production | called once per selected expert by the installed callback |
| `q38_moe_cuda_route` | legacy/diagnostic | performs a host route path and is not the callback's authoritative route |
| `q38_moe_cuda_expert_q2_workspace` | legacy/diagnostic for S2 | valid kernel API, but it represents one expert and includes a different orchestration boundary |
| `q38_moe_cuda_shared_f32` | test-only/diagnostic | not used by `full_moe`'s current shared BF16 matrix path |
| `q38_moe_ref.c` | reference | useful scalar reference, but does not model the production BF16 router/effective-weight distinction exactly |

## Fixture provenance gate

The repository currently contains one complete real MoE trace in
`artifacts/m6/transformers_reference.json` (`layer2_moe_trace`) and routing
summaries for all 48 layers. It does not contain complete real hidden,
router-weight, and selected Q2 slice captures for early/middle/late layers.
The S2 extractor is therefore separate from the benchmark and must be run
only when an authorized compact tensor source is available. The benchmark
must fail closed when a fixture is missing; synthetic tensors are useful for
kernel smoke tests but are not promotion-grade MoE fixtures.
