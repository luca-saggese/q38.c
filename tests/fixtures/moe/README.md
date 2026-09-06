# Compact MoE fixture pack

This directory is reserved for promotion-grade early, middle, and late MoE
captures. The benchmark expects these subdirectories:

```text
early/
middle/
late/
```

Each layer directory must contain the following binary files:

| File | Shape and type |
|---|---|
| `hidden.f32` | `[2560]` FP32 |
| `router.bf16` | `[512, 2560]` BF16 |
| `selected_experts.u16` | `[10]` uint16 |
| `selected_weights.f32` | `[10]` FP32 |
| `selected_gate_up.q2_k` | `[10, 1280, 10]` Q2_K blocks |
| `selected_down.q2_k` | `[10, 640, 10]` Q2_K blocks (production-transposed storage) |
| `shared_gate.bf16` | `[640, 2560]` BF16 |
| `shared_up.bf16` | `[640, 2560]` BF16 |
| `shared_down.bf16` | `[2560, 640]` BF16 |
| `shared_gate_weight.bf16` | `[2560]` BF16 |
| `router_logits_pre.f32` | `[512]` FP32 |
| `router_logits_effective.f32` | `[512]` FP32 |
| `selected_weights_pre.f32` | `[10]` FP32 |
| `expected_routed.f32` | `[2560]` FP32 |
| `expected_shared.f32` | `[2560]` FP32 |
| `expected.f32` | `[2560]` FP32 |

The selected Q2 tensors are stored in route order, not expert-ID order. The
fixture extractor must write provenance metadata separately and must refuse to
label replayed or synthetic tensors as real captures. Until all three complete
real captures exist, `make bench-moe` must fail closed rather than benchmark
an old wrapper or a synthetic substitute.
