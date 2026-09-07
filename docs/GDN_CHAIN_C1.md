# GDN_CHAIN_C1

GDN_CHAIN_C1 promotes the existing resident GDN device island as the
explicit production contract for single-token decode. The chain keeps the
input projection results, recurrent state, convolution history, gated output,
and output-projection workspace on the device. The host-visible boundary is
one input upload and one output download per layer.

The projection entry point `q38_cuda_bf16_matvec_device()` is enqueue-only and
reuses the existing `bf16_matvec_kernel`. It performs no allocation, host
transfer, or synchronization. The four input projections and output
projection therefore remain device-to-device within the CUDA stream; the
existing fused recurrent and history-update kernels are unchanged.

## Structural gate

The early, middle, and late fixtures passed with finite outputs, recurrent
state, convolution history, deterministic repeat behavior, and the existing
GDN tolerances. Each measured layer reports seven launches, one final stream
wait, 10,240 input H2D bytes, and 10,240 output D2H bytes. There are no
intermediate projection transfers or projection synchronizations.

Normal decode state remains device-resident. Host state synchronization is
still available only through the explicit diagnostic/state-snapshot path.

## Isolated benchmark

The fixture benchmark uses 100 warmup iterations and 1,000 measured
iterations, with persistent allocations excluded from the timed loop.

| Fixture | GDN_CHAIN_C1 median | Integrated-style C2 median | Reduction |
| --- | ---: | ---: | ---: |
| early | 580.515 us | 656.148 us | 11.5% |
| middle | 591.475 us | 637.315 us | 7.2% |
| late | 569.123 us | 667.540 us | 14.7% |

The chain is correctness-green on all three fixtures. The complete-layer
10% promotion threshold is met for the early and late fixtures, while the
middle fixture is already dominated by the same resident projection work and
does not meet that threshold in isolation. No kernel tuning was started.
