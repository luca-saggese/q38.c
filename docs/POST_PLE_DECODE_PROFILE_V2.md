# PERF-V2 — Post-PLE Decode Critical Path

## Capture

The diagnostic harness was rebuilt with the missing `q38_residency_plan.o`
link dependency and run once with one warmup path plus **10 generated
interactions**. Decode samples 1–9 were measured in the same model-resident
process. The Nsight Systems report is
`artifacts/perf/current/nsys_perf_v2/q2_decode_10.nsys-rep`.

Correctness remained GREEN: generated IDs and final logits hash matched across
measured runs, logits were finite, fallback was false, and PLE critical stall
was zero.

## Exclusive token partition

Mean measured decode wall was **129.767 ms/token** (`7.706 tok/s`). The new
partition is non-overlapping and reconciles exactly:

| Partition | ms/token | Wall |
|---|---:|---:|
| Embedding/input | 0.012 | 0.01% |
| Decoder layers | 122.502 | 94.40% |
| Full PLE | 0.791 | 0.61% |
| Final GR | 0.197 | 0.15% |
| LM head | 5.615 | 4.33% |
| Argmax | 0.511 | 0.39% |
| Runtime/bookkeeping | 0.139 | 0.11% |
| **Total** | **129.767** | **100.00%** |

Residual is `0.000 ms` and `0.000%` for the aggregate capture.

## Decoder layers

The 36 GDN layers account for `82.735 ms/token`; the 12 QSA layers account for
`39.767 ms/token`. Their internal exclusive timing is:

| Group | GR | Mixer | MoE | Orchestration | Total |
|---|---:|---:|---:|---:|---:|
| GDN (36) | 17.279 | 49.868 | 15.226 | 0.362 | 82.735 |
| QSA (12) | 5.680 | 28.861 | 5.106 | 0.120 | 39.767 |

The fastest layer was GDN layer 24 at `2.252 ms`; median layer wall was
`2.312 ms`; the slowest was QSA layer 3 at `3.438 ms`.

## Runtime device-boundary evidence

The q2 runtime observer reports per measured token:

- `12,311,808` H2D bytes;
- `8,843,520` D2H bytes;
- `0` D2D bytes;
- `739` real CUDA synchronizations;
- sync attribution: 665 matrix D2H, 48 grouped MoE D2H, 12 QSA QKV,
  12 matrix-batch D2H, and 2 argmax.

This is direct evidence of repeated host/device boundaries in the current
forward path. The q2 callback count is retained only as observational
telemetry; it is not used as the CUDA kernel count.

Nsight Systems reports the complete process capture separately: 11,889 kernel
launches, 22,061 `cudaMemcpyAsync` calls, 9,082 stream synchronizations, and
155 event synchronizations. Those totals include startup, warmup, and measured
decode, so they are not presented as per-token decode values.

## CUDA timeline

The captured GPU interval was `3,170.104 ms`, with `1,827.733 ms` of kernel
execution and a timeline busy fraction of `57.66%`. There were 11,888
inter-kernel gaps; the longest was `12.985 ms` and the p95 was `0.554 ms`.
The kernel distribution was:

| Duration | Count | GPU time (ms) |
|---|---:|---:|
| `<5 us` | 2,697 | 6.731 |
| `5–20 us` | 2,666 | 39.044 |
| `20–100 us` | 3,582 | 236.247 |
| `100 us–1 ms` | 2,705 | 590.347 |
| `>1 ms` | 239 | 955.365 |

The dominant kernel was `matrix_batch_generic_kernel` with `1,081.379 ms`
across 675 instances in the complete trace, followed by
`bf16_matvec_kernel` with `348.848 ms` across 6,939 instances.

The trace did not contain NVTX ranges. Therefore, CUDA-only stage attribution
is not claimed; the stage/layer attribution in this report comes from the
exclusive q2 timing tree and sync-reason instrumentation. A future trace can
add NVTX ranges, but no optimization or second model load was performed here.

## Bottleneck shortlist

1. **GDN mixer — 49.868 ms/token (38.43%)**: 36 GDN mixer stages dominate the
   decoder. Validate with the existing GDN fixture and an NVTX-labeled trace.
2. **QSA mixer — 28.861 ms/token (22.24%)**: 12 QSA mixer stages remain
   critical. Validate with the QSA layer fixture and CUDA boundary trace.
3. **MoE — 20.332 ms/token (15.67%)**: routed/shared expert execution and
   grouped reduction work. Validate with the existing MoE fixture.
4. **GR — 22.958 ms/token (17.69%)**: repeated GR projection segments across
   decoder layers. Validate with the GR C1/C2/C3 fixtures.
5. **LM head — 5.615 ms/token (4.33%)**: final vocabulary projection remains a
   distinct stage. Validate with the resident LM-head microbenchmark.

No optimization was implemented in PERF-V2.
