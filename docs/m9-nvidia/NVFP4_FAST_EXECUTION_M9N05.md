# M9N-05 NVIDIA NVFP4 fast execution

M9N-05 compares fast fixture-only CUDA candidates on GB10/SM121 against the
frozen `NVFP4_CUDA_ORACLE_V1` from M9N-04. The oracle source file was not
modified.

No full model, PLE, MTP, server, router, or decoder path was loaded or
executed. The canonical NVFP4 pack was not modified.

## Benchmark setup

The harness is `tests/nvfp4/nvfp4_fast_bench.cu`. It uses resident device
buffers populated once before timing and CUDA events with 100 warmup and 100
measured iterations. The generated top-10 bundles are local-only:

```text
artifacts/m9-nvidia/m9n05-bundles/{early,middle,late}/
```

They are regenerated from the pinned checkpoint by
`tools/q38_nvfp4_bundle_fixture.py` and are ignored by Git. The single-expert
inputs are the frozen M9N-04 full-expert fixtures. The routed bundles use the
existing real selected-ID and route-weight captures; the middle NVFP4 weights
are layer 24 while its existing hidden capture is labeled layer 23, so that
provenance is explicit in the bundle metadata.

Effective bandwidth counts packed weight bytes plus E4M3 block-scale bytes:

```text
one projection:  921,600 bytes
single expert:  2,764,800 bytes
top-10 bundle:  27,648,000 bytes
```

The previously observed GB10 read ceiling is `237.234382 GB/s`.

## Candidate results

| Stage | C0 full expert | C2 fused gate/up | C0 top-10 | C4 grouped top-10 |
|---|---:|---:|---:|---:|
| early | `78.368 us` | `70.080 us` | `891.328 us` | `481.632 us` |
| middle | `78.304 us` | `70.176 us` | `911.808 us` | `471.520 us` |
| late | `80.192 us` | `70.720 us` | `905.632 us` | `473.536 us` |

`NV-C1` direct cooperative matvec was slower than the frozen C0 full-expert
path at approximately `85.6–88.0 us`, so it is not promoted independently.
`NV-C2` fuses gate/up and reduces the single-expert launch count from 7 to 4.
`NV-C4` performs the complete top-10 path with four launches:

```text
quantize input
grouped gate/up + SiLU
grouped down activation quantization
grouped down weighted deterministic accumulation
```

The custom projection timings for NV-C1 were:

| Stage | Gate | Up | Down | Fused gate/up (C2) |
|---|---:|---:|---:|---:|
| early | `31.008 us` | `31.744 us` | `24.992 us` | `45.504 us` |
| middle | `31.136 us` | `30.752 us` | `25.696 us` | `45.440 us` |
| late | `31.008 us` | `31.808 us` | `25.024 us` | `45.504 us` |

Top-10 wall improvement versus C0 is:

| Stage | Improvement | Speedup |
|---|---:|---:|
| early | `45.964672%` | `1.850641x` |
| middle | `48.287359%` | `1.933763x` |
| late | `47.712094%` | `1.912488x` |

The promoted fixture-only candidate is therefore `NV-C4`.

## Correctness

Every custom candidate comparison passed against the frozen oracle with
`max_abs <= 1e-5`. Selected IDs were preserved exactly, and all cases had
zero NaN and zero Inf.

| Stage | Single-expert max_abs / RMSE | Top-10 max_abs / RMSE |
|---|---:|---:|
| early | `9.31322575e-09` / `2.14790317e-09` | `2.23517418e-08` / `3.16000218e-09` |
| middle | `1.49011612e-08` / `3.37355720e-09` | `1.11758709e-08` / `2.19258599e-09` |
| late | `4.47034836e-08` / `1.05205134e-08` | `2.23517418e-08` / `4.58665625e-09` |

## cuBLASLt result

The separate cuBLASLt probe initializes successfully, but the installed CUDA
13 cuBLASLt interface exposes scalar A/B scale pointers and no binding for the
checkpoint's canonical per-16-value E4M3 block scales. No permanent checkpoint
swizzle or conversion was performed, and the candidate is recorded as
`not_promotable` rather than substituting an incorrect W4A16/scalar-scale
operation.

M9N-05 stops here. No full decoder integration or full-model inference is
included.
