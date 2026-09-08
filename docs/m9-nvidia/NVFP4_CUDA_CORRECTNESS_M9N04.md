# M9N-04 NVIDIA NVFP4 CUDA correctness

M9N-04 adds a readable CUDA correctness harness for the already frozen NVIDIA
NVFP4 contract. It does not change the importer, the materialized pack, CUDA
residency, or the external PLE path.

## Scope

The harness uses the existing four-row/64-value fixtures for C0-C2 and adds
compact full-expert fixtures for expert 0 at layers 0, 24, and 47. The full
fixtures contain the real packed U8 weights, E4M3 block scales, scalar
metadata, and a `[2560]` F32 hidden activation. They do not contain dense
dequantized expert weights.

The hidden activations are the existing real q38 runtime captures:

```text
tests/fixtures/moe/early/hidden.f32
tests/fixtures/moe/middle/hidden.f32
tests/fixtures/moe/late/hidden.f32
```

The NVFP4 weights and metadata are ranged reads from the pinned NVIDIA
checkpoint revision. Each fixture's `metadata.json` records source tensor
names, shard, byte offsets, shapes, dtypes, and SHA-256 payload hashes.

## CUDA path

The C0 primitive reads packed FP4 weights and scales directly, applies
`weight_scale_2`, dequantizes the packed activation with its E4M3 block
scales, and accumulates in FP32. C1 is the CUDA activation quantizer. C2 is
the end-to-end activation-quantize plus projection path.

C3 executes one expert as sequential readable kernels:

```text
F32 hidden
  -> gate activation quantization -> gate NVFP4 matvec
  -> up activation quantization   -> up NVFP4 matvec
  -> SiLU(gate) * up
  -> down activation quantization
  -> down NVFP4 matvec
```

Scratch buffers are allocated once for the fixture run. No projection call
performs `cudaMalloc` or `cudaFree`, and no host-side dense weight conversion
is performed. The harness uses the fixed `1e-5` maximum absolute comparison
tolerance from the M9N-02 validation path; it does not select tolerance from
the observed result. The isolated NVFP4 contract specifies FP32 accumulation
but does not define a BF16 boundary between gate, up, SiLU, and down, so C3
keeps those intermediate values in FP32 rather than introducing an
uncontracted conversion.

## Results

All three full experts produced exact packed activation bytes and exact
activation scale bytes for gate, up, and down. Every result had zero NaN and
zero Inf. The largest full-expert error was the late gate projection:
`max_abs = 5.48362732e-06`, below the fixed tolerance.

| Stage | Layer | Gate max_abs / RMSE | Up max_abs / RMSE | SiLU*up max_abs / RMSE | Down max_abs / RMSE |
|---|---:|---:|---:|---:|---:|
| early | 0 | `1.31130219e-06` / `2.69203235e-07` | `1.13248825e-06` / `2.20650582e-07` | `9.53674316e-07` / `6.31518873e-08` | `5.96046448e-08` / `6.821277e-09` |
| middle | 24 | `2.02655792e-06` / `3.70096929e-07` | `2.14576721e-06` / `3.63647575e-07` | `6.85453415e-07` / `9.18684795e-08` | `6.70552254e-08` / `1.05629903e-08` |
| late | 47 | `5.48362732e-06` / `8.31878410e-07` | `4.29153442e-06` / `6.90055332e-07` | `2.02655792e-06` / `2.53529557e-07` | `1.34110451e-07` / `2.89109996e-08` |

The machine-readable report is
`artifacts/m9-nvidia/nvfp4_cuda_correctness_m9n04.json`.

## Reproduction

```text
make m9n-nvfp4-cuda-test
make m9n-nvfp4-cuda-full-test
```

The full test uses only the compact fixture payloads under
`tests/nvfp4/fixtures/full_expert_{early,middle,late}/`. It does not load the
full checkpoint or run model inference.

M9N-05 Tensor Core, cuBLASLt, CUTLASS, grouped top-10 execution, and
production performance work are intentionally not included.
