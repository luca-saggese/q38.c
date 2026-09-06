# Q38 GR Contract

This document freezes the semantics of the production Q38 grouped-residual
(GR) subsystem before any GR optimization is promoted. Reference 0 remains
unchanged; this contract applies to isolated GR fixtures and future
candidate comparisons.

## Production call chain

The canonical forward path is:

```text
q38_forward_full_with_backend_config
  -> full_gr_read
     -> full_rms per branch
     -> full_matvec_batch(input_mix_weight_down)
     -> SiLU(/4)
     -> full_matvec_batch(input_mix_weight_up)
     -> sigmoid gates
     -> four-branch average
  -> GDN/QSA or MoE
  -> full_gr_write
     -> full_rms per branch
     -> full_matvec_batch(block_inject_weight)
     -> 2*sigmoid(/4)
     -> residual update
```

`full_gr_read` and `full_gr_write` in `q38_forward.c` are the production
implementation. The standalone `q38_gr_ref.c` and `cuda/q38_gr.cu` entry
points are reference/test implementations and are not called by the
canonical session forward path.

The layer graph uses GR in these positions:

- attention GR read before GDN/QSA;
- attention GR write after GDN/QSA;
- MLP GR read before MoE;
- MLP GR write after MoE;
- a final read-only GR using the global input-mix tensors.

There is no final write for the read-only GR.

## Shapes and parameter families

| Item | Shape |
|---|---:|
| Hidden width per branch | `2560` |
| Branch count | `4` |
| Concatenated hidden width | `10240` |
| Low-rank dimension | `320` |
| `hc_norm` | `[10240]` |
| `input_mix_weight_down` | `[320, 10240]` |
| `input_mix_weight_up` | `[10240, 320]` |
| `block_inject_weight` | `[4, 10240]` |

The same tensor family is bound independently for attention GR and MLP GR on
each layer. All currently bound GR tensors are GGUF BF16 (`type = 30`). The
runtime converts each tensor value to float32 before scalar arithmetic or
backend dispatch.

## Exact production read formula

For token `t`, branch `b` and channel `d`, let:

- `R[t,b,d]` be the residual input;
- `G[b,d]` be `hc_norm`;
- `D[r,i]` be `input_mix_weight_down`;
- `U[i,r]` be `input_mix_weight_up`;
- `i = b * 2560 + d`;
- `H = 2560`, `B = 4`, `K = 320`;
- `epsilon = 1e-6`.

Production normalization uses double-precision accumulation for the sum of
squares, then writes float32 values:

```text
mean_sq[b] = sum_d float64(R[t,b,d] * R[t,b,d]) / H
N[t,b,d] = R[t,b,d] *
           (1 / sqrt(float32(mean_sq[b]) + epsilon)) *
           (1 + G[b,d])
```

The `1 + hc_norm` multiplier is part of the production `full_rms(..., true)`
call. It must not be replaced with `hc_norm` alone.

The low-rank read projection and activation are:

```text
z[t,r] = sum_i D[r,i] * N[t,i]
a[t,r] = SiLU(z[t,r] / 4)
       = (z[t,r] / 4) * sigmoid(z[t,r] / 4)
u[t,i] = sum_r U[i,r] * a[t,r]
Q[t,i] = sigmoid(u[t,i])
```

The GR read output is:

```text
input[t,d] = (1 / 4) * sum_b Q[t,b*H+d] * N[t,b*H+d]
```

## Exact production write formula

For block output `X[t,d]`, `block_inject_weight` `I[b,i]` produces:

```text
v[t,b] = sum_i I[b,i] * N[t,i]
s[t,b] = 2 * sigmoid(v[t,b] / 4)
updated[t,b,d] = R[t,b,d] + s[t,b] * X[t,d]
```

The update is float32 and does not normalize or add another residual after
this operation.

## Dtypes and accumulation

- Inputs, normalized activations, intermediates, outputs, and block output:
  float32.
- GGUF GR weights: BF16.
- BF16 weights are converted to float32 at scalar access/backend boundaries.
- Production RMS sum-of-squares accumulation: float64, then converted to
  float32 for the reciprocal square root.
- Production matrix accumulation is owned by the configured matrix backend;
  the backend receives the BF16 tensor and float32 activation contract.
- No GR quantized tensor family is accepted by the binding code.

## Implementation classification

| Component | Classification |
|---|---|
| `q38_forward.c:full_gr_read/full_gr_write` | `PRODUCTION` |
| `q38_forward.c` layer placement | `PRODUCTION` |
| `q38_weights.c` GR binding | `PRODUCTION` |
| `q38_gr_ref.c` | `REFERENCE`, legacy direct-gamma semantics |
| `cuda/q38_gr.cu` | `REFERENCE/TEST_ONLY`, not canonical dispatch |
| `tests/gr/test_m3_gr_*` | `TEST_ONLY` |

The existing standalone reference/CUDA implementations use a direct gamma
multiplier, whereas production uses `1 + hc_norm`. New GR fixtures and the
independent oracle in this directory use the production contract above.
This discrepancy is recorded here deliberately; it is not silently treated
as an optimization opportunity.

## Fixture and correctness policy

Each fixture records its layer, GR family, tensor shapes, input provenance,
weight dtype, and expected-output provenance. At least early, middle, and late
layer shapes are covered. Captured runtime activations are preferred; any
zero-filled or replayed branch is explicitly marked in fixture metadata.

Every candidate must report:

- max absolute difference;
- max relative difference;
- RMSE;
- NaN/Inf count;
- kernel-only and end-to-end GR-call timings;
- kernel launches and host synchronizations;
- H2D/D2H/D2D bytes.

No candidate may enter the production path unless all fixtures pass without a
fallback, new transfer, or new synchronization.
