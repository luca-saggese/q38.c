# GDN contract

Status: `S3-C3 / GDN_OPT_V1`

This document freezes the standalone contract used by the GDN fixture suite.
It is derived from the current forward implementation and is not a
description of the older M3 CUDA-only probes.

## Production call-chain

The current full-forward path is:

```text
q38_forward_full
  -> full_gr_read
  -> full_gdn
       -> q38_forward_cuda_gdn_layer_backend (single-token CUDA path)
            -> BF16 projections
            -> fused device convolution/SiLU, Q/K/V preparation, FP32 gates,
               recurrence, recurrent RMSNorm, and sigmoid(Z) gate
            -> device history writeback
            -> device output projection
       -> full_matvec_batch (multi-token fallback)
  -> full_gr_write
```

For the CUDA runtime configuration, `full_matvec_batch` calls
`q38_forward_cuda_matrix_batch_backend`. Decode-time single-token GDN now uses the persistent device island above;
chunked multi-token prefill retains the generic batch/CPU path because it
amortizes one launch across the chunk. The single-token island performs one
final output D2H and one stream synchronization; recurrent state and
convolution history are not transferred per token.

The device state is authoritative while the single-token island is active.
The production session installs a state-sync callback that copies the complete
device state and history back to the host only before an explicit decode trace
snapshot. This preserves the measured no-state-transfer token path while
keeping diagnostics and host snapshots coherent. The C3 backend is installed
strictly for the CUDA session; it does not silently fall back to the stale host
mirror after a device-path failure.

| Component | Classification | Evidence |
|---|---|---|
| `full_gdn` in `q38_forward.c` | **PRODUCTION** | Called for every `Q38_LAYER_LINEAR_ATTENTION` layer |
| `full_matvec_batch` GDN projection calls | **PRODUCTION** | Used by `full_gdn` for all five projections |
| `q38_forward_cuda_matrix_batch_backend` | **PRODUCTION** | Installed by the CUDA session backend for non-island and prefill paths |
| `q38_cuda_matrix_batch_generic` | **PRODUCTION** | Retained for multi-token prefill and non-GDN matrix batches |
| `q38_forward_cuda_gdn_layer_backend` | **PRODUCTION S3-C3 / GDN_OPT_V1** | Complete single-token GDN device island |
| `q38_cuda_gdn_project`, `q38_cuda_gdn_fused_recurrent`, `q38_cuda_gdn_history_update` | **PRODUCTION S3-C3 / GDN_OPT_V1** | Invoked by the single-token GDN island |
| `q38_gdn_ref_run` | **PRODUCTION REFERENCE** | Current CPU recurrence called by `full_gdn` |
| `tests/test_m3_*` GDN paths | **TEST_ONLY** | Deterministic synthetic M3 coverage |

## Geometry and state

- Model layers: 48.
- GDN layers: 36; full-attention layers are excluded from this suite.
- Fixture layers: 0 (early), 24 (middle), 46 (late).
- Input hidden: `[tokens, 2560]`, F32 activation.
- Output: `[tokens, 2560]`, F32 activation.
- QKV projection: `[10240, 2560]`, logical output-by-input.
- Z projection: `[6144, 2560]`, logical output-by-input.
- A and B projections: `[48, 2560]`, logical output-by-input.
- Q/K heads: 16; value heads: 48; head dimension: 128.
- QKV stream order: `Q[2048], K[2048], V[6144]`.
- Convolution: depthwise causal kernel 4, logical `[tap, channel]`, 10240
  channels, persistent history `[3, 10240]`.
- Recurrent state: F32 `[48, 128, 128]`, persistent per GDN layer.
- Output projection: `[2560, 6144]`, logical output-by-input.
- Recurrent scale: `1 / sqrt(128)`.

The fixture captures the state immediately before and immediately after the
target token. The initial state is established by a real prefix run, so the
standalone benchmark does not assume a zero recurrent state or zero
convolution history.

## Exact operation order

For one token, with `H = 2560`, `Cqkv = 10240`, `Cz = 6144`,
`heads = 48`, and `D = 128`:

1. Project the hidden input to `qkv`, `z`, `a`, and `b`.
2. Apply causal depthwise convolution to `qkv` using the previous
   `[3, Cqkv]` history. For tap `k`, the source is
   `x[t - (3-k)]`; then write the final three input rows as the next history.
3. Apply SiLU to each convolved QKV element.
4. Split QKV into 16-head Q, 16-head K, and 48-head V.
5. Repeat each key head three times into value-head order
   `value_head -> key_head = value_head / 3`.
6. L2-normalize every repeated Q and K head with
   `1 / sqrt(sum(x*x) + 1e-6)`.
7. For each value head, compute:

   ```text
   beta = sigmoid(b)
   a' = a + dt_bias
   decay = exp(-exp(A_log) * softplus(a'))
   ```

8. Apply the recurrent delta rule in token order:

   ```text
   Sbar       = decay * S_prev
   prediction = Sbar^T * k
   delta      = (v - prediction) * beta
   S_next     = Sbar + k * delta^T
   y          = (1 / sqrt(128)) * S_next^T * q
   ```

9. RMS-normalize each recurrent output head using `norm.weight`, then gate
   each element with `sigmoid(z)`.
10. Project the gated `[48, 128]` output through `out_proj`.
11. Residual/GR writeback is outside the standalone GDN contract.

The CPU fixture oracle implements this sequence directly and independently
from the CUDA kernels. It compares the output, next recurrent state, and next
convolution history against the captured production values.

## Dtypes and layouts

Activations, state, history, and expected vectors are F32. GGUF weight files
retain their captured storage type and physical layout. Projection tensors are
decoded as logical row-major `[rows, cols]`; `conv1d.weight` is decoded from
the GGUF physical `[10240, 1, 4]` layout as `[channel, tap]`. The fixture
metadata records each tensor type and byte size.

## Benchmark baseline

The baseline benchmark keeps the current production boundary:

- resident fixture weights are uploaded once;
- each projection uploads its host input, uses the current generic
  matrix-batch CUDA kernel, downloads its output, and synchronizes;
- convolution, gate/activation, split/repeat/norm, recurrence, and output
  gating execute on the host as in `full_gdn`;
- no model mmap, tokenizer, session, or full-chain forward is involved.

Stage accounting must cover at least 97% of complete standalone GDN wall time.
No candidate is selected until this baseline identifies the dominant stage.

S3-C1 was promoted after real early/middle/late fixtures showed approximately
53--55% lower isolated layer wall time. S3-C2 then moved the conv/history,
FP32 recurrence, state update, post-gate, and output projection onto one
persistent device island. S3-C3 fused the non-projection stages into one
value-head kernel plus a history-writeback kernel, reducing the complete
isolated path from nine to seven launches. Relative to C2, C3 measured
approximately 11% lower wall time across the three fixtures, reduced the
median range from 641--659 us to 567--588 us, and retained one
synchronization plus one 10,240-byte H2D and one 10,240-byte D2H.
Correctness gates are zero non-finite values, output `max_abs <= 5e-3`,
output `max_rel <= 1e-2`, next-history `max_abs <= 1e-4`, and next-state
`max_abs <= 1e-5`.
