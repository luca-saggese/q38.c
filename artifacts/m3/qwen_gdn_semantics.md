# Qwen3.8-Flash-Next GR/GDN semantics

This document is the reference freeze for the M3 GR/GDN implementation. It
separates model semantics from storage and kernel layout. A layout described as
`logical` is required for numerical equivalence; a layout described as
`physical` is an implementation choice and must not be treated as a model
equation.

## Reference identity

The source references used for this freeze are:

- `ggml-org/llama.cpp` PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742),
  `src/models/qwen4exp.cpp`, `src/models/qwen3next.cpp`, and
  `conversion/qwen4exp.py`, read from `refs/pull/27742/head`.
- Current `master`:
  `ggml/src/ggml-et/et-kernels/src/gated_delta_net_f32.c`.
  The file currently resolves to blob
  `c09c7742528fadd77ba67b165d6df22e343d8f1f`.
- Local frozen checkpoint:
  `/home/lvx/q38model`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

The llama.cpp source is the executable graph reference. The ET kernel is the
recurrence and state-storage reference. `conversion/qwen4exp.py` is the
converter reference for tensor names, split tensors, PLE metadata, and
checkpoint-to-GGUF transformations.

## Frozen shapes and layer schedule

The local checkpoint has 48 layers. The recurrent/GDN layers are every layer
except 3, 7, 11, ..., 47. Those twelve layers are full-attention layers.
For every GDN layer:

| Parameter | Value | Reference |
|---|---:|---|
| key heads | 16 | `qwen4exp.cpp::build_layer_attn_linear` |
| value heads | 48 | `qwen4exp.cpp::build_layer_attn_linear` |
| key head dimension | 128 | local `config.json`; `qwen4exp.cpp` |
| value head dimension | 128 | local `config.json`; `qwen4exp.cpp` |
| convolution kernel | 4 | `qwen4exp.cpp::build_layer_attn_linear` |
| recurrent-state dtype | F32 | `gated_delta_net_f32.c`; `qwen3next.cpp` |
| output gate | sigmoid | `qwen4exp.cpp::build_norm_gated` |

For Qwen3.8-Flash-Next, `d_inner = 48 * 128 = 6144` is not used as an
inference assumption: the graph obtains the dimensions from the loaded
hyperparameters and asserts the relevant equalities. The local inventory
confirms the corresponding tensor shapes below.

## GDN tensor shapes and projection order

The following shapes are logical tensor shapes. GGUF tensor dimensions may be
stored in the transposed convention used by the loader; the role and matrix
operation below are the authoritative interpretation.

| Logical role | Local/ GGUF tensor | Logical shape | Reference |
|---|---|---|---|
| input QKV projection | `linear_attn.in_proj_qkv.weight` | `[10240, 2560]` output-by-input | `qwen4exp.cpp::load_arch_tensors`; `build_qkvz` |
| output-gate projection | `linear_attn.in_proj_z.weight` | `[6144, 2560]` output-by-input | `qwen4exp.cpp::load_arch_tensors`; `build_qkvz` |
| beta projection | `linear_attn.in_proj_b.weight` | `[48, 2560]` output-by-input | `qwen4exp.cpp::load_arch_tensors`; `build_layer_attn_linear` |
| decay-input projection | `linear_attn.in_proj_a.weight` | `[48, 2560]` output-by-input | `qwen4exp.cpp::load_arch_tensors`; `build_layer_attn_linear` |
| decay bias | `linear_attn.dt_bias` | `[48]` | `qwen4exp.cpp::load_arch_tensors` |
| log decay | `linear_attn.A_log` | `[48]` | `qwen4exp.cpp::load_arch_tensors` |
| depthwise causal convolution | `linear_attn.conv1d.weight` | `[4, 10240]` logical `[kernel, channels]` | `qwen4exp.cpp::load_arch_tensors`; `build_layer_attn_linear` |
| recurrent output norm | `linear_attn.norm.weight` | `[128]` | `qwen4exp.cpp::load_arch_tensors` |
| recurrent output projection | `linear_attn.out_proj.weight` | `[2560, 6144]` output-by-input | `qwen4exp.cpp::load_arch_tensors` |

The fused QKV output is logically:

```text
qkv_mixed[t] = W_qkv x[t]
qkv_mixed[t] shape = [key_dim * 2 + value_dim]
               = [16*128*2 + 48*128]
               = [10240]
```

`in_proj_z` separately produces:

```text
z[t] shape = [value_heads * value_head_dim] = [6144]
```

The reference projection and recurrence order is:

1. `qkv_mixed = in_proj_qkv(x)` and `z = in_proj_z(x)`.
2. `b = in_proj_b(x)`, reshape to `[1, 48, T, S]`, then
   `beta = sigmoid(b)`.
3. `a = in_proj_a(x)`, reshape to `[48, T, S]`.
4. `a_biased = a + dt_bias`.
5. `a_softplus = softplus(a_biased)`.
6. `g = -exp(A_log) * a_softplus`.
7. Prepend/read the convolution history and apply causal depthwise
   convolution with kernel size 4 to `qkv_mixed`.
8. Apply SiLU to the convolved Q/K/V stream.
9. Split the result into Q, K, and V; L2-normalize Q and K.
10. Repeat/interleave Q and K to the 48-value-head shape.
11. Run the recurrent delta rule using `g` and `beta`.
12. Reshape `z` to `[128, 48, T, S]`; apply RMSNorm to the recurrence output,
    then multiply elementwise by `sigmoid(z)`.
13. Flatten to `[6144, T, S]` and apply `out_proj`.

The order above is taken from
`qwen4exp.cpp::build_layer_attn_linear`. It is not inferred from tensor
dimensions.

### Legacy Qwen3Next projection path

`qwen3next.cpp` also contains a legacy path for files with `ssm_in` and
`ssm_beta_alpha` instead of the split Qwen4Exp tensors. Its explicit order is:

- `ssm_in(x)` produces a fused legacy QKVZ stream.
- The stream is split into Q, K, V, and Z.
- `ssm_beta_alpha(x)` is reshaped to
  `[2 * value_heads / key_heads, key_heads, T, S]`, then split into B and A.
- `beta = sigmoid(B)`.
- `alpha = A + ssm_dt`; `a_softplus = softplus(alpha)`;
  `g = -A_log.exp() * a_softplus`.
- Convolution, SiLU, Q/K L2 normalization, recurrent update, gated norm, and
  output projection follow the same order.

The legacy path is a compatibility path, not evidence that the current
checkpoint uses a different semantic recurrence.

## Q/K/V logical shapes and V-head mapping

Immediately before recurrence, the logical tensors are:

```text
q_conv: [128, 16, T, S]
k_conv: [128, 16, T, S]
v_conv: [128, 48, T, S]
g:      [1,   48, T, S]
beta:   [1,   48, T, S]
```

Here `T` is the token count in the current sequence chunk and `S` is the
number of equal-length sequences in the ubatch.

The 16-to-48 mapping is explicitly implemented in
`qwen3next.cpp::build_layer_attn_linear` and the corresponding Qwen4Exp
function. The code reshapes each key tensor to
`[128, 1, 16, T*S]`, repeats the inserted dimension by
`value_heads / key_heads`, and reshapes back to
`[128, 48, T, S]`. Therefore the semantic mapping is:

```text
value_head h = 3 * key_head + repeat
key_head     = h / 3
repeat       = h % 3
q_conv[h]    = q_conv_original[h / 3]
k_conv[h]    = k_conv_original[h / 3]
```

This is a repeat-interleave operation, not a modulo broadcast and not a
ratio-based deduction. The source reshape/repeat sequence is the authority.
The fused GDN path may accept the un-repeated 16-head tensors through an
explicit broadcast configuration, but that is a kernel optimization; the
logical recurrence still uses the mapping above.

## GDN equations

For each value head `h`, with `d = 128`, the logical state is a matrix
`S[h] ∈ R^(d × d)`. For token `t`, Qwen3.8 uses:

```text
decay_t[h] = exp(g_t[h])
             = exp(-exp(A_log[h]) * softplus(a_t[h] + dt_bias[h]))

Sbar[h]     = decay_t[h] * S_prev[h]
prediction  = Sbar[h]^T * k_t[h]
delta[h]    = (v_t[h] - prediction) * beta_t[h]
S_next[h]   = Sbar[h] + k_t[h] * delta[h]^T
y_t[h]      = S_next[h]^T * q_t[h]
```

The scalar-gate form above is the Qwen3.8 path: `g` and `beta` have one value
per value head. The ET kernel also supports a KDA form where `g` has `d`
values per head; that optional `kda` branch is not the Qwen3.8 scalar path
unless a future checkpoint explicitly selects it.

The source order is independently visible in
`gated_delta_net_f32.c::entry_point`:

1. decay each state row;
2. compute `dot(decayed_state_row, k)`;
3. compute `(v - dot) * beta`;
4. write the outer-product update `k * delta`;
5. read the updated state with `q`;
6. multiply the read by `scale` (`1/sqrt(d)` supplied by the caller).

The source kernel stores the state transposed relative to the mathematical
matrix, as documented by its header comment:

```text
physical_state[j * d + i] = logical_state[i, j]
```

Consequently, the ET kernel's `dot(S_row, k)` and `dot(S_row, q)` correspond to
the transposed access convention above; this does not change the logical
equations.

## Recurrent-state logical shape and physical layout

### Logical layout

The logical persistent recurrent state is:

```text
state_logical[sequence, value_head, row_i, column_j]
    shape = [S, 48, 128, 128]
    dtype = float32
```

The llama.cpp graph obtains a recurrent state row from
`build_rs(..., hparams.n_embd_s(), n_seqs)` and reshapes it in
`qwen4exp.cpp::build_layer_attn_linear` and
`qwen3next.cpp::build_layer_attn_linear` to:

```text
[head_v_dim, head_v_dim, num_v_heads, n_seqs]
= [128, 128, 48, S]
```

The logical state bytes per sequence and GDN layer are therefore
`48 * 128 * 128 * sizeof(float) = 3,145,728` bytes. This byte count follows
the explicit source reshape and dtype, not a dimensionality guess.

### Upstream physical layout

The current `gated_delta_net_f32.c` kernel receives:

```text
state_in: [d*d*H, K, S]
```

and uses a per-head plane of `d*d` floats. Its physical state indexing is:

```text
head_state_offset = (sequence * H + head) * d * d
state[j * d + i]   = logical_state[i, j]
```

The kernel's output tensor also contains attention output followed by state
snapshot planes:

```text
dst = [d*H, T*S + d*S*K]
```

with the live state in slot 0 and optional reverse-chronological snapshots in
later slots. Snapshot slots are a kernel/testing contract, not model
semantics.

### Our GB10 physical layout

For the first correct GB10 implementation, use an explicit contiguous
per-layer allocation:

```text
gb10_state[sequence][value_head][row_i][column_j] : float32
stride(column_j) = 1
stride(row_i)    = 128
stride(value_head)= 128*128
stride(sequence)  = 48*128*128
```

This is our implementation layout, chosen for simple address auditing and
chunk-invariance tests. It is not claimed to be the upstream CUDA/ET layout.
A transpose into the ET convention may be used at a kernel boundary only when
the conversion is explicit and covered by the same scalar oracle. No GB10
layout optimization may alter the logical state definition.

## Convolution state layout

The GDN convolution consumes the concatenation of the prior causal history and
the current QKV stream. In the Qwen4Exp path:

```text
conv_channels = key_dim * 2 + value_dim
              = 16*128*2 + 48*128
              = 10240
conv_kernel_size = 4
history_columns  = conv_kernel_size - 1 = 3
```

`qwen4exp.cpp::build_conv_state_at` explicitly:

1. treats the recurrent row as a tensor of
   `[state_cols, channels, n_seqs]`;
2. concatenates that history with the transposed current QKV stream;
3. applies the causal convolution;
4. writes the last `state_cols` columns back to the same recurrent row.

Thus the logical convolution state is:

```text
conv_history[sequence][history_column][channel]
    shape = [S, 3, 10240]
    dtype = the convolution activation type
```

The source uses tokens/columns as the first logical dimension in the
`[state_cols, channels, sequence]` view. The corresponding upstream physical
row is contiguous in channel-major storage within each history column:

```text
physical_conv[offset(sequence, column, channel)]
    = physical_conv[(sequence * 3 + column) * 10240 + channel]
```

This is distinct from the recurrent matrix state. The convolution history is
not a scratch buffer and must survive ubatch boundaries.

The converter stores the Qwen4Exp convolution tensor as
`linear_attn.conv1d.weight`. `qwen4exp.py::modify_tensors` removes the
singleton channel dimension with `squeeze()`; the graph then interprets the
result as `[kernel, channels]`. That conversion operation is a storage
transformation, not a change to convolution semantics.

## Output gate and gated normalization

The Qwen4Exp-specific difference from Qwen3.5 is explicit in
`qwen4exp.cpp::build_norm_gated`:

```text
normalized = RMSNorm(recurrent_output, norm.weight)
gate       = sigmoid(z)
output     = normalized * gate
```

`z` is the `[128, 48, T, S]` view of `in_proj_z(x)` and is not passed through
SiLU. The source comment explicitly contrasts this with Qwen3.5's SiLU output
gate. `qwen3next.cpp` retains the older `SiLU(z)` implementation for the
legacy Qwen3Next path; that is not the Qwen4Exp semantic path.

## Reference state initialization and reset

The recurrent memory interface supplies the state tensor to
`build_rs`/`build_recurrent_attn`. The scalar ET recurrence consumes
`state_in` and writes a new state; it does not define model-specific
checkpoint initialization. For a new single-sequence M3 session, q38 must
initialize both the recurrent matrix and convolution history to the reference
reset value, which is zero for the current llama.cpp recurrent-memory path.
This reset must be tested, not replaced by implicit uninitialized allocation.

## Prefill, decode, and chunk boundaries

The same recurrence is used for prefill and decode. The difference is the
number of tokens in the current ubatch:

- prefill processes `T > 1` tokens while carrying state through the token loop;
- decode normally processes `T = 1`.

`build_conv_state_at` prepends the stored history and writes the final three
columns back after every chunk. `gated_delta_net_f32.c::entry_point` updates
the recurrent state token-by-token in order. Therefore a single `[N]` chunk
and any partition of that same stream must produce the same state after every
boundary, modulo the declared floating-point tolerance.

## Gated Residual equations

GR is implemented by the Qwen4Exp hyper-connection functions rather than by
the GDN kernel:

- `qwen4exp.cpp::build_hc_mix`
  - logical input: `[hidden_size, hc_count, T] = [2560, 4, T]`;
  - `RMSNorm` over each stream;
  - multiply by `hc_norm.weight` with logical shape `[10240]`;
  - down projection `[320, 10240]`;
  - divide by `hc_count`, apply SiLU;
  - up projection `[10240, 320]`, apply sigmoid;
  - multiply the normalized streams by the gate;
  - mean-collapse the four streams to `[2560, T]`;
  - independently produce the injection tensor `[4, T]` using
    `block_inject_weight` `[4, 10240]`.
- `qwen4exp.cpp::build_hc_combine`
  - `w = 2 * sigmoid(inject / hc_count)`;
  - broadcast the block output over the four streams;
  - `residual_next = residual + block_output * w`.

The two GR modules per layer are `attn_hyper_connection` and
`mlp_hyper_connection`. Their tensor shapes are verified in
`qwen4exp.cpp::load_arch_tensors`; the final
`hyper_connection_mixer` performs the output mix and is used instead of a
separate output norm.

## Explicit non-semantic physical-layout boundary

The following are implementation details and must not be used as equations:

- GGML `nb[]` strides and contiguous conversions;
- the ET kernel's transposed state rows and snapshot planes;
- fused-GDN broadcast of 16 key heads to 48 value heads;
- the order of source-shard bytes in Safetensors or GGUF;
- the GB10 state tiling, shared-memory staging, vector width, or fusion plan.

M3's correctness oracle must compare logical Q/K/V, gate, beta, state-before,
state-after, and output tensors. A physical layout is accepted only after an
explicit address map to the logical tensors is documented and tested.

## Explicit unresolved reference semantics

The requested GDN semantic fields are fixed by the llama.cpp and ET source
references. The original model-report-only freeze did not publish the
following items; the source references above now resolve the semantic parts,
while the final GB10 implementation choices remain open:

1. the final GB10 tile/vector layout;
2. whether q38 uses a direct logical state layout or an explicit transpose at
   the CUDA boundary;
3. workspace reuse and kernel fusion after the unfused correctness baseline.

None of these may be used to change the equations, head mapping, state shape,
or convolution-history semantics above.
