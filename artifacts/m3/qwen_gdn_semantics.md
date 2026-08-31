# Qwen3.8-Flash-Next GR/GDN reference freeze

This document records only semantics verified from the frozen configuration and
the official Qwen technical report. It is not a license to infer an on-disk
packing convention that the reference does not publish.

## Reference identity

- Repository: `QwenLM/Qwen3.8-Flash-Next`
- Repository revision checked: `69885871a64393807d988b27b1b5e380e8f28526`
- Reference: `tech_report.pdf`, SHA-256
  `04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d`
- Local model configuration: `/home/lvx/q38model/config.json`
- Local model revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`

## Frozen shapes and layer schedule

The local configuration verifies 48 layers with layers 3, 7, ..., 47 using
full attention. The remaining layers use GDN with 16 key heads, 48 value
heads, key/value head dimensions 128, and causal convolution kernel 4.
The residual stream has four branches and GR bottleneck rank 320. GDN state
dtype is not encoded as a GGUF tensor type; M3 keeps recurrent state in F32.

The real GGUF projection tensors are:

| Role | GGUF tensor suffix | Shape |
| --- | --- | --- |
| GDN fused input projection | `linear_attn.in_proj_qkv.weight` | `[10240, 2560]` |
| GDN output projection | `linear_attn.out_proj.weight` | `[2560, 6144]` |
| GDN decay vector | `linear_attn.A_log` | `[48]` |
| GDN decay bias | `linear_attn.dt_bias` | `[48]` |
| GDN gate/update projection | `linear_attn.in_proj_a.weight`, `in_proj_b.weight` | `[48, 2560]` |
| GDN output gate projection | `linear_attn.in_proj_z.weight` | `[6144, 2560]` |
| GDN convolution | `linear_attn.conv1d.weight` | `[10240, 1, 4]` |

GR tensors occur independently for `attn_hyper_connection` and
`mlp_hyper_connection`: `block_inject_weight.weight` `[4,10240]`,
`hc_norm.weight` `[10240]`, `input_mix_weight_down.weight` `[320,10240]`,
and `input_mix_weight_up.weight` `[10240,320]`.

## GDN equations (technical report §§2.1.1, Equations 1--11)

For each head, `q_t ∈ R^128`, `k_t ∈ R^128`, `v_t ∈ R^128`, and
`S_t ∈ R^(128×128)`:

```text
S~_{t-1} = α_t S_{t-1}
e_t      = v_t - S~_{t-1}^T k_t
S_t      = S~_{t-1} + β_t k_t e_t^T
y_t      = S_t^T q_t
```

The equivalent update is
`S_t = α_t (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T`.
The report defines:

```text
q_t = L2Norm(SiLU(ShortConv(W_q x_t)))
k_t = L2Norm(SiLU(ShortConv(W_k x_t)))
v_t = SiLU(ShortConv(W_v x_t))
β_t = sigmoid(W_β x_t)
α_t = exp(-exp(A) softplus(W_α x_t + b_α))
o_t = W_o [ sigmoid(W_z x_t) ⊙ RMSNorm(y_t) ]
```

The report states that the convolution is short, depthwise, causal, and
applied before the recurrence. It states that q/k are L2-normalized and that
the output gate is sigmoid. Initial state and reset are zero for the scalar
micro-oracle unless a later reference dump proves otherwise.

## Gated Residual equations (technical report §2.2, Equations 29--34)

For each of the four branches, with branch/channel gain `γ_i`:

```text
R̂_i = RMSNorm(R_i; γ_i)
G   = unvec(sigmoid(W_u SiLU((1/n_r) W_d vec(R̂))))
x   = (1/n_r) Σ_i G_i ⊙ R̂_i
s   = 2 sigmoid((1/n_r) W_w vec(R̂))
R'_i = R_i + s_i y
```

The report specifies `n_r = 4`, bottleneck rank `r = d/8 = 320`, an
elementwise read gate, and one scalar write gate per branch. It says GR
replaces pre-normalization, has no branch mixing operator, and has separate
modules for attention and MLP blocks.

## Explicit unresolved reference semantics

The official report and frozen checkpoint do **not** publish:

1. The physical slice order inside fused `in_proj_qkv` and `conv1d` tensors.
2. The exact mapping of 16 key heads to 48 value heads.
3. The serialized state layout/stride and whether the checkpoint uses a
   transpose or interleaving convention.
4. The exact GR tensor-to-equation assignment for `block_inject_weight` and
   `hc_norm`, beyond their names and shapes.
5. A runnable Qwen3.8 reference forward implementation or intermediate dumps.

These are blocking for a native projection/recurrence implementation. The
llama.cpp GDN recurrence is suitable only as a scalar recurrence oracle after
the Qwen projection and packing choices are frozen; it cannot resolve these
items.
