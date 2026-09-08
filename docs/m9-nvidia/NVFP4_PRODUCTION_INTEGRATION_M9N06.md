# M9N-06 NVIDIA NVFP4 production integration

M9N-06 integrates the promoted `NV-C4` grouped top-10 primitive into the
existing device decoder chain. The frozen `NVFP4_CUDA_ORACLE_V1`, checkpoint
contract, Q2 path, BF16 router, shared BF16 expert, GDN/QSA/GR chain, and
file-backed PLE strategy are unchanged.

## Integration

The materialized `Q38_NVFP4_PACK_V1` is exposed through a pointer-backed
synthetic descriptor model consumed by the existing binder. The payloads are
not repacked or expanded:

```text
q38_session_eval
  -> q38_forward_token
  -> q38_forward_cuda_decoder_layer_chain_backend
  -> existing GR
  -> existing GDN/QSA
  -> BF16 router and deterministic top-k
  -> NV-C4 grouped NVFP4 routed experts
  -> existing BF16 shared expert
  -> existing GR
```

Q2 tensors continue to select the existing Q2 grouped primitive. NVIDIA
NVFP4 tensors select NV-C4. The native CUDA residency regions are used
directly for packed weights, E4M3 block scales, `weight_scale_2`, and
`input_scale`. No per-token expert pointer-table rebuild or per-call
allocation was added.

The NVIDIA PLE sidecar remains file-backed. Its FP8 E4M3 rows and BF16 global
scale are bound through the existing PLE store. MTP and vision auxiliary
descriptors are intentionally excluded from the runtime descriptor model.

## Static preflight

The host-only preflight passed before the CUDA smoke:

```text
native NVFP4 preflight: PASS
layers: 48
routed experts per layer: 512
projections: gate/up/down
BF16 tensors: bound
PLE: NVIDIA FP8 rows + BF16 global scale
MTP: skipped
vision: skipped
Q2/NVFP4 dispatch: separated
```

## Correctness status

The frozen CUDA fixture regression remains green for early, middle, and late
fixtures. Activation payload and scale bytes remain exact; all projection and
single-expert comparisons remain within the frozen `1e-5` tolerance with zero
NaN and zero Inf.

The M9N-06 hotfix then added the missing shared-MoE `device_moe_mid`
workspace and replaced the five synchronous whole-region copies with two
128 MiB pinned staging buffers, asynchronous H2D transfers, reuse events, and
final cleanup before inference. The host-only preflight and the normal
non-model test gates remained green.

The single authorized post-hotfix smoke used MTP and vision disabled,
file-backed PLE, greedy decoding, `--prefill-reference`, and
`--max-tokens 4`. It completed without a crash or numerical exception:

```text
prompt tokens: 11
cuda_prepare_ms: 83317.870
runtime_init_ms: 83399.558
first token: 271
generated IDs: [271, 248068, 271, 248069]
prefill_ms: 1296.283
decode_median_ms: 113.657
peak CUDA allocated bytes: 76540162048
peak RSS bytes: 36908294144
NaN: 0
Inf: 0
fallback used: false
```

The generated-token sequence is now available, but no NVIDIA/vLLM reference
run was performed in this hotfix. Therefore token equality against the
external reference and `NVIDIA_NVFP4_REFERENCE_0` remain **not established**.

## Acceptance disposition

| Gate | Status |
|---|---|
| Native descriptor/binder preflight | PASS |
| 48 layers / 512 routed experts | PASS |
| Direct native NVFP4 component binding | PASS |
| BF16 router and shared expert preserved | PASS |
| File-backed NVIDIA PLE binding | PASS |
| MTP and vision excluded | PASS |
| Frozen NVFP4 CUDA fixtures | PASS |
| Full-model generated tokens | PASS: 4 IDs emitted, no NaN/Inf, no fallback |
| NVIDIA/vLLM token equality | NOT RUN |
| `NVIDIA_NVFP4_REFERENCE_0` | NOT RUN |

M9N-06 stops here. Tensor Core work, cuBLASLt follow-up, MTP, FP8 KV,
YaRN, and PLE I/O tuning are not included.
