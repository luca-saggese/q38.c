# Decoder Layer Chain Audit

## DECODER_LAYER_CHAIN: PROMOTED

Final end-to-end A/B result with separate model loads, identical model,
prompt, greedy decoding, PLE, residency, CUDA configuration, and generation
length:

| Path | Decode latency | Throughput |
|---|---:|---:|
| OLD | 110.393 ms/token | 9.059 tok/s |
| DEVICE | 76.282 ms/token | 13.109 tok/s |

Improvement: **-30.90% latency / +44.71% throughput**. The generated token
sequence was identical. The previous regression was caused by single-token
MoE routing, not by the decoder-layer device chain itself.

The generic multi-token routing path remains unchanged.

## Scope

This is a static comparison of the two single-token decoder paths:

- **Old path:** `runtime->backend.decoder_layer_chain = NULL`
- **Chain path:** `q38_forward_cuda_decoder_layer_chain_backend`

The static comparison below records the pre-promotion differences between the
two paths. The production default now uses
`q38_forward_cuda_decoder_layer_chain_backend`.

## Call-graph comparison

| Operation | Old path | Chain path | CUDA kernels / device work | Copies | Synchronization | Host callbacks / allocation |
|---|---|---|---|---|---|---|
| Layer entry | `full_forward` layer loop | `q38_forward_cuda_decoder_layer_chain_backend` | Old path performs each stage through host-visible backends; chain enqueues one device island | Old path has stage boundaries; chain uploads `host_input` and downloads `host_output` | Old path is owned by each backend; chain has a final stream wait | Chain lazily ensures persistent hidden, GR, MoE workspaces |
| GR read (attention) | `full_gr_read` -> `q38_forward_cuda_gr_read_backend` | `q38_forward_cuda_gr_read_device` | Chain: cooperative normalize/down, cooperative low-rank/up, branch-read kernel | Old backend exposes host buffers; chain stays device-resident | Chain read primitive is enqueue-only | Chain uses persistent `device_gr_*` buffers |
| GDN mixer | `full_gdn` -> `q38_forward_cuda_gdn_layer_backend` | `q38_forward_cuda_gdn_layer_device` | Both use the existing GDN CUDA primitives; orchestration differs | Old path crosses the host ABI; chain uses device buffers | Chain primitive is intended to be enqueue-only | Chain reuses context-wide GDN workspaces |
| QSA mixer | `full_qsa` -> `q38_forward_cuda_qsa_chain_backend` | `q38_forward_cuda_qsa_chain_device` | Both use QSA projection/prepare/append/select/gather/attention/output stages, but through separate orchestration implementations | Old path has host input/output boundaries; chain has device input/output boundaries | Both have caller-owned boundaries; chain runs inside the layer island | Both use persistent state, but the device path has its own chain state/workspace path |
| GR write (attention) | `full_gr_write` -> `q38_forward_cuda_gr_write_backend` | `q38_forward_cuda_gr_write_device` | Normalize, inject matvec, writeback | Old backend exposes host stage buffers; chain remains device-resident | Chain write primitive is enqueue-only | Chain reuses persistent GR buffers |
| GR read (MLP) | `full_gr_read` -> `q38_forward_cuda_gr_read_backend` | `q38_forward_cuda_gr_read_device` | Same GR device primitive as first read | Old host-visible boundary; chain device pointer | Chain enqueue-only | No per-token allocation intended, but `ensure_gr_buffers()` is reached |
| Router | `full_moe` router matrix path | Device BF16 matvec plus `q38_moe_cuda_route_weights` | One router matvec plus device route-weight work | Old route logits become host-visible for routing; chain keeps route buffers on device | Chain route path is enqueue-only | Chain owns persistent route buffers |
| Grouped routed MoE | `full_moe` and `q38_forward_cuda_moe_layer_q2_backend` | `q38_moe_cuda_q2_grouped_indexed_deterministic` | Grouped Q2 kernels plus deterministic reduction | Old path uses host route/input/output ABI; chain uses device route IDs/weights/output | Chain has no explicit stage wait | Chain uses persistent grouped-MoE buffers |
| Shared expert | `full_moe` shared projections | Device BF16 projections, SiLU/mul, projection, add | Device matvecs plus shared activation/reduction kernels | Old shared activations use host buffers; chain remains device-resident | Chain enqueue-only | Persistent shared-expert buffers |
| GR write (MLP) | `full_gr_write` -> host-visible backend | `q38_forward_cuda_gr_write_device` | Same device GR write primitive | Old host output boundary; chain only downloads at layer exit | Chain final layer boundary wait | Persistent buffers |
| Layer state | Host `q38_forward_state` updates in old orchestration | Device mixer state plus host state advancement hooks | QSA/GDN state kernels and state advancement | Old state is host-visible at subsystem boundaries; chain retains device state during the island | Chain state synchronization is not performed between stages | QSA chain state is persistent per layer |

The old path also performs host-side residual glue and explicit buffer clears
between stages. The chain removes those host-visible stage buffers but adds a
large device-layer boundary and a separate device orchestration path.

## Static sync/copy/allocation audit

### Chain-only or chain-specific boundaries

`q38_forward_cuda_decoder_layer_chain_backend` contains:

- one `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, ...)` for layer input;
- one `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, ...)` for layer output;
- one final `cudaStreamSynchronize` after the output download;
- lazy `ensure_buffer` calls for hidden, GR, router, route, grouped-MoE, and
  shared-expert workspaces.

The device GR/GDN/QSA/MoE entry points are intended to enqueue work without
intermediate synchronization. The chain therefore does not satisfy the
stronger “no layer-boundary wait” form of the device-island goal: it still
waits once per complete layer.

### Old-path boundaries

The old path calls the host-visible GR, mixer, router, and MoE backends from
`q38_forward.c`. Those backends own their own input/output transfers and
waits. The old path also performs host residual glue and `memset` operations
between GR and mixer/MoE stages. Exact dynamic transfer byte counts are not
claimed here because this document is intentionally static.

### Static red flags

1. **Complete-layer wait:** the chain unconditionally downloads and waits at
   every layer, preventing overlap across consecutive layers.
2. **Separate QSA implementation:** the chain calls
   `q38_forward_cuda_qsa_chain_device`, while the old path calls
   `q38_forward_cuda_qsa_chain_backend`; these are not a single shared
   orchestration implementation.
3. **Repeated workspace checks:** the chain reaches `ensure_gr_buffers()` and
   `ensure_qsa_chain_workspace()` from the layer path. Even when they do not
   allocate, the checks and ownership path are repeated across 48 layers.
4. **Additional device orchestration:** the chain adds device router, route
   selection, grouped-MoE, shared-expert, and layer ping-pong coordination
   that the old path does not use.
5. **QSA launch shape:** the device QSA path records nine kernel launches per
   chain decode. The old host-boundary path has a different orchestration and
   must not be assumed equivalent from the shared kernel names alone.

## QSA comparison

| QSA stage | Host-boundary backend | Device chain |
|---|---|---|
| Projection | `q38_forward_cuda_qsa_chain_backend` obtains resident projections and uses the QSA chain decode with host boundary buffers | `q38_forward_cuda_qsa_chain_device` obtains the same projection tensors and uses device input/output buffers |
| Cache prepare | QSA CUDA chain prepare kernels | Same underlying chain prepare kernels |
| Cache append | Persistent device chain state | Persistent per-layer device chain state |
| Selection | Device selection workspace/state | Device selection workspace/state |
| Gather/attention | QSA chain CUDA implementation behind the backend | QSA chain CUDA implementation behind the device entry point |
| Output projection | Device output projection followed by host download in the backend | Device output projection followed by layer-chain continuation |
| State advancement | Host QSA state is advanced after the chain call | Host state is advanced after the device chain call |
| Capacity policy | Centralized `ensure_qsa_chain_capacity` | Centralized `ensure_qsa_chain_capacity` |
| Synchronization | Boundary synchronization owned by backend | Complete decoder-layer output synchronization owned by layer chain |

The recent cache-growth defect is fixed centrally, but the two entry points
remain distinct orchestration paths. They must not be treated as equivalent
until fixture-level output/state parity and isolated timing prove it.

## Isolated benchmark status

No model load or full-model benchmark was performed. The repository has GDN
and QSA fixture benches, but no trusted complete decoder-layer old-vs-chain
fixture harness that simultaneously captures GR, mixer, router, grouped MoE,
state, copies, and waits. Therefore this audit does not fabricate:

- complete-layer wall time;
- exact extra sync count;
- exact extra copy bytes;
- exact extra launch count;
- QSA or GDN microseconds/layer deltas.

Those values require a dedicated complete-layer fixture harness before any
optimization or promotion decision.

## Root-cause ranking from static evidence

1. **Per-layer D2H plus `cudaStreamSynchronize`:** the chain cannot overlap
   the next layer while preserving its current host-output ABI.
2. **Duplicated QSA orchestration:** the device entry point is a separate
   path from the host-boundary backend, so common primitive use does not prove
   equivalent scheduling or workspace behavior.
3. **New device router/MoE orchestration:** route selection and grouped
   reduction are now serialized inside the layer island and may add launches
   relative to the old backend.
4. **Persistent-workspace ownership checks:** repeated ensure paths and
   layer-level ping-pong setup add host orchestration even when no allocation
   occurs.

## Confirmed routing root cause

Before promotion, the device chain routed a single decode token through the
generic `q38_topk_cuda()` path. That kernel mapped one CUDA thread per token;
with `token_count == 1`, only one thread performed the serial top-10 search
over all 512 experts, including repeated selected-expert checks. The separate
route-weight kernel also scanned the 512 logits serially for that token.

The specialized cooperative single-token routing kernel now performs the
BF16-effective-logit conversion, deterministic top-10 selection, selected
softmax normalization, and output ID/weight writes in one launch. The generic
multi-token path remains in use for `token_count > 1`.

This removed the pathological per-layer routing bottleneck and restored the
device chain's end-to-end advantage shown above.
