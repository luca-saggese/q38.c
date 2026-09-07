# DS4 Main Forward Orchestration Audit

## Scope

This is a static comparison of the local `main` donor and current q38. No model was loaded and no benchmark was run. The donor contains both ordinary CUDA operation paths and optional CUDA graph paths; graph use is conditional and must not be assumed for every deployment.

## DS4 forward structure

The donor CUDA implementation exposes:

* CUDA graph capture/replay for decode islands (`ds4_cuda.cu:839-1065`), including a layer-top island and an attention-output-through-MoE-tail island.
* Device kernels for RMS normalization, RoPE, attention score/value processing, routed MoE packing, expert MMQ, and output combination.
* Device-side handoff helpers such as `ds4_gpu_moe_handoff_pack_tensor()` (`ds4_cuda.cu:3177-3224`).
* Explicit device cache lookup and cross-device paths; host readback is used for selected diagnostics, compatibility transfers, or cross-device fallback, not as the normal operation boundary.

The donor's public `ds4.c` layer is mostly an orchestration/ABI surface. For example, graph functions are stubs there (`ds4.c:131-135`) and are implemented by `ds4_cuda.cu`.

## q38 forward structure

The q38 forward chain is host-orchestrated in `q38_forward.c`:

```text
full_forward
  -> GR read backend
  -> GDN or QSA backend
  -> GR write backend
  -> MoE backend
  -> residual/norm/glue
  -> LM head
  -> host logits
  -> argmax
```

The runtime installs separate backend callbacks in `q38_session.c:101-115`. The callback boundaries return host arrays, so each stage can impose a device-to-host fence even when the next stage immediately uploads the result.

## Stage comparison

| Stage | DS4 main | q38 current | Static assessment |
|---|---|---|---|
| GR | Device kernels and graph-island support keep normalization, low-rank work, branch read/write, and subsequent layers on CUDA where the selected path supports it. | `q38_forward_cuda_gr_read_backend()` uploads residual, launches kernels, then copies both `normed` and `input` D2H and synchronizes (`cuda/q38_forward_cuda.cu:2287-2375`). `gr_write_backend()` similarly uploads host residual/block and copies host outputs D2H (`2382-2446`). | q38 has a clear GPU -> D2H -> sync -> CPU orchestration boundary at both GR halves. |
| GDN | Donor has device kernels and graph-compatible operation infrastructure; exact Qwen3.8 GDN naming is not exposed as one donor callback in the audited surface. | `q38_forward_cuda_gdn_layer_backend()` keeps projections/recurrent/history/out projection on one stream, but uploads input and copies the full output D2H before returning (`2047-2202`). | q38 avoids per-projection host round trips inside the GDN island, but still exits to host once per layer. |
| QSA/attention | DS4 has device attention kernels, fused RoPE/score/value stages, optional score-split graphs, and graph replay (`ds4_cuda.cu:6977-8870`). | `q38_forward_cuda_qsa_qkv_backend()` performs device QKV work but returns host output after a stream fence (`2453-2546`). | q38 has a host boundary around QSA output even when internal kernels are device-resident. |
| MoE | DS4 provides device-side handoff packing and MMQ/MoE kernels, plus optional routed-MoE decode graphs (`ds4_cuda.cu:1150-1200,3177-3224`). | The grouped Q2 path uploads host input/route IDs/weights, executes grouped kernels, copies the full 2560-float result D2H, and synchronizes (`q38_forward_cuda.cu:856-960`). | q38's grouped kernel fixes expert weight residency but still exposes a host boundary per MoE layer. |
| Norm/residual/glue | DS4 has device normalization and combine kernels in the CUDA implementation. | q38 performs several residual/mixing decisions in `q38_forward.c`; CUDA GR callbacks return host buffers before the next host operation. | q38 uses CPU scalar orchestration between device islands. |
| LM head | DS4 keeps output operations in the CUDA graph/command path where enabled; exact final graph composition is configuration-dependent. | `q38_forward_cuda_matrix_backend()` copies logits D2H and synchronizes (`q38_forward_cuda.cu:1755-1900`), then q38 performs host argmax unless the dedicated CUDA argmax path is used. | q38 always has a logits visibility boundary for the normal backend contract. |
| Argmax | DS4 has device-side output kernels and graph infrastructure; host readback is not the fundamental layer-to-layer mechanism. | CUDA argmax exists, but it still copies one token ID D2H and synchronizes (`q38_forward_cuda.cu:2580-2613`). | Final token selection is a small but explicit D2H/sync boundary. |

## q38 boundary inventory

The current CUDA source contains synchronization at:

* matvec row and matrix completion;
* grouped MoE completion;
* GDN output;
* GR read and write;
* QSA completion;
* final argmax;
* GDN trace-state synchronization.

The corresponding calls are visible at `q38_forward_cuda.cu:813,960,1090,1195,1226,1409,1697,1744,1883,2021,2202,2375,2446,2546,2596,2612`.

The dominant structural pattern is therefore:

```text
GPU kernels
  -> cudaMemcpyAsync(DeviceToHost)
  -> cudaStreamSynchronize
  -> CPU q38_forward.c orchestration
  -> cudaMemcpyAsync(HostToDevice)
  -> next GPU island
```

This is especially direct in:

* GR read/write (`q38_forward_cuda.cu:2332-2375`, `2413-2446`);
* routed MoE (`856-960`);
* GDN (`2047-2202`);
* QSA (`2453-2546`).

## DS4 versus q38 synchronization interpretation

DS4 is not synchronization-free. It has stream/event fences for staged model uploads, cross-device transfers, command completion, diagnostics, and graph capture/replay. The important difference is placement: DS4's normal forward data path is device-resident and can be captured/replayed, while q38's backend ABI requires host-visible stage outputs.

The audit does **not** prove that every DS4 layer is one monolithic graph launch, nor that all DS4 configurations use graphs. It proves that the donor has the mechanisms to keep major decode islands on CUDA and that q38 currently inserts host-visible callback boundaries between them.

## Port implications

1. Preserve the existing q38 kernels, but change the ABI from `float *host_input/host_output` to device-buffer variants for GR, GDN, QSA, and MoE.
2. Add a device-resident layer chain before attempting graph capture. Graph capture is a second step, not a substitute for correct ownership.
3. Keep D2H only at final logits/token selection and explicit diagnostics.
4. Add a microbench that counts H2D/D2H bytes, stream synchronizations, and stage transitions for one decode token. The acceptance target is a reduction in inter-stage D2H/H2D pairs without changing generated IDs or logits hashes.

