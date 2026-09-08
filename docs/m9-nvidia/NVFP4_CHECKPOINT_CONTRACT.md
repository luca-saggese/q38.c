# M9-NVIDIA NVFP4 checkpoint contract

**Project:** q38.c  
**Checkpoint:** `nvidia/Qwen3.8-Flash-Next-NVFP4`  
**Milestone:** `M9-NVIDIA`  
**Audit mode:** metadata and safetensors-header inspection only  
**Audit commit:** `e5ba107b7`  
**Date:** 2026-09-08

## 1. Scope and stop condition

This document freezes the checkpoint facts needed before writing an importer.
The audit downloaded only:

```text
README.md
config.json
hf_quant_config.json
model.safetensors.index.json
```

For every safetensors shard, the first eight bytes and the JSON header were
read with HTTP `Range` requests. No tensor payload was downloaded. No CUDA
code, importer, runtime loader, model load, or inference run was performed.

The machine-readable record for every indexed tensor is:

```text
artifacts/m9-nvidia/nvfp4_tensor_inventory.json
```

The copied milestone specification is:

```text
docs/m9-nvidia/M9-NVIDIA_NVFP4_BRINGUP.md
```

## 2. Frozen Q2 reference

M9-NVIDIA does not change or rebaseline the existing Q2 arm.

```text
reference: Q2_DEVICE_CHAIN_V1
latency:   76.282 ms/token
throughput: 13.109 tok/s
policy:    frozen for the entire M9-NVIDIA bring-up
```

The Q2 reference remains the rollback and performance reference while the
NVIDIA checkpoint is inspected and later brought up independently.

## 3. Source and architecture facts

The metadata identifies:

| Field | Value |
|---|---|
| Architecture | `Qwen4ExpForConditionalGeneration` |
| Model type | `qwen4_exp` |
| Hidden size | 2560 |
| Decoder layers | 48 |
| Routed experts per layer | 512 |
| Routed experts per token | 10 |
| MoE intermediate size | 640 |
| Native context | 262,144 |
| Linear-attention/GDN layers | 36 |
| Full-attention/QSA layers | 12 |
| Main expert quantization | W4A4 floating-point, group size 16 |
| MTP quantization | W8A8 floating-point, group size 128, dynamic activations |
| PLE n-gram quantization | W8 per-tensor floating-point |
| ModelOpt producer | `0.46.0.dev281+g73d778422` |

The main model is split over ten shards. The MTP and PLE payload is in:

```text
model-fp8-mtp-ple.safetensors
```

## 4. Classification contract

Every tensor is assigned exactly one of the following classes:

| Class | Definition |
|---|---|
| `main BF16` | BF16 tensors in the main shards, excluding `model.visual.*` and explicit `mtp.*` tensors |
| `main NVFP4 weight` | Main routed-expert `.weight` tensors; safetensors dtype is `U8` |
| `NVFP4 weight_scale` | Main routed-expert `.weight_scale`; dtype is `F8_E4M3` |
| `NVFP4 weight_scale_2` | Main routed-expert `.weight_scale_2`; dtype is `F32` |
| `NVFP4 input_scale` | Main routed-expert `.input_scale`; dtype is `F32` |
| `PLE FP8` | PLE n-gram FP8 shard tensors and their BF16 tensor scale |
| `MTP FP8` | All explicit `mtp.*` tensors, including FP8 expert weights, BF16 inverse scales, and BF16 MTP tensors stored in main shards |
| `vision` | `model.visual.*` tensors |
| `other` | Explicit I64 PLE control metadata |

The `PLE FP8` class includes one BF16 scalar scale because it is metadata for
the FP8 PLE payload. Likewise, `MTP FP8` includes the BF16
`weight_scale_inv` tensors and the 29 BF16 MTP tensors that are stored in the
main shards. These choices keep the inventory exhaustive without hiding
mixed-precision ownership.

The gate is strict:

```text
indexed tensors:       299,545
classified tensors:    299,545
unknown/unclassified:        0
```

The three explicitly classified `other` tensors are:

```text
model.language_model.layers.1.ple.ple_embedding.layer_multipliers
model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets
model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes
```

They are I64 control metadata, not model weights. They account for 280 bytes.

## 5. Byte totals

All values below are exact tensor-header payload byte totals unless marked as
file bytes. File size is not used as a residency estimate.

### 5.1 Shards and payload

| Quantity | Bytes | GiB |
|---|---:|---:|
| Main shard files, ten files | 78,962,697,648 | 73.539743 |
| Main shard tensor payload | 78,922,711,032 | 73.502502 |
| `model-fp8-mtp-ple.safetensors` file | 53,717,551,730 | 50.028369 |
| Sidecar tensor payload | 53,717,135,362 | 50.027981 |
| Index `metadata.total_size` | 132,639,846,394 | 123.530482 |

The index total equals main-shard payload plus sidecar payload. The small
file-to-payload deltas are safetensors headers/alignment and must not be
treated as resident tensor memory.

### 5.2 Requested category totals

| Category | Tensor count | Tensor bytes | GiB |
|---|---:|---:|---:|
| `main BF16` | 1,067 | 9,895,397,120 | 9.215807 |
| `main NVFP4 weight` | 73,728 | 60,397,977,600 | 56.250000 |
| `NVFP4 weight_scale` | 73,728 | 7,549,747,200 | 7.031525 |
| `NVFP4 weight_scale_2` | 73,728 | 294,912 | 0.000275 |
| `NVFP4 input_scale` | 73,728 | 294,912 | 0.000275 |
| **NVFP4 scales, combined** | **221,184** | **7,550,337,024** | **7.031799** |
| `PLE FP8` | 129 | 51,200,245,762 | 47.683945 |
| `MTP FP8` | 3,101 | 2,698,026,496 | 2.512733 |
| `vision` | 333 | 897,862,112 | 0.836199 |
| `other` | 3 | 280 | 0.000000 |

In particular:

```text
main-shard total file bytes:  78,962,697,648
PLE/MTP sidecar file bytes:   53,717,551,730
NVFP4 weight bytes:            60,397,977,600
NVFP4 scale bytes:              7,550,337,024
BF16 main-model bytes:          9,895,397,120
PLE tensor bytes:              51,200,245,762
MTP tensor bytes:               2,698,026,496
vision tensor bytes:              897,862,112
```

## 6. Main NVFP4 expert headers

The inspected experts are expert 0 from layer 0, layer 24, and layer 47.
They are early, middle, and late representatives.

| Position | Layer/expert | Source shard |
|---|---|---|
| Early | `layers.0.mlp.experts.0` | `model-00001-of-00010.safetensors` |
| Middle | `layers.24.mlp.experts.0` | `model-00005-of-00010.safetensors` |
| Late | `layers.47.mlp.experts.0` | `model-00009-of-00010.safetensors` |

The exact tensor-name patterns are:

```text
model.language_model.layers.0.mlp.experts.0.{gate_proj,up_proj,down_proj}...
model.language_model.layers.24.mlp.experts.0.{gate_proj,up_proj,down_proj}...
model.language_model.layers.47.mlp.experts.0.{gate_proj,up_proj,down_proj}...
```

All three inspected positions have identical stored headers:

| Projection | Weight dtype | Stored weight shape | Weight bytes | `weight_scale` dtype/shape | `weight_scale_2` dtype/shape | `input_scale` dtype/shape |
|---|---|---:|---:|---|---|---|
| `gate_proj` | `U8` | `[640, 1280]` | 819,200 | `F8_E4M3` / `[640, 160]` | `F32` / `[]` | `F32` / `[]` |
| `up_proj` | `U8` | `[640, 1280]` | 819,200 | `F8_E4M3` / `[640, 160]` | `F32` / `[]` | `F32` / `[]` |
| `down_proj` | `U8` | `[2560, 320]` | 819,200 | `F8_E4M3` / `[2560, 40]` | `F32` / `[]` | `F32` / `[]` |

The corresponding scale payload sizes are:

```text
gate/up weight_scale: 640 * 160 F8_E4M3 = 102,400 bytes
down   weight_scale: 2560 * 40 F8_E4M3 = 102,400 bytes
weight_scale_2:      scalar F32       = 4 bytes
input_scale:         scalar F32       = 4 bytes
```

The U8 storage is packed NVFP4 payload, not an assertion that the runtime
should interpret the values as ordinary uint8 weights. The scale cardinality
is consistent with:

```text
gate/up inferred logical element shape: [640, 2560]
down   inferred logical element shape: [2560, 640]
```

Those logical shapes are inventory inferences from the packed byte count,
group size, and architecture. They are **not** yet a physical-orientation
contract. In particular, the importer must not assume that the stored
`[out, packed_in]` orientation, scale layout, or any cuBLASLt swizzle is
already established. The stored header shape and dtype remain authoritative
until semantic dequantization tests define the physical ABI.

## 7. PLE contract

PLE remains permanently file-backed for the initial q38 bring-up.

Representative exact sidecar tensors:

| Tensor | Dtype | Stored shape | Bytes | Source |
|---|---|---:|---:|---|
| `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight` | `F8_E4M3` | `[2500012, 160]` | 400,001,920 | `model-fp8-mtp-ple.safetensors` |
| `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale` | `BF16` | `[1]` | 2 | `model-fp8-mtp-ple.safetensors` |

There are 128 FP8 n-gram shard tensors with the same stored shape. The 128
FP8 payloads plus the BF16 scalar scale account for the complete
`PLE FP8` category:

```text
128 * 400,001,920 + 2 = 51,200,245,762 bytes
```

The six BF16 PLE compute tensors remain in `main BF16`:

```text
model.language_model.layers.1.ple.key_proj.weight
model.language_model.layers.1.ple.value_proj.weight
model.language_model.layers.1.ple.norm_key.weight
model.language_model.layers.1.ple.norm_query.weight
model.language_model.layers.1.ple.norm_conv.weight
model.language_model.layers.1.ple.conv1d.weight
```

This is an ownership distinction, not a residency decision for the whole PLE
subsystem. The n-gram embedding remains file-backed; later bring-up must
preserve the existing asynchronous PLE policy.

## 8. MTP quantization classes

The MTP expert block is present in the sidecar and is disabled/skipped for
the initial M9 arm.

For each of 512 MTP experts and each of `gate_proj`, `up_proj`, and
`down_proj`, the sidecar contains:

| Tensor family | Count | Dtype | Representative stored shape | Scale family |
|---|---:|---|---|---|
| MTP expert weights | 1,536 | `F8_E4M3` | gate/up `[640, 2560]`; down `[2560, 640]` | matching `weight_scale_inv` |
| MTP inverse scales | 1,536 | `BF16` | gate/up `[5, 20]`; down `[20, 5]` | associated with the weight |

The sidecar also contains 29 BF16 MTP auxiliary tensors in the main shards,
including MTP attention, shared-expert, embedding, and hyper-connection
weights. These are included in the `MTP FP8` ownership class for exhaustive
inventory accounting, but they are not FP8 expert payloads.

MTP is not imported, resident, or executed by this metadata-only milestone.

## 9. Inventory and validation gates

The inventory must continue to satisfy all of the following before importer
work starts:

```text
tensor count == 299,545
unknown/unclassified count == 0
sum(category tensor bytes) == 132,639,846,394
sum(main shard payload) == 78,922,711,032
sidecar payload == 53,717,135,362
main shard payload + sidecar payload == index metadata.total_size
```

The JSON inventory records, for every tensor:

```text
name
source shard
dtype
stored shape
stored byte range and byte count
component class
layer and expert identifiers when present
projection when present
precision class
residency policy
associated scale tensor names
```

## 10. Explicit non-assumptions

The following are intentionally not frozen by this document:

- NVFP4 nibble ordering inside each U8 payload;
- physical matrix orientation after packing;
- scale byte encoding conversion details;
- cuBLASLt scale/swizzle layout;
- activation quantization staging;
- CUDA kernel selection;
- main-model loader ownership and residency implementation;
- MTP execution;
- full-model inference behavior.

Those require byte-level payload fixtures and semantic reference checks in a
later milestone. This milestone stops at the checkpoint contract and
complete header inventory.
