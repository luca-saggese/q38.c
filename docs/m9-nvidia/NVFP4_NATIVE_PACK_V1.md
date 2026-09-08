# Q38 native NVFP4 pack

M9N-03 adds the first q38-native descriptor format for the NVIDIA checkpoint:

```text
Q38_NVFP4_PACK_V1
```

The offline importer is `tools/q38_nvfp4_pack.py`. It reads safetensors
headers directly and never imports Python `safetensors`, Transformers,
Hugging Face Hub, or ModelOpt. Materialized payloads are copied with a
bounded 16 MiB buffer; no FP4 dequantization or cuBLASLt/CUTLASS swizzle is
performed.

## Native layout

The routed expert table is indexed as:

```text
[layer][projection][expert][component]
```

with projection order `gate`, `up`, `down` and component order:

```text
weight U8
weight_scale F8_E4M3
weight_scale_2 F32
input_scale F32
```

The canonical checkpoint row layouts remain unchanged:

```text
gate/up: [640, 1280] packed weight, [640, 160] block scales
down:    [2560, 320] packed weight, [2560, 40] block scales
```

BF16 tensors have a separate descriptor table and retain their original
dtype, shape, and source bytes. MTP tensors are indexed but disabled.
Vision tensors are indexed but disabled.

## PLE backing

PLE is never copied into the native pack. The pack records the sidecar
filename, source offsets, shapes, and bytes for all 128 FP8 n-gram shards.
The sidecar is opened read-only and mapped as the external backing store.
Shard IDs must be resolved through the descriptor table: the safetensors
header places `shard_100` and `shard_101` before `shard_98` and `shard_99`
physically even though all shard payloads are contiguous.

## Current local mode

The local filesystem has approximately 28.8 GB available while a fully
materialized main pack requires approximately 78.8 GB in addition to the
downloaded checkpoint. The importer therefore fails closed before creating a
partial file in materialized mode and currently produces a source-backed
descriptor pack:

```text
artifacts/m9-nvidia/q38_nvfp4_source_backed.pack
artifacts/m9-nvidia/q38_nvfp4_pack_manifest.json
```

The source-backed pack is a valid load/bind artifact, but it is not a
materialized CUDA residency image. Its planned main payload is
`77,843,711,744` bytes and its actual CUDA payload allocation in the
load-only check is zero. A materialized load-only gate must be run on a
filesystem with enough free space; no inference or NVFP4 kernel is part of
this milestone.

## Integrity and round-trip

The versioned summary is regenerated from the local headers:

```text
artifacts/m9-nvidia/nvfp4_tensor_inventory_summary.json
```

It records the pinned checkpoint revision, exhaustive category totals,
source-shard totals, region offsets/bytes, and `unknown_count = 0`. The
early, middle, and late M9N-02 fixtures retain exact packed weight,
weight-scale, and scalar bytes after native binding; the independent
dequantization result remains `max_abs = 0`, `mismatch = 0`.
