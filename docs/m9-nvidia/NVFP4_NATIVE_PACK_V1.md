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

## Materialized load-only result

The complete materialized pack is:

```text
artifacts/m9-nvidia/q38_nvfp4_materialized.pack
```

Its file size is `78,751,653,215` bytes. The main CUDA residency image
excludes the optional indexed vision region and contains
`77,843,711,744` bytes:

```text
NVFP4 weights: 60,397,977,600 bytes
NVFP4 scales:   7,549,747,200 bytes
BF16 main:      9,895,397,120 bytes
```

The load-only path allocates five CUDA regions and copies the materialized
payload without dequantization or swizzling. The completed gate reported
`77,843,711,744` allocated and copied bytes, `0` PLE resident bytes, and no
inference. The host RSS after the load-only process was
`17,879,400,448` bytes; this is separate from the CUDA allocation accounting.

## Integrity and round-trip

The versioned summary is regenerated from the local headers:

```text
artifacts/m9-nvidia/nvfp4_tensor_inventory_summary.json
```

It records the pinned checkpoint revision, exhaustive category totals,
source-shard totals, materialized region offsets/bytes, complete region
hashes, and `unknown_count = 0`. The early, middle, and late M9N-02 fixtures
retain exact packed weight, weight-scale, and scalar bytes when read from both
source-backed and materialized packs; the independent dequantization result
remains `max_abs = 0`, `mismatch = 0`.
