# PLE file-backed storage layout

This document describes the physical storage path only. The logical PLE
hashing, head order, and injection contract remain in
`docs/qwen_ple_semantics.md`.

## Verified local GGUF layout

The checked-in runtime artifact
`artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf` contains
128 tensors named:

```text
model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<n>.weight
```

where `<n>` is `0..127`. Each shard is GGUF `Q8_0`, has dimensions
`[2500012, 160]` (rows, row width), and occupies 170 bytes per row. Shard
indices, not tensor-directory order, define the physical-to-logical order.
The loader validates every shard name, dimension, type, and byte extent before
it exposes a row.

`q38_ple_store_bind_gguf()` retains only the GGUF mapping and shard
descriptors. `q38_ple_store_read_row()` and `q38_ple_store_read_rows()` copy
requested quantized rows into caller-owned buffers in logical ID order. They
never allocate a table-sized buffer or dequantize the table. A bounded
`q38_ple_cache` may be layered over those row reads.

## Format boundary

The local safetensors checkpoint is the source used by the offline conversion
and golden-vector tools. This C runtime has no safetensors parser or shard
binding API; implementing one would require a separate verified format
contract. Therefore the file-backed runtime loader intentionally accepts the
verified converted GGUF layout only. It does not claim safetensors runtime
support or invent offsets from safetensors metadata.

## Residency evidence boundary

The M4-C10 host test measures GGUF mapping size, requested row copies, and the
configured bounded cache. It does not report CUDA staging, page faults, or
full-forward overlap: the repository still lacks a complete PLE-enabled
forward runtime and no CUDA-visible staging API exists for this loader.
Those fields remain explicitly unmeasured rather than synthesized.
