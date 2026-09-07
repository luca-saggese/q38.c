# PLE Projection C1: Resident Grouped Q8 Projection

## Result

C1 is **GREEN** on the isolated production-Q8 fixture and is promoted as
`PLE_PROJ_C1`.

The fixture uses Q8_0 tensors extracted directly from the production GGUF,
not the source BF16 tensors. `PLE_PROJ_Q8_ORACLE_V1` decodes each Q8_0 row
(`FP16 scale + 32 signed int8 values`) to FP32 and computes the complete
CPU reference projection for `early`, `middle`, and `late`.

| Case | Complete wall/token | CUDA execution | Max abs | RMSE |
|---|---:|---:|---:|---:|
| early | 0.339477 ms | 0.004270 ms | 5.96e-08 | 2.32e-09 |
| middle | 0.328387 ms | 0.003140 ms | 1.19e-07 | 2.60e-09 |
| late | 0.386792 ms | 0.003154 ms | 1.19e-07 | 2.41e-09 |

The previous production projection attribution was `176.868638 ms/token`.
The isolated C1 projection wall is approximately `0.352 ms/token`, an
approximately 508x reduction in the isolated projection path and well above
the required 10% gate. This is more than an order-of-magnitude improvement;
the isolated result does not indicate a second matrix-backend bottleneck.

## Correctness fixture

`tools/extract_ple_projection_q8_fixture.py` parses GGUF metadata and maps
only the required tensor payload slices. It does not invoke a model loader.
The fixture contains:

* `key_proj [10240,2560] Q8_0`;
* `value_proj [2560,2560] Q8_0`;
* 48 Q8_0 PLE rows for the three captured cases;
* complete expected `key[10240]` and `value[2560]` outputs.

The independent oracle is implemented by the extractor and is not the CUDA
production implementation. The benchmark validates all complete outputs for
all three cases for finiteness, determinism, maximum absolute error, and
RMSE.

## Residency contract

The classification is now explicit:

| Class | Tensors | Residency |
|---|---|---|
| `PLE_EMBEDDING_TABLE` | `.ple.ple_embedding.ngram_embedding.shard_*` | Permanently mmap/file-backed; never fully resident |
| `PLE_DENSE_WEIGHTS` | `.ple.key_proj.weight`, `.ple.value_proj.weight` | Ordinary persistent CUDA residency allowed |
| `PLE_METADATA` | `ngram_heads_offsets`, `ngram_heads_vocab_sizes`, `layer_multipliers` | Small ordinary tensors; persistent residency allowed |
| Other PLE tensors | norms and `conv1d` | Existing ordinary tensor handling; unchanged by C1 |

`full_is_file_backed_ple_embedding()` now excludes dense projection weights
from the file-backed matrix-backend bypass. CUDA residency uses the same
embedding-table predicate, so key/value weights enter the ordinary persistent
execution set while the large embedding shards remain excluded.

## Production backend path

The existing backend supports the exact C1 contract. No new kernel was
written:

```text
full_matvec(key_proj)   -> resident q38_forward_cuda_matrix_backend()
full_matvec(value_proj) -> resident q38_forward_cuda_matrix_backend()
  -> q38_cuda_gdn_project(Q38_GDN_WEIGHT_Q8_0, ...)
```

The isolated benchmark allocates device weights, input, and outputs once,
uploads both weights before timing, uploads the input once per iteration,
launches two Q8 matrix calls, copies the complete outputs once, and performs
one stream synchronization. Across the timed iteration:

```text
matrix calls/token       2
row callbacks/token      0
projection weight H2D    0 bytes
input H2D                10240 bytes
output D2H               51200 bytes
syncs                    1
```

The output dimensions are complete projection outputs, not scalar row
results. The permanent file-backed invariant applies only to the large PLE
embedding table.

## Scope boundary

C1 does not implement fused dequantization, a combined key/value kernel, or a
specialized Qwen3.8 PLE kernel. Those remain C2/C3 candidates only if a
production end-to-end measurement later shows a material residual after this
structural fix.
