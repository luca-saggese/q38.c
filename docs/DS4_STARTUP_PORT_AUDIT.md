# DS4 Main Startup and Residency Port Audit

## Scope and method

This is a read-only source audit of the local `main` branch (`c1d4597a80e300b803dc642519718f2c999589da`) against the current q38 branch. No model was loaded and no benchmark was run. Counts below are statically provable call-site properties; they are not runtime measurements.

## DS4 startup call graph

The donor startup path is:

```text
GGUF model open/mmap
  -> ds4_repack_map_file()
       -> open(O_RDONLY)
       -> optional open(/proc/self/fd/<fd>, O_DIRECT)
       -> mmap(MAP_SHARED)
  -> ds4_repack_collect_catalog()
       -> parse metadata/tensor records from the mmap
  -> ds4_gpu_build_derived_artifacts()
       -> q2k aligned artifacts (optional)
       -> iq2 aligned artifacts (optional)
       -> q8 aligned artifacts (optional)
          -> repack_thread_count()
          -> worker pool
             -> pread() through ds4_repack_read_stage()
             -> cudaMemcpyAsync(H2D)
             -> repack CUDA kernel
             -> cudaStreamSynchronize()
       -> publish g_derived_ranges
       -> mark complete IQ2/Q2K expert replacement
  -> ds4_gpu_set_model_map()/set_model_fd()
       -> artifact map can leave raw mmap unpinned
       -> otherwise optional full copy or cudaHostRegister()
  -> accelerator_cache_model_tensors()
       -> accelerator_prepare_model_tensor_spans()
       -> sort and merge nearby tensor spans
       -> ds4_gpu_cache_model_range()
       -> optional q8/f16 selective cache
  -> final startup command/session synchronization
```

Relevant donor locations are `cuda/mmq/ds4_repack.cu:52-93,208-329,566-680`, `ds4_cuda.cu:3750-3845,4354-4474`, and `ds4.c:2890-3059`.

## DS4 mechanisms

| Mechanism | DS4 main | q38 current | Port candidate |
|---|---|---|---|
| Raw model access | GGUF mmap is retained as the source map. Repack uses a second read descriptor only for staged conversion. | `q38_gguf_open()` uses a private read-only mmap (`q38_gguf.c:345-395`). | Keep mmap for metadata/file-backed PLE, but do not use it as the upload unit for every resident tensor. |
| Direct I/O | `ds4_repack_map_file()` opens an `O_DIRECT` descriptor when available; reads are aligned down/up to filesystem alignment and issued with `pread()` (`ds4_repack.cu:67-91,306-329`). | No `O_DIRECT` path exists in the q38 GGUF loader or residency uploader. | Add a startup-only aligned staged reader, gated by a fixture comparison against buffered reads. |
| Read granularity | Repack uses large chunks, default `copy_chunk_bytes = 256 MiB`; Q8/IQ2 chunks are block-aligned and Q2K chunks are row-pair aligned (`ds4_cuda.cu:4390-4396`, `ds4_repack.cu:680-718,893-930,1007-1048`). | q38 residency walks tensors and copies each tensor's exact payload (`q38_forward_cuda.cu:1096-1229`). | Batch adjacent tensor spans into large transfers. |
| Worker count | `run_repack_jobs()` creates a bounded thread pool from `repack_thread_count(jobs.size())`; each worker owns one CUDA stream, pinned staging memory, and device scratch (`ds4_repack.cu:566-629`). | q38 residency has one context stream and no startup worker pool (`q38_forward_cuda.cu:1145-1228`). | Parallelize file staging/repack, but publish artifacts only after an explicit join. |
| Upload batching | Each worker overlaps file staging, H2D, and conversion for a chunk; artifact allocation occurs once per tensor (`ds4_repack.cu:629-678`). | Each tensor allocates and uploads separately (`q38_forward_cuda.cu:1159-1171`). | Use span/chunk uploads and persistent destination allocation. |
| Synchronization | Repack synchronizes once per converted chunk to protect the next reuse of scratch. The general mapped-range path uses stream/event synchronization around staged uploads (`ds4_cuda.cu:2226-2430`). | q38 synchronizes after every tensor upload and once again at the end (`q38_forward_cuda.cu:1190-1228`). | First remove per-tensor initialization fences; retain one final readiness fence. |
| Artifact ownership | `g_derived_ranges` owns device artifacts. Q8 is additive to raw storage; IQ2/Q2K can replace raw expert residency when every candidate is built (`ds4_cuda.cu:4380-4473`). | `persistent_tensor` entries own one CUDA allocation for every non-PLE tensor, while the mmap remains open (`q38_forward_cuda.cu:1145-1229`). | Introduce explicit `source`, `derived`, and `replacement` ownership states. |
| Raw residency | When complete aligned expert replacement is active, DS4 explicitly leaves the model mmap unpinned (`ds4_cuda.cu:3750-3757`). | q38 keeps the mmap as the host source while also allocating resident CUDA copies; PLE embedding shards remain file-backed by contract. | Avoid pinning or duplicating raw expert spans once replacement artifacts are complete. |
| Tensor mapping | DS4 sorts tensor spans and merges spans within 64 KiB and a configured maximum span before caching (`ds4.c:2945-3003`). | q38 indexes tensors individually and does not merge adjacent payload ranges during residency preparation. | Port span coalescing before any H2D operation. |
| Final synchronization | DS4 exposes command flush/end/synchronize boundaries (`ds4_cuda.cu:3684-3691`) rather than requiring a fence after each tensor. | q38 has `cudaStreamSynchronize()` inside the per-tensor residency loop plus a final fence. | Preserve one startup completion barrier after all uploads. |

## Exact static findings

* DS4 has one mmap for catalog/source access, one optional direct-I/O descriptor, and one staged pinned buffer plus one device scratch allocation per active repack worker.
* DS4 performs one device artifact allocation per selected source tensor, not one allocation per chunk. Chunk loops reuse the worker staging and scratch buffers.
* DS4 read order follows the catalog's tensor records and is therefore sequential per tensor; the merged mapping cache is range-oriented. The source does not prove that all selected tensor spans are globally contiguous, so “fully sequential model read” would be an overclaim.
* DS4 Q8 artifacts are explicitly additive. IQ2/Q2K artifacts are replacement candidates only after complete coverage is established.
* q38 current residency performs `cudaMalloc` + `cudaMemcpyAsync` + `cudaStreamSynchronize` inside the tensor loop. This is a per-tensor startup fence and is the clearest structural explanation for poor startup scaling.
* q38's duplicate check rejects duplicate host pointers, but it does not eliminate the normal raw-mmap plus resident-device-copy coexistence. The mmap remains required for file-backed PLE and fallback accesses.

## Port shortlist, ordered by expected impact

### 1. Batched aligned startup uploader

* **Donor:** `ds4_repack_map_file`, `ds4_repack_read_stage`, `run_repack_jobs`, `accelerator_prepare_model_tensor_spans`.
* **q38 target:** `q38_forward_cuda_enable_all_non_ple_residency()` and startup initialization in `q38_runtime_init()`.
* **Risk:** tensor offset/quant-block alignment and failure cleanup.
* **Expected startup impact:** very high; removes per-tensor fences and reduces syscall/API overhead.
* **Expected decode impact:** neutral directly; improves only initialization unless the same artifact is reused.
* **Required validation:** synthetic GGUF with adjacent/non-adjacent spans, exact byte/hash comparison, allocation/fence counters, and a no-model-load replay.

### 2. Persistent aligned derived artifacts for routed experts

* **Donor:** `ds4_gpu_build_derived_artifacts()`, `g_derived_ranges`, `ds4_gpu_model_range_replaced()`.
* **q38 target:** resident expert registration and `q38_forward_cuda_expert_backend()` / `q38_forward_cuda_moe_layer_q2_backend()`.
* **Risk:** quant layout, expert indexing, fallback to raw tensors, and artifact lifetime.
* **Expected startup impact:** high when repack is parallel and raw expert residency is removed.
* **Expected decode impact:** high for routed MoE by eliminating repeated host payload discovery/copy.
* **Required validation:** per-expert fixture parity for Q2K/IQ2, route-id permutations, replacement coverage audit, and resident-vs-raw checksum tests.

### 3. Explicit raw/derived ownership replacement

* **Donor:** `cuda_span_fully_replaced()`, `ds4_gpu_model_range_replaced()`, and the unpinned-mmap branch in `ds4_gpu_set_model_map()`.
* **q38 target:** `persistent_tensor`, `q38_exec_tensor`, and residency teardown in `q38_forward_cuda.cu`.
* **Risk:** fallback paths may still need raw mmap; partial replacement must never return an invalid device pointer.
* **Expected startup impact:** high memory-pressure reduction and less duplicate residency.
* **Expected decode impact:** medium to high, depending on expert/cache hit rate.
* **Required validation:** ownership table, partial-range negative tests, and teardown/reload tests.

### 4. Span coalescing for ordinary resident tensors

* **Donor:** sorted/merged `accelerator_tensor_span` preparation in `ds4.c`.
* **q38 target:** the tensor loop in `q38_forward_cuda_enable_all_non_ple_residency()`.
* **Risk:** accidentally copying metadata gaps or PLE embedding ranges.
* **Expected startup impact:** medium to high.
* **Expected decode impact:** neutral unless used to create derived resident ranges.
* **Required validation:** merged-range byte coverage and exclusion tests.

### 5. Startup readiness barrier redesign

* **Donor:** DS4 command flush/end/synchronize API and worker join.
* **q38 target:** residency initialization and `q38_runtime_init()`.
* **Risk:** allowing forward execution before a tensor is ready.
* **Expected startup impact:** medium after batching; it is not useful without ports 1-4.
* **Expected decode impact:** neutral.
* **Required validation:** injected delayed worker, readiness state machine, and first-token race test.

## Conclusion

The strongest donor pattern is not “copy the whole model faster.” It is: catalog once, read large aligned ranges, convert in parallel into persistent artifacts, publish ownership once, and fence once. q38 currently allocates and synchronizes at tensor granularity, while preserving the mmap as a live source. That is the primary startup port target.

