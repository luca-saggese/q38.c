# S4C PLE critical-path attribution

S4C keeps the existing forward timing tree and the PLE scheduler metrics
separate. They are not interchangeable:

| Field | Scope | Additive to wall? |
|---|---|---|
| `exclusive_forward_timing.ple_ms` | Exclusive CPU wall span of the `ple_injection` timing-tree node after child spans are subtracted | Yes, as the forward timing-tree category |
| `ple_elapsed_ms` | Async prefetch worker elapsed time from job start to page-cache warming completion | No |
| `ple_overlap_ms` | Worker elapsed time overlapped by the main thread before the injection wait | No |
| `ple_critical_stall_ms` | Main-thread wait at `q38_forward_state_wait_ple` | Yes |

Therefore `ple_ms ~= 177.5 ms`, `ple_elapsed_ms ~= 0.055 ms`, and
`ple_critical_stall_ms == 0` can coexist: the first includes the full PLE
injection computation and file-backed row consumption on the forward path,
the second covers only the asynchronous prefetch worker, and the third says
the worker completed before the consumer reached its wait.

The S4C attribution fields are:

- request construction: `request_build_ms`, `history_ngram_ms`,
  `index_lookup_ms`;
- asynchronous storage: `async_submit_ms`, `file_io_ms`, `worker_exec_ms`,
  `worker_cpu_ms`, `result_publish_ms`;
- consumer path: `decode_dequant_ms`, `accumulation_ms`, `injection_ms`,
  `wait_at_injection_ms`.

Worker fields are reported separately and must not be added to the main
critical path. `wait_at_injection_ms` is the only scheduler wait that enters
critical-path accounting. The PLE table remains mmap/file-backed; the
scheduler stores row IDs and transient 64 KiB read buffers only.

The `UNKNOWN` matrix traffic is classified as PLE file-backed lookup when its
record has `ple_file_backed_access=true`. The replay fixture
`tests/test_s4c_ple_replay` consumes the captured rows from
`artifacts/m4/ple_injection_vectors.json` against the real PLE shard layout
in the runtime GGUF and reports accesses, bytes, read syscalls, read-size
range, cache hits/misses, worker CPU time, and worker wall time. It uses
buffered transient `pread`; direct/aligned I/O is intentionally not tested in
S4C.
