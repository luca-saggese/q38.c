# PLE Production Path Audit

This audit is static. `artifacts/perf/current/ple_timeline_validation_v1.json`
is not used to infer the production critical path. Its T7/T8 interval is a
fixture observation, not evidence that layer 1 is blocked for that interval.

## Production call path

The production decode path is:

```text
q38_session_eval_timed()
  -> q38_decode_step_with_backend_config_timed()
     -> q38_forward_full()
        -> q38_forward_state_prefetch_ple()
           -> q38_ple_ngram_ids_ref()
           -> q38_ple_scheduler_submit()
        -> decoder layer loop
           -> layer_number == 1:
              -> full_ple()
                 -> q38_forward_state_wait_ple()
                    -> q38_ple_scheduler_wait()
                 -> q38_ple_store_read_row()
                 -> PLE projections, gating, convolution
                 -> PLE contribution + hidden
        -> remaining decoder work and final head
```

The corresponding production locations are:

| Operation | Location |
|---|---|
| Session entry | `q38_session.c:q38_session_eval_timed()` (295-338) |
| Request creation/history and n-gram IDs | `q38_forward.c:q38_forward_state_prefetch_ple()` (1590-1630) |
| Request submit | `q38_forward.c:q38_forward_state_prefetch_ple()` (1631-1642), calling `q38_ple_scheduler_submit()` |
| Worker result publication | `q38_ple_prefetch.c:scheduler_worker()` (179-218) |
| Result check/wait | `q38_forward.c:q38_forward_state_wait_ple()` (1645-1650), calling `q38_ple_scheduler_wait()` |
| PLE injection | `q38_forward.c:full_ple()` (1659-1857) |
| Row lookup from the mapped GGUF | `q38_forward.c:full_ple()` (1704-1712), calling `q38_ple_store_read_row()` |

## Layer and injection point

The frozen configuration declares `ple_layer = 2`
(`q38_model_config.c:31`, validated at line 100). The production forward does
not read that field to select the injection. It hard-codes
`layer_number == 1` (`q38_forward.c:2224-2236`), which is the zero-based layer
corresponding to configured one-based layer 2.

The exact order is:

```text
embedding
for layer = 0 .. 47:
  layer 0:
    GR read -> mixer (GDN) -> GR write -> GR read -> MoE
    -> GR write -> layer output
  layer 1:
    PLE wait
    PLE row lookup/decode/projections/gating/convolution
    PLE contribution + hidden is copied into streams
    GR read -> mixer (GDN) -> GR write -> GR read -> MoE
    -> GR write -> layer output
  layers 2 .. 47: normal decoder layer sequence
final GR -> LM head -> logits
```

Therefore:

- **injection layer:** configured layer 2, runtime zero-based index 1;
- **injection point inside layer:** at the layer-1 entry, before the first
  `full_gr_read()` (`q38_forward.c:2238-2240`);
- it is **before norm, before mixer, before residual, before MoE**, and before
  all layer-1 decoder operations;
- it is not after the layer residual or after MoE.

## Request ownership

The request is for the same forward token that consumes it:

- `q38_forward_state_prefetch_ple()` receives the current `tokens[t]`;
- it computes IDs from the current token plus the copied prior history
  (`q38_forward.c:1611-1629`);
- `full_ple()` recomputes and consumes those IDs for the same `tokens[t]`
  (`q38_forward.c:1704-1711`);
- the session appends the token to session history only after forward returns
  (`q38_session.c:337-340`).

Thus the production association is **request token N -> consumed token N**,
not request N -> token N+1. The n-gram context is the prior state history plus
the current token: `q38_ngram_history_context()` receives the history and
`current_token` (`q38_ple_ref.c:47-48`). The prefetch loop then appends the
current token to its local history before preparing the next token in a
multi-token request (`q38_forward.c:1627-1629`).

## Lookup derivation

The production hash configuration is:

- n-gram size: 3;
- heads per n-gram: 8;
- supported n-gram orders: 2 and 3;
- rows per token: `heads_per_ngram * (ngram_size - 1)`;
- rows per token: `8 * (3 - 1) = 16`.

`q38_ple_ngram_ids_ref()` implements exactly two branches, `n = 2` and
`n = 3`, with eight heads in each branch (`q38_ple_ref.c:49-62`). The forward
allocates and submits `token_count * Q38_PLE_MAX_HEADS`, with
`Q38_PLE_MAX_HEADS = 16` (`q38_forward.c:1603-1607`, `1625-1626`).

Therefore the statically derivable production result is:

```text
accesses/token = 2 n-gram branches * 8 heads = 16 rows
```

The configured `split_ngram_parts = 128` is metadata (`q38_model_config.c:30`)
and is not multiplied into the request loop. The code therefore cannot derive
12,800 accesses/token. **12,800 is a discrepancy with this production path**
and must not be used as proof of the runtime lookup count.

Logical bytes are:

```text
logical_bytes/token = 16 rows * store.row_bytes
```

For the runtime Q8_0 PLE row size of 170 bytes, this is **2,720 logical bytes
per token**. For BF16 rows of 320 bytes, it is **5,120 logical bytes per
token**. A claim of 34,816,000 bytes/token is not derivable from this
single-token production lookup.

## File access and cache architecture

### Application and OS cache

There is no persistent application row cache. The scheduler retains only the
request's sorted row IDs and physical blocks; the job is freed after consume
(`q38_ple_prefetch.c:399-407`, `434-464`). The cache key is effectively the
mapped file page range represented by a coalesced physical block, and cache
lifetime is the OS page-cache/mmap lifetime. There is no application eviction
policy.

`mincore()` checks page residency (`q38_ple_prefetch.c:101-121`). On a miss,
the worker issues `madvise(MADV_WILLNEED)` and uses a transient 64 KiB buffer
(`q38_ple_prefetch.c:124-147`). No PLE table-sized application copy is
created.

### Actual operations

There are two distinct production access paths:

1. **Asynchronous warm path:** sorted adjacent blocks are coalesced
   (`q38_ple_prefetch.c:376-397`) and warmed with buffered `pread()`
   (`q38_ple_prefetch.c:150-170`).
2. **Synchronous consumer path:** each PLE row is copied from the existing
   GGUF `mmap()` with `memcpy()` (`q38_ple.c:149-165`).

`O_DIRECT` is not used. Reads are buffered; there is no alignment contract for
`pread()` beyond the normal file-descriptor API. The warm path uses a maximum
64 KiB request size, while a row is 170 bytes for Q8_0. Coalescing can make a
warm read larger than one row, but the consumer still performs mapped row
copies.

The accounting meanings are:

| Quantity | Meaning |
|---|---|
| Logical bytes | submitted logical rows multiplied by `row_bytes` |
| App-cache misses | coalesced block ranges found non-resident by `mincore()` |
| `pread` bytes | bytes returned by the buffered `pread()` calls |
| Physical NVMe bytes | **not observable from this code** |

`q38_ple_scheduler_stats.physical_bytes` is therefore a misnamed
`pread`-returned-byte counter, not physical NVMe traffic
(`q38_ple_prefetch.h:38-45`, `q38_ple_prefetch.c:155-160`).

## Replay semantic correction

The replay must not use an arbitrary T7/T8 span as the production wait. The
consumer arrival is T7 (`consume_ms`), and worker completion is T6
(`ready_ms`). The corrected definitions are:

```text
wait_at_consume = max(0, T6 - T7)
useful_overlap =
    min(worker_elapsed, max(0, T7 - T1))
```

The scheduler now records `wait_ms`/`wait_at_injection_ms` from worker
completion versus consume arrival and computes `overlap_ms` using the
submit-to-consume window (`q38_ple_prefetch.c:443-460`). The replay output
uses the same definitions and reports `submit_to_consume_window_ms` and
`useful_overlap_ms`; T8 remains only the condition-variable return timestamp.

This correction is semantic only. It does not claim that the fixture timeline
represents the production critical path.
