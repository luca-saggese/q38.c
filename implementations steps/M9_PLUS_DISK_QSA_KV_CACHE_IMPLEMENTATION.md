# q38.c — M9+ Disk QSA/KV Cache & Durable Session Store

**Placement:** after core M9 session/checkpoint ABI is stable; may proceed in parallel with server work.  
**Target:** Qwen3.8-Flash-Next / `qwen4_exp` on DGX Spark / GB10 / CUDA.  
**PLE invariant:** PLE remains permanently SSD/file-backed and is not part of this cache.

## 0. Executive decision

q38 should add a disk-backed QSA/KV session cache inspired by ds4's Disk KV Cache, adapted to Qwen3.8 semantic state.

The first implementation is **not** per-attention random NVMe paging. It is:

```text
active session
  -> live semantic state in RAM/unified memory
  -> periodic durable prefix checkpoints on NVMe

inactive/restarted session
  -> sequentially load best matching checkpoint
  -> restore semantic state
  -> prefill only uncached suffix
  -> continue decode
```

This matches what ds4 actually implements today. A later optional track may test live segmented cold-QSA reads from NVMe.

## 1. What ds4 does and what we copy conceptually

Current ds4 Disk KV Cache:

- one live in-memory session graph/KV state;
- disk entries are prefix checkpoints;
- cache keys are SHA-1 of rendered prefix bytes;
- restored payload contains exact token history + serialized graph state;
- ordinary sequential file I/O is used rather than mmap;
- save reasons include `cold`, `continued`, `evict`, `shutdown`;
- default policy is approximately min 512 tokens, cold max 30k, trim 32, align 2048, continued interval 10k;
- save writes a temporary file and atomically renames it;
- disk budget + eviction policy use hits, age, token count, size, reason and prefix supersession;
- payload ABI and model/context compatibility are validated before restore.

Primary references:

- https://github.com/antirez/ds4/blob/main/README.md
- https://github.com/antirez/ds4/blob/main/ds4_kvstore.c
- https://github.com/antirez/ds4/blob/main/ds4.h
- https://github.com/antirez/ds4/issues/444
- https://github.com/antirez/ds4/issues/176
- https://github.com/antirez/ds4/issues/805
- https://github.com/antirez/ds4/issues/847

Important q38 correction: ds4's production Disk KV Cache is primarily a durable **prefix checkpoint / session resume** mechanism. It is not evidence that its attention kernel directly streams historical KV from SSD every token.

## 2. q38 goals

```text
avoid repeated long prefill
survive worker/server restart
support many inactive sessions without retaining all state in RAM
resume matching prefixes quickly
use large sequential NVMe I/O
preserve exact semantic state
bound disk usage
avoid CUDA interference during save
```

Phase-1 non-goals:

- random NVMe reads inside every QSA attention operation;
- `O_DIRECT`;
- `io_uring`;
- compressed checkpoint payloads;
- cross-model reuse;
- MTP scratch persistence.

## 3. State to serialize

Persist only semantic session state:

```text
token history
position / committed count

for every GDN layer:
  recurrent S state
  convolution history
  conv indices/position if required

for every QSA layer:
  raw K
  raw V
  compressed K/V or compressed representation
  indexer state
  pending/raw ring
  logical lengths/frontiers/write pointers
  continuation-critical metadata

PLE:
  n-gram/token semantic history and counters only

optional:
  current logits if required for exact resume
  deterministic RNG state once sampling is supported
```

Never persist:

```text
model weights
resident weight arenas
PLE weight pages/cache
CUDA workspace
MoE/QSA scratch
CUDA pointers
CUDA graph instances
profiling data
```

## 4. Module split

Create:

```text
q38_kvstore.h
q38_kvstore.c
```

Responsibilities:

```text
q38_session:
  semantic serialization/deserialization

q38_kvstore:
  directory
  keys
  metadata
  prefix lookup
  atomic files
  disk budget
  eviction
  checkpoint policy

server/CLI:
  lifecycle policy: cold/continued/evict/shutdown
```

## 5. File ABI

Use a q38-specific format, not ds4's exact binary format.

```c
#define Q38_SESSION_PAYLOAD_MAGIC   UINT32_C(0x3853514b)
#define Q38_SESSION_PAYLOAD_VERSION 1u

typedef struct {
    uint32_t magic;
    uint16_t file_version;
    uint16_t payload_abi;

    uint32_t model_arch_id;
    uint32_t weight_abi;
    uint32_t quant_recipe_id;
    uint32_t ctx_size;

    uint64_t token_count;
    uint64_t payload_bytes;

    uint64_t created_unix;
    uint64_t last_used_unix;

    uint32_t reason;
    uint32_t flags;

    uint8_t model_sha256[32];
    uint8_t quant_manifest_sha256[32];
    uint8_t tokenizer_sha256[32];

    uint8_t reserved[64];
} q38_kv_file_header;
```

Header must be fixed-size and endian-defined.

## 6. Identity safety

Do not repeat the unsafe structural-ID-only reuse reported in ds4 issue #805.

Default compatibility requires exact:

```text
model artifact SHA-256
quant manifest SHA-256
tokenizer SHA-256
weight ABI
payload ABI
architecture ID
```

Different model weights => reject.  
Different tokenizer => reject.  
Different state ABI => reject.  
Cross-Q2/Q4 reuse => reject initially.

## 7. Key strategy

Phase 1:

```text
key = SHA-256(model identity || payload ABI || exact token IDs[0..N))
```

File:

```text
<hash>.qkv
```

Phase 2 may add rendered-byte prefix matching like ds4 for chat clients that resend equivalent text with different token-boundary spelling.

## 8. Sequential file layout

```text
fixed header
exact token IDs
optional rendered-prefix bytes
section table
GDN recurrent state
GDN conv state
QSA raw K/V
QSA compressed/index state
QSA pending metadata/state
PLE semantic history
optional logits/RNG
checksums/footer
```

Do not mmap checkpoint files in phase 1. Use buffered sequential read/write.

## 9. Section table

```c
typedef enum {
    Q38_KV_SECTION_TOKENS = 1,
    Q38_KV_SECTION_GDN,
    Q38_KV_SECTION_QSA,
    Q38_KV_SECTION_PLE_HISTORY,
    Q38_KV_SECTION_LOGITS,
    Q38_KV_SECTION_RNG,
} q38_kv_section_type;

typedef struct {
    uint32_t type;
    uint32_t version;
    uint64_t offset;
    uint64_t bytes;
    uint64_t checksum;
} q38_kv_section;
```

Bounds-check every offset/length before restoring.

## 10. Session serialization API

```c
uint64_t q38_session_payload_bytes(const q38_session *s);

int q38_session_stage_payload(
    q38_session *s,
    q38_session_payload_file *out,
    char *err,
    size_t err_len);

int q38_session_write_staged_payload(
    const q38_session_payload_file *payload,
    FILE *fp,
    char *err,
    size_t err_len);

int q38_session_load_payload(
    q38_session *s,
    FILE *fp,
    uint64_t payload_bytes,
    char *err,
    size_t err_len);
```

This deliberately mirrors the useful ds4 separation: engine/session owns semantic snapshot content; kvstore owns persistence policy.

## 11. KV store API

```c
typedef enum {
    Q38_KV_REASON_COLD = 1,
    Q38_KV_REASON_CONTINUED,
    Q38_KV_REASON_EVICT,
    Q38_KV_REASON_SHUTDOWN,
    Q38_KV_REASON_EXPLICIT,
} q38_kv_reason;

typedef struct {
    uint64_t budget_bytes;
    uint32_t min_tokens;
    uint32_t cold_max_tokens;
    uint32_t continued_interval_tokens;
    uint32_t boundary_trim_tokens;
    uint32_t boundary_align_tokens;
    uint32_t max_staged_jobs;
    uint64_t max_staged_bytes;
} q38_kvstore_options;
```

Core functions:

```c
int q38_kvstore_open(...);
void q38_kvstore_close(...);

int q38_kvstore_try_restore(
    q38_kvstore *store,
    q38_runtime *rt,
    q38_session *session,
    const uint32_t *prompt,
    uint32_t prompt_tokens,
    q38_kvstore_result *result,
    char *err,
    size_t err_len);

int q38_kvstore_queue_save(
    q38_kvstore *store,
    q38_runtime *rt,
    q38_session *session,
    const uint32_t *tokens,
    uint32_t tokens_len,
    uint32_t store_len,
    q38_kv_reason reason,
    char *err,
    size_t err_len);
```

## 12. Checkpoint-safe frontier

Never snapshot:

```text
mid-layer
mid-QSA update
mid-GDN recurrence
partial prefill chunk commit
uncommitted speculative token
partial PLE-history update
```

Add:

```c
bool q38_session_checkpoint_safe(const q38_session *s);
```

A save is allowed only when:

```text
semantic CUDA state is committed
session position == token history length
no MTP verification transaction active
no partial chunk semantic commit active
```

## 13. Avoid ds4's mid-prefill failure mode

A real ds4 CUDA issue shows that a continued checkpoint overlapping long prefill can destabilize the CUDA path.

q38 phase-1 rule:

```text
the I/O worker never calls CUDA
the inference thread never writes the large checkpoint file
```

Flow:

```text
CUDA/inference thread
  -> checkpoint-safe frontier
  -> stage immutable semantic payload
  -> enqueue bounded job
  -> resume inference immediately

I/O worker
  -> sequentially write temp file
  -> flush/close
  -> atomic rename
```

## 14. Bounded asynchronous I/O worker

```c
typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    q38_checkpoint_job queue[Q38_KV_QUEUE_CAP];
    uint32_t head;
    uint32_t tail;

    uint64_t queued_bytes;
    bool stop;
} q38_kv_io_worker;
```

Policies:

```text
continued checkpoint + queue full -> skip, never stall decode
explicit/evict -> caller policy may wait
shutdown -> drain queue
```

No unbounded staged snapshots.

## 15. Atomic save

```text
<hash>.qkv.tmp.<pid>.<counter>
   -> write header
   -> tokens
   -> sections
   -> payload
   -> checksums
   -> fflush
   -> close
   -> rename to <hash>.qkv
```

Never update a valid checkpoint in place.

## 16. Checkpoint reasons

Adopt ds4's useful operational model:

```text
cold       stable prefix after a long initial prompt
continued  periodic growing-conversation checkpoint
evict      before active session state is replaced
shutdown   clean process shutdown
explicit   explicit session-save API
```

Initial policy can be ds4-inspired:

```text
min_tokens = 512
cold_max_tokens = 30000
continued_interval_tokens = 10000
boundary_trim_tokens = 32
boundary_align_tokens = prefill_chunk
```

These are defaults to benchmark, not immutable model facts.

## 17. Stable boundary trimming/alignment

```c
static uint32_t q38_kv_stable_prefix(
    uint32_t tokens,
    uint32_t min_tokens,
    uint32_t trim,
    uint32_t align)
{
    if (tokens <= min_tokens + trim)
        return tokens;

    uint32_t stable = tokens - trim;
    if (align)
        stable -= stable % align;

    return stable >= min_tokens ? stable : tokens;
}
```

Align continued/cold checkpoints to q38 chunked-prefill semantic frontiers.

## 18. Prefix lookup

Phase 1 exact token prefix:

```text
prompt
 -> longest stored exact token prefix
 -> restore
 -> prefill suffix only
```

Startup scans fixed headers only and builds metadata index.

```c
typedef struct {
    char hash_hex[65];
    char *path;
    uint64_t token_count;
    uint64_t file_bytes;
    uint64_t created_at;
    uint64_t last_used;
    uint32_t hits;
    uint32_t ctx_size;
    q38_kv_reason reason;
    q38_checkpoint_identity identity;
} q38_kv_entry;
```

Do not read payloads during directory scan.

## 19. Restore path

```text
find longest compatible prefix
open file
read/validate fixed header
validate model identity
validate exact token prefix
sequentially read payload
restore session state
validate restored state
update hit metadata
prefill suffix only
```

Pseudo-code:

```c
int q38_kvstore_try_restore(...) {
    q38_kv_entry *e = find_longest_prefix(...);
    if (!e) return 0;

    FILE *fp = fopen(e->path, "rb");
    if (!fp) return 0;

    q38_kv_file_header h;
    if (read_header(fp, &h) != 0) goto invalid;
    if (!identity_matches(rt, &h)) goto incompatible;
    if (!prefix_matches(fp, prompt, prompt_tokens, &h)) goto invalid;

    if (q38_session_load_payload(
            session, fp, h.payload_bytes,
            err, sizeof(err)) != 0)
        goto invalid;

    if (!q38_session_validate_restored_state(session))
        goto invalid;

    touch_entry(e);
    fclose(fp);
    return (int)h.token_count;

invalid:
    fclose(fp);
    quarantine_or_unlink(e);
    return 0;

incompatible:
    fclose(fp);
    return 0;
}
```

## 20. Restore for NVMe throughput

Payload order should match restore order:

```text
tokens
GDN recurrent all layers
GDN conv all layers
QSA raw state
QSA compressed/index state
QSA pending metadata
PLE history
optional logits/RNG
```

Use large contiguous sections. Avoid interleaving tiny records.

File alignment:

```text
>=4 KiB
optionally 64 KiB for large sections
```

Initial implementation uses normal buffered sequential I/O, not `O_DIRECT`.

## 21. GPU state restore

On-disk payload is pointer-free host representation.

Restore using few large copies:

```text
NVMe sequential read -> host staging
GDN family -> batched H2D
QSA family -> batched H2D
metadata -> restore CPU/session fields
```

Measure separately:

```text
NVMe read ms
host validation ms
H2D state-restore ms
suffix prefill ms
total resume ms
```

## 22. Disk budget and eviction

CLI/server:

```text
--kv-disk-dir PATH
--kv-disk-space-mb N
--kv-cache-min-tokens N
--kv-cache-cold-max-tokens N
--kv-cache-continued-interval-tokens N
--kv-cache-boundary-trim-tokens N
--kv-cache-boundary-align-tokens N
```

Eviction score:

```text
effective_hits = hits * exp2(-age / half_life)

score =
  (effective_hits + 1)
  * token_count
  / file_bytes

cold/evict/shutdown anchors get a positive reason weight
old continued prefixes superseded by incoming longer prefixes get penalized
```

Evict lowest score until enough space exists.

Pre-evict before writing so the new checkpoint is not immediately self-evicted.

## 23. Active-session tier

Initial M9 server model:

```text
1 active session in RAM/unified memory
many durable prefix checkpoints on NVMe
```

Session switch:

```text
A active
 -> queue evict checkpoint
 -> reset/reuse active state
 -> lookup/restore B checkpoint
 -> prefill B suffix
```

Later allow a small pool of active sessions + large NVMe cold store.

## 24. Interaction with PLE

Only persist:

```text
PLE semantic token/ngram history
```

Never persist:

```text
PLE weight pages
PLE page cache
PLE decoded rows
PLE async request queue
```

After restore:

```text
restore PLE semantic history
restart normal SSD-backed PLE prefetch
```

First resumed token must produce the same PLE lookup sequence as uninterrupted execution.

## 25. Interaction with MTP

Initial:

```text
checkpoint forbidden while speculative transaction is unresolved
```

Checkpoint only committed accepted state.

Do not serialize draft scratch.

## 26. Resume equivalence test

For prefix P and suffix S:

```text
Path A:
fresh -> prefill(P+S)

Path B:
fresh -> restore(P) -> prefill(S)
```

Compare:

```text
token IDs exact
position exact
GDN recurrent state
GDN conv history
QSA raw K/V
QSA compressed/index
QSA pending/raw state
QSA logical frontiers
PLE semantic history
final hidden
final logits
argmax
```

Raw serialization should permit bit-exact state for most sections.

## 27. Corruption and incompatibility tests

Test:

```text
bad magic
bad file version
bad payload ABI
different model SHA
different tokenizer SHA
truncated header
truncated payload
invalid section offset
checksum mismatch
wrong token prefix
integer overflow
unknown section
```

Required behavior:

```text
never crash
never partially continue
invalidate partial restore
ignore/quarantine bad file
fall back to normal prefill
```

## 28. Benchmark suite

Create:

```text
tests/test_m9_kv_disk_resume.c
```

Prefix sizes:

```text
512
2k
8k
32k
64k
128k
262k when supported
```

For each:

```text
checkpoint MiB
save ms / GB/s
load ms / GB/s
H2D restore ms
suffix prefill ms
resume total
cold prefill total
speedup
```

Main metric:

```text
saved_ms = cold_prefill_ms - resume_total_ms
```

## 29. Telemetry

```text
kv_disk_lookup_count
kv_disk_hit_count
kv_disk_miss_count
kv_disk_prefix_tokens_restored
kv_disk_suffix_tokens_prefilled

kv_disk_bytes_read
kv_disk_read_ms
kv_disk_read_gbps

kv_disk_bytes_written
kv_disk_write_ms
kv_disk_write_gbps

kv_disk_checkpoint_queued
kv_disk_checkpoint_skipped_queue_full
kv_disk_checkpoint_skipped_unsafe
kv_disk_checkpoint_skipped_too_small

kv_disk_evictions
kv_disk_evicted_bytes

kv_disk_corrupt_entries
kv_disk_incompatible_entries

resume_total_ms
cold_prefill_equivalent_ms
resume_speedup
```

## 30. Known ds4 failure modes q38 must test explicitly

1. **Disk full / immediate self-eviction**  
   New cache entry must not be written then instantly evicted by the same admission pass.

2. **Different weights, same model shape**  
   Reject using exact model fingerprint.

3. **Checkpoint during long CUDA prefill**  
   Snapshot only at safe frontier; disk I/O thread never uses CUDA.

4. **Long context**  
   Soak 32k/64k/128k/262k with periodic saves.

5. **Restart**  
   Save -> destroy process -> restart -> restore -> exact continuation.

## 31. Optional later: true live QSA cold tier

This is **not** what ds4's current prefix cache proves, but q38 can experiment later.

Potential hierarchy:

```text
L0 recent/hot QSA state -> unified memory
L1 bounded host cache
L2 old QSA segments -> NVMe
```

Only prototype if QSA historical access is:

```text
predictable
large/coalescible
prefetchable before consumption
```

Segment by layer/token range, e.g. 4096-token ranges.

No live-SSD tier in initial M9 acceptance.

## 32. NVMe optimization order

After correctness-first buffered baseline:

```text
1. larger stdio/read-write buffers
2. pread/readv
3. batched section reads
4. io_uring
5. O_DIRECT only if page cache is measured harmful
```

Never assume `O_DIRECT` is faster.

## 33. Suggested commits

```text
M9-KV-C00 docs: freeze disk-KV design and payload ABI
M9-KV-C01 session: enumerate serialized semantic state
M9-KV-C02 session: exact in-memory snapshot save/load
M9-KV-C03 session: staged payload API
M9-KV-C04 kvstore: file header, identities, exact-token key
M9-KV-C05 kvstore: sequential atomic save/load
M9-KV-C06 kvstore: startup metadata index
M9-KV-C07 kvstore: longest-prefix restore + suffix prefill
M9-KV-C08 tests: restore-vs-fresh state equivalence
M9-KV-C09 kvstore: disk budget + eviction
M9-KV-C10 server: cold/continued/evict/shutdown policy
M9-KV-C11 kvstore: bounded async I/O worker
M9-KV-C12 prefill: safe aligned checkpoint frontiers
M9-KV-C13 telemetry: NVMe throughput + resume speedup
M9-KV-C14 soak: restart/disk-full/corruption/long-context
M9-KV-C15 docs: disk-KV acceptance PASS
```

## 34. Acceptance

Complete when q38 proves:

```text
long session saved to NVMe
process/session destroyed
checkpoint restored by large sequential reads
only suffix prefills
GDN/QSA/PLE semantic state equals uninterrupted execution
final logits/tokens equal
disk budget/eviction safe
normal checkpoint writes do not materially block inference
wrong-model/corrupt checkpoints are rejected
PLE remains independent SSD-backed subsystem
```

The initial product value is **prefix reuse + inactive-session capacity + restart persistence**.

True live QSA/KV paging from NVMe is a later, separately benchmarked optimization.
