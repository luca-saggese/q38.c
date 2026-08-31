Qwen3.8-Flash-Next / DGX Spark prototype 

# M4S — Implementation Specification 

_PLE / n-gram SSD-backed: NVMe-resident quantized table, bounded RAM cache, asynchronous fetch e deterministic equivalence sul DGX Spark._ 

**Metodo:** nessun comportamento multimodale o I/O viene dedotto per analogia. Checkpoint/config e reference processor/model sono autoritativi; ogni optimization layer deve avere un fallback semplice e un test di equivalenza. 

## **1. Stato rispetto alla roadmap esistente** 

M4 ha già separato PLE da normali weights e previsto cache/residency/prefetch. M7 ha previsto un ADR per decidere se usare SSD. Tuttavia non esiste ancora una milestone che renda il path SSD un requisito implementato e accettato. M4S chiude precisamente questo gap. 

**Importante:** Qwen dichiara ufficialmente che la tabella n-gram può stare in host memory ed essere prefetched asincronamente. Non dichiara che debba stare su SSD. L’NVMe-backed PLE è una nostra estensione di runtime per il <u>budget 128 GB dello Spark, quindi va validata indipendentemente.</u> 

## **2. Definition of Done** 

- La PLE quantizzata può rimanere file-backed su NVMe senza richiedere residency completa in RAM/unified memory. 

- Lookup SSD-backed e lookup fully-resident producono gli stessi PLE row values/hidden/logits entro la tolleranza del quant format. 

- Esiste una bounded RAM cache con budget esplicito; nessuna cache può espandersi fino a residentare accidentalmente l’intera PLE. 

- Prefill usa batching/dedup/coalescing degli accessi; decode usa prefetch deterministico basato sugli n-gram IDs già noti. 

- I/O e compute possono sovrapporsi senza cambiare semantics o accumulation order. 

- Il path funziona con Q2 PLE bootstrap e Q4 PLE candidate. 

- La modalità SSD è selezionabile esplicitamente e produce telemetry su hit rate, bytes letti, wait time, page-cache effects e tail latency. 

## **3. Architettura proposta** 

```
GGUF / dedicated PLE extent on NVMe
```

```
       |
```

```
       +-- row/block address map (read-only)
       |
```

- `+-- bounded quantized-page cache in host RAM` 

```
                |
```

- `+-- miss -> async read/coalesced read` 

```
                |
```

```
                +-- hit
```

```
                v
          quant row/block batch
                |
          CUDA decode/accumulate
                |
          layer-2 PLE injection
```

## **4. I/O policy: baseline prima, sofisticazione dopo** 

|**Mode**|**Uso**|**Decisione iniziale**|
|---|---|---|
|mmap + OS page cache|baseline SSD-backed|PRIMO path da implementare; semplice e<br>osservabile|
|pread/preadv batched|fallback/controlled I/O|implementare se mmap page faults/tail sono<br>problematici|



Target: GB10 / 128 GB coherent unified memory / CUDA only 

|||Qwen3.8-Flash-Next /DGXSparkprototype<br>|
|---|---|---|
|**Mode**|**Uso**|**Decisione iniziale**|
|io_uring|optional optimization|solo se supportato/utile su DGX OS e<br>benchmark mostra gain|
|O_DIRECT|non iniziale|evitare finché page cache non è dimostrata<br>problematica; alignment/complexity elevati|
|whole-file cudaHostRegister|vietato|contrario al budget|
|whole-file cudaMallocManaged|vietato|annullalo scopo SSD-backed|



## **5. Data layout requirements** 

- Il converter deve garantire che ogni row/block PLE sia indirizzabile senza leggere/decomprimere l’intera tabella. 

- Preferire un PLE extent contiguo o pochi extents con metadata row_offset/row_bytes deterministici. 

- Se GGUF alignment/block quant rende una row non self-contained, definire l’unità minima di fetch come quant block/page e registrarla nel manifest. 

- Non cambiare il quant format solo per favorire I/O senza misurare l’impatto qualitativo. 

## **6. Cache hierarchy** 

- `L0: current microbatch row references       (transient)` 

- `L1: bounded decoded-row cache (optional, tiny)` 

- `L2: bounded quantized-page host RAM cache      (primary)` 

- `L3: OS page cache / file mapping               (if mmap mode)` 

- `L4: NVMe file` 

L1 è opzionale e piccolo perché decoded rows costano più memoria. L2 deve essere la cache principale. LRU/Clock/2Q sono candidati; la policy finale è scelta con trace replay, non per preferenza teorica. 

## **7. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M4S-C00|Freeze PLE file layout|docs/<br>ple_storage_layout.md|Documenta extents,<br>row/block bytes, alignment,<br>qtype.|complete|
|M4S-C01|Storage interface|q38_ple_storage.*|Abstract<br>read_rows/prefetch_rows;<br>resident backend remains<br>oracle.|same API|
|M4S-C02|mmap backend|q38_ple_storage_mmap.*|File-backed quant pages,<br>no eager touching.|cold lookup correct|
|M4S-C03|Bounded host cache|q38_ple_page_cache.*|Hard byte cap +<br>deterministic eviction<br>bookkeeping.|never exceeds cap|
|M4S-C04|Batched row planner|q38_ple_plan.*|IDs -> sort/dedup/coalesce<br>physical ranges while<br>preserving logical order.|same logical rows|
|M4S-C05|Async read path|q38_ple_async.*|Double-buffered/coalesced<br>fetch for next work unit.|cold/warm equivalence|
|M4S-C06|CUDA staging|q38_ple_cuda.cu|Decode fetched quant<br>blocks directly; bounded<br>staging buffers.|no persistent mirror|
|M4S-C07|Decode prefetch|q38_decode.c|Use known local token<br>history to issue next<br>required PLE reads as<br>early as semantics allow.|no token divergence|
|M4S-C08|Prefill pipeline|q38_prefill.c|Chunk N compute overlaps<br>N+1 PLE fetch;<br>configurable depth.|chunk invariance|
|M4S-C09|pread backend|q38_ple_storage_pread.*|Controlled alternative to<br>mmap for A/B.|correctness same|
|M4S-C10|I/O profiler|q38_ple_io_stats.*|bytes/read count/queue<br>wait/cache hit/page<br>faults/tail.|complete telemetry|
|M4S-C11|Cache policy sweep|scripts/<br>ple_cache_sweep.sh|Budgets/policies over fixed<br>trace.|best policy selected|
|M4S-C12|Q2/Q4 compatibility|tests|Both quant policies through<br>same SSDbackend.|pass|
|M4S-C13|Failure injection|tests|Short read, EIO, truncated<br>file, checksum mismatch ->|safe failure|



Target: GB10 / 128 GB coherent unified memory / CUDA only 

||||Qwen3.8-Fla|sh-Next /DGXSparkprototype|
|---|---|---|---|---|
|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
||||hard failure.||
|M4S-C14|Spark sustained test|scripts/ple_ssd_soak.sh|Long prefill+decode; cache<br>churn and repeated<br>sessions.|no leak/thrash|
|M4S-C15|M4S acceptance|artifacts/m4s|Resident vs SSD<br>equivalence, memory,<br>latency, soak.|100% pass|



## **8. Prefill access planning** 

```
for chunk tokens:
```

   `1. compute exact ngram IDs` 

   `2. map IDs -> physical quant blocks` 

   `3. build unique physical fetch set` 

   `4. coalesce adjacent ranges` 

   `5. async fetch missing ranges` 

   `6. retain logical ID -> cache slot mapping` 

   `7. execute PLE accumulation in original semantic order` 

- **Cruciale:** Deduplicare I/O non significa riordinare la matematica. Due token che puntano alla stessa row possono condividere <u>il fetch, ma l’ordine di accumulation/injection resta quello del reference.</u> 

## **9. Decode prefetch** 

Nel decode, una volta committed il token t, la local history necessaria per i successivi n-gram IDs è in parte nota. Il runtime deve calcolare e inviare il prefetch il prima possibile senza predire un token non ancora accettato. Con MTP, solo token committed possono aggiornare la semantic history; draft non committed non devono contaminare la cache semantica. 

## **10. Cache budget matrix** 

|**Budget host cache**|**Scopo**|
|---|---|
|1GiB|minimum functionalSSD mode|
|2 GiB|small-cache baseline|
|4GiB|balanced candidate|
|8 GiB|high-cache candidate if M8 headroom permits|
|>8 GiB|solo se trace datamostraforte beneficio enoncompromette Q4/M10|



I budget sono esperimenti, non valori raccomandati a priori. Il selected budget viene scelto da hit rate, wait time e peak memory reali. 

## **11. Test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M4S-T01|Physical row mapping|exact offsets/blocks|
|M4S-T02|Residentvsmmap cold|same decodedrows/hidden/logits|
|M4S-T03|Cold vs warm|same outputs|
|M4S-T04|Cachehard cap|neverexceeded|
|M4S-T05|Dedup/coalesce planner|same logical row sequence|
|M4S-T06|Chunk invariance|all M4partitiontests pass|
|M4S-T07|Q2 SSD path|pass|
|M4S-T08|Q4SSDpath|pass|
|M4S-T09|Async on/off|same outputs|
|M4S-T10|mmapvs pread|same outputs|
|M4S-T11|Fault injection|clear fatal error, no silent corruption|
|M4S-T12|Memory reduction|resident pressure materially lower than full-<br>resident baseline|
|M4S-T13|Sustainedworkload|no cache growth/thrashcatastrophe|
|M4S-T14|Tail latency report|p50/p95/p99 measured|



Target: GB10 / 128 GB coherent unified memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **12. Performance metrics** 

```
per request / per phase:
  ple_ids
  unique_physical_blocks
  cache_hits / misses
  nvme_read_ops
  nvme_bytes
  coalesced_bytes
  io_wait_us
  overlap_hidden_us
  decode_stall_us
  prefill_stall_us
  p50/p95/p99 lookup latency
  host_cache_bytes
  staging_bytes
```

## **13. Acceptance rule** 

M4S è accettata quando il path SSD riduce in modo sostanziale la residency PLE rispetto al resident baseline, mantiene correctness identica e ha una penalità di throughput/tail latency documentata e accettabile. Non è richiesto che SSD sia più veloce del full-resident path: il suo scopo principale è liberare memoria per il Q4 selettivo e la multimodalità. 

## **14. UNKNOWN** 

- Pattern reale di locality degli n-gram IDs su workload target. 

- Quanto il filesystem/page cache del DGX OS renda mmap migliore o peggiore di pread batched. 

- Bandwidth/latency NVMe effettiva dello specifico Spark e del drive installato. 

- Cache size ottimale per Q2 PLE e Q4 PLE. 

- Se io_uring produce beneficio sufficiente da giustificare complessità. 

## **15. Riferimenti** 

**Official Qwen3.8-Flash-Next config:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Transformers Qwen4-Exp model documentation:** https://huggingface.co/docs/transformers/model_doc/qwen4_exp 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Hugging Face Transformers vision utilities:** https://github.com/huggingface/transformers/blob/main/src/transformers/ vision_utils.py 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB coherent unified memory / CUDA only 

