Qwen3.8-Flash-Next / DGX Spark prototype 

# M4 — Implementation Specification 

_PLE / n-gram embedding: semantic correctness, Q2 bootstrap residency, chunk invariance e memory strategy sul DGX Spark._ 

**Regola metodologica:** ogni dettaglio non verificato dal checkpoint/config/reference implementation rimane UNKNOWN. Le ottimizzazioni vengono introdotte solo dopo un oracle di correttezza. 

## **1. Definition of Done** 

- La PLE/n-gram del modello text-only viene interpretata con formula di indexing/hash e ordine di aggregazione verificati dal reference. 

- Il layer di iniezione PLE è identificato e validato esattamente; nessun posizionamento viene dedotto dal solo nome del tensor. 

- La stessa token stream produce gli stessi PLE IDs, hidden intermedi e logits indipendentemente dal partizionamento del prefill. 

- PLE ha un subsystem separato da routed experts e normali weight matrices: storage, cache, quantizzazione e profiling sono indipendenti. 

- Il formato Q2 bootstrap della PLE rientra nel memory plan M1 oppure viene servito con residency/paging controllato senza creare copie persistenti. 

- Cold-cache e warm-cache producono output numericamente equivalenti. 

- Ogni errore CUDA fatale durante lookup/prefill termina il processo di test; non si continua con un context CUDA avvelenato. 

## **2. Parametri già verificabili dal config** 

|**Parametro**|**Valore**|
|---|---|
|ngram_size|3|
|ngram_vocab_size_base|20,000,000|
|heads_per_ngram|8|
|make_ngram_vocab_size_divisible_by|128|
|PLE layer IDs|[2]|
|<br>PLEembedding dim|2560|
|PLE convolution kernel|4|
|Texthiddensize|2560|



**NON dedurre:** Il config non basta per stabilire formula hash, mapping unigram/bigram/trigram, ordine delle heads, packing fisico della tabella, o posizione esatta dell'iniezione rispetto a norm/GR/GDN. Questi elementi devono essere estratti dal <u>reference implementation congelato.</u> 

## **3. Design del subsystem PLE** 

```
typedef struct {
    q38_tensor_ref tensor;       /* file-backed GGUF tensor/table */
    q38_quant_type qtype;
    uint64_t rows;
    uint32_t row_width;
    uint32_t row_bytes;
```

```
    q38_ple_cache cache;
    q38_ple_stats stats;
} q38_ple_store;
```

```
typedef struct {
    uint32_t prev_token_1;
    uint32_t prev_token_2;
    bool have_prev_1;
    bool have_prev_2;
} q38_ngram_history;
```

- Il `q38_ple_store` non espone una GEMM API. Espone lookup/prefetch/dequant-row. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

- La history token è session state, non cache. Deve sopravvivere ai confini di ubatch/prefill chunk. 

- La cache PLE è ricostruibile e non fa parte della semantica del modello. 

- Il loader non materializza l'intera PLE in FP16/BF16. 

## **4. Fasi di correttezza** 

1. Congelare il reference Qwen4Exp usato per PLE e documentare funzioni/tensor names rilevanti. 

2. Implementare `q38_ngram_ids_ref()` in scalar C, usando esclusivamente la formula verificata. 

3. Produrre golden IDs per sequenze sintetiche che includano BOS, token ripetuti, token massimi e chunk boundary. 

4. Implementare row addressing e dequantizzazione scalar della PLE Q2. 

5. Implementare CUDA lookup semplice, senza cache e senza prefetch. 

6. Validare il punto esatto di injection al layer 2 con hidden dumps reference. 

7. Solo dopo introdurre cache, deduplica e prefetch. 

## **5. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M4-C00|Freeze PLE semantics|docs/<br>qwen_ple_semantics.md|Documenta formula<br>hash/index, n-gram<br>composition, head<br>aggregation, injection point<br>e tensor layout dal<br>reference congelato.|documento completo;<br>UNKNOWN residui marcati|
|M4-C01|Token history|q38_session.h/.c|Aggiunge history minima<br>per ngram_size=3 e<br>reset/append semantics.|unit test boundary|
|M4-C02|Scalar n-gram oracle|q38_ple_ref.c/.h|Implementa IDs/hash<br>scalar verificati; nessuna<br>CUDA.|golden IDs pass|
|M4-C03|PLE tensor descriptor|q38_ple.h/.c|Bind dedicato, row<br>geometry, quant type, file<br>offset; separato da<br>q38_weights generic.|strict bind pass|
|M4-C04|Scalar row decoder|q38_ple_ref.c|<br>Dequant row Q2 con<br>oracle; supporta anche<br>candidato Q4 per future-<br>proof.|Q2/Q4 row tests|
|M4-C05|CUDA naive lookup|q38_ple_cuda.cu|Lookup per ID, decode row<br>e accumulo/injection senza<br>cache/prefetch.|CUDA vs scalar|
|M4-C06|Injection validation|q38_forward_probe.c|Esegue fino a layer 2 con<br>dump prima/dopoPLE.|hidden golden pass|
|M4-C07|Chunk invariance harness|tests/test_ple_chunking.*|Partizioni<br>deterministiche/random<br>della stessa token stream.|IDs/hidden/logits invariant|
|M4-C08|PLE cache v1|q38_ple_cache.c/.h|Cache bounded e<br>deterministic; key=row ID,<br>value=dequant o quant<br>block secondo memory<br>plan.|cold==warm|
|M4-C09|Prefill batching|q38_ple_cuda.cu|Batch IDs, dedupe solo<br>degli accessi storage;<br>preserva<br>semantics/accumulation<br>order.|same output; fewer<br>bytes/lookup|
|M4-C10|Unified-memory residency<br>experiment|q38_ple_residency.c;<br>scripts/ple_memory_matrix<br>.sh|Confronta mmap file-<br>backed, bounded host<br>cache e bounded CUDA-<br>visible staging. Vietata full<br>dequant.|memory/perf report|
|M4-C11|Async prefetch optional|q38_ple_prefetch.c/.cu|Prefetch del prossimo<br>chunk solo se profiling<br>mostra beneficio; feature<br>flag off di default fino a<br>pass.|correctness + measurable<br>benefit|
|M4-C12|Failure handling|q38_cuda.cu; tests|CUDA fatal -> immediate<br>process/test failure;<br>nessuna continuation.|fault injection pass|
|M4-C13|M4 acceptance|tests/; artifacts/m4|Automatizza semantics,<br>chunking, cache<br>equivalence, memory and<br>CUDA health.|100% pass|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **6. Hash/index test corpus** 

|**Caso**|**Scopo**|
|---|---|
|[BOS]|inizializzazione history|
|[a,b,c]|prima occorrenza completa trigram|
|[a,a,a,a]|collisioni/repetition behavior|
|token IDs 0,1,vocab-1|boundary arithmetic|
|chunk [a]+[b,c]|history across chunk|
|chunk[a,b]+[c]|second boundaryform|
|1-token chunks|massimo stress sul state continuity|
|randomstream 1ktoken|property test controreference|



## **7. Chunk invariance: protocollo obbligatorio** 

```
tokens = fixed_stream(N)
```

```
partitions:
  [N]
  [1,1,1,...]
  [2,N-2]
  [3,5,N-8]
  [4,4,4,...]
  random(seed=1)
  random(seed=2)
compare after every chunk:
  ngram IDs
  PLE selected rows
  PLE accumulated vector
  hidden before/after layer-2 injection
  downstream probe logits
```

**Gate forte:** Una differenza dovuta al boundary è un bug anche se il testo generato sembra plausibile. Un issue qwen4exp corrente in llama.cpp mostra regressioni specifiche con prompt multi-segmento; per questo M4 tratta chunk invariance come <u>proprietà primaria, non come smoke test.</u> 

## **8. Quantizzazione PLE: strategia M4** 

- Q2 è ammesso per bootstrap, ma la PLE mantiene una regola distinta nel quant manifest. 

- Il decoder deve supportare già un candidato Q4 per M8, come fatto per gli experts. 

- La scelta Q2 PLE viene accettata solo se non forza una copia dequant persistente che annulla il vantaggio memoria. 

- L'errore PLE va misurato separatamente da routed expert quantization usando hidden del layer 2 e logits downstream. 

|**Esperimento**|**Cosa cambia**|**Misura**|
|---|---|---|
|P0|PLE BF16/referencerows|oracle semantico|
|P1|PLE Q2|row reconstruction error+hidden drift|
|P2|PLEQ4|future quality baseline|
|P3|Q2 cold cache vs Q2 warm cache|deve essere numericamente identico|



## **9. Residency e memoria su GB10** 

Il DGX Spark usa memoria coerente unificata, ma questo non rende gratuiti mapping, page residency o transienti. Gli issue ds4 sul GB10 mostrano OOM durante startup e problemi dovuti a host registration/cache. M4 quindi misura residency invece di assumere che mmap o unified memory siano automaticamente ottimali. 

|**Modalità**|**Uso M4**|**Regola**|
|---|---|---|
|Filemmap quantized|baseline|consentito;nessun whole-filehostregister|
|Bounded quant row cache|baseline candidata|budget esplicito e statistiche hit/miss|
|<br>Bounded dequantrowcache|solo test|<br>budget piccolo;mai whole-tablemirror|
|cudaHostRegister whole PLE|VIETATO|rischio memory pressure/OOM|
|<br>cudaMalloc/cudaMallocManagedwholePLE|VIETATOM4|<br>nessun fullduplication/materialization|
|SSD explicit prefetch|opzionale|solo se profiling dimostra bisogno/beneficio|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **10. Profiler PLE** 

```
q38_ple_stats:
  lookup_count
  unique_row_count
  cache_hits
  cache_misses
  quant_bytes_touched
  bytes_prefetched
  bytes_dequantized
  lookup_us
  dequant_us
  prefetch_wait_us
  peak_cache_bytes
  page_fault counters where available
```

## **11. M4 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M4-T01|PLE semantics doc complete|formula/index/layout/injection mapped to<br>reference|
|M4-T02|N-gram IDs golden|exact integer match|
|M4-T03|Historyreset/append|exact state transitions|
|M4-T04|Q2 row decode|within calibrated quant error|
|M4-T05|Q4 rowdecodefuture-proof|candidateworksnow|
|M4-T06|CUDA lookup vs scalar|within quant/numeric tolerance|
|M4-T07|Layer-2 injection|reference hidden match within expected Q2<br>error|
|M4-T08|Chunk invarianceIDs|exact|
|M4-T09|Chunk invariance hidden/logits|within numeric tolerance|
|M4-T10|Cold/warmequivalence|same outputs|
|M4-T11|Cache bound|never exceeds configured bytes|
|M4-T12|No persistentfulldequant|allocationaudit pass|
|M4-T13|Memory gate|runtime remains within project limits|
|M4-T14|CUDA fatal handling|fault terminates test/process cleanly|



## **12. Acceptance artifacts** 

```
artifacts/m4/
  qwen_ple_semantics.md
  ple_golden_ids.json
  ple_row_quant_tests.json
  ple_injection_golden.json
  ple_chunk_invariance.json
  ple_cache_stats_cold.json
  ple_cache_stats_warm.json
  ple_memory_matrix.json
  ple_prefetch_benchmark.json
  cuda_health.txt
  acceptance.txt
```

## **13. Stop conditions** 

- Se hash/indexing non è verificato dal reference: non implementare CUDA. 

- Se la cache cambia l'ordine matematico e altera hidden oltre tolleranza: rimuovere l'ottimizzazione. 

- Se il prefetch aumenta peak memory senza beneficio misurabile: non abilitarlo. 

- Se la PLE Q2 introduce una perdita troppo elevata ma il runtime è corretto: non confonderla con un bug; documentare e usare Q4 PLE temporaneamente. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **14. UNKNOWN** 

- Formula hash e mapping fisico esatti del checkpoint/reference congelato. 

- Packing effettivo dei 128 split e relationship con heads_per_ngram. 

- Ordine esatto di aggregation/injection rispetto a GR/norm/GDN. 

- Migliore policy di residency su GB10: deve emergere dal memory matrix. 

- Q2 type ottimale della PLE: non definito a priori. 

## **15. Riferimenti** 

**Qwen3.8-Flash-Next config.json:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**llama.cpp qwen4exp support / source tree:** https://github.com/ggml-org/llama.cpp 

**llama.cpp qwen4exp multi-segment issue #27797:** https://github.com/ggml-org/llama.cpp/issues/27797 **llama.cpp qwen4exp CUDA issue #27763:** https://github.com/ggml-org/llama.cpp/issues/27763 

**ds4 DGX Spark full GGUF registration issue #293:** https://github.com/antirez/ds4/issues/293 

**ds4 DGX Spark OOM regression #585:** https://github.com/antirez/ds4/issues/585 

**ds4 DGX Spark fatal CUDA context issue #759:** https://github.com/antirez/ds4/issues/759 **ds4 discrete/unified registration analysis #791:** https://github.com/antirez/ds4/issues/791 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

