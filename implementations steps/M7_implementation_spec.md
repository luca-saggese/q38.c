Qwen3.8-Flash-Next / DGX Spark prototype 

# M7 — Implementation Specification 

_Ottimizzazione specifica DGX Spark: residency, bandwidth, expert locality, prefill/decode scheduling e stabilità del runtime Q2._ 

**Metodo:** correctness, quantization error e performance sono tre assi separati. Nessuna ottimizzazione è accettata se rende più difficile localizzare una divergenza. 

## **1. Definition of Done** 

- Il runtime M6 corretto viene profilato e ottimizzato sul solo GB10, senza introdurre genericità per altre GPU. 

- Ogni ottimizzazione ha una baseline, una metrica primaria e una verifica numerica post-change. 

- Peak memory resta entro i gate di progetto; nessuna ottimizzazione crea transienti o mirror che riducano il margine necessario al futuro Q4 selettivo. 

- Decode e prefill hanno tuning separato: il primo è tipicamente bandwidth-sensitive, il secondo può essere più compute/scheduling-sensitive. 

- Expert locality, PLE residency e QSA state vengono misurati separatamente prima di scegliere cache o SSD streaming. 

- Il risultato M7 è una baseline Q2 stabile e misurata che rende possibile dimensionare M8 Q4 selettivo con dati reali. 

## **2. Principio: misurare prima di ottimizzare** 

Su DGX Spark, ds4 ha riportato decode al ~85-90% della banda memoria fisicamente misurata sul proprio workload, suggerendo che micro-ottimizzazioni decode possono rapidamente diventare inutili quando il limite è bandwidth. Questo non dimostra che Qwen3.8 abbia lo stesso comportamento, ma impone di classificare ogni fase come bandwidth-, compute-, launcho latency-bound prima di modificarla. 

**Regola:** Nessun 'kernel faster' entra nel branch principale senza mostrare beneficio sul benchmark GB10 e mantenere i <u>golden M6.</u> 

## **3. Metriche obbligatorie** 

|**Subsystem**|**Metriche**|
|---|---|
|Global memory|MemAvailable, RSS, mapped bytes, resident bytes, CUDA/unified peak,<br>page faults|
|Decode|<br>t/s, bytes/tokenstimati/misurati,kernel launchcount, time/token|
|Prefill|t/s, ubatch size, kernel time distribution, workspace peak|
|MoE|expert pairs, unique experts/layer, bytes touched, reuse distance, cache<br>hit rate|
|PLE|rows/token, uniquerows, cold/warm hitrate, bytes touched, prefetch wait|
|GDN|state read/write bytes, projection time, recurrence time|
|QSA|indexertime, top-ktime, gathertime, attentiontime, cache bytes|
|Allocator|alloc/free count, largest transient, fragmentation indicators|



## **4. Benchmark harness** 

```
./q38-bench --suite spark-q2 --json artifacts/m7/baseline.json
```

```
cases:
```

```
  S:   prompt 32,    decode 256
```

```
  M:   prompt 4096,  decode 256
```

```
  L:   prompt 16384, decode 256
  XL:  prompt 65536, decode 128
  REP: same M prompt x5
  UB:  fixed prompt, ubatch sweep
```

- Prompt corpus deterministico e versionato. 

- 

- 

- Warm-up separato dalle misure. 

- Minimo 5 run per caso; riportare mediana e dispersione. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

- Registrare temperatura/power mode se disponibili per ridurre benchmark noise. 

## **5. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M7-C00|Freeze M6 baseline|artifacts/m7/baseline_*|Copia commit, GGUF<br>checksum, platform state,<br>t/s,memory e golden.|baseline reproducibile|
|M7-C01|Profiler hooks|q38_profile.*;<br>q38_cuda_timing.cu|Timestamp per subsystem<br>e bytes accounting.|overhead debug controllato|
|M7-C02|Allocator telemetry|q38_memory.*|Classifica<br>persistent/transient/cache/<br>workspace; peak per<br>phase.|complete accounting|
|M7-C03|Decode classification|scripts/profile_decode.sh|Determina bandwidth vs<br>computevslaunch limit.|report|
|M7-C04|Prefill ubatch sweep|scripts/profile_prefill.sh|Sweep ubatch incl. tail<br>sizes,memory/time/error.|best safe region|
|M7-C05|MoE locality study|q38_moe_stats.*|Unique expert/reuse<br>distance per layer e token<br>window.|locality report|
|M7-C06|Expert cache experiment|q38_expert_cache.*|Bounded cache/residency<br>variants solo se locality lo<br>giustifica.|gain + memory cap|
|M7-C07|PLE residency tuning|q38_ple_residency.*|Mmap/cache/prefetch<br>matrix usando M4<br>subsystem.|best measured policy|
|M7-C08|QSA layout tuning|q38_qsa_cuda.cu|KV/index layout e gather<br>locality; no semantic<br>changes.|M5 exact IDs preserved|
|M7-C09|GDN launch reduction|q38_gdn_cuda.cu|Fuse only safe adjacent<br>stages; debug fallback<br>retained.|same golden + lower time|
|M7-C10|MoE decode specialization|q38_moe_decode.cu|Batch=1/small batch path<br>tuned to GB10.|gain measured|
|M7-C11|MoE prefill scheduling|q38_moe_dispatch.cu|Sort/group expert-token<br>pairs, workspace reuse,<br>safe padding.|ubatch fuzz still pass|
|M7-C12|Load/startup memory pass|q38_gguf.*; q38_memory.*|Elimina transienti, no<br>whole-file registration,<br>lazy/bounded residency.|startup peak gate|
|M7-C13|SSD decision ADR|docs/ADR-ssd.md|Decide YES/NO per<br>PLE/experts basandosi su<br>memory/perfdata.|signed decision|
|M7-C14|Long-context memory pass|tests/bench|1k→4k→16k→64k; target<br>superiore solo se utile.|no corruption/OOM|
|M7-C15|Regression sweep|make m0...m7-acceptance|Riesegue tutti i gate<br>numericie safety.|all pass|
|M7-C16|M7 acceptance|artifacts/m7/final_*|Confronto baseline/final,<br>memoryheadroomper M8.|accepted|



## **6. Memory budgets M7** 

|**Fase**|**Gate operativo**|
|---|---|
|Startup peak|≤ 112 GiB|
|Prefillpeak|≤ 116 GiB|
|Steady-state desiderato|≤ 108 GiB|
|Headroomobiettivo per M8 experiments|≥ 12GiB; preferibile>16 GiB|
|Whole-file cudaHostRegister|0|
|Persistentfulldequantmirror|0|
|Swap thrash|0|



Se M7 migliora throughput ma riduce headroom al punto da rendere impossibile il Q4 selettivo, non è un successo di progetto. 

## **7. Decode optimization order** 

1. Misurare bytes/token e bandwidth effettiva. 

2. Eliminare copie e letture duplicate. 

3. Ridurre kernel launches solo se launch overhead è significativo. 

4. Ottimizzare routed expert read/dequant locality. 

5. Ottimizzare GDN recurrence memory accesses. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

6. Ottimizzare QSA solo se compare nel top timing breakdown. 

7. Fermarsi quando il bottleneck è chiaramente bandwidth e i cambi non producono guadagno robusto. 

**Contesto:** Nel workload ds4 su GB10 è stato riportato decode vicino all'85-90% della banda misurata. Per Qwen questo è <u>una ipotesi da verificare, non un target da copiare.</u> 

## **8. Prefill optimization order** 

8. Ubatch sweep con correctness/safety già M6. 

9. Ridurre padding/tail waste dei quant kernels. 

10. Batch expert-token pairs e minimizzare scatter/gather. 

11. Overlap PLE lookup/prefetch solo se M4 profiling lo giustifica. 

12. Workspace reuse e stream scheduling. 

13. Fusioni solo dopo aver preservato probe intermedi in build debug. 

## **9. Expert locality e selective residency** 

Con 512 routed experts e top-10, l'efficacia di una cache dipende dalla distribuzione reale del routing. M7 non assume che gli expert siano uniformemente visitati né che esista una hot set stabile. 

```
per layer and rolling window:
  routed_pairs
  unique_experts
  top-N expert frequency
  reuse_distance histogram
  bytes_if_no_cache
  bytes_if_cache(K)
  observed hit_rate(K)
```

|**Decisione**|**Condizione**|
|---|---|
|No expert cache|reuse basso o memoria troppo costosa|
|<br>Smallbounded cache|<br>hot set stabile e gain misurato|
|Full routed residency|solo se già naturale nel Q2 mapping e non riduce headroom|
|SSDexpert streaming|solo se Q4 memory plan lorichiederà;nonanticiparlo senza dati|



## **10. PLE residency decision** 

- Confrontare cold/warm performance e page-fault behavior. 

- Se mmap/unified memory è sufficiente, non introdurre un I/O scheduler custom. 

- Se prefetch aiuta solo su prefill ma peggiora decode/peak memory, abilitarlo per-mode. 

- Cache PLE e cache experts restano indipendenti. 

## **11. SSD ADR** 

```
ADR-ssd.md must answer separately:
```

```
PLE:
  resident/mmap/cache/SSD?
  measured memory saved?
  measured t/s impact?
```

```
Routed experts:
  resident/mmap/cache/SSD?
  Q2 need?
  projected Q4 need?
```

```
Decision:
  ENABLE / DO NOT ENABLE
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  with benchmark evidence
```

SSD non è un requisito del prototipo. È uno strumento da attivare solo se migliora il trade-off memoria/prestazioni, soprattutto in vista del Q4. 

## **12. Long-context staging** 

|**Stage**|**Prompt**|**Output**|**Gate**|
|---|---|---|---|
|LC1|1k|128|baseline state|
|LC2|4k|128|no growthbug|
|LC3|16k|128|memory slope measured|
|LC4|64k|128|<br>no corruption/OOM|
|LC5|>64k fino a target utile|64|solo se tempo/memoria lo<br>consentono|



Non è necessario raggiungere 262k in M7 per chiudere la milestone se il costo di test è sproporzionato; è però obbligatorio modellare la crescita della memoria e dimostrare che non esiste crescita superlineare inattesa nei componenti che dovrebbero essere bounded/linear. 

## **13. Regression discipline** 

- Ogni commit performance deve rieseguire almeno i golden del subsystem modificato. 

- Ogni 3 commit performance: rieseguire M6 end-to-end + ubatch fuzz subset. 

- Prima di M7 acceptance: rieseguire acceptance M0..M6 completa. 

- Le build optimized conservano un `Q38_DEBUG_REFERENCE` mode con kernel/reference più semplici per diagnosi. 

## **14. M7 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M7-T01|Baseline reproducibility|median t/s and memory within expected noise<br>band|
|M7-T02|Profiler accounting|major phases sum plausibly to wall time|
|M7-T03|Allocatoraccounting|persistent/transient totals complete|
|M7-T04|Decode bottleneck classification|documented with data|
|M7-T05|Prefillubatchsweep|safe optimum identified;M6fuzzstillpasses|
|M7-T06|Expert cache experiment|bounded memory; enable only if gain|
|M7-T07|PLEpolicy|cold/warm/prefetchcompared|
|M7-T08|QSA optimization|selected IDs exact vs M5|
|M7-T09|GDNoptimization|M3 state/output goldenpreserved|
|M7-T10|MoE optimized path|M6 router/expert/ubatch goldens preserved|
|M7-T11|Startupmemory|≤112GiB|
|M7-T12|Prefill peak|≤116 GiB|
|M7-T13|No swap thrash|pass|
|M7-T14|Long-context staged|through agreed stage, no corruption/OOM|
|M7-T15|Full regression M0-M6|100% pass|
|M7-T16|Performance delta|final report shows per-case gain/loss|



## **15. Acceptance artifacts** 

```
artifacts/m7/
  baseline_platform.json
  baseline_memory.json
  baseline_bench.json
  decode_profile.json
  prefill_ubatch_matrix.json
  moe_locality.json
  expert_cache_matrix.json
  ple_residency_matrix.json
  qsa_profile.json
  gdn_profile.json
  startup_memory.json
  long_context.json
  ADR-ssd.md
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  final_memory.json
  final_bench.json
  regression.txt
  acceptance.txt
```

## **16. Stop conditions** 

- Se un'ottimizzazione richiede >4 GiB di nuovo persistent memory per un guadagno marginale, deve essere giustificata esplicitamente o respinta. 

- Se cambia selected expert IDs o QSA IDs, è correctness regression. 

- Se il throughput migliora ma aumenta startup/prefill peak oltre i gate, revert o redesign. 

- Se profiling dimostra bandwidth saturation, non continuare a fondere kernel senza un modello di bytes/token. 

- Se SSD introduce thrashing o tail latency elevata, disabilitarlo: non è un obiettivo in sé. 

## **17. UNKNOWN** 

- Bottleneck dominante reale Qwen3.8 Q2 sul GB10. 

- Distribuzione/hotness reale dei 512 routed experts. 

- Beneficio reale del prefetch PLE con memoria coerente unificata. 

- Necessità effettiva di SSD prima del passaggio a Q4. 

- Throughput target finale: deve essere derivato dalla baseline M6/M7, non inventato. 

## **18. Riferimenti** 

**Qwen3.8-Flash-Next config.json:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**llama.cpp Qwen3.8/qwen4exp support:** https://github.com/ggml-org/llama.cpp 

**llama.cpp issue #27792 - MoE mul_mat_id ubatch tail OOB:** https://github.com/ggml-org/llama.cpp/issues/27792 

**llama.cpp issue #27763 - qwen4exp GPU layer corruption on Blackwell SM110:** https://github.com/ggml-org/llama.cpp/issues/27763 

**llama.cpp issue #27797 - qwen4exp multi-segment regression:** https://github.com/ggml-org/llama.cpp/issues/27797 

**ds4 issue #773 - GB10 decode near measured memory bandwidth:** https://github.com/antirez/ds4/issues/773 

**ds4 issue #293 - whole GGUF CUDA registration OOM on Spark:** https://github.com/antirez/ds4/issues/293 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

