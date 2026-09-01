Qwen3.8-Flash-Next / DGX Spark prototype 

# M3 — Implementation Specification 

_Gated Residual + Gated DeltaNet: correctness-first su CUDA/GB10, state FP32, Q2 weights, chunk invariance obbligatoria._ 

**Regola metodologica:** reference-first, nessun guessing. Ogni divergenza va attribuita a tokenizer, quantizzazione, binding, kernel o stato prima di procedere. 

## **1. Scopo e Definition of Done** 

- Implementare esattamente il Gated Residual (GR) di Qwen3.8 e il path Gated DeltaNet (GDN) usato dai layer indicati dal `layer_types` congelato. 

- Portare in CUDA sul GB10 il forward GDN per prefill corto e decode token-by-token senza dipendere da un backend CPU completo. 

- Mantenere lo state ricorrente nel dtype richiesto dal config/reference: FP32 finché una milestone successiva dimostri la sicurezza di altro. 

- Ottenere equivalenza quant-reference↔q38 su intermedi GR/GDN entro tolleranze calibrate. 

- Dimostrare chunk invariance: stessa token stream, partizionamenti di prefill diversi, stesso state/logits entro tolleranza. 

- Misurare e documentare persistent state bytes/layer/session e workspace bytes; nessuna stima non verificata. 

- Non implementare ancora QSA, PLE, MoE completo o generazione end-to-end: M3 deve isolare GR/GDN. 

## **2. Cosa sappiamo e cosa no** 

|**Elemento**|**Stato**|**Conseguenza**|
|---|---|---|
|GR 4 branches, lowrank 320|Verificato dal config/descrizione Qwen|Dimensionare metadata e test; formula ancora<br>dareference code/report|
|GDN key heads 16, value heads 48, dims<br>128/128, conv kernel 4|Verificato dal config|Shape validator fisso|
|GDN recurrent state dtype FP32|Verificato dal config|State allocation FP32|
|GDN recurrence generica decay/delta/outer-<br>product/attention|Confermata dal kernel reference llama.cpp|Usabile come oracle concettuale, non prova del<br>packing/projections Qwen|
|Exact Qwen gating/projection order|UNKNOWN finché non congelato reference<br>graph|Blocking prima di M3-C02|
|Exact state packing /head grouping|UNKNOWN|Blocking prima delCUDA layoutfinale|
|**No guessing:**Non codificare `state = h<br>dall’implementazione reference e poi in|eads × dk × dv` o grouping 48↔16 per intui<br>serire assert statici/runtime.|zione. Derivare le dimensioni|



## **3. Oracle semantico della recurrence** 

Il kernel F32 corrente di llama.cpp descrive la recurrence GDN in quattro passi: decay dello state, calcolo del delta rispetto al valore predetto dallo state, update outer-product, lettura attention tramite q. Questo è utile per costruire test micro-kernel. Il graph Qwen3.8 deve comunque stabilire come vengono prodotti q/k/v, gate/decay/beta, scale, head grouping e convolutional preprocessing. 

## **4. Architettura software M3** 

```
q38_gr.h/.cu
  q38_gr_read(...)
  q38_gr_write(...)
  q38_gr_collapse(...)
```

```
q38_gdn.h/.cu
  q38_gdn_project(...)
  q38_gdn_conv_update(...)
  q38_gdn_recurrence(...)
  q38_gdn_output(...)
```

```
q38_state.h/.c
  q38_gdn_state_desc
  q38_session_state
```

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
q38_gdn_ref.c
  test-only scalar/F32 oracle for recurrence
```

## **5. Stato persistente: regole** 

- Separare `persistent recurrent state`, `conv history`, GR forward activation
  workspace, e generic workspace. Il GR non è stato semantico persistente;
  l'accounting deve escluderlo da `persistent_bytes`.

- Lo state non deve vivere dentro scratch buffers riciclabili tra layer. 

- Reset deve inizializzare esattamente come il reference, non semplicemente memset(0) se il reference richiede altro. 

- Snapshot/rewind non viene implementato in M3. Tuttavia lo state struct deve essere serializzabile senza pointer interni opachi per non precluderlo. 

- Una sessione M3 è single-sequence. Batching multi-sequence è fuori scope finché la recurrence single-sequence è corretta. 

## **6. Commit plan M3** 

|**Commit**|**Scopo**|**File/area**|**Modifiche**|**Gate**|
|---|---|---|---|---|
|M3-C00|Freeze equations|docs/<br>qwen_gdn_semantics.md|Trascrive<br>formule/ordine/layout dal<br>reference con link+revision|review checklist completa|
|M3-C01|GR weight binding|q38_weights.*|Aggiunge campi reali GR<br>dopo mapping verificato|strict bind|
|M3-C02|GR scalar oracle|q38_gr_ref.c|read/write/collapse F32<br>test-only|golden tiny vectors|
|M3-C03|GR CUDA baseline|q38_gr.cu|kernel non fused, FP32<br>accum dove richiesto|CUDA vs oracle|
|M3-C04|GDN weight binding|q38_weights.*|Projection/conv/gate<br>tensors reali, exact shapes|strict bind|
|M3-C05|State descriptor|q38_state.*|Calcola bytes da verified<br>shapes; alloc/reset|state report exact|
|M3-C06|GDN recurrence oracle|q38_gdn_ref.c|decay/delta/update/read;<br>no optimization|micro tests|
|M3-C07|Projection + conv CUDA|q38_gdn.cu|Q2/Q8/BF16 dispatch<br>secondo manifest|intermediate golden|
|M3-C08|Recurrence CUDA<br>baseline|q38_gdn.cu|state FP32, layout leggibile<br>prima di tuning|token-step compare|
|M3-C09|Layer 0 probe|q38_forward_probe.c|GR + GDN su primo layer<br>GDN|hidden/state golden|
|M3-C10|Multi-GDN probe|forward probe|3 GDN consecutivi o<br>sequenza reale dal frozen<br>pattern|golden pass|
|M3-C11|Chunk invariance harness|tests/m3/<br>chunk_invariance.c|stessa stream con<br>partizioni molteplici|state/output pass|
|M3-C12|Spark profiling baseline|Nsight/telemetry hooks|kernel/bytes/launch count;<br>nessuna tuning ancora|report prodotto|
|M3-C13|Safe local fusion|CUDA|solo fusioni dimostrate<br>equivalenti|accuracy unchanged|
|M3-C14|M3 acceptance|make m3-acceptance|archive artifacts|100% pass|



## **7. qwen_gdn_semantics.md: contenuto obbligatorio** 

- Nomi tensor reali e mapping GGUF. 

- Equazioni complete in ordine di esecuzione. 

- Shapes logiche e physical layout per token/head/dimension. 

- Mapping tra 16 key heads e 48 value heads. 

- Semantica e posizione della convolution kernel=4. 

- Definizione esatta di gate decay e beta/update gate. 

- Scale attention e dtype di ogni accumulo/state. 

- Initial state e reset semantics. 

- Differenze tra prefill e decode se esistono. 

- Reference revision e line/function anchors sufficienti a ritrovare il codice. 

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **8. Micro-test GDN** 

|**Test**|**Input**|**Controlla**|
|---|---|---|
|Zero state|1 token, state zero|prima update/read|
|Zero beta/update|non-zero state|nessun delta update se semanticamente<br>previsto|
|Strong decay|gatenoto|decay applicato prima dell’update|
|Prediction exact|v==S·k|delta circa zero|
|Single basisk/q|basisvectors|orientamento/layout state|
|Two timesteps|t0,t1 controllati|ordine di update/read|
|Convboundary|1..5 token|historykernel=4|
|Head mapping|pattern diversi per head|16→48 grouping corretto|



## **9. Chunk invariance suite** 

È il gate più importante di M3 perché un GDN può produrre output plausibili pur aggiornando male lo state ai confini dei microbatch. 

```
Token stream length N:
  [N]
  [1] * N
  [2, N-2]
  [3, 5, N-8]
  [4, 4, ...]
  [127, 1, 128, ...]
  deterministic random partitions
```

```
Compare after every boundary:
  conv history
  recurrent state selected slices/checksum
  GR state
  layer output
```

- Partizioni devono includere boundary < conv kernel, ==4 e >4. 

- Confrontare state subito dopo ogni chunk, non solo logits finali. 

- Se l’errore cresce monotonamente con i token, fermarsi: potrebbe essere dtype/order/state update, non “normale Q2”. 

## **10. Layout CUDA su GB10** 

Prima versione: layout esplicito e semplice, scelto per corrispondere al reference. Solo dopo correctness si valuta transpose/fusion. La documentazione di llama.cpp mostra che il layout dello state GDN può essere determinante per cache/coalescing; issue #20436 discute regressioni e differenze di access pattern. Questo giustifica profilare, non copiare automaticamente un layout. 

|**Fase**|**Scelta**|
|---|---|
|Baseline|state FP32; no compressed state|
|Projection weights|<br>Q2 routed/core policy M1 non cambia; GDN core tipicamente Q8/BF16<br>secondo manifest|
|Workspace|preallocato/arena;misurato|
|Decode|priorità a basso launch count dopo baseline|
|Prefill|<br>kernelseparato consentito|
|Fusion|solo dopo layer-level equivalence|



## **11. Performance counters M3** 

- State bytes per GDN layer e totale per sessione. 

- Conv history bytes. 

- GR state/workspace bytes. 

- Bytes letti dai pesi per token/layer. 

- Kernel launches per token/layer. 

- Tempo projection, conv, recurrence, output projection separati. 

- Effective bandwidth del recurrence kernel e occupancy solo come diagnostica, non come KPI isolato. 

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **12. Test matrix M3** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M3-T01|Equation/reference audit|0 UNKNOWN bloccanti nel path implementato|
|M3-T02|GRscalar vectors|goldenexact/within F32tolerance|
|M3-T03|GR CUDA|oracle agreement|
|M3-T04|State byte accounting|calcolo==allocazioni misurate|
|M3-T05|GDN recurrence scalar|micro cases pass|
|M3-T06|Projection/convCUDA|intermediate goldenpass|
|M3-T07|GDN one-token CUDA|state/output pass|
|M3-T08|GDN 2/4/5-token|conv/state boundary pass|
|M3-T09|Layer 0 complete|quant-reference agreement|
|M3-T10|Consecutive GDN layers|intermediate agreement|
|M3-T11|Chunk invariance fixed partitions|pass at every boundary|
|M3-T12|Chunk invariancerandom 100 partitions|no unexplained divergence|
|M3-T13|Reset determinism|fresh==reset session|
|M3-T14|Longrecurrence smoke|noNaN/Inf/unbounded unexpected drift|
|M3-T15|CUDA error checks|0 illegal access/race in supported tooling|
|M3-T16|Memory gate|noregressionoverallowed budget|
|M3-T17|Q2 vs quant-reference|implementation error<<quantization error|



## **13. Acceptance artifacts** 

```
artifacts/m3/
  qwen_gdn_semantics.md
  gr_golden.json
  gdn_microtests.json
  layer0_intermediates.json
  multigdn_intermediates.json
  state_layout.json
  state_memory.json
  chunk_invariance.json
  cuda_profile_baseline.json
  cuda_profile_fused.json   # solo se M3-C13 eseguito
  memory.json
  checksums.txt
  acceptance.txt
```

## **14. Failure diagnosis order** 

1. Tokenizer/tokens: devono già essere esatti da M2. 

2. Weight binding e dequant: ricontrollare tensor/shape/quant block. 

3. GR read input e read gate. 

4. Projection outputs q/k/v/gate/beta e conv output. 

5. State prima del decay. 

6. State dopo decay. 

7. Delta/prediction. 

8. State dopo update. 

9. Attention/read output. 

10. GR write/output. 

11. Solo dopo considerare precisione/quantizzazione come causa. 

## **15. Cose deliberatamente escluse da M3** 

- QSA: M5. 

- PLE injection: M4. 

- MoE end-to-end: M6; eventuali dense/probe placeholders possono alimentare hidden golden. 

- Snapshot/restore e rewind: hardening successivo; issue llama.cpp #22384 mostra perché recurrent checkpoint semantics meritano una milestone propria. 

- Multi-sequence batching, TP, distributed, MTP e vision. 

- State FP16/BF16/FP8: nessuna riduzione prima di long-context ablation. 

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **16. UNKNOWN / blocking questions** 

- Formula completa del GR Qwen3.8 e parametro esatto dei read/write gates. 

- Projection/gating order esatto attorno al GDN e alla convolution. 

- Packing fisico dello state e mapping 16 key heads ↔ 48 value heads. 

- Eventuali differenze Qwen3.8 rispetto ai GDN delle precedenti famiglie Qwen3.x. 

- Shape precise di ogni tensor GDN/GR nel GGUF congelato. 

## **17. Riferimenti** 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next config:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**llama.cpp gated_delta_net_f32 oracle:** https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-et/et-kernels/src/ gated_delta_net_f32.c 

**llama.cpp recurrent checkpoint issue #22384:** https://github.com/ggml-org/llama.cpp/issues/22384 

**llama.cpp GDN cache-layout performance issue #20436:** https://github.com/ggml-org/llama.cpp/issues/20436 

**ds4 core donor:** https://github.com/antirez/ds4/blob/main/ds4.c 

**ds4 CUDA donor:** https://github.com/antirez/ds4/blob/main/ds4_cuda.cu 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 
