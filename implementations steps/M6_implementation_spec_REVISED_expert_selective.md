Qwen3.8-Flash-Next / DGX Spark prototype 

# M6 — Implementation Specification 

_MoE completo, shared expert, routing top-10 e primo forward end-to-end text-only Q2 su DGX Spark._ 

**Metodo:** correctness, quantization error e performance sono tre assi separati. Nessuna ottimizzazione è accettata se rende più difficile localizzare una divergenza. 

## **1. Definition of Done** 

- Tutti i 48 layer eseguono il forward text-only completo con GDN/QSA/GR/PLE già validati e MoE Qwen-specific. 

- Il router produce top-10 expert IDs compatibili con il reference e weights entro la tolleranza numerica definita. 

- Routed experts, shared expert e router restano domini distinti sia nel binder sia nella quantizzazione. 

- Il Q2 bootstrap usa quantizzazione aggressiva principalmente sui routed experts; nessuna cache persistente dequantizzata dell'intero expert set. 

- Prefill e decode hanno percorsi funzionali separati ma semanticamente equivalenti. 

- Il runtime genera una continuazione completa su DGX Spark senza OOM, illegal access o contaminazione di session state. 

- La divergenza runtime vs reference quant/dequant è localizzabile al singolo layer e, per il MoE, al singolo expert/token pair. 

## **2. Parametri MoE fissati dal checkpoint/config** 

|**Parametro**<br>**Valore operativo**|
|---|
|Routed experts<br>512|
|Experts selected/token<br>10|
|MoE intermediate size<br>640|
|Shared expertintermediate size<br>640|
|Hidden size<br>2560|
|Layers<br>48|
|**Da non assumere:**La normalizzazione esatta dei router weights, eventuali bias/scaling, ordine gate/up/down, activation e<br>shared-expert scaling devono essere verificati nel reference congelato. Il config da solo non è sufficiente.|



## **3. Architettura del MoE runtime** 

```
hidden
```

   - `-> router projection / scores` 

   - `-> top-10 IDs + weights` 

   - `-> routed expert dispatch` 

   - `-> gate/up` 

   - `-> activation` 

   - `-> down` 

   - `-> weighted combine` 

   - `-> shared expert` 

   - `-> exact Qwen combine/scaling` 

   - `-> GR write` 

- Il router non deve dipendere dal quant format degli expert. 

- Gli expert IDs sono interi semanticamente osservabili: in debug devono essere dumpabili per token/layer. 

- Il shared expert non viene inserito nello stesso array dei 512 routed expert: struttura separata. 

- Le matrici routed restano file-backed/quantized; dequantizzazione per tile/operation. 

## **4. Strutture proposte** 

```
typedef struct {
    q38_tensor *router;
```

- `q38_tensor *router_bias;      /* NULL se il reference non lo usa */ q38_expert_bank routed;` 

```
    q38_shared_expert shared;
```

- `} q38_moe_weights;` 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
typedef struct {
    uint16_t expert_id[10];
    float    weight[10];
} q38_route10;
```

```
typedef struct {
    uint64_t routed_pairs;
    uint64_t unique_experts;
    uint64_t bytes_touched;
    uint64_t dequant_bytes;
    uint64_t expert_cache_hits;
    uint64_t expert_cache_misses;
} q38_moe_stats;
```

## **5. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M6-C00|Freeze MoE semantics|docs/<br>qwen_moe_semantics.md|Documenta router, top-k<br>normalization, activation,<br>expert combine e shared<br>expert dal reference.|documento completo|
|M6-C01|Strict MoE binder|q38_moe.h/.c;<br>q38_weights.c|Bind separato<br>router/routed/shared; exact<br>shape/quant validation.|bind audit pass|
|M6-C02|Router scalar oracle|q38_moe_ref.c/.h|Scores, top-k e<br>normalization scalar su<br>hiddengolden.|exact IDs|
|M6-C03|Router CUDA|q38_moe_cuda.cu|Projection + deterministic<br>top-k; debug dump<br>IDs/weights.|CUDA vs scalar|
|M6-C04|Single expert scalar oracle|q38_moe_ref.c|Gate/up/activation/down<br>per expert e token.|golden expert output|
|M6-C05|Q2 expert CUDA primitive|q38_moe_cuda.cu|<br>Matvec/matmul Q2 routed<br>expert, dequant per tile; no<br>global mirror.|Q2 vs quant oracle|
|M6-C06|Shared expert path|q38_moe_cuda.cu|Precisione M1 policy, path<br>separato.|shared golden|
|M6-C07|Dispatch/combine prefill|q38_moe_dispatch.cu|Batch expert-token pairs,<br>stable token routing,<br>weighted combine.|per-pair audit|
|M6-C08|Decode path|q38_moe_decode.cu|Specializzazione<br>batch=1/small batch senza<br>cambiaremath.|same output|
|M6-C09|Ubatch fuzz harness|tests/test_moe_ubatch.*|Dimensioni 1..1024 con<br>focus su tail/block<br>boundaries.|no OOB; same results|
|M6-C10|Layer MoE integration|q38_forward.c|Integra MoE nel singolo<br>layer con GR.|layer golden|
|M6-C11|4-layer superblock full|q38_forward_probe.c|3×GDN + QSA + PLE<br>where applicable+MoE.|superblock golden|
|M6-C12|48-layer Q2 forward|q38_forward.c|Completa text-only forward<br>e final head.|logit probes pass|
|M6-C13|Decode loop|q38_decode.c|Sampling greedy iniziale;<br>state append/reset.|golden continuation|
|M6-C14|Session contamination<br>tests|tests/test_sessions.*|<br>reset/reuse/new process<br>equivalence.|pass|
|M6-C15|Memory/perf baseline|scripts/m6_baseline.sh|Short/medium/long prompt;<br>memory + t/s + expert<br>stats.|report produced|
|M6-C16|M6 acceptance|tests/; artifacts/m6|Automatizza per-stage,<br>ubatch, end-to-end,<br>memory, CUDA health.|100% pass|



## **6. Router correctness** 

- Confrontare scores pre-top-k, selected IDs, selected raw scores e normalized weights. 

- IDs devono coincidere esattamente col reference salvo tie formalmente equivalente. 

- La normalizzazione dei pesi deve essere verificata, non dedotta da implementazioni DeepSeek. 

- Testare hidden sintetici che producano expert ties, valori molto vicini e score estremi. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
for layer in selected_layers:
  dump:
    router_scores[512]
    top10_ids
    top10_raw
    top10_weights
```

## **7. Quantizzazione selettiva degli routed experts** 

|**Classe**|**M6 Q2**|**Regola**|
|---|---|---|
|Router|Q8/BF16|mai Q2 in M6|
|Routed gate/up|Q2-classverificataM1/M2|dequant pertile|
|Routed down|Q2-class propria|non assumere stesso schema gate/up|
|Shared expert|Q8/BF16|separato dai routed|
|Core GDN/QSA/GR|policy M1|immutata in M6|
|PLE|policyM4|immutatain M6|



M6 non ottimizza ancora la ricetta qualitativa: verifica che l'errore end-to-end sia spiegabile dalla quantizzazione selettiva. Il confronto chiave è Reference BF16 -> Reference con stessi pesi quant/dequant -> q38 Q2. 

## **8. Ubatch fuzzing: gate obbligatorio** 

Il path MoE deve essere testato su molte dimensioni di ubatch. Un issue recentissimo di llama.cpp su un checkpoint Qwen4Exp ha mostrato un out-of-bounds nel path `mul_mat_id` per specifici ubatch (ad esempio 508 e 1016), dovuto al calcolo del tail/padding. Non copiamo quel codice, ma il failure mode è direttamente rilevante. 

```
mandatory ubatch set:
  1..64
  127,128,129
  255,256,257
  507,508,509
  511,512,513
  1015,1016,1017
  random sizes up to 2048
```

```
for each:
  route
  dispatch
  expert matmul
  combine
  cuda bounds instrumentation
```

- Il contenuto del prompt non deve influenzare la safety del buffer. 

- Testare expert IDs al limite: 0, 1, 510, 511. 

- Testare token counts non multipli del block size del quant format. 

## **9. End-to-end validation ladder** 

1. Single token / single layer. 

2. Single token / 4-layer superblock. 

3. Single token / 48 layers. 

4. 16-token prefill / 48 layers. 

5. 512-token prefill con ubatch variabili. 

6. Greedy decode 32 token. 

7. Greedy decode 256 token. 

8. Prompt 4k + decode. 

9. Prompt 16k + decode se M4/M5 memory gates consentono. 

Ad ogni rung si salvano logits top-N e first-divergence layer. Non passare al rung successivo se la divergenza non è classificata come quantization-only o numerical-tolerance-only. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **10. Failure localization** 

```
if end-to-end diverges:
  1 tokenizer
  2 PLE IDs/injection
  3 GR
  4 GDN/QSA
```

- `5 router scores` 

- `6 top10 IDs 7 per-expert outputs` 

- `8 combine/shared expert` 

- `9 final norm/head` 

## **11. M6 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M6-T01|MoE semantics freeze|documento reference completo|
|M6-T02|Routerscores|within numeric tolerance|
|M6-T03|Top-10 IDs|exact|
|M6-T04|Router normalizedweights|withintolerance|
|M6-T05|Single routed expert Q2|matches quant/dequant oracle|
|M6-T06|Shared expert|matchesreference policy|
|M6-T07|Dispatch/combine|per-pair reconstruction correct|
|M6-T08|Ubatch fuzz 1..2048 sampled|no OOB/illegalaccess|
|M6-T09|Expert boundary IDs|0/511 safe and correct|
|M6-T10|Full layer|hidden withinexpected Q2error|
|M6-T11|4-layer superblock|hidden/logits pass|
|M6-T12|48-layer logits|first-divergence classified|
|M6-T13|Greedy continuation|stable deterministic output|
|M6-T14|Chunk invariance|same committed stream ->same outputs|
|M6-T15|Session reset/reuse|no contamination|
|M6-T16|Memory gate|withinproject peak limits|
|M6-T17|CUDA health|fatal errors terminate process|



## **12. Acceptance artifacts** 

```
artifacts/m6/
  qwen_moe_semantics.md
  router_goldens.json
  expert_q2_goldens.json
  shared_expert_goldens.json
  dispatch_combine_goldens.json
  ubatch_fuzz.json
  layer_goldens.json
  superblock_goldens.json
  full48_logits.json
  greedy_continuations.json
  session_tests.json
  memory_baseline.json
  throughput_baseline.json
  expert_stats.json
  cuda_health.txt
  acceptance.txt
```

## **13. Stop conditions** 

- Se top-10 IDs differiscono dal reference senza tie spiegato: fermarsi prima del full model. 

- Se un ubatch specifico causa illegal access: M6 fallisce anche se la dimensione default funziona. 

- Se la Q2 end-to-end divergence è molto maggiore della reference quant/dequant divergence: trattarla come bug runtime. 

- Se per far funzionare il MoE serve una dequant cache globale: design da correggere. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **14. UNKNOWN** 

- Formula esatta di router normalization/scaling della revisione Qwen congelata. 

- Activation/packing esatto delle routed e shared expert matrices nel GGUF scelto. 

- Migliore scheduling prefill per 512 experts sul GB10: da profilare solo in M7. 

- Impatto qualitativo Q2 definitivo: M6 lo misura, non lo presume. 

## **15. Riferimenti** 

**Qwen3.8-Flash-Next config.json:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**llama.cpp Qwen3.8/qwen4exp support:** https://github.com/ggml-org/llama.cpp 

**llama.cpp issue #27792 - MoE mul_mat_id ubatch tail OOB:** https://github.com/ggml-org/llama.cpp/issues/27792 

**llama.cpp issue #27763 - qwen4exp GPU layer corruption on Blackwell SM110:** https://github.com/ggml-org/llama.cpp/issues/27763 

**llama.cpp issue #27797 - qwen4exp multi-segment regression:** https://github.com/ggml-org/llama.cpp/issues/27797 **ds4 issue #773 - GB10 decode near measured memory bandwidth:** https://github.com/antirez/ds4/issues/773 **ds4 issue #293 - whole GGUF CUDA registration OOM on Spark:** https://github.com/antirez/ds4/issues/293 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **16. REVISIONE R1 — Telemetria per expert-wise quantization** 

M6 non sceglie ancora quali expert meritano Q4, ma deve produrre i dati necessari per farlo. Le statistiche sono raccolte per (layer, expert) e, in calibration mode, per projection gate/up/down. 

### **16.1 Statistiche obbligatorie** 

```
per layer, expert:
  activation_count
  selected_fraction
  sum_router_weight
  mean_router_weight
  max_router_weight
  output_l2_sum
  output_l2_mean
```

```
calibration mode:
  gate_up_reconstruction_error_q2
  down_reconstruction_error_q2
  cosine_error_q2
```

- La telemetria non deve cambiare routing o ordine delle somme. 

- I contatori devono evitare overflow su soak lunghi. 

- Il profiling detailed può essere disabilitato; i contatori base restano disponibili. 

### **16.2 Expert sensitivity probe** 

```
for selected calibration tokens:
```

```
  y_ref = expert_forward(reference_or_high_precision)
  y_q2  = expert_forward(q2)
```

```
  record:
    relative_l2(y_ref, y_q2)
    cosine(y_ref, y_q2)
    weighted_error = router_weight * relative_l2
```

È un proxy locale, non la misura finale di quality loss. Serve a creare una shortlist per M8E. 

### **16.3 Aggiornamenti commit M6** 

|**Commit**|**Modifica aggiuntiva**|**Gate**|
|---|---|---|
|M6-C03|Router CUDA espone IDs/weights a telemetry<br>hook.|No output change.|
|M6-C05|Q2 expert primitive supporta calibration<br>compare mode.|Oracle match.|
|M6-C07|<br>Dispatch registra usage e router-weight stats<br>per expert.|Counts reconcile.|
|M6-C15|<br>Baselineinclude expert_usage_by_layer.json.|48×512table complete.|
|M6-C16|Acceptance verifica telemetry accounting.|activations=routed_pairs.|



### **16.4 Nuovi test M6** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M6-T18|Expert usage accounting|somma activation_count= routed_pairs|
|M6-T19|Router-weight accounting|aggregate stats match debug recomputation|
|M6-T20|Telemetry disabled equivalence|logitsidenticaltelemetry on/off|
|M6-T21|Sensitivity sample determinism|same seed->same sample/error metrics|



### **16.5 Acceptance artifact aggiuntivi** 

```
artifacts/m6/
  expert_usage_by_layer.json
  expert_router_weight_stats.json
  expert_q2_local_error_sample.json
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

