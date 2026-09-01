Qwen3.8-Flash-Next / DGX Spark prototype 

# M5 — Implementation Specification 

_Qwen Sparse Attention: indexer, compressed selection, RoPE/positioning, KV/index state e CUDA correctness sul DGX Spark._ 

**Regola metodologica:** ogni dettaglio non verificato dal checkpoint/config/reference implementation rimane UNKNOWN. Le ottimizzazioni vengono introdotte solo dopo un oracle di correttezza. 

## **1. Definition of Done** 

- I 12 layer full-attention/QSA vengono identificati dal `layer_types` congelato, non da una formula hardcoded se il config diverge. 

- Q/K/V main path, indexer path, compression grouping, scoring, top-k selection, causal tail e final attention sono verificati separatamente. 

- Main KV e indexer/compressed state hanno strutture distinte e accounting separato. 

- RoPE/partial rotary e position semantics sono verificati rispetto al reference Qwen4Exp; nessun codice YaRN/DeepSeek viene riusato per analogia. 

- La selezione QSA è deterministica e invariabile rispetto al chunking della stessa token stream. 

- CUDA naive/reference-compatible passa prima di qualunque fused/flash path. 

- Il runtime gestisce crescita del context senza corrompere cache o CUDA context. 

## **2. Parametri verificabili dal config** 

|**Parametro**<br>**Valore**|
|---|
|full_attention_interval<br>4|
|layer_types<br>patterndichiaratonelconfig:3linear_attention + 1 full_attention ripetuto|
|num_attention_heads<br>24|
|num_key_value_heads<br>2|
|head_dim<br>256|
|partial_rotary_factor<br>0.25|
|rope_theta<br>10,000,000|
|indexer_n_heads<br>4|
|indexer_kv_heads<br>1|
|indexer_head_dim<br>128|
|indexer_compress_ratio<br>4|
|indexer_budget<br>2048|
|max_position_embeddings<br>262,144|
|**Interpretazione prudente:**2048 è il budget dichiarato dall'architettura. Il significato esatto di '512 gruppi × 4 token' è<br>plausibile dato il ratio, ma M5 lo considera verificato solo se il reference implementation conferma grouping, tail e selection<br>semantics.|



## **3. State model** 

```
typedef struct {
    q38_kv_store main_kv;          /* K/V used by selected attention */
    q38_qsa_index_store index;     /* compressed/indexer representation */
    uint64_t position;
```

```
    uint64_t committed_tokens;
} q38_qsa_state;
```

```
typedef struct {
    q38_tensor *q;
    q38_tensor *k;
    q38_tensor *v;
    q38_tensor *out;
```

```
    q38_qsa_indexer_weights indexer;
    q38_rope_params rope;
} q38_qsa_weights;
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

- 

   - La cache main K/V e la cache indexer sono allocate e resettate separatamente. 

- Gli indici selezionati sono transient/output diagnostico, non state persistente salvo diversa semantica del reference. 

- 

- La posizione deve avanzare con i token committed, non con dimensioni interne del chunk. 

## **4. Decomposizione del forward QSA** 

```
hidden
```

- `-> Q/K/V main projections` 

- `-> partial RoPE / positional transform` 

- `-> indexer projections` 

- `-> compression/group construction` 

- `-> index scores` 

- `-> deterministic top-k / budget selection` 

- `-> causal tail handling` 

- `-> gather selected main K/V` 

- `-> attention over selected rows` 

- `-> output projection` 

Ogni freccia sopra deve avere un probe separato. Se il primo end-to-end QSA diverge, il debug deve poter localizzare il primo stage diverso. 

## **5. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M5-C00|Freeze QSA semantics|docs/<br>qwen_qsa_semantics.md|Documenta tensor names,<br>RoPE, indexer,<br>compression, top-k, causal<br>tail, gather e state update<br>dal reference.|nessun UNKNOWN<br>bloccante|
|M5-C01|QSA state/binder|q38_qsa.h/.c;<br>q38_weights.c|Bind strict dei tensor QSA<br>e state structures separate.|shape/type tests|
|M5-C02|RoPE reference|q38_rope_ref.c/.h|Implementa text-only<br>positional transform<br>verificato; partial factor<br>incluso.|golden positions|
|M5-C03|Main QKV CUDA|q38_qsa_cuda.cu|Projection + RoPE senza<br>sparse selection.|Q/K/V golden|
|M5-C04|Indexer scalar oracle|q38_qsa_ref.c/.h|Compression/indexer/<br>scoring scalar per piccoli<br>context.|score golden|
|M5-C05|Deterministic top-k|q38_topk_ref.c;<br>q38_topk_cuda.cu|Tie-breaking esplicito e<br>reference-compatible.|index exact match|
|M5-C06|CUDA indexer naive|q38_qsa_cuda.cu|Indexer/compress/score<br>CUDA non fused.|CUDA vs scalar|
|M5-C07|Causal tail|q38_qsa_ref.c; tests|Implementa gruppi<br>incompleti e token recenti<br>esattamente come<br>reference.|boundary matrix pass|
|M5-C08|Gather + dense-selected<br>attention|q38_qsa_cuda.cu|Gather main K/V<br>selezionati e attention<br>reference-style.|attention golden|
|M5-C09|KV/index state growth|q38_qsa_cache.c/.h|Append/extend/reset e<br>capacity growthcontrollati.|sequential tests|
|M5-C10|Chunk invariance|tests/test_qsa_chunking.*|Confronta index state,<br>selected IDs e output su<br>partizioni diverse.|pass|
|M5-C11|First 4-layer superblock|q38_forward_probe.c|3×GDN M3 + 1×QSA M5,<br>GR incluso; ancora senza<br>full MoE finale se non<br>necessario al probe.|hidden golden|
|M5-C12|GB10 memory layout<br>profiling|scripts/qsa_profile.sh|Misura cache bytes, gather<br>bandwidth, launch count,<br>positionscaling.|report|
|M5-C13|Safe optimization pass|q38_qsa_cuda.cu|Solo fusioni/locality<br>improvements che<br>mantengono probe<br>equivalence.|same outputs + measured<br>gain|
|M5-C14|Long-context staged tests|tests/test_qsa_long.*|1k→4k→16k→64k; 262k<br>solo se memory/time<br>budget lo consente.|no corruption/OOM|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

||||Qwen3.8-Fl|ash-Next /DGXSparkprototype|
|---|---|---|---|---|
|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|M5-C15|M5 acceptance|tests/; artifacts/m5|Automatizza oracle,<br>boundary, chunk, long-<br>context, memory e CUDA-<br>healthtests.|100% pass|



## **6. RoPE / position validation** 

- Confrontare Q e K prima e dopo rotary, non solo logits finali. 

- Testare posizioni 0,1,2,3,4,31,32,127,128,511,512,2047,2048,4095,4096. 

- Testare un context esteso con posizione >32k e >64k prima di dichiarare long-context stabile. 

- Partial rotary factor=0.25 non implica automaticamente quale quarto delle dimensioni sia ruotato: verificare layout/interleaving del reference. 

**No guessing:** mrope_interleaved / section metadata può essere rilevante anche per il text-only path. M5 deve verificare <u>come il reference costruisce position IDs e rotary dimensions nel caso solo testo.</u> 

## **7. Compression e causal-tail boundary matrix** 

|**Context length**|**Perché testarlo**|
|---|---|
|1,2,3|prima del primo gruppo completo|
|4|primo gruppo completo con ratio4|
|5,6,7|gruppo completo+tail parziale|
|8|due gruppicompleti|
|2047,2048,2049|attorno al budget nominale|
|4095,4096,4097|power-of-two/allocatorboundaries|
|random non multipli di 4|stress grouping|



Per ogni caso salvare: compressed/index representation, score vector, top selected group/token IDs, causal tail contribution e output attention. 

## **8. Top-k determinism** 

```
topk requirements:
  identical selected IDs to reference
  explicit tie-breaking rule
  no dependence on CUDA thread scheduling
  stable across prefill chunking
  stable across cold/warm cache
```

Per QSA un top-k 'quasi uguale' non è sufficiente: selezionare righe diverse cambia il graph effettivamente eseguito. Gli score floating possono tollerare errore numerico; gli IDs selezionati devono invece seguire il reference salvo casi di tie formalmente equivalenti, che devono essere testati esplicitamente. 

## **9. Cache model e memoria** 

|**Componente**|**Accounting separato**|
|---|---|
|Main Kcache|bytes/token/layer, capacity,resident bytes|
|Main V cache|bytes/token/layer, capacity, resident bytes|
|Indexer/compressed state|bytes/token/group/layer|
|Temporary score buffers|peak workspace|
|Top-k IDs/offsets|peak workspace|
|Gathered selected KV|workspace; non duplicare più del necessario|



- M5 non sposta automaticamente QSA state su SSD. Prima misurare il footprint reale: solo 12 layer usano QSA e ci sono 2 KV heads. 

- Il context target massimo 262k viene affrontato incrementalmente; un test 262k non deve essere usato per il primo debug. 

- Ogni growth/reallocation deve aggiornare memory telemetry e preservare state. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **10. CUDA failure discipline** 

Gli issue ds4 su GB10 mostrano che un illegal memory access può avvelenare il CUDA context e lasciare vivo il processo. Nei test M5 qualsiasi `cudaErrorIllegalAddress`, launch failure o equivalente è fatale: dump diagnostico, exit non-zero, nessun tentativo di continuazione. 

```
CUDA_CHECK_FATAL(call):
  if error:
    dump last_qsa_phase
    dump position/chunk/cache sizes
    dump selected indices bounds
    fflush logs
    _exit(nonzero)
```

## **11. QSA chunk invariance** 

```
same token stream, same committed sequence
```

```
compare:
  position counter
  index/compressed state
  main K/V append ranges
  score vectors (tolerance)
  selected IDs (exact)
  gathered positions (exact)
  attention output (tolerance)
  downstream probe logits (tolerance)
```

**Rischio reale:** Il supporto qwen4exp appena introdotto in llama.cpp ha già report di regressioni multi-segment e GPUspecifiche. Non prova che il nostro design avrà gli stessi bug, ma giustifica un acceptance gate molto più forte su state <u>continuity e chunking.</u> 

## **12. M5 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M5-T01|QSAsemanticsfreeze|referencemapping completo|
|M5-T02|Strict bind|tutti i tensor QSA attesi, zero extras non<br>classificati|
|M5-T03|RoPE golden|Q/K post-rotary entro tolleranza|
|M5-T04|MainQKV|projectiongolden|
|M5-T05|Indexer scalar|compressed/scoring golden|
|M5-T06|Top-k IDs|exactmatch|
|M5-T07|Tie handling|deterministico/reference-compatible|
|M5-T08|Causalboundarymatrix|tutti icasipassano|
|M5-T09|CUDA indexer vs scalar|entro tolleranza; same IDs|
|M5-T10|Gatherbounds|nessunout-of-range; exact positions|
|M5-T11|Selected attention|output golden|
|M5-T12|Cache append/reset|state transitions corretti|
|M5-T13|Chunk invariance|selected IDs exact; output/logits within<br>tolerance|
|M5-T14|4-layer superblock|reference hidden match entro Q2 budget|
|M5-T15|Long staged context|no corruption/OOMthroughagreed stage|
|M5-T16|Memory accounting|all QSA persistent/transient bytes reported|
|M5-T17|FatalCUDAbehavior|faultinjectionexitsnon-zero|



## **13. Performance pass: cosa ottimizzare e cosa non toccare** 

|**Priorità**|**Ottimizzazione**|**Condizione**|
|---|---|---|
|1|layout main K/V per gather coalescing|solo dopo M5-T11|
|2|indexer projection/compress fusion|golden intermedi ancora disponibili in debug<br>build|
|3|top-k kernelspecialized perbudget|IDs devonorestare exact|
|4|fused gather+attention|solo con fallback naive per validation|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

<u>Qwen3.8-Flash-Next / DGX Spark prototype</u> **<u><mark>Condizione</mark></u>** <u>nessuna aliasing corruption</u> 

**<mark>Priorità</mark>** 

**<u><mark>Ottimizzazione</mark></u>** 

<u>workspace reuse</u> 

<u>5 workspace reuse nessuna aliasing corruption</u> Non introdurre FlashAttention generica per 'velocizzare' prima di avere la semantica sparse corretta. La QSA non è full attention con un parametro top-k; l'indexer e la selezione fanno parte del modello. 

## **14. Acceptance artifacts** 

```
artifacts/m5/
  qwen_qsa_semantics.md
  rope_goldens.json
  qkv_goldens.json
  indexer_goldens.json
  topk_goldens.json
  causal_boundary_matrix.json
  qsa_chunk_invariance.json
  qsa_cache_memory.json
  qsa_long_context.json
  qsa_profile_naive.json
  qsa_profile_optimized.json
  cuda_health.txt
  acceptance.txt
```

## **15. Stop conditions** 

- Se selected IDs differiscono dal reference e non è un tie formalmente spiegato: fermare le ottimizzazioni. 

- Se chunking cambia index state o selected IDs: bug di correctness, non 'numerical noise'. 

- Se un fused kernel passa end-to-end ma fallisce gli intermediate probes: non accettarlo. 

- Se long-context OOM avviene prima del budget previsto: correggere accounting/layout prima di introdurre SSD. 

## **16. Historical UNKNOWNs resolved by the official references**

- Formula exact indexer/compression/grouping is frozen in `docs/qwen_qsa_semantics.md`.

- Causal-tail semantics are frozen as complete groups plus the incomplete visible tail.

- Top-k tie-breaking is explicit and stable: score descending, cell ID ascending.

- Text-only mRoPE uses the T coordinate with interleaved `[11,11,10]` sections and 64 rotary dimensions.

- GB10 layout remains a performance question; the correctness graph uses separate growing caches.

## **17. Riferimenti** 

**Qwen3.8-Flash-Next config.json:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json **Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next **Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next **llama.cpp qwen4exp support / source tree:** https://github.com/ggml-org/llama.cpp **llama.cpp qwen4exp multi-segment issue #27797:** https://github.com/ggml-org/llama.cpp/issues/27797 **llama.cpp qwen4exp CUDA issue #27763:** https://github.com/ggml-org/llama.cpp/issues/27763 **ds4 DGX Spark full GGUF registration issue #293:** https://github.com/antirez/ds4/issues/293 **ds4 DGX Spark OOM regression #585:** https://github.com/antirez/ds4/issues/585 **ds4 DGX Spark fatal CUDA context issue #759:** https://github.com/antirez/ds4/issues/759 **ds4 discrete/unified registration analysis #791:** https://github.com/antirez/ds4/issues/791 **NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 
