Qwen3.8-Flash-Next / DGX Spark prototype 

# M1 — Implementation Specification 

_Tensor inventory, formato GGUF runtime-only e Q2 bootstrap con quantizzazione selettiva degli experts._ 

**Regola metodologica:** ogni fatto architetturale non verificato resta marcato come UNKNOWN e diventa un task di verifica. Nessun comportamento viene dedotto per analogia con DeepSeek/GLM. 

## **1. Definition of Done** 

- Un tensor inventory completo del checkpoint text model Qwen3.8-Flash-Next, con ogni tensor classificato per ruolo. 

- Un `quant_manifest.json` versionato e senza fallback impliciti. 

- Un GGUF runtime-only Q2 che esclude vision e MTP, viene aperto dallo skeleton M0 e rispetta il memory gate sul DGX Spark. 

- Routed experts quantizzati in modo aggressivo Q2-class; core sensibile e shared expert mantenuti a precisione maggiore finché non misurati. 

- PLE gestita come classe separata con policy Q2 iniziale ammessa, ma mai confusa con gli experts. 

- Loader strict: shape, dtype, quant type, metadata architetturali e tensor count sono controllati prima di qualunque allocazione pesante. 

- Nessuna copia persistente dequantizzata dell'intero modello e nessuna full-file CUDA host registration. 

## **2. Architettura da codificare nel validator** 

|**Parametro**<br>**Valore verificato dal config ufficiale**|
|---|
|model_type<br>qwen4_exp|
|textmodel_type<br>qwen4_exp_text|
|hidden_size<br>2560|
|num_hidden_layers<br>48|
|num_experts<br>512|
|num_experts_per_tok<br>10|
|moe_intermediate_size<br>640|
|shared_expert_intermediate_size<br>640|
|hc_count<br>4|
|hc_lowrank<br>320|
|GDN key heads/value heads<br>16 / 48|
|GDN key/valuehead dim<br>128 /128|
|GDN conv kernel<br>4|
|GDNstate dtype<br>float32|
|QSA heads / KV heads / head dim<br>24 / 2 / 256|
|QSA indexer heads/KV/dim<br>4/1/128|
|QSA compress ratio / budget<br>4 / 2048|
|ngram_size / basevocab<br>3 /20,000,000|
|heads_per_ngram / split parts<br>8 / 128|
|PLE layer<br>2|
|vocab_size<br>248,320|
|nativemaxpositions<br>262,144|
|**Attenzione al layer pattern:**Il config corrente deve essere preso come autorità al momento del freeze. Non codificare il<br>pattern da una descrizione precedente: validare l'array `layer_types` reale del checkpoint/converter scelto e generare da esso<br>la tabella compile-time.|



## **3. Quantizzazione: policy M1** 

Q2 è deliberatamente un formato di bring-up. L'obiettivo non è ottenere subito la massima qualità, ma far rientrare il modello con margine e preservare abbastanza precisione nei componenti che determinano routing/state. La policy deve essere per classe semantica. 

|**Classe**|**M1 policy**|**Motivo**|**Fallback consentito**|
|---|---|---|---|
|Routed experts gate/up|<br>IQ2_XXS o Q2_K candidato, da<br>scegliere con kernel/layout reali|massa dominante; ds4 ha donor<br>kernels Q2|solo altro Q2 supportato; no BF16<br>globale|
|Routed experts down|<br>Q2_K candidato|<br>down può avere sensitività/layout<br>diversa|IQ2/Q2 alternativo dopo test|
|Router|Q8_0 oBF16|top-ksensibile;footprint piccolo|BF16|
|Shared expert|Q8_0 o BF16|sempre attivo; non degradarlo|BF16|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

|||Qwe|n3.8-Flash-Next /DGXSparkprototype|
|---|---|---|---|
|**Classe**|**M1policy**|**Motivo**|**Fallback consentito**|
|||insieme ai routed||
|GDN/QSA/GRprojections|Q8_0/BF16|state/attentionsensibili|BF16|
|Norm/scales/bias-like|F32/BF16 come reference|footprint minimo|nessuna quant aggressiva|
|Embedding/output|Q8_0/BF16|qualitàlogits|BF16|
|PLE|Q2-class separata, oppure più alta<br>seilQ2 rowdecoder nonè pronto|domina storage ma è lookup, non<br>expert GEMM|Q4/Q8 con compensazione<br>storage|



Non usare il nome 'Q2' del file per implicare che tutto sia a 2 bit. Il filename deve descrivere almeno `Q2Experts` e la precisione delle altre macro-classi. 

## **4. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Lavoro**|**Gate**|
|---|---|---|---|---|
|M1-C00|Freeze reference|MODEL_BASELINE.md|HF revision, config<br>checksum, converter/llama<br>revision, tokenizer revision.|hashes presenti|
|M1-C01|Inventory extractor|tools/q38_inventory.py o C<br>tool|Esporta tensor<br>name/shape/dtype/bytes/s<br>ource shard. Nessuna<br>quant.|inventory completo|
|M1-C02|Semantic classifier|tools/q38_classify.py;<br>tensor_classes.json|Classifica<br>core/routed/shared/PLE/vis<br>ion/MTP. Unknown = hard<br>error.|0 unclassified|
|M1-C03|Qwen fixed config|q38_model_config.h/.c|Costanti + validator<br>metadata/layer_types.<br>Vision/MTP riconosciuti e<br>marcati excluded.|validator unit tests|
|M1-C04|Quant manifest schema|tools/<br>quant_manifest.schema.jso<br>n; quant_manifest_q2.json|Pattern ordinati, expected<br>matches, quant type,<br>rationale,include/exclude.|schema + match-count<br>tests|
|M1-C05|Quant compatibility probe|tests/test_quant_blocks.cu|Verifica<br>Q2_K/IQ2_XXS/Q8 kernels<br>donor su shape/block<br>alignment Qwen reali.|selected formats pass|
|M1-C06|Converter subset|tools/convert_q38_gguf.py<br>wrapper/patch|Produce embedding +<br>layer 0..3 + output,<br>runtime-only.|loader strict pass|
|M1-C07|Subset memory/load|q38 --inspect/--memory-<br>plan|Misura mmap, page faults,<br>CUDA metadata allocation,<br>zerofull registration.|memory report|
|M1-C08|Full Q2 conversion|artifact GGUF + manifest|48 layer text-only, no<br>vision/MTP.|inventory reconcile 100%|
|M1-C09|Strict binder skeleton|q38_weights.c/.h|Bind per-role, senza<br>forward; type/shape exact.<br>PLE handle separato.|all required tensors bound|
|M1-C10|Spark residency<br>experiment|tools/<br>run_memory_matrix.sh|<br>cold/warm mmap, optional<br>madvise modes; non host-<br>register wholefile.|peak under gate|
|M1-C11|M1 acceptance|tests/ + artifacts/m1|Automatizza checksum,<br>inventory, manifest,<br>memory, quant block tests.|100% pass|



## **5. Tensor inventory schema** 

```
{
  "name": "blk.0....",
  "shape": [ ... ],
  "gguf_type": "...",
  "source_dtype": "...",
  "elements": 0,
  "bytes_source": 0,
  "class": "routed_expert|shared_expert|gdn|qsa|gr|ple|embed|output|...",
  "layer": 0,
  "role": "...",
  "included_runtime": true,
  "quant_rule": "experts_gate_up_q2"
}
```

- 

   - Il report deve sommare bytes ed elementi per class e layer. 

- Ogni pattern del manifest dichiara `expected_min_matches` e `expected_max_matches` (idealmente exact count). 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

- Un tensor non matchato non eredita una quantizzazione di default: conversion failure. 

## **6. GGUF runtime-only** 

- Conservare metadata/tokenizer necessari alla modalità text-only. 

- Escludere tensor vision in modo esplicito e registrare numero/bytes esclusi. 

- Escludere MTP iniziale in modo esplicito. 

- Non eliminare metadata necessari a riconoscere che il checkpoint originale è multimodale; aggiungere metadata runtimelocali se serve distinguere 'text-only stripped artifact'. 

- PLE deve rimanere indicizzabile senza forzare una materializzazione completa in GPU memory. 

## **7. Loader/binder structure** 

```
typedef struct {
    q38_tensor *token_embd;
    q38_layer  layer[48];
    q38_ple_store ple;
    q38_tensor *output_norm;
    q38_tensor *output;
} q38_weights;
```

```
typedef struct {
    q38_layer_kind kind;   /* derived from frozen layer_types table */
    q38_gr_weights gr_attn;
    union {
        q38_gdn_weights gdn;
        q38_qsa_weights qsa;
    } core;
    q38_moe_weights moe;
    q38_gr_weights gr_ffn;
} q38_layer;
```

I nomi tensor specifici dentro `q38_gdn_weights`, `q38_qsa_weights` e `q38_gr_weights` non vanno inventati in M1. Devono essere generati dopo la lettura del mapping GGUF effettivamente scelto; il binder può essere introdotto incrementalmente. 

## **8. Memory accounting** 

Il modello non viene accettato solo perché `stat(file) < 128 GB`. M1 deve produrre quattro numeri distinti: file bytes, mapped virtual bytes, resident set, e peak unified-memory pressure durante le operazioni M1. 

|**Gate**|**Valore operativo M1**|
|---|---|
|MemAvailable prima del load|registrato; test non valido se macchina già sotto forte pressione|
|Peakprocess/unified pressure|target≤ 108 GiBduranteinspect/bindM1|
|Headroom dopo bind|target ≥ 12 GiB|
|Swap-in/out sustained|0 peracceptance|
|Whole-file cudaHostRegister|vietato|
|Persistent dequantmirror|vietato|



Questi sono limiti di progetto conservativi, non limiti hardware NVIDIA. Possono essere aggiornati solo con un ADR (Architecture Decision Record) basato su misure. 

## **9. Quant block compatibility tests** 

- Per ogni candidato Q2: verificare block size divisibility sulle dimensioni effettive delle matrici routed expert. 

- Verificare byte addressing per expert index 0, 1, 510, 511; l'ultimo expert è importante per scoprire stride/overflow. 

- Confrontare CUDA dequant/dot con una decode scalar oracle per centinaia di blocchi random e boundary values. 

- Testare gate/up/down separatamente: non assumere stesso layout fisico. 

- Q4_K viene testato già in M1 anche se non è ancora target, per evitare di scoprire in M8 che il binder/kernel non supporta il futuro formato. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **10. Selective quant audit** 

```
q38 --inspect model-q2.gguf --quant-audit
```

```
Expected output classes:
  routed_expert:  <quant types + bytes>
  router:         <quant types + bytes>
  shared_expert:  <quant types + bytes>
  gdn:            <quant types + bytes>
  qsa:            <quant types + bytes>
  gated_residual: <quant types + bytes>
  ple:            <quant types + bytes>
  embedding:      <quant types + bytes>
  output:         <quant types + bytes>
```

L'audit fallisce se una classe sensibile finisce accidentalmente nella regola experts o se shared/routed vengono confusi. 

## **11. M1 test matrix** 

|**ID**|**Test**|**Procedura**|**PASS**|
|---|---|---|---|
|M1-T01|Reference freeze|checksum files|revisioni e hash completi|
|M1-T02|Inventory completeness|inventoryvs source tensors|100% accounted|
|M1-T03|Classify strict|inject fake tensor|conversion fails|
|M1-T04|Vision/MTPstrip|compareinventories|excluded counts/bytesmatch|
|M1-T05|Layer pattern|metadata config vs fixed table|exact match|
|M1-T06|Q2blockdecode|CUDA vs scalaroracle|withincalibrated quant tolerance|
|M1-T07|Q4 future-proof block|same on Q4 candidate|passes before M8|
|M1-T08|Expertindexing|all512expert boundaries sampled|no stride/indexerrors|
|M1-T09|Selective audit|quant class report|no policy violation|
|M1-T10|Subset GGUFbind|layers 0..3|strict bind succeeds|
|M1-T11|Full Q2 inspect/bind|48 layers text-only|strict bind succeeds|
|M1-T12|Memory gate|cold + warm runs on Spark|≤108 GiB M1 target; ≥12 GiB<br>headroom|
|M1-T13|Nofull hostregister|instrument/trace|0whole-fileregistrations|
|M1-T14|Deterministic conversion|repeat conversion where feasible|same manifest + deterministic<br>metadata/tensorchecksums|



## **12. Acceptance artifacts** 

```
artifacts/m1/
  model_baseline.json
  source_inventory.json
  runtime_inventory.json
  tensor_classes.json
  quant_manifest_q2.json
  quant_audit.json
  quant_block_tests.json
  memory_cold.json
  memory_warm.json
  excluded_tensors.json
  checksums.txt
  acceptance.txt
```

## **13. Exact sequence on DGX Spark** 

1. Eseguire M0 acceptance e salvare platform.json. 

2. Copiously verify free memory; stop desktop/extra workloads se necessario e documentare lo stato. 

3. Eseguire inspect sull'artifact subset, poi full Q2. 

4. Eseguire bind strict senza forward. 

5. Eseguire quant block CUDA tests Q2 e Q4 candidate. 

6. Eseguire memory cold boot test dopo drop-caches solo se operativamente sicuro e autorizzato; altrimenti usare reboot/test controllato e documentarlo. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

7. Eseguire warm test immediatamente dopo. 

8. Generare quant-audit e confrontarlo automaticamente col manifest. 

9. Chiudere M1 solo quando tutti gli artifact sono archiviati insieme al commit runtime e al checksum GGUF. 

## **14. Decisione Q2: cosa NON fare** 

- Non abbassare router/shared/core a Q2 solo per far entrare il file. Se il budget non torna, prima verificare PLE, stripping e loader residency. 

- Non usare una dequant cache FP16 permanente: annullerebbe il vantaggio memoria e ds4 ha già mostrato che i transient sono pericolosi su GB10. 

- Non considerare un output linguisticamente plausibile una validazione del quant/kernel. 

- Non usare un GGUF Q2 pubblico senza sapere esattamente quali tensor sono stati quantizzati in quale formato. 

## **15. UNKNOWN / blocking questions prima di M2** 

- Mapping tensor GGUF qwen4_exp esatto della revisione congelata. 

- Compatibilità effettiva dei donor kernel IQ2_XXS/Q2_K/Q4_K ds4 con le shape degli expert Qwen. 

- Quant type PLE migliore per il bootstrap. 

- Dimensione finale del runtime-only Q2 e relativo resident working set sul GB10. 

- Eventuali metadata GGUF custom necessari per rappresentare PLE in modo efficiente. 

## **16. Riferimenti** 

**ds4 Makefile (main):** https://github.com/antirez/ds4/blob/main/Makefile 

**ds4 core (main):** https://github.com/antirez/ds4/blob/main/ds4.c 

**ds4 public API (main):** https://github.com/antirez/ds4/blob/main/ds4.h 

**ds4 GPU API (main):** https://github.com/antirez/ds4/blob/main/ds4_gpu.h 

**ds4 SSD API (main):** https://github.com/antirez/ds4/blob/main/ds4_ssd.h 

**ds4 issue #293 - full GGUF host registration OOM on Spark:** https://github.com/antirez/ds4/issues/293 

**ds4 issue #721 - Q4 load transient on GB10:** https://github.com/antirez/ds4/issues/721 

**Qwen3.8-Flash-Next config:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**NVIDIA DGX Spark specs:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

