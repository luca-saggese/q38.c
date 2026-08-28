Qwen3.8-Flash-Next / DGX Spark prototype 

# M8 — Implementation Specification 

_Selective Q4: sensitivity-driven quantization, memory solver, quality gates e ricetta finale DGX Spark._ 

**Metodo:** nessuna ricetta di quantizzazione o accelerazione viene promossa perché 'sembra funzionare'. Deve superare simultaneamente correctness, quality, memory e stability gates sul DGX Spark. 

## **1. Definition of Done** 

- Esiste una ricetta Q4 selettiva formalmente definita per classe tensor, non una semplice quantizzazione uniforme. 

- I routed experts sono il dominio principale che passa da Q2 a Q4; router, shared expert, GDN/QSA/GR, norms e output cambiano precisione solo tramite ablation. 

- PLE mantiene una policy indipendente e può restare Q2 se questa scelta offre il miglior trade-off globale. 

- La ricetta finale rispetta i memory gates dello Spark con headroom sufficiente per context state, workspace e M9. 

- La qualità Q4 migliora in modo misurabile la baseline Q2 M7 e non introduce regressioni di correctness. 

- Ogni cambio di quant type è tracciabile a un esperimento di sensitivity e a un delta memoria/qualità/prestazioni. 

- Il formato finale è riproducibile da un quant manifest versionato e verificato da `q38 --quant-audit`. 

## **2. Perché 'Q4' non è una specifica sufficiente** 

I primi GGUF pubblici Qwen3.8-Flash-Next mostrano footprint sostanzialmente diversi a seconda della ricetta. Un Q4_K_M pubblico è descritto attorno a 120 GB e usa Q6_K per embedding/output e Q5_0 per la PLE a causa delle shape; un UDIQ4_XS è riportato attorno a 93.7 GB. Questi numeri sono riferimenti esterni, non target del nostro runtime. 

**Conseguenza:** M8 ottimizza una composizione di quant types per tensor class. Non useremo 'Q4_K_M' o 'IQ4_XS' come <u>scorciatoia progettuale.</u> 

## **3. Objective function** 

```
minimize:
    quality_loss(recipe)
```

```
subject to:
    peak_startup_bytes <= 112 GiB
    peak_prefill_bytes <= 116 GiB
    steady_state_target <= 108 GiB (preferred)
    no swap thrash
    no full dequant mirror
    all correctness gates pass
secondary objectives:
    maximize decode t/s
    maximize prefill t/s
    minimize startup latency
```

Il solver non deve essere sofisticato: può iniziare come enumerazione guidata di ricette discrete. Ciò che conta è che il budget sia calcolato per classi/tensor reali e non da un bits-per-parameter medio. 

## **4. Classi di sensitivity** 

|**Classe**|**Baseline M7**|**Prima prova M8**|**Priorità qualità**|
|---|---|---|---|
|Routed expert gate/up|Q2-class|Q4_K/IQ4-class|MEDIA-ALTA|
|Routed expert down|Q2-class|Q4_K/IQ4-class separata|ALTA|
|Router|Q8/BF16|immutato|MOLTO ALTA|
|Shared expert|Q8/BF16|immutato; poiQ6/Q8 ablation|ALTA|
|GDN projections/state-related<br>weights|Q8/BF16|immutato|MOLTO ALTA|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

|||Q|wen3.8-Flash-Next /DGXSparkprototype|
|---|---|---|---|
|**Classe**|**Baseline M7**|**Prima prova M8**|**Priorità qualità**|
|QSA projections/indexer|Q8/BF16|immutato|MOLTO ALTA|
|GatedResidual|Q8/BF16|immutato|MOLTOALTA|
|Norm/scales|BF16/F32|immutato|MOLTO ALTA|
|Embedding/output|Q8/BF16|Q6/Q8 ablationonly|ALTA|
|PLE|Q2-class policy M4|Q2 vs Q4 dedicated ablation|MEDIA/UNKNOWN|



## **5. Ricette iniziali da misurare** 

|**Ricetta**|**Experts**|**PLE**|**Core sensibile**|**Scopo**|
|---|---|---|---|---|
|R0|Q2|Q2|M7precision|baselineM7|
|R1|Q4|Q2|M7 precision|misura valore puro del Q4<br>experts|
|R2|Q4|Q4|M7 precision|misura valore PLE Q4|
|R3|Q4 down + Q4 gate/up<br>variant B|Q2|<br>M7 precision|quant-type sensitivity<br>experts|
|R4|Q4 experts|Q2|shared expert Q6/Q8|recupero memoria se<br>necessario|
|R5|Q4 experts|Q2|embedding/output Q6<br>candidate|trade-off memoria/logits|



La ricetta finale può non coincidere con nessuna R0-R5. Queste sono prove controllate per isolare gli effetti. 

## **6. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M8-C00|Freeze M7 baseline|artifacts/m8/m7_baseline_*|Blocca commit, GGUF,<br>benchmark e quality<br>corpus.|reproducible|
|M8-C01|Sensitivity harness|tools/<br>q38_quant_sensitivity.py;<br>q38_eval|Permette override di quant<br>class per esperimento.|class-isolated experiments|
|M8-C02|Memory solver|tools/<br>q38_memory_solver.py|Calcola file/resident/peak<br>projection per ricetta da<br>inventoryreale.|matches measured R0|
|M8-C03|Q4 expert kernels<br>finalization|q38_moe_cuda.cu;<br>q38_quant.cu|Completa Q4 candidate<br>gate/up/down e ubatch<br>fuzz.|M6 goldens + Q4|
|M8-C04|R1 artifact|quant_manifest_R1.json;<br>GGUF|Q4 routed experts, resto<br>invariato.|load + audit|
|M8-C05|R1 quality/perf|artifacts/m8/R1_*|Logit/perplexity/task/<br>memory/t/s vs R0.|report|
|M8-C06|PLE Q4 ablation|q38_ple*; R2 manifest|Q4 row decoder/storage<br>policy se necessario.|M4 goldens pass|
|M8-C07|Core sensitivity probes|R4/R5 manifests|Shared/embed/output<br>reductions una classe per<br>volta.|no mixed attribution|
|M8-C08|Recipe search|tools/<br>q38_recipe_search.py|Enumera candidate entro<br>memory gate usando<br>measured deltas.|candidate shortlist|
|M8-C09|Final recipe build|quant_manifest_q4_selecti<br>ve.json|Produce artifact finale<br>reproducibile.|audit 100%|
|M8-C10|Full regression|make m0...m8-acceptance|Riesegue all correctness,<br>chunk, ubatch e CUDA-<br>healthgates.|all pass|
|M8-C11|Quality acceptance|artifacts/m8/final_quality_*|Confronta R0, final Q4,<br>reference quant/dequant.|threshold pass|
|M8-C12|Memory/perf acceptance|artifacts/m8/<br>final_memory_bench_*|Spark cold/warm,<br>short/long, startup/prefill<br>peak.|gates pass|
|M8-C13|M8 release manifest|RELEASE_Q4.md;<br>checksums|Documenta esatta<br>composizione e<br>motivazione.|complete|



## **7. Memory solver schema** 

```
recipe component:
  class
  tensor_count
  source_bytes
  quant_type
  quantized_bytes
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  alignment_overhead
  required_workspace_bytes
  expected_residency_policy
```

```
solver output:
  model_file_bytes
  mmap_bytes
  predicted_resident_bytes
  persistent_state_bytes
  max_workspace_bytes
  cache_budget_bytes
  projected_peak_bytes
```

- Il solver viene calibrato contro R0/M7 misurato prima di usarlo per R1+. 

- Se previsione e misura differiscono >5% sul peak, correggere il modello prima di prendere decisioni. 

- PLE e routed expert cache/residency sono input espliciti, non hidden constants. 

## **8. Quality methodology** 

|**Metrica**|**Uso**|
|---|---|
|Layer/logit error|localizza distorsione tecnica|
|KL/log-prob delta|confronto distribuzionale|
|<br>Perplexity su corpusfisso|sensibilità globale|
|Greedy deterministic suite|regressioni evidenti e reproducibility|
|Coding/reasoningmultilingual mini-suite|quality sanity|
|Tool/structured-format suite|stabilità formato se target server la richiede|



**No soglia inventata:** Le soglie di acceptance devono essere fissate dopo aver misurato R0 e R1 sul corpus scelto. Il <u>documento non impone percentuali arbitrarie senza dati.</u> 

## **9. Ablation discipline** 

- Cambiare una sola macro-classe per esperimento, salvo recipe-search finale. 

- Ogni esperimento salva quant manifest e checksum. 

- Se R1 migliora qualità ma peggiora t/s, misurare se il costo deriva da bandwidth, decoder Q4 o maggiore residency. 

- Se PLE Q4 porta poco beneficio qualitativo ma costa molta memoria, mantenerla Q2 è una soluzione pienamente valida. 

- Se una classe sensibile a precisione inferiore libera poca memoria, non quantizzarla per principio estetico. 

## **10. Q4 expert validation** 

```
for every routed expert quant candidate:
  scalar block decode
  CUDA block decode
  expert ids 0,1,510,511
  gate/up/down separately
  ubatch fuzz M6
  layer MoE golden
  48-layer regression
```

Il Q4 non è accettato se passa solo un benchmark end-to-end. Deve mantenere tutti i test strutturali costruiti in M1/M2/M6. 

## **11. M8 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M8-T01|Memory solver calibration|R0 predicted vs measured within agreed error|
|M8-T02|Q4blockdecode|scalar/CUDApass|
|M8-T03|Q4 expert boundaries|all 512 indexing safe|
|M8-T04|Q4ubatch fuzz|M6matrixpass|
|M8-T05|R1 quant audit|only intended classes changed|
|M8-T06|R1 memory|withinSparkgates|
|M8-T07|R1 correctness regression|M0-M7 pass|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

<u>Qwen3.8-Flash-Next / DGX Spark prototype</u> 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M8-T08|PLE Q2 vs Q4 ablation|measured and attributable|
|M8-T09|Final recipe quant audit|100% tensoraccounted|
|M8-T10|Final startup peak|≤112 GiB|
|M8-T11|Finalprefillpeak|≤116 GiB|
|M8-T12|No swap/full mirror|pass|
|M8-T13|Finalquality|meets thresholds establishedfromdata|
|M8-T14|Final throughput|documented vs R0; no unexplained regression|
|M8-T15|Chunk/ubatch/stateregression|allprior invariance tests pass|



## **12. Acceptance artifacts** 

```
artifacts/m8/
  m7_baseline_manifest.json
  sensitivity_results.json
  memory_solver_calibration.json
  quant_manifest_R1.json
  R1_quality.json
  R1_memory_bench.json
  ple_ablation.json
  core_ablation.json
  recipe_search.json
  quant_manifest_q4_selective.json
  final_quant_audit.json
  final_quality.json
  final_memory.json
  final_bench.json
  full_regression.txt
  RELEASE_Q4.md
  checksums.txt
  acceptance.txt
```

## **13. Stop conditions** 

- Se il Q4 final artifact entra solo eliminando il memory headroom operativo, non è accettabile. 

- Se una classe sensibile viene quantizzata più aggressivamente solo per raggiungere un'etichetta nominale Q4, revert. 

- Se la PLE Q4 non giustifica i byte aggiuntivi, mantenerla Q2. 

- Se Q4 speculative/MTP tests vengono usati per giudicare M8: separare il problema; MTP appartiene a M9. 

## **14. UNKNOWN** 

- Ricetta Q4 ottimale reale sul nostro checkpoint/converter. 

- Beneficio qualitativo marginale di PLE Q4 rispetto a routed expert Q4. 

- Quant type Q4 migliore per gate/up vs down expert. 

- Footprint finale misurato: non derivare dai GGUF pubblici perché le ricette differiscono. 

## **15. Riferimenti** 

**Qwen3.8-Flash-Next config.json:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**Public Q4_K_M GGUF example (~120 GB):** https://huggingface.co/vumpt/Qwen3.8-Flash-Next-GGUF **llama.cpp qwen4exp SM121 sustained-load issue #27780:** https://github.com/ggml-org/llama.cpp/issues/27780 **llama.cpp qwen4exp SM110 corruption issue #27763:** https://github.com/ggml-org/llama.cpp/issues/27763 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype **llama.cpp qwen4exp multi-segment issue #27797:** https://github.com/ggml-org/llama.cpp/issues/27797 

**llama.cpp speculative Q4 divergence issue #25618:** https://github.com/ggml-org/llama.cpp/issues/25618 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

