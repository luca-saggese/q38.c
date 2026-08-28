Qwen3.8-Flash-Next / DGX Spark prototype 

# M9 — Implementation Specification 

_Hardening: session persistence, checkpoint/replay, MTP speculative decoding, server single-worker e soak testing sul DGX Spark._ 

**Metodo:** nessuna ricetta di quantizzazione o accelerazione viene promossa perché 'sembra funzionare'. Deve superare simultaneamente correctness, quality, memory e stability gates sul DGX Spark. 

## **1. Scope e Definition of Done** 

M9 è deliberatamente divisa in tre sottotracce. M9A hardening state/session è obbligatoria. M9B server è obbligatoria se il prototipo deve essere servito via API. M9C MTP è opzionale e viene promosso solo se rimane lossless rispetto al target Q4. 

|**Track**|**Scopo**|**Obbligatorietà**|
|---|---|---|
|M9A|Session persistence, checkpoint/replay, reset,<br>multi-turn correctness|OBBLIGATORIA|
|M9B|Single-worker server, health, request isolation,<br>fatal CUDA recovery|OBBLIGATORIA per serving|
|M9C|1-layer MTPspeculative decoding|OPZIONALE/performance|



- Il target Q4 M8 rimane l'oracle autorevole: MTP non può cambiare il risultato greedy verificato dal target. 

- Un CUDA illegal access/abort non viene 'recuperato' nello stesso context: il worker viene terminato e ricreato dal supervisor. 

- Session serialization include solo semantic state; cache ricostruibili (PLE/expert hot cache) non vengono persistite. 

- Soak tests multi-request sul vero DGX Spark sono gate formali, non demo manuali. 

## **2. State inventory per sessione** 

|**State**|**Persistente semanticamente?**|**Checkpoint**|
|---|---|---|
|Token history|SI|sempre|
|GDN recurrent state|SI|sempre|
|GDN conv history|SI|sempre|
|QSA main K/V|SI|sempre o tramitereplay policy|
|QSA index/compressed state|SI|sempre|
|Position/committed tokencounters|SI|sempre|
|Gated Residual transient between layers|NO a token boundary|non serializzare|
|PLEcache|NO|ricostruibile|
|Expert cache|NO|ricostruibile|
|Temporary CUDA workspace|NO|ricostruibile|



## **3. Rewind: strategia checkpoint + replay** 

GDN è ricorrente: non assumiamo che lo state possa essere invertito. Il rewind affidabile viene quindi implementato tramite checkpoint periodici + replay deterministico della token stream successiva. 

```
checkpoint at token K stores:
  committed_position
  token_history
  GDN recurrent + conv state
  QSA KV/index state (or an explicitly validated compact form)
  model/quant manifest checksum
  session format version
```

```
rewind to N:
  choose latest checkpoint K <= N
  restore K
  replay tokens K..N
  verify state checksum in debug mode
```

- Checkpoint interval è un parametro di performance/storage, non di semantica. 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

- Il formato deve essere versionato e rifiutare manifest/model checksum incompatibili. 

- Compressione dello snapshot è consentita solo dopo byte-exact/semantic restore tests. 

## **4. Commit plan M9A/M9B** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M9-C00|Freeze M8 release|RELEASE_Q4.md;<br>checksums|Il model artifact M8 diventa<br>target oracle.|immutable baseline|
|M9-C01|Session state API|q38_session.*|Create/reset/append/clone<br>introspection con<br>ownership chiaro.|unit pass|
|M9-C02|Checkpoint format v1|q38_checkpoint.*|Header versionato + per-<br>subsystem state<br>serialization.|roundtrip pass|
|M9-C03|Replay engine|q38_replay.*|Restore K + deterministic<br>replayfino aN.|rewind golden|
|M9-C04|Checkpoint interval study|scripts/<br>checkpoint_matrix.sh|Trade-off bytes, latency,<br>rewind cost.|policy chosen|
|M9-C05|Multi-turn harness|tests/test_multiturn.*|Conversation segmentation<br>variants e cache-hit<br>repeats.|same target result|
|M9-C06|Worker process model|q38_worker.*;<br>q38_supervisor.*|Single CUDA worker<br>process supervised<br>externally.|health pass|
|M9-C07|Minimal HTTP/API server|q38_server.*|OpenAI-like subset o API<br>minima; no parallel GPU<br>contexts initially.|functional pass|
|M9-C08|Request isolation|q38_server.*; tests|Session IDs/state<br>separate, cancellation<br>cleanup.|no contamination|
|M9-C09|Fatal CUDA recovery|q38_supervisor.*|Worker exit on CUDA fatal;<br>supervisor restarts; model<br>reload policy measured.|fault injection pass|
|M9-C10|Sustained-load harness|scripts/soak.sh|Multi-turn/tool-format/long<br>prompts over<br>hours/requests.|no abort/corruption|
|M9-C11|Server memory stability|artifacts/m9/<br>memory_soak.json|RSS/unified memory<br>plateau; cache bounds<br>respected.|no leak trend|



## **5. Perché M9 richiede soak test sullo Spark** 

Un issue qwen4exp aperto il 27 agosto 2026 riporta un abort del graph builder proprio su DGX Spark SM121 durante carico multi-request sostenuto, mentre richieste singole risultavano corrette. Questo non implica che q38 erediterà il bug, ma dimostra che single-prompt acceptance non è sufficiente per questa architettura/hardware. 

**Gate M9:** Il server non è considerato stabile finché non passa una batteria sostenuta che mescola multi-turn, reset, prompt <u>lunghi, tool/structured formatting e cache reuse.</u> 

## **6. MTP: facts verificati dal config** 

|**Parametro MTP**|**Valore**|
|---|---|
|mtp_num_hidden_layers|1|
|mtp.num_hidden_layers|1|
|mtp.hybrid|true|
|mtp.layer_types|[full_attention]|
|mtp_use_dedicated_embeddings|false|
|mtp_use_hidden_state_from_layer|null|
|mtp.rope_theta|10,000,000|



Questi campi confermano l'esistenza di un singolo layer MTP ibrido/full-attention, ma non definiscono da soli l'algoritmo di proposal/verification né la forma esatta delle projection/injection. M9C parte solo dopo freeze semantico dal reference. 

## **7. Commit plan M9C — MTP opzionale** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M9-C12|Freeze MTP semantics|docs/<br>qwen_mtp_semantics.md|Tensor mapping, input<br>hidden, tokenconditioning,|document complete|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

<u>Qwen3.8-Flash-Next / DGX Spark prototype</u> 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
||||position/state e logits.||
|M9-C13|MTP binder|q38_mtp.*; q38_weights.c|Carica tensor MTP<br>precedentemente esclusi<br>dal runtime artifact o<br>artifact companion.|strict bind|
|M9-C14|MTP reference path|q38_mtp_ref.*|Proposal logits/token su<br>short goldens.|reference match|
|M9-C15|MTP CUDA path|q38_mtp_cuda.cu|1-layer proposal path; no<br>speculation yet.|golden pass|
|M9-C16|Target verification loop|q38_speculative.*|Draft proposes, target Q4<br>verifies; greedy semantics<br>lossless.|exact target output|
|M9-C17|Acceptance-rate profiler|q38_spec_stats.*|Accepted tokens, rejected,<br>target calls, speedup.|report|
|M9-C18|Q4 path-dependence<br>stress|tests/test_spec_q4.*|Vanilla vs speculative over<br>deterministic suite/chunk<br>sizes.|exact greedy equality|
|M9-C19|MTP performance decision|docs/ADR-mtp.md|Enable only if robust<br>speedup after memory<br>cost.|ENABLE/DISABLE|



## **8. MTP losslessness gate** 

Speculative decoding deve essere semanticamente un'accelerazione, non un diverso decoder. Un issue recente di llama.cpp riporta divergenza greedy tra vanilla e MTP/DSpark su target Q4, mentre BF16 non divergeva. Per q38 la regola è quindi più severa: sul nostro target Q4, greedy vanilla e speculative devono produrre esattamente la stessa sequenza di token per l'intera suite. 

```
for every deterministic prompt:
  vanilla_tokens = decode_target_q4(temp=0, top_k=1)
  spec_tokens    = decode_mtp_q4(temp=0, top_k=1)
```

```
  REQUIRE vanilla_tokens == spec_tokens
```

```
also compare at each verify step:
  target logits path
  committed tokens
  session state position
```

Se la divergenza è path-dependent per effetti numerici Q4, MTP rimane DISABLED anche se la velocità è buona. Non accetteremo 'quasi lossless'. 

## **9. MTP memory policy sullo Spark** 

- MTP era escluso dagli artifact M1-M8: M9 deve prima misurare il costo reale di aggiungerlo. 

- Preferire artifact companion o sezione chiaramente separata per poter avviare il target senza MTP. 

- Il peak con MTP deve rimanere entro i gate; se richiede sacrificare precisione del target Q4, MTP viene respinto. 

- MTP cache/workspace non deve duplicare indiscriminatamente QSA/GDN target state. 

## **10. Server architecture** 

```
q38-supervisor
  |
  +-- q38-worker (owns the only CUDA context)
        |
        +-- model Q4 target
        +-- optional MTP
        +-- session table
        +-- bounded PLE/expert caches
```

```
fatal CUDA:
  worker exits
  supervisor marks in-flight request failed
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  new worker is spawned/reloaded
```

In M9 non implementerei più worker CUDA paralleli sullo stesso Spark. Prima si deve dimostrare stabilità e memory headroom del singolo worker; concurrency può essere multiplexata tramite request scheduling/session interleaving con un solo owner del device. 

## **11. Multi-turn / segmentation invariance** 

|**Scenario**|**Confronto**|
|---|---|
|single serialized prompt vs messages segmented|same tokenizer stream->same result|
|sessioncontinuevsreconstruct+replay|same committed tokenstream|
|checkpoint restore vs live session|same state/logits|
|cache coldvswarm|same output|
|server request reuse vs fresh process|same session semantics|



Questo gate è particolarmente importante perché qwen4exp ha avuto un report recente di output degenerato con prompt multisegmento pur funzionando su single segment. 

## **12. Soak matrix** 

|**Suite**|**Carico**|**Gate**|
|---|---|---|
|S1|1000 short single-turn requests|0 crash/corruption;memory plateau|
|S2|500 multi-turn sessions|no cross-session state|
|S3|mixed4k/16kprompts|noleak/fragmentationtrend|
|S4|checkpoint/rewind repeated|deterministic replay|
|S5|tool/structured-output battery|no graph/state abort|
|S6|MTP enabled if candidate|vanilla/spec exact greedy equality|



Le quantità possono essere adattate al tempo disponibile, ma l'accettazione deve includere abbastanza iterazioni da esercitare repeated allocation, reset e cache eviction. 

## **13. M9 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M9-T01|Session reset/clone|semantic state correct|
|M9-T02|Checkpoint roundtrip|restore equals live|
|M9-T03|Rewind+replay|same targetlogits/tokens|
|M9-T04|Model/manifest mismatch|checkpoint rejected|
|M9-T05|Multi-turnsegmentation|same stream ->same output|
|M9-T06|Request isolation|no state leakage|
|M9-T07|CUDA fatal injection|workerdies, supervisor recovers|
|M9-T08|Sustained load|no abort/corruption|
|M9-T09|Memory soak|no unbounded growth|
|M9-T10|MTP reference|proposal path correct, if enabled|
|M9-T11|MTPQ4 losslessness|vanilla/spec tokensequence exact|
|M9-T12|MTP memory gate|target precision/headroom preserved|
|M9-T13|MTPspeedup|positiverobust gainor DISABLEdecision|
|M9-T14|Full M0-M8 regression|100% pass after hardening|



## **14. Acceptance artifacts** 

```
artifacts/m9/
  session_format_v1.md
  checkpoint_roundtrip.json
  rewind_replay.json
  multiturn_matrix.json
  worker_fault_recovery.json
  soak_results.json
  memory_soak.json
  server_bench.json
  qwen_mtp_semantics.md           # if MTP attempted
  mtp_goldens.json                # if attempted
  mtp_losslessness.json           # if attempted
  mtp_perf.json                   # if attempted
  ADR-mtp.md
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  full_regression.txt
  acceptance.txt
```

## **15. Final release criteria** 

- Q4 target M8 correctness e quality invariati. 

- Session state deterministico e checkpoint format versionato. 

- Nessuna memory growth non bounded durante soak. 

- CUDA fatal failure isolata a worker process. 

- MTP solo se esattamente lossless sul greedy target Q4 e realmente più veloce. 

- Ogni artifact release contiene model checksum, quant manifest checksum, runtime commit e platform report. 

## **16. UNKNOWN** 

- Semantica esatta MTP Qwen3.8 oltre ai campi config. 

- Costo reale in memoria dei tensor MTP quantizzati secondo la policy futura. 

- Acceptance rate e speedup MTP sul GB10. 

- Intervallo checkpoint ottimale per GDN/QSA state. 

- Throughput/concurrency server ottimale con un solo CUDA owner. 

## **17. Riferimenti** 

**Qwen3.8-Flash-Next config.json:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**Public Q4_K_M GGUF example (~120 GB):** https://huggingface.co/vumpt/Qwen3.8-Flash-Next-GGUF 

**llama.cpp qwen4exp SM121 sustained-load issue #27780:** https://github.com/ggml-org/llama.cpp/issues/27780 **llama.cpp qwen4exp SM110 corruption issue #27763:** https://github.com/ggml-org/llama.cpp/issues/27763 **llama.cpp qwen4exp multi-segment issue #27797:** https://github.com/ggml-org/llama.cpp/issues/27797 **llama.cpp speculative Q4 divergence issue #25618:** https://github.com/ggml-org/llama.cpp/issues/25618 **NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB unified coherent memory / CUDA only 

