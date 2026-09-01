# q38.c — Code Review aggiornata

Branch: `qwen38-spark-proto`

Scope:
- stato corrente dopo le correzioni GR/state/tokenizer;
- Qwen3.8-Flash-Next;
- DGX Spark / GB10;
- CUDA-only runtime;
- GGUF q38 derivato intenzionalmente dai tensor originali Qwen3.8 secondo la specifica del progetto.

Questa review **non riapre** finding già corretti. L'obiettivo è identificare le macro-problematiche ancora aperte prima di proseguire troppo in M5/M6.

---

# Executive summary

Lo stato attuale è sensibilmente migliore rispetto alla review precedente.

Le criticità GR e GDN multi-layer state risultano corrette, e il tokenizer ora è nativo.

Le principali aree ancora da chiudere sono:

| Priorità | Finding | Azione richiesta |
|---|---|---|
| P0 | M4-C06 non valida ancora realmente l'iniezione PLE sugli hidden state | Implementare un vero forward probe PLE e golden indipendenti |
| P1 | Il PLE store file-backed è ancora un baseline mmap/memcpy, non una pipeline SSD ottimizzata completa | Mantenerlo come reference path e profilare prima di ottimizzare |
| P1 | Manca ancora un vero forward integrato sui primi layer | Costruire un probe layer 0–3 con hidden intermedi |
| P1 | QSA è ancora uno skeleton; attenzione particolare allo state del gruppo compresso/pending | Congelare semantics completa prima del kernel |
| P1/P2 | Il tokenizer nativo va stressato su edge case JSON/chat/multimodali | Ampliare parity suite |
| P2 | Special token ID hardcoded nel tokenizer | Verificare sempre contro tokenizer/config locale |
| P2 | `expected_tensor_count()` contiene logica fragile / numeri magici | Rafforzare binding semantico per layer type |
| P2 | M4S: non assumere che una grande PLE cache abbia buon hit rate | Misurare hit-rate reale prima di investire in caching |

---

# 1. P0 — M4-C06 non valida ancora realmente l'iniezione PLE

## File da verificare

- `q38_forward_probe.c`
- `q38_ple.c`
- `q38_ple.h`
- test/golden M4
- eventuali generatori reference Python/Transformers/llama.cpp

## Problema

Il current `q38_forward_probe.c` sembra ancora trattare:

```text
hidden_before_ple
hidden_after_ple
ple_contribution_vector
```

come assenti / `null`.

Il probe corrente valida soprattutto:

```text
tokens
  -> ngram history
  -> ngram row IDs
  -> chunk invariance degli IDs
```

Questo è utile, ma **non dimostra ancora il punto di injection PLE nel forward reale**.

M4-C06 deve validare esplicitamente:

```text
hidden_before_ple
        ↓
lookup delle righe PLE reali
        ↓
dequant / aggregation
        ↓
PLE contribution
        ↓
injection nel layer corretto
        ↓
hidden_after_ple
```

contro un reference indipendente.

## Perché è P0

Se l'injection point, l'ordine di aggregazione o la contribution vector sono sbagliati:

- gli ngram IDs possono comunque essere perfetti;
- i test di cache/residency possono essere perfetti;
- il modello finale può comunque essere matematicamente sbagliato.

È quindi un gate semantico indispensabile.

## Fix richiesto

Implementare un **reference probe minimale e indipendente** usando:

- checkpoint originale Qwen3.8 locale;
- reference Qwen/Transformers;
- `llama.cpp` come reference semantico secondario;
- lettura file-backed delle sole righe PLE effettivamente richieste.

Non caricare l'intero checkpoint.

Per un piccolo corpus golden servono solo:

```text
token IDs
ngram IDs
row IDs
hidden_before_ple
PLE rows selezionate
PLE contribution[2560]
hidden_after_ple[2560]
```

## Golden obbligatori

Per ogni caso:

```json
{
  "tokens": [...],
  "ngram_ids": [...],
  "row_ids": [...],
  "hidden_before_ple": [...],
  "ple_contribution": [...],
  "hidden_after_ple": [...],
  "source_model_revision": "...",
  "reference_revision": "...",
  "checksums": {...}
}
```

## Test richiesti

### PLE-INJ-T01 — single small sequence

Un caso deterministico semplice.

### PLE-INJ-T02 — chunked vs non-chunked

Stessa token stream, diverse partizioni.

Confrontare:
- ngram IDs;
- row IDs;
- PLE contribution;
- hidden after injection.

### PLE-INJ-T03 — repeated ngram

Sequenza con token ripetuti per verificare lookup/aggregation.

### PLE-INJ-T04 — CUDA vs external golden

Non basta confrontare CUDA contro q38 scalar.

Il confronto deve essere:

```text
external reference
==
q38 scalar/reference
==
q38 CUDA
```

entro la tolleranza del quant format.

## Exit criterion

M4-C06 è davvero chiuso solo quando esiste almeno un hidden-state golden indipendente prima/dopo PLE.

---

# 2. P1 — Il PLE file-backed store è ancora un baseline, non la pipeline SSD finale

## Stato attuale

Il nuovo store PLE file-backed è una buona base:

- 128 shard;
- row addressing;
- niente materializzazione completa;
- single-row/batch reads;
- residency/cache tests.

Il path corrente però è ancora sostanzialmente:

```text
NVMe
 ↓
mmap
 ↓
OS page cache / page fault
 ↓
CPU memcpy
```

con lettura della riga tramite mapping.

Questo è corretto come **reference/baseline implementation**.

Non va però descritto come pipeline SSD completa M4S.

## Mancano ancora, se dimostrati utili

- planner degli accessi per physical block;
- deduplica/coalescing I/O;
- bounded quantized RAM cache;
- pinned staging;
- `cudaMemcpyAsync`;
- overlap I/O/compute;
- eventuale `preadv` / `io_uring`;
- telemetry dettagliata del wait I/O.

## Regola importante

Non ottimizzare ancora alla cieca.

Prima misurare:

```text
page faults / token
quant bytes touched / token
unique PLE rows / token
cache hit rate
CPU memcpy time
I/O wait
CUDA transfer wait
```

## Decisione

Mantenere il path mmap come reference path anche dopo le ottimizzazioni.

Ogni path SSD/cache ottimizzato deve produrre gli stessi output del baseline.

---

# 3. P1 — Manca ancora un vero forward integrato sui primi layer

## Problema

I componenti sono stati implementati e testati in modo abbastanza rigoroso:

- GR;
- GDN;
- tokenizer;
- PLE indexing/store;
- state.

Ma manca ancora una prova forte che i subsystem siano collegati **nell'ordine esatto del modello**.

Prima di investire molto codice in QSA/MoE, creare un integrated forward probe.

## Probe raccomandato

Primo target:

```text
embedding
  ↓
layer 0
  ↓
layer 1 / PLE injection se applicabile secondo artifact/layout reale
  ↓
layer 2
  ↓
layer 3 QSA
```

Il mapping layer va preso dal frozen config/reference reale, non da questa rappresentazione schematica.

## Dump richiesti

Per ogni stage:

```text
input hidden
norm output
GR read output
GDN/QSA input
GDN/QSA output
router/MoE input quando disponibile
GR write output
layer final hidden
```

## Obiettivo

Arrivare a un test:

```text
reference hidden after layer N
==
q38 hidden after layer N
```

per i primi 4 layer.

Questo riduce drasticamente il rischio di scoprire solo a M6 che:
- norm placement;
- GR placement;
- PLE injection;
- residual ordering

sono incompatibili tra loro.

---

# 4. P1 — QSA è ancora early-stage: congelare attentamente anche lo state "pending"

## Stato

L'attuale QSA sembra ancora principalmente composto da:

- weight validation;
- state descriptors;
- K/V storage;
- index storage;
- position counters.

Non è una criticità in sé: è normale a questo punto.

## Rischio da evitare

Con:

```text
compression ratio = 4
```

non basta necessariamente modellare solo:

```text
completed index groups
```

Bisogna verificare esplicitamente anche la semantica del:

```text
partial / pending current compression group
```

e la posizione corrente dentro quel gruppo.

Questo diventa particolarmente importante per:

- chunked prefill;
- multi-turn;
- speculative verification;
- rewind/replay.

## Required semantics freeze

Il futuro `docs/qwen_qsa_semantics.md` deve esplicitare:

```text
main K/V state
compressed/index state
current pending group
tokens already accumulated in pending group
when compression commits
causal tail semantics
position advancement
reset/restore behavior
```

## No guessing

Non dedurre la semantica solo da:

```text
budget = 2048
ratio = 4
```

Verificare upstream Qwen4Exp/Transformers/llama.cpp.

---

# 5. P1/P2 — Tokenizer nativo: migliorato, ma serve una parity suite più aggressiva

## Stato

Il tokenizer ora è nativo e non dipende più da:

- fork;
- Python runtime;
- Transformers nel path di produzione.

Questo finding della review precedente è quindi **risolto**.

## Rischio residuo

Il tokenizer contiene parsing/config/template logic custom.

Queste parti possono essere "quasi corrette" ma fallire su edge case.

## Ampliare i test

Aggiungere casi per:

### Unicode

- combining marks;
- NFC/NFD;
- emoji;
- CJK;
- RTL;
- caratteri fuori BMP.

### JSON / escaping

- `\"`
- `\\`
- newline escape;
- Unicode escaped;
- nested arrays/objects.

### Chat

- empty system;
- no system;
- multiple assistant turns;
- empty content;
- content arrays;
- structured markup;
- tool-style messages.

### Multimodal markers

Verificare già ora il riconoscimento corretto di:
- image token;
- video token;
- vision start/end;
- eventuali placeholder.

M10 implementerà il path vision, ma il tokenizer deve già preservare esattamente i token speciali.

## Gate

```text
native token IDs
==
frozen HF/Python token IDs
```

100% per tutta la suite.

---

# 6. P2 — Special token IDs hardcoded

## Problema

Il tokenizer contiene alcuni special token / ID hardcoded nel codice.

Per un runtime Qwen3.8-specifico questo non è necessariamente sbagliato.

Il rischio è un artifact/tokenizer revision mismatch.

## Correzione consigliata

All'avvio:

1. leggere gli ID dal tokenizer/config locale;
2. verificare che coincidano con gli expected Qwen3.8 fixed IDs;
3. fallire esplicitamente se divergono.

Quindi:

```text
hardcoded expected constant
+
runtime verification
```

va bene.

Solo:

```text
hardcoded constant senza verifica
```

è più fragile.

---

# 7. P2 — `expected_tensor_count()` è fragile

## Problema

Una logica equivalente a:

```c
full += (i % 4 == 3) ? 9 : 9;
```

è semanticamente identica a:

```c
full += 9;
```

e suggerisce che il conteggio tensor globale non distingue realmente QSA da GDN.

Non è necessariamente un bug immediato, ma il validator rischia di dare una falsa sicurezza.

## Raccomandazione

Preferire validazione semantica per layer type.

Esempio:

```text
if GDN layer:
    require exact GDN tensor set

if QSA layer:
    require exact QSA tensor set

if PLE layer:
    require exact PLE tensor set

for every layer:
    require exact GR + MoE tensor set
```

Il tensor count globale può restare come sanity check, ma non come prova primaria di completezza.

---

# 8. P2 — Non assumere una hot-set PLE utile prima di misurarla

## Rischio

La PLE usa lookup hash/sparse su una tabella enorme.

È possibile che la distribuzione degli accessi abbia locality limitata.

Se così fosse, una grande LRU cache potrebbe:

- consumare molti GiB;
- avere hit-rate basso;
- ridurre il memory headroom;
- complicare il runtime senza migliorare il throughput.

## M4S benchmark minimo

Confrontare:

```text
A. plain mmap
B. mmap + small quantized cache
C. larger cache
D. coalesced read planner
E. pinned staging
F. async prefetch
```

Misurare almeno:

```text
hit rate
bytes saved
memory cost
prefill t/s
decode t/s
p95 lookup latency
```

## Regola

Se la cache non produce un beneficio robusto, non mantenerla per principio.

---

# 9. Non-finding — GGUF q38

Il GGUF q38 non è una criticità.

Il formato è intenzionalmente:

```text
original Qwen3.8 tensors
        ↓
q38 converter
        ↓
q38 GGUF
        ↓
q38 runtime
```

`llama.cpp` viene usato come reference semantico.

Non è richiesto che il q38 GGUF abbia il tensor reorder/layout usato dal converter ufficiale llama.cpp.

## Raccomandazione opzionale

Documentare nel metadata:

```text
q38.weight_abi = "qwen38-original-layout-v1"
q38.quant_manifest_sha256 = "..."
q38.source_model_revision = "..."
```

per evitare ambiguità future.

---

# 10. Corrective gate consigliato prima di proseguire molto in M5

Suggerisco un gate:

```text
M4/M5-INTEGRATION-AUDIT
```

## Checklist

- [ ] Hidden-state golden indipendente per PLE prodotto.
- [ ] `hidden_before_ple` validato.
- [ ] PLE contribution vector validato.
- [ ] `hidden_after_ple` validato.
- [ ] `q38_forward_probe` esegue davvero la PLE injection.
- [ ] Chunked/non-chunked PLE hidden equivalence verificata.
- [ ] PLE mmap path conservato come baseline reference.
- [ ] Integrated forward probe layer 0–3 implementato o almeno avviato.
- [ ] QSA semantics document include pending compression-group state.
- [ ] Tokenizer extended parity suite verde.
- [ ] Tensor binder semantic validation rafforzata.
- [ ] M0–M4 regression completa verde.

---

# 11. Regole operative per l'agente

1. Non fermarti perché manca un probe, loader, API, golden o semantic document se è implementabile usando i reference disponibili: implementalo.
2. Fermati solo quando manca realmente un'informazione non ricavabile da:
   - checkpoint originale;
   - Qwen/Transformers;
   - llama.cpp;
   - documentazione congelata;
   - repository locale.
3. Non sostituire un golden indipendente con un confronto q38-vs-q38.
4. Ogni ottimizzazione deve conservare un path reference più semplice.
5. Non avanzare a M6 se l'integrazione dei primi layer non è stata verificata con hidden intermedi.
6. Non chiamare "SSD optimized" un path che è ancora solo mmap/page-cache.
7. Non inventare QSA state semantics da ratio/budget.
8. Preservare il GGUF q38 original-tensor ABI.
9. Continuare a usare il DGX Spark come unico target hardware.

---

# 12. Priorità consigliata da qui

Ordine suggerito:

```text
1. Chiudere davvero PLE injection golden/probe        P0
2. Integrated forward probe primi layer               P1
3. Freeze completa QSA semantics                      P1
4. Implementare QSA reference + CUDA naive            P1
5. Rafforzare tokenizer parity suite                   P1/P2
6. Rafforzare binder semantic validation               P2
7. Solo dopo: PLE SSD/prefetch optimization           P2/performance
8. Poi proseguire verso M6 MoE/end-to-end
```

---

# 13. Exit criterion

La review può essere considerata chiusa quando:

```text
M0 PASS
M1 PASS
M2 PASS
M3 PASS
M4 semantic PLE injection PASS
M4 residency baseline PASS
M5 QSA semantics frozen
first integrated layer probe PASS
```

A quel punto ha senso proseguire in modo aggressivo verso QSA/MoE e la prima vera inferenza end-to-end M6.
