Qwen3.8-Flash-Next / DGX Spark prototype 

# M2 — Implementation Specification 

_Strict binding, tokenizer parity, golden-vector infrastructure e primitive CUDA/oracle necessarie prima del graph Qwen._ 

**Regola metodologica:** reference-first, nessun guessing. Ogni divergenza va attribuita a tokenizer, quantizzazione, binding, kernel o stato prima di procedere. 

## **1. Scopo e Definition of Done** 

- Il GGUF Q2 prodotto da M1 viene trasformato da semplice inventory in `q38_weights` completamente bindato e validato per i tensori necessari a text-only target inference. 

- Tokenizer + chat-template producono gli stessi token IDs del reference congelato per corpus plain, chat, Unicode e tool-like markup scelto. 

- Esiste un protocollo golden-vector che permette di confrontare output intermedi a livello di tensor/layer/token senza affidarsi al testo generato. 

- Primitive CUDA elementari (dequant, RMSNorm, activation, matvec/matmul selezionati, reductions) dispongono di oracle scalar/F32 o reference dump. 

- Il runtime M2 non esegue ancora il modello intero. Deve però poter testare embedding -> primitive -> final norm/lm-head su vettori controllati. 

- Q2 resta formato runtime. I test includono Q4 candidate block decoders per non creare incompatibilità futura. 

- Peak memory M2 resta entro il gate M1 e nessun test crea mirror dequantizzato permanente del modello. 

## **2. Principio: tre reference distinti** 

|**Reference**|**Serve a**|**Non prova**|
|---|---|---|
|HF/Qwen BF16/FP32|Tokenizzazione, semantica graph, hidden/logits<br>golden|Che il nostro Q2 debba coincidere<br>numericamente col BF16|
|Reference quant-dequant|Isola errore introdotto dai pesi quantizzati|Correttezza del kernel CUDA/state machine|
|Runtime Q2|Implementazione reale Spark|Correttezza se confrontato solo via testo<br>plausibile|



**Regola:** Ogni test numerico deve specificare quale coppia confronta: BF16 vs quant-reference misura quant error; quant- <u>reference vs q38 misura implementation error.</u> 

## **3. Nuove strutture dati** 

```
typedef struct {
    q38_tensor *token_embd;
    q38_layer   layer[48];
    q38_ple_store ple;        /* handle, non materializzazione */
    q38_tensor *output_norm;
    q38_tensor *output;
} q38_weights;
```

```
typedef struct {
    uint32_t token_count;
    uint32_t *tokens;
    uint64_t prompt_hash;
    uint64_t model_hash;
} q38_token_batch;
```

```
typedef struct {
    char stage[64];
    int layer;
    int token_pos;
    enum q38_dtype dtype;
    uint64_t elements;
    uint64_t checksum;
} q38_golden_meta;
```

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **4. File e moduli M2** 

|**Modulo**|**Responsabilità**|**Vincolo**|
|---|---|---|
|q38_weights.c/.h|Strict tensor binding e validator semantico|Niente fuzzy fallback sui nomi|
|q38_tokenizer.c/.h|TokenizerGGUF +template text-only|Tokenparity prima delgraph|
|q38_quant.c/.h|Type metadata, block decoding/dispatch|Nessuna global dequant cache|
|q38_cuda_primitives.cu/.h|Primitive elementariSpark|Nolayersemantics qui|
|q38_oracle.c/.h|Scalar/F32 test-only|Non backend utente|
|q38_golden.c/.h|Dump/load/comparevector|Metadata+checksumobbligatori|
|tests/m2/*|Unit+numerical tests|Deterministici|
|tools/golden_export/*|Export dal reference|Revisionato con MODEL_BASELINE|



## **5. Commit plan M2** 

|**Commit**|**Scopo**|**File/area**|**Modifiche**|**Gate**|
|---|---|---|---|---|
|M2-C00|Golden format|q38_golden.*, schema<br>JSON/binario|roundtrip + checksum||
|M2-C01|Strict weight skeleton|q38_weights.*|required/optional/excluded<br>tensor sets; exact<br>type+shape|subset Q2 bind|
|M2-C02|Tokenizerparity|q38_tokenizer.* + vectors|token IDs exact su corpus|100% exact|
|M2-C03|Quant decode oracle|q38_quant.* +q38_oracle.*|Q2/Q4 block decode scalar|random/boundary blocks|
|M2-C04|CUDA dequant/vector ops|q38_cuda_primitives.cu|dequant/reductions/copy<br>kernels|CUDA vs scalar|
|M2-C05|RMSNorm + activations|CUDA + oracle|FP32 accumulation dove<br>richiesto dal reference|golden pass|
|M2-C06|Matvec/matmul dispatch|CUDA|Q2 expert-compatible +<br>Q8/BF16 core path|shape matrix pass|
|M2-C07|Embedding path|q38_forward_probe.c|token lookup -> hidden<br>dump|reference compare|
|M2-C08|Final norm + LM head<br>probe|q38_forward_probe.c|hidden golden -> logits|top-k + numeric compare|
|M2-C09|Full strict bind|q38_weights.*|48 layers + PLE handle; no<br>forward|all runtime tensors<br>accounted|
|M2-C10|Memory regression|telemetry|repeat bind/probe under<br>unified-memory budget|no regression|
|M2-C11|M2acceptance|makem2-acceptance|archive allartifacts|100% pass|



## **6. Strict binding contract** 

- Ogni tensor richiesto ha: canonical semantic role, exact shape, accepted quant/dtype set, expected layer kind e expected multiplicity. 

- Un tensor presente ma non consumato deve essere in `excluded_tensors.json`; un tensor sconosciuto è hard error. 

- Non usare alias multipli silenziosi. Se il mapping GGUF upstream cambia, aggiornare il mapping con un commit/ADR esplicito. 

- Il binder non alloca il payload: restituisce views/offset sul GGUF o handles storage-specific. 

- PLE resta `q38_ple_store`: non viene bindata come normale dense matrix CUDA. 

## **7. Tokenizer parity suite** 

|**Categoria**|**Casi minimi**|**PASS**|
|---|---|---|
|ASCII/plain|whitespace,newline, punctuation|exact token IDs|
|Unicode|accenti italiani, CJK, emoji, combining marks|exact token IDs|
|Longrepetitive|pattern ripetutie boundary tokenizer|exact token IDs|
|Chat template|system/user/assistant; empty system se<br>supportato|exact IDs inclusi special token|
|Tool-like markup|JSON/code/XML-like strings|exact IDs|
|BOS/EOS/special|single/paired specialtokencases|exact policy|



Il corpus deve essere versionato come testo + expected token IDs. Non usare solo `decode(encode(x)) == x`: può passare anche con segmentation diversa. 

## **8. Golden-vector protocol** 

```
goldens/<case>/
  manifest.json
  tokens.u32
```

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  embedding.bin
  layer00_input.bin
  ...
  final_hidden.bin
  logits_topk.json
manifest.json:
  model_revision
  tokenizer_revision
  source_dtype
  quant_manifest_hash
  prompt_sha256
  stage_name
  shape
  dtype
```

- Per M2 sono obbligatori almeno embedding e LM-head probe. 

- Per M3 il formato viene riusato per GR/GDN intermedi; non inventare un secondo sistema di dump. 

- Per tensori enormi salvare slice/checksum + statistiche e una selezione deterministica di elementi, non copie integrali inutili. 

## **9. Calibrazione delle tolleranze** 

Non fissare una tolleranza unica. Registrare almeno max-abs, mean-abs, RMS error, relative error con epsilon dichiarato, cosine similarity e — per logits — top-k overlap/rank. Le soglie di implementation error vengono ricavate confrontando due esecuzioni reference equivalenti e devono essere più strette della distanza BF16↔Q2. 

## **10. CUDA primitive matrix** 

|**Primitiva**|**Dtype/path M2**|**Test critico**|
|---|---|---|
|Q2 block decode|candidate selected in M1|first/last block, odd row strides|
|Q4blockdecode|future target|compatibility only|
|Q8/BF16 load|core sensitive tensors|alignment+vectorized tails|
|RMSNorm|BF16/F32accumulationasverified|zero/small/largemagnitude|
|SiLU / elementwise|FP32 reference|NaN/Inf behavior documented|
|Reduction|FP32accumulation whereneeded|determinismtolerance|
|Matvec|decode-oriented shapes|quant rows+non-multiple tails|
|Matmul|prefillprobe shapes|smallbatches before tuning|



## **11. Memory rules M2** 

- Workspace di ogni primitive viene allocato tramite arena/pool misurabile e rilasciabile. 

- Nessuna API `get_dequantized_tensor()` che possa accidentalmente mantenere decine di GB. 

- Quant block test usa buffer piccoli; embedding/LM-head probe non materializza il resto del modello. 

- Il binder full-model deve mantenere il peak entro il gate M1 (~108 GiB operativo) e ≥12 GiB di headroom. 

## **12. M2 test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M2-T01|Golden binary roundtrip|metadata/checksum identici|
|M2-T02|Strictmissing tensor|hard errorpreciso|
|M2-T03|Strict wrong shape/type|hard error prima del kernel|
|M2-T04|Tokenizercorpus|100% token IDs exact|
|M2-T05|Q2 scalar decode|stable oracle|
|M2-T06|Q2CUDAdecode|entroimplementationtolerancevs scalar|
|M2-T07|Q4 candidate decode|entro implementation tolerance|
|M2-T08|RMSNorm/activation|goldenpass|
|M2-T09|Matvec quant|random+adversarial shapes pass|
|M2-T10|Embedding probe|quant-reference agreement|
|M2-T11|LM-head probe|top-k/rank+numeric gates|
|M2-T12|Repeatedfullbind20x|noRSS/fd/arena growth|
|M2-T13|Memory peak|M1 gate preserved|
|M2-T14|CUDA fatal error policy|test harness exits, non continua con context<br>poisoned|



GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **13. Acceptance artifacts** 

```
artifacts/m2/
  tokenizer_vectors.json
  golden_format_version.json
  binding_report.json
  quant_q2_oracle.json
  quant_q4_oracle.json
  primitive_accuracy.json
  embedding_probe.json
  lm_head_probe.json
  memory.json
  checksums.txt
  acceptance.txt
```

## **14. UNKNOWN / stop conditions** 

- Il canonical tensor mapping qwen4_exp della revisione congelata va letto e registrato; non derivarlo dai nomi DeepSeek/GLM. 

- Dtype/accumulation esatta di ogni projection/norm va verificata nel reference prima del golden export. 

- Se un donor kernel ds4 passa dequant unit test ma assume semanticamente un layout expert diverso, non riusarlo in M3/M6. 

- M2 non è chiuso se tokenizer e tensor binding sono “quasi giusti”: devono essere esatti a livello discreto. 

## **15. Riferimenti** 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Qwen3.8-Flash-Next config:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**llama.cpp gated_delta_net_f32 oracle:** https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-et/et-kernels/src/ gated_delta_net_f32.c 

**llama.cpp recurrent checkpoint issue #22384:** https://github.com/ggml-org/llama.cpp/issues/22384 

**llama.cpp GDN cache-layout performance issue #20436:** https://github.com/ggml-org/llama.cpp/issues/20436 

**ds4 core donor:** https://github.com/antirez/ds4/blob/main/ds4.c 

**ds4 CUDA donor:** https://github.com/antirez/ds4/blob/main/ds4_cuda.cu 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

GB10 / 128 GB unified coherent memory / CUDA only / Q2 bring-up -> selective Q4 

