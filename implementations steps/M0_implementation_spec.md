Qwen3.8-Flash-Next / DGX Spark prototype 

# M0 — Implementation Specification 

_Fork-and-prune ds4 fino a un runtime skeleton Qwen3.8-Flash-Next specifico per DGX Spark._ 

**Regola metodologica:** ogni fatto architetturale non verificato resta marcato come UNKNOWN e diventa un task di verifica. Nessun comportamento viene dedotto per analogia con DeepSeek/GLM. 

## **1. Definition of Done** 

- Repository nuovo derivato da un commit ds4 congelato e registrato in BASELINE.md. 

- Build supportata: DGX Spark GB10 / Linux aarch64 / CUDA. Nessun Metal, ROCm o backend CPU esposto all'utente. 

- Nessun supporto DeepSeek, GLM, DSpark draft, MTP, vision, distributed pipeline o tensor parallel nel binary M0. 

- Un unico executable `q38` (nome provvisorio) capace di: platform probe, GGUF inspect, tensor inventory, memory-plan dry run. 

- Il processo rifiuta esplicitamente GPU non GB10/SM121 invece di degradare silenziosamente. 

- Nessuna allocazione o host-registration dell'intero GGUF durante `--inspect`/`--memory-plan`. 

- Test M0 ripetibili e report JSON di ambiente/memoria. 

## **2. Baseline hardware** 

|**Parametro**<br>**Valore operativo**|
|---|
|Target<br>NVIDIA DGX Spark|
|SoC<br>GB10 GraceBlackwell|
|CPU<br>ARM64, 20 core|
|Memoria<br>128 GB LPDDR5xcoherent unified system memory|
|Bandwidth<br>273 GB/s|
|GPU target<br>SM 12.1;verificare aruntimeilcompute capabilityreale|
|CUDA<br>Versione installata sullo Spark; salvare driver/runtime/toolkit nel report|
|**Nota:**I 128 GB sono un pool condiviso. Non trattare 128 GB come VRAM dedicata. Gli issue ds4 #293/#721 mostrano che<br>full host registration e transienti di load possono trasformare un file apparentemente compatibile in OOM.|



## **3. Strategia di fork** 

M0 non deve essere una sequenza di refactoring cosmetici. L'obiettivo è ridurre la superficie prima di introdurre Qwen. Il donor code deve essere identificabile ma il prodotto risultante non deve continuare a portarsi dietro decisioni di compatibilità ds4. 

```
git remote add ds4-upstream https://github.com/antirez/ds4.git
git fetch ds4-upstream
```

```
git checkout -b qwen38-spark-proto <PINNED_DS4_COMMIT>
```

```
# Commit 0 produce anche:
BASELINE.md
THIRD_PARTY_NOTES.md
```

## **4. Commit plan** 

|**Commit**|**Scopo**|**File principali**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M0-C00|Freeze donor|BASELINE.md,<br>LICENSE/notes|Registra commit ds4, data,<br>compiler/CUDA baseline.<br>Nessuncambiofunzionale.|tree clean; baseline<br>identificabile|
|M0-C01|Collapse build|Makefile|Elimina Darwin/Metal,<br>ROCm, cuda-generic, cpu<br>targets; conserva solo<br>`spark`, `test`, `clean`.<br>Riduce executable a `q38`<br>+test binaries.|make clean && make spark|
|M0-C02|Remove non-target<br>products|Makefile; ds4_server.c;<br>ds4_agent.c; ds4_web.c;<br>ds4_distributed.*; ds4_tp.*;|Rimuove<br>server/agent/distributed/TP<br>. CLI minimale è sufficiente|grep/link audit senza<br>symbol rimossi|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

<u>Qwen3.8-Flash-Next / DGX Spark prototype</u> 

|**Commit**|**Scopo**|**File principali**|**Modifica**|**Gate**|
|---|---|---|---|---|
|||ds4_gpu_mgpu.*; relativi<br>test|per bring-up.||
|M0-C03|Narrow public API|ds4.h -> q38.h; ds4_gpu.h<br>-> q38_cuda.h|Backend enum sparisce;<br>options ridotte a<br>model_path, context hint,<br>prefill hint, inspect,<br>memory-plan, verbose.|header compila da C e C+<br>+ smoke|
|M0-C04|Platform guard|q38_platform.c/.h;<br>q38_cuda.cu|Interroga CUDA device;<br>richiede un solo device e<br>GB10/SM121; registra<br>total/free memory. Rifiuto<br>esplicito altrove.|platform probe test|
|M0-C05|Preserve GGUF core,<br>isolate it|ds4.c -> q38_gguf.c/.h|Estrae parser GGUF v3,<br>mmap, metadata, tensor<br>directory e type-size logic;<br>elimina graph/model-family<br>binding.|inspect su GGUF piccolo|
|M0-C06|Memory telemetry|q38_memory.c/.h|RSS, MemAvailable,<br>mmap bytes, CUDA<br>free/total, internal<br>allocations, peak counters;<br>JSONsnapshot.|schema + monotonic peak<br>tests|
|M0-C07|Inspection CLI|q38.c|Implementa --platform, --<br>inspect, --list-tensors, --<br>memory-plan; nessun<br>inference path.|CLI golden output|
|M0-C08|Delete dead architecture<br>code|tree-wide|Elimina<br>DeepSeek/GLM/DSpark/M<br>TP/chat assumptions e<br>relativi test. Non lasciare<br>#ifdef morti.|ripgrep deny-list|
|M0-C09|M0 acceptance|tests/|Automatizza tutti i gate in<br>`make m0-acceptance`.|100% pass su Spark|



## **5. File-by-file: cosa salvare e cosa eliminare** 

|**File ds4**|**Decisione M0**|**Dettaglio**|
|---|---|---|
|Makefile|RISCRIVERE|Tenere soltanto Spark/CUDA; ds4 attuale<br>costruisce molteplici frontend/backend.|
|ds4.c|DONOR → SPLIT|Salvare GGUF parser, tensor descriptors,<br>tokenizer primitives utili e quant structs; non<br>trascinare graph monolitico.|
|ds4.h|RISCRIVERE|API minimale Qwen prototype; eliminare<br>backend enum, distributed, TP, MTP, steering,<br>Metal test hooks.|
|ds4_cuda.cu|DONOR|Conservare inizializzazione CUDA, alloc/tensor<br>primitives e quant kernels solo dopo audit.<br>Niente assunzione dicompatibilitàMoE.|
|ds4_gpu.h|RISCRIVERE|Esporre solo primitive CUDA che M0/M1 usano.<br>L'attuale header contiene semantica<br>attention/cache specifica.|
|ds4_ssd.c/.h|PARCHEGGIARE|<br>Tenere nel tree ma fuori dal link M0. Verrà<br>riammesso quando un memory plan lo richiede.|
|ds4_layer_pack.*|DELETE M0|Single GPU, single node: non serve placement<br>multi-GPU/CPU spill.|
|ds4_distributed.*/ ds4_tp.*|DELETE|Fuoriscope.|
|ds4_metal.m|DELETE|Fuori scope.|
|ds4_rocm*|DELETE|<br>Fuoriscope.|
|ds4_cli.c|DONOR → MINIMAL|Riutilizzare parsing base solo se costa meno di<br>riscriverlo.|
|ds4_server/agent/web/kvstore|DELETE M0|Non necessari al bring-up.|
|tests/*|REBUILD|<br>Salvare solo test generici realmente<br>indipendenti; creare nuova suite M0.|



## **6. Build contract** 

```
make spark
# deve produrre:
./q38
./tests/test_platform
./tests/test_gguf
./tests/test_memory
```

Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
# nessun auto-detection di backend:
```

```
./q38 --platform
./q38 --inspect model.gguf
./q38 --memory-plan model.gguf
```

Non hardcodare `-arch=sm_121` prima di aver verificato che la versione NVCC installata sullo Spark accetti la sintassi desiderata. Il configure/build deve stampare il valore scelto e fallire se non può generare codice adatto al device. 

## **7. Platform probe richiesto** 

```
struct q38_platform_info {
    int cuda_device_count;
    int cuda_device;
    int cc_major, cc_minor;
    uint64_t cuda_total_bytes;
    uint64_t cuda_free_bytes;
    uint64_t mem_total_bytes;
    uint64_t mem_available_bytes;
    char device_name[128];
    char driver_version[32];
    char runtime_version[32];
};
```

- M0 non deve chiamare cudaHostRegister sul mapping modello. 

- M0 non deve `cudaMalloc` in proporzione al file durante inspect. 

- Managed memory non è automaticamente la scelta corretta: va introdotta solo dietro una API esplicita e misurata. 

## **8. Memory telemetry schema** 

```
{
```

```
  "phase": "gguf_mapped",
  "rss_bytes": ...,
  "mem_available_bytes": ...,
```

```
  "cuda_free_bytes": ...,
  "cuda_total_bytes": ...,
  "model_file_bytes": ...,
  "model_mapped_bytes": ...,
```

```
  "cuda_allocated_bytes": ...,
```

```
  "peak_internal_bytes": ...
}
```

## **9. M0 test matrix** 

|**ID**|**Test**|**Procedura**|**PASS**|
|---|---|---|---|
|M0-T01|Build|clean tree -> make spark|success; no unwanted backend<br>objects|
|M0-T02|Platformaccept|runon DGXSpark|GB10/SM121accepted|
|M0-T03|Platform refuse|inject/mock wrong CC|clear refusal, no fallback|
|M0-T04|GGUF v3inspect|small valid GGUF|metadata/tensors enumerated|
|M0-T05|Bad GGUF|truncated/bad offsets|safe failure, no OOB|
|M0-T06|Inspect memory|large sparse/dummy GGUF|RSS does not scale as private<br>copy of payload|
|M0-T07|Nohost-register|trace/instrumentloader|zerowhole-filehostregistration|
|M0-T08|Deny-list|ripgrep tree|no operational<br>DeepSeek/GLM/Metal/ROCm/MTP<br>/DSpark symbols|
|M0-T09|Leak|100inspect open/closeloops|stableRSS/fds|
|M0-T10|JSON report|--platform-json|machine-readable deterministic<br>keys|



Target: GB10 / 128 GB unified coherent memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **10. Acceptance command** 

```
make clean
make spark
make m0-acceptance
```

```
./q38 --platform --json > artifacts/m0/platform.json
```

```
./q38 --inspect test.gguf --json > artifacts/m0/inspect.json
```

```
./q38 --memory-plan test.gguf --json > artifacts/m0/memory.json
```

## **11. Stop conditions** 

- Se `--inspect` richiede host-registration dell'intero mapping: fermarsi e correggere M0. 

- Se per mantenere un helper bisogna conservare un intero graph DeepSeek/GLM: riscrivere l'helper. 

- Se il build supporta hardware diverso da Spark 'per comodità': rimuoverlo in M0; la genericità potrà essere reintrodotta solo con un requisito esplicito futuro. 

## **12. UNKNOWN / verifiche obbligatorie** 

- Commit esatto ds4 donor: deve essere deciso e scritto in BASELINE.md. 

- Versioni effettive driver/CUDA/DGX OS: catturarle sul dispositivo reale. 

- Quali primitive CUDA ds4 sono semanticamente generiche: audit funzione-per-funzione; nessuna assunzione. 

## **13. Riferimenti** 

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

