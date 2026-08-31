Qwen3.8-Flash-Next / DGX Spark prototype 

# M10 — Implementation Specification 

_Restore multimodale completo: immagini e video, vision encoder, visual token injection, multimodal positions/mRoPE, quantizzazione e memory gates sul DGX Spark._ 

**Metodo:** nessun comportamento multimodale o I/O viene dedotto per analogia. Checkpoint/config e reference processor/model sono autoritativi; ogni optimization layer deve avere un fallback semplice e un test di equivalenza. 

## **1. Scope e Definition of Done** 

- Il runtime finale accetta testo, immagini e video secondo le capabilities del checkpoint Qwen3.8-Flash-Next; la modalità text-only M8/M9 rimane invariata e regression-tested. 

- Vision preprocessing, patchification, vision encoder, spatial merge, projection a hidden 2560, placeholder replacement e multimodal position IDs sono validati contro il reference processor/model. 

- Image e video path vengono testati separatamente e poi insieme a conversazioni multi-turn. 

- Il vision path ha una propria policy di quantizzazione: non eredita automaticamente Q2/Q4 dagli expert. 

- Nessun tensor vision viene più strippato dall’artifact release finale; il loader può però supportare un artifact text-only separato per debug. 

- Il memory planner M8 viene esteso per includere vision weights, pixel/patch buffers, visual-token activations e mRoPE workspace. 

- Qualsiasi ottimizzazione CUDA multimodale mantiene un reference path non fused per diagnosi. 

## **2. Parametri vision verificati nel checkpoint** 

|**Parametro**|**Checkpoint Qwen3.8-Flash-Next**|
|---|---|
|language_model_only|false|
|visiondepth|27|
|vision hidden size|1152|
|vision intermediate size|4304|
|vision heads|16|
|input channels|3|
|patch size|16|
|spatial merge size|2|
|temporal patch size|2|
|visionpositionalembeddings|2304|
|vision output hidden size|2560|
|visionactivation|gelu_pytorch_tanh|
|image token ID|248056|
|<br>video token ID|248057|
|vision start/end IDs|248053 / 248054|



**Config-specific override:** La classe Transformers generica Qwen4ExpVisionConfig documenta un default out_hidden_size diverso, ma il config del checkpoint Qwen3.8-Flash-Next dichiara 2560. Il validator usa sempre il valore del checkpoint <u>congelato, mai il default della classe.</u> 

## **3. Pipeline multimodale da implementare** 

```
media input
```

- `-> decoder / frame sampler` 

- `-> reference-compatible resize/normalize` 

- `-> patchify / temporal grouping` 

- `-> vision encoder (27 layers)` 

- `-> spatial merge (2x2)` 

- `-> projection/output hidden = 2560` 

- `-> replace image/video placeholders in text embedding stream` 

- `-> build mm token types + multimodal positions` 

- `-> text decoder (existing M8/M9 target)` 

Target: GB10 / 128 GB coherent unified memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **4. Moduli nuovi** 

|**Modulo**|**Responsabilità**|
|---|---|
|q38_media.*|Input image/video metadata, decode abstraction, bounds and ownership.|
|q38_vision_preprocess.*|Resize/normalize/patchify/grid_thw; exactreferencematching.|
|q38_vision_weights.*|Strict bind/quant audit dei tensor vision.|
|q38_vision_ref.*|CPU/scalar or small-reference probes for patch/position math; non<br>production backend.|
|q38_vision_cuda.cu|VisionencoderCUDA,inizialmentenon fused.|
|q38_vision_merge.cu|Spatial merge+projection to 2560.|
|q38_mm_tokens.*|Placeholderspans,image/video token replacement, tokentypes.|
|q38_mrope.*|Multimodal position construction and rotary application verified from<br>reference.|
|q38_mm_session.*|Media-aware session metadata; no duplication of immutable media<br>unlessrequired.|



## **5. Commit plan** 

|**Commit**|**Scopo**|**File/artefatti**|**Modifica**|**Gate**|
|---|---|---|---|---|
|M10-C00|Freeze multimodal<br>reference|docs/<br>qwen_mm_semantics.md|Pin processor/model<br>revision; document exact<br>preprocessing, grid,<br>placeholder, mRoPE and<br>fusion semantics.|no blocking UNKNOWN|
|M10-C01|Restore vision inventory|q38_inventory; quant<br>manifest|Reintroduce all vision<br>tensors stripped in M1 and<br>classify them.|100% accounted|
|M10-C02|Strict vision binder|q38_vision_weights.*|Exact<br>names/shapes/dtypes; no<br>default assumptions.|bind pass|
|M10-C03|Image preprocessing<br>oracle|q38_vision_preprocess.*;<br>tests|<br>Image<br>resize/normalize/patch<br>order/grid_thw exact vs<br>processor.|pixel/patch goldens|
|M10-C04|Video preprocessing oracle|same + video tests|Frame sampling/grouping,<br>temporal patching,<br>videogridthwexact.|video goldens|
|M10-C05|Patch embedding CUDA|q38_vision_cuda.cu|__<br>First projection/embedding<br>stage, debug dumps.|reference hidden pass|
|M10-C06|27-layer vision encoder|q38_vision_cuda.cu|Attention+MLP+norm<br>stages, initially<br>straightforward.|per-layer probes|
|M10-C07|Spatial merge/project|q38_vision_merge.cu|Merge 2x2 and output<br>hidden 2560.|merged hidden pass|
|M10-C08|Placeholder injection|q38_mm_tokens.*|Replace visual placeholder<br>positions with visual<br>embeddings exactly.|span exactness|
|M10-C09|Multimodal<br>positions/mRoPE|q38_mrope.*|Implement<br>image/video/text positional<br>axes/interleaving from<br>reference.|position+Q/K goldens|
|M10-C10|Image end-to-end|q38_forward_mm.*|One image + text through<br>fullQ4target.|logit/token goldens|
|M10-C11|Video end-to-end|q38_forward_mm.*|Short video+text.|logit/token goldens|
|M10-C12|Multi-image/mixed media|tests/test_mm_mix.*|Multiple images,<br>image+video, multiple<br>media turns.|no span/state corruption|
|M10-C13|Vision quant sensitivity|tools/<br>q38_mm_quant_sensitivity.<br>py|BF16/Q8/Q6/Q4 candidate<br>by vision tensor class.|quality+memory report|
|M10-C14|GB10 memory/perf pass|scripts/mm_profile.sh|Pixel budget, visual token<br>count, peak memory, prefill<br>t/s.|gates pass|
|M10-C15|Multimodal soak/server|tests/scripts|Repeated mixed media<br>through M9<br>worker/supervisor.|no leak/crash|
|M10-C16|Full regression|make m0...m10-<br>acceptance|Text-only must remain<br>identical; multimodal suite<br>passes.|100% pass|



Target: GB10 / 128 GB coherent unified memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

## **6. Preprocessing: no guessing** 

- Non fissare normalization constants, resize policy, min/max pixels o frame sampling da ricordi della famiglia Qwen-VL: estrarli dal processor/config congelato. 

- Verificare l’ordine fisico delle patch, non solo la shape finale. 

- image_grid_thw e video_grid_thw sono semantic state del preprocessing e devono coincidere esattamente col reference. 

- Il numero di visual tokens dopo spatial merge deve essere derivato dal reference per ogni media item e confrontato con gli span placeholder. 

## **7. Multimodal positions / mRoPE** 

Il text config dichiara mrope_interleaved=true e mrope_section=[11,11,10]. Questo non è sufficiente per ricostruire le position IDs multimodali. M10 congela e replica la funzione reference che costruisce gli assi temporale/spaziali e il mapping nel decoder. 

```
goldens per media item:
  input token IDs
  mm_token_type_ids
  image_grid_thw / video_grid_thw
  position IDs for every decoder token
  rotary sections/interleaving
  Q/K before and after rotary at selected layers
```

## **8. Vision quantization policy** 

|**Classe vision**|**Bootstrap**|**Candidate finale**|**Nota**|
|---|---|---|---|
|Patch embedding|Q8/BF16|Q8/Q6/Q4 ablation|Input-sensitive.|
|<br>VisionattentionQKV/O|Q8/BF16|<br>Q6/Q4ablation|Per-layersensitivity.|
|Vision MLP|Q8/BF16|Q4/Q6 candidate|Larger weight mass; likely quant<br>target.|
|Vision norms|BF16|BF16|Small and sensitive.|
|Merge/project to 2560|Q8/BF16|Q8/Q6|Directly enters text decoder; high<br>caution.|
|Positionembeddings|BF16/Q8|BF16/Q8|Low footprint.|



**Policy:** Non useremo Q2 vision come default. Il text model può restare Q4 selettivo mentre il vision tower usa una ricetta <u>diversa; il memory solver decide se serve scendere ulteriormente.</u> 

## **9. Test matrix** 

|**ID**|**Test**|**PASS**|
|---|---|---|
|M10-T01|Vision tensor inventory|100% classified/bound|
|M10-T02|Image preprocessing|pixel/patch/grid exactvsreference|
|M10-T03|Video preprocessing|frame/patch/grid exact|
|M10-T04|Vision layerprobes|withincalibratednumeric/quant tolerance|
|M10-T05|Spatial merge/project|hidden 2560 match|
|M10-T06|Placeholderspans|exact positions/counts|
|M10-T07|Multimodal position IDs|exact integers|
|M10-T08|mRoPEQ/K|withintolerance|
|M10-T09|Image+text logits|reference-compatible|
|M10-T10|Video+textlogits|reference-compatible|
|M10-T11|Multi-image|no corruption|
|M10-T12|Mixedimage/video|no corruption|
|M10-T13|Text-only regression|bit/within established tolerance identical to M9|
|M10-T14|Memory peak|withinSpark release gate|
|M10-T15|Server soak|no crash/leak/state contamination|



## **10. Acceptance artifacts** 

```
artifacts/m10/
  qwen_mm_semantics.md
  vision_inventory.json
  image_preprocess_goldens.json
  video_preprocess_goldens.json
```

Target: GB10 / 128 GB coherent unified memory / CUDA only 

Qwen3.8-Flash-Next / DGX Spark prototype 

```
  vision_layer_goldens.json
  merge_project_goldens.json
  mm_position_goldens.json
  image_e2e.json
  video_e2e.json
  mixed_media.json
  vision_quant_sensitivity.json
  mm_memory_profile.json
  mm_bench.json
  mm_soak.json
  text_regression.txt
  acceptance.txt
```

## **11. Stop conditions** 

- Se preprocessing o mRoPE non coincidono col reference, non tentare di compensare nel decoder. 

- Se il vision path richiede abbassare la precisione del text target Q4 già accettato, prima ottimizzare vision residency/quant; il target text non viene degradato automaticamente. 

- Se video eccede il memory budget, introdurre un media token/pixel budget esplicito e documentato, non truncation silenziosa. 

- Se una ottimizzazione CUDA multimodale passa solo end-to-end ma diverge negli intermediate goldens, non accettarla. 

## **12. UNKNOWN** 

- Exact image normalization/resize/pixel budget della revision processor congelata. 

- Exact video sampling/fps/frame budget della revision processor congelata. 

- Exact multimodal position construction and mRoPE interleaving for this checkpoint. 

- Exact vision tensor naming/layout del GGUF converter scelto. 

- Best vision quant recipe on GB10. 

## **13. Riferimenti** 

**Official Qwen3.8-Flash-Next config:** https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json 

**Transformers Qwen4-Exp model documentation:** https://huggingface.co/docs/transformers/model_doc/qwen4_exp 

**Qwen3.8-Flash-Next technical overview:** https://qwen.ai/blog?id=qwen3.8-flash-next 

**Qwen3.8-Flash-Next repository:** https://github.com/QwenLM/Qwen3.8-Flash-Next 

**Hugging Face Transformers vision utilities:** https://github.com/huggingface/transformers/blob/main/src/transformers/ vision_utils.py 

**NVIDIA DGX Spark:** https://www.nvidia.com/products/workstations/dgx-spark/ 

Target: GB10 / 128 GB coherent unified memory / CUDA only 

