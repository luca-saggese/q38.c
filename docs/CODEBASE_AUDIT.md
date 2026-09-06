# q38 Codebase Audit (CLEAN-01)

**Snapshot:** 2026-09-06  
**Scope:** static inventory only; no production refactor or model inference was
performed for this audit. The canonical benchmark process was left untouched.

This document is the baseline for the aggressive consolidation work. It
records the current reachable production chain, the symbols exported by the
existing `q38` binary, source-file roles, and cleanup candidates. A symbol
being referenced by an archived or obsolete test is not sufficient to retain
it in production.

## Target contract

- Model: Qwen3.8-Flash-Next / `qwen4_exp`
- Hardware: DGX Spark / GB10 / SM121
- OS/toolchain: Linux ARM64 / CUDA
- Production quantization: Q2 today; Q4 only where the current runtime already
  requires it
- PLE policy: permanently SSD/file-backed
- Required public behavior: canonical correctness, Reference 0 benchmarks, and
  the CLI/session runtime
- Explicitly out of scope for the final production tree: CPU, Metal, ROCm,
  other model families, generic backend plugins, and completed milestone
  scaffolding

## Observed production call graph

The current CLI and canonical benchmark paths are:

```text
q38 main
  -> q38_runtime_init
       -> q38_gguf_open
       -> q38_tokenizer_init
       -> q38_weights_bind_subset
       -> q38_forward_cuda_context_create
       -> q38_forward_cuda_enable_all_non_ple_residency
       -> q38_forward_cuda_prepare_lm_head
  -> q38_session_create
  -> q38_session_prefill_reference
     or q38_session_prefill_chunked
       -> q38_forward_full*
  -> q38_session_eval_timed
       -> q38_decode_step_with_matrix_batch_moe_layer_backend_timed
          -> q38_forward_full_with_matrix_batch_moe_layer_backend
             -> layer semantics in q38_forward.c
             -> q38_forward_cuda_matvec_backend
             -> q38_forward_cuda_matrix_backend
             -> q38_forward_cuda_matrix_batch_backend
             -> q38_forward_cuda_expert_backend
             -> q38_forward_cuda_moe_layer_q2_backend
             -> q38_forward_cuda_qsa_qkv_backend
  -> q38_session_destroy
  -> q38_runtime_destroy
```

This confirms the requested architectural problem: backend selection is still
passed through the decode and forward layers as callback arguments. The final
runtime should install these operations once in the GB10 runtime context and
expose one token-forward entry point.

The current session implementation also contains two compiled personalities:
`q38_session.c` is built once normally as `q38_session.o` and once with
`-DQ38_SESSION_RUNTIME` as `q38_runtime.o`. This is a CLEAN-08 consolidation
candidate, not a reason to preserve both APIs.

## Build and linkage inventory

The current `Makefile` contains a production target plus the historical M0
through M9 acceptance graph. The `q38` target currently links production
objects together with reference/profiling support:

```text
q38.o q38_gguf.o q38_memory.o q38_platform.o q38_tokenizer.o
q38_decode.o q38_forward.o q38_ple_prefetch.o q38_moe.o q38_weights.o
q38_model_config.o q38_ple.o q38_qsa.o q38_state.o q38_session.o
q38_runtime.o q38_quant.o q38_ple_ref.o q38_gdn_ref.o q38_gr_ref.o
q38_replay.o q38_profile.o q38_residency.o
q38_cuda.o q38_forward_cuda.o q38_qsa_cuda.o q38_cuda_primitives.o
q38_gdn.o q38_moe_cuda.o q38_cuda_timing.o q38_profile_cuda.o
q38_topk_cuda.o
```

Immediate consequences:

| Finding | Classification | Cleanup action |
|---|---|---|
| `q38_runtime.o` and `q38_session.o` compile the same source twice | DUPLICATE | Merge runtime/session implementation and remove the preprocessor personality |
| `q38_ple_ref.o`, `q38_gdn_ref.o`, and `q38_gr_ref.o` are linked into `q38` | REFERENCE_ORACLE leakage | Move reference implementations under `tests/reference/` and keep them out of `q38` |
| `q38_replay.o` and `q38_profile.o` are linked into `q38` | PROD_SUPPORT / diagnostic leakage | Retain only if the canonical CLI directly requires them; otherwise move to test tooling |
| M0-M9 targets remain in the default Makefile | LEGACY_UNUSED build surface | Replace with the five final targets requested by the cleanup |
| `tools/q38_quantize` depends on `to_be_deleted/gguf-tools/quants.c` | LEGACY_UNUSED donor dependency | Extract only required code or remove the target |

## Source-file inventory

LOC values are from the audit snapshot and include source files, not generated
objects or binaries.

### Production candidate files

| File | LOC | Current role | Symbols / callers | Decision |
|---|---:|---|---|---|
| `q38.c` | 1515 | CLI, inspection, generation, trace plumbing | `main`, CLI helpers; calls runtime/session | KEEP, later simplify CLI |
| `q38.h` | 85 | public core constants/types | included by CLI/platform/memory | KEEP |
| `q38_gguf.c/h` | 484/89 | GGUF mapping and metadata | runtime, weights, tests | KEEP |
| `q38_tokenizer.c/h` | 395/31 | tokenizer runtime | CLI/session/tests | KEEP |
| `q38_weights.c/h` | 703/117 | weight binding and validation | runtime/forward/tests | KEEP; absorb MoE/QSA validation where appropriate |
| `q38_model_config.c/h` | 119/63 | fixed model configuration | weights/tests | KEEP |
| `q38_session.c/h` | 402/101 | runtime and session APIs | CLI/canonical benchmark/tests | KEEP, consolidate into one runtime file |
| `q38_session_types.h` | 29 | shared session types | session/forward | KEEP or merge into runtime header |
| `q38_state.c/h` | 257/113 | recurrent/QSA/PLE state storage | forward/session/reference tests | KEEP only state required by production |
| `q38_forward.c/h` | 2307/352 | layer semantics and callback maze | session, CLI, many historical tests | KEEP, collapse to one token graph |
| `q38_forward_cuda.cu/h` | 1594/175 | CUDA context, residency, matrix/QSA/MoE dispatch | session/forward/CLI | KEEP, simplify as CUDA runtime island |
| `q38_cuda.cu/h` | 106/45 | platform CUDA probe | CLI/platform/tests | KEEP or merge into runtime CUDA support |
| `q38_cuda_primitives.cu/h` | 334/42 | CUDA math primitives | CUDA forward/GDN/MoE/tests | KEEP |
| `q38_gdn.cu/h` | 499/122 | production GDN CUDA execution | forward CUDA/tests | KEEP |
| `q38_gr.cu/h` | 116/27 | production GR execution | forward/tests | KEEP |
| `q38_moe_cuda.cu/h` | 449/84 | production MoE CUDA execution | forward CUDA/tests | KEEP, consolidate and rename production kernels |
| `q38_ple.c/h` | 232/83 | file-backed PLE store and gather policy | weights/forward/session/tests | KEEP as an independent subsystem |
| `q38_ple_prefetch.c/h` | 425/77 | asynchronous PLE scheduler | state/forward/tests | KEEP if production scheduler remains required |
| `q38_ple_cuda.cu/h` | 259/42 | CUDA PLE helpers | current linkage/tests | AUDIT before KEEP; merge if no direct production caller |
| `q38_ple_stage.cu/h` | 117/42 | PLE staging helper | current build rules | AUDIT; likely merge/delete after call graph proof |
| `q38_residency.c/h` | 118/59 | residency accounting helpers | weights/tests | KEEP only for canonical runtime counters |
| `q38_quant.c/h` | 119/46 | host quant/dequant helpers | forward/weights/tests | KEEP only for formats used by Q2/Q4 runtime |
| `q38_topk_cuda.cu/h` | 98/27 | CUDA greedy/top-k support | forward CUDA/tests | KEEP canonical greedy path |

### Reference, diagnostic, or historical candidates

| File | LOC | Current role | Decision |
|---|---:|---|---|
| `q38_gdn_ref.c/h` | 106/60 | CPU/reference GDN | MOVE to `tests/reference/`; do not link into `q38` |
| `q38_gr_ref.c/h` | 81/35 | CPU/reference GR | MOVE to `tests/reference/`; do not link into `q38` |
| `q38_moe_ref.c/h` | 205/69 | CPU/reference MoE | MOVE to `tests/reference/`; do not link into `q38` |
| `q38_qsa_ref.c/h` | 199/46 | CPU/reference QSA | MOVE to `tests/reference/` if canonical tests still require it |
| `q38_ple_ref.c/h` | 202/70 | CPU/reference PLE | MOVE to `tests/reference/` if retained |
| `q38_rope_ref.c/h` | 70/33 | reference RoPE | MOVE to `tests/reference/` |
| `q38_topk_ref.c/h` | 36/19 | reference top-k | MOVE to `tests/reference/` |
| `q38_oracle.c/h` | 44/30 | oracle helpers | MOVE to `tests/reference/` |
| `q38_golden.c/h` | 159/57 | milestone golden format | MOVE to `tests/reference/` or DELETE after canonical fixtures are checked |
| `q38_replay.c/h` | 286/60 | replay/snapshot diagnostics | MOVE out of production unless a live CLI requirement is proven |
| `q38_profile.c` / `q38_profile.h` | 271/139 | broad profiler callbacks and JSON | Reduce to lightweight canonical counters or move to diagnostics |
| `q38_profile_cuda.cu` | 100 | CUDA profiler wrappers | MOVE to diagnostics if not required by Reference 0 |
| `q38_cuda_timing.cu/h` | 54/35 | generic CUDA timing wrapper | Merge into canonical perf support or delete |
| `q38_memory.c/h` | 104/49 | legacy memory tracker | AUDIT; likely merge with runtime memory metadata |
| `q38_platform.c/h` | 89/39 | platform probing | KEEP only CLI metadata; no backend abstraction |
| `q38_moe.c/h` | 88/38 | host MoE binding/slicing | MERGE binding/validation into weights; remove standalone file if callers permit |
| `q38_qsa.c/h` | 142/61 | host dynamic QSA cache/state | KEEP only if it is production canonical; otherwise merge state or move to reference |
| `q38_qsa_candidate.cu/h` | 64/25 | optional candidate QSA plugin | RENAME/merge only if it is the production path; otherwise delete plugin surface |
| `ds4_ssd.c/h` | 210/36 | donor SSD/cache planning API | DELETE or rewrite as `q38_ssd.c/h`; no `ds4_*` symbols may remain |

### CUDA source relocation inventory

The requested final layout places all CUDA source under `cuda/`. Root CUDA
files currently include:

```text
q38_cuda.cu
q38_cuda_primitives.cu
q38_cuda_timing.cu
q38_forward_cuda.cu
q38_gdn.cu
q38_gr.cu
q38_moe_cuda.cu
q38_ple_cuda.cu
q38_ple_stage.cu
q38_profile_cuda.cu
q38_qsa_candidate.cu
q38_qsa_cuda.cu
q38_topk_cuda.cu
```

The existing `cuda/mmq/` tree is a donor/kernel subtree and contains
`ds4_*`-named files. It must be isolated, renamed, or removed as part of the
donor cleanup; moving files without removing the donor namespace would not
satisfy the final gate.

## Current global symbol inventory

This is the `nm -C --defined-only q38` snapshot from the existing production
binary. CRT/linker symbols are omitted below; every `q38_*` global symbol in
the binary is listed. The binary itself predates this cleanup and therefore
also demonstrates the current leakage of reference, replay, profiling, and
legacy wrapper APIs.

### CLI, model, tokenizer, and runtime support

```text
main
q38_gguf_type_name q38_gguf_type_nbytes q38_gguf_tensor_data
q38_gguf_open q38_gguf_close q38_gguf_find_kv q38_gguf_get_string
q38_gguf_get_u32 q38_gguf_get_u64 q38_gguf_get_bool
q38_tokenizer_init q38_tokenizer_destroy q38_tokenizer_encode
q38_tokenizer_encode_chat_json q38_tokenizer_decode q38_token_batch_free
q38_tokenizer_verify_specials
q38_model_config_default q38_model_config_validate
q38_weights_bind_subset q38_weights_release q38_weights_validate_bound
q38_expert_store_init_uniform q38_expert_store_init_mixed
q38_half_to_float q38_quant_dequantize_row
q38_runtime_init q38_runtime_destroy q38_session_create q38_session_reset
q38_session_destroy q38_session_eval q38_session_eval_timed
q38_session_prefill q38_session_prefill_reference q38_session_prefill_chunked
q38_session_emit q38_session_stream_token q38_session_eos_token
q38_session_context_remaining q38_session_state_init
q38_session_state_validate q38_ngram_history_reset
q38_ngram_history_append q38_ngram_history_context
q38_state_alloc q38_state_reset q38_state_free
q38_state_recurrent_slot q38_state_conv_history_slot
q38_gdn_slot_for_layer
```

**Classification:** CLI/runtime symbols are `PUBLIC_API`, `PROD_HOT`, or
`PROD_SUPPORT`. `q38_session_prefill_reference`, `q38_session_eval`, and the
ngram helpers are duplicate/compatibility surfaces to reassess after the
canonical path is frozen.

### Forward and decode callback maze

```text
q38_decode q38_decode_stream q38_decode_stream_with_backend
q38_decode_stream_with_matrix_backend q38_decode_emit_trace
q38_decode_step_with_matrix_moe_layer_backend
q38_decode_step_with_matrix_moe_layer_backend_timed
q38_decode_step_with_matrix_batch_moe_layer_backend_timed
q38_forward_matrix_from_tensor q38_forward_qsa_state_init
q38_forward_qsa_ref q38_forward_qsa_ref_timed
q38_forward_state_init q38_forward_state_reset q38_forward_state_destroy
q38_forward_state_prefetch_ple q38_forward_state_wait_ple
q38_forward_state_get_ple_prefetch_stats
q38_forward_full q38_forward_full_with_backend
q38_forward_full_with_matrix_backend
q38_forward_full_with_matrix_moe_layer_backend
q38_forward_full_with_matrix_batch_moe_layer_backend
q38_forward_cuda_matvec_backend q38_forward_cuda_matrix_backend
q38_forward_cuda_matrix_batch_backend q38_forward_cuda_expert_backend
q38_forward_cuda_moe_layer_q2_backend q38_forward_cuda_qsa_qkv_backend
q38_forward_cuda_set_qsa_candidate
```

**Classification:** `q38_forward_full*`, `q38_decode_step_with_*`, and the
`*_backend` callback typedefs are `DUPLICATE` / `PROD_HOT` transitional
surfaces. The target replacement is one runtime-installed
`q38_forward_token()` operation with no per-token callback list.

### CUDA execution and model residency

```text
q38_cuda_init q38_cuda_cleanup q38_cuda_probe
q38_cuda_get_shared_memory_info q38_cuda_dequantize_row
q38_cuda_q2_matvec q38_cuda_bf16_matvec q38_cuda_bf16_matvec_configured
q38_cuda_rms_norm q38_cuda_silu
q38_forward_cuda_context_create q38_forward_cuda_context_destroy
q38_forward_cuda_enable_all_non_ple_residency
q38_forward_cuda_prepare_lm_head q38_forward_cuda_get_residency_stats
q38_forward_cuda_stream q38_forward_cuda_set_allocation_observer
q38_forward_cuda_set_telemetry_observer
q38_forward_cuda_set_residency_progress_observer
q38_forward_cuda_set_stage_context
q38_forward_cuda_record_route q38_forward_cuda_get_expert_layer_calls
q38_forward_cuda_greedy_argmax
q38_qsa_cuda_project_main q38_qsa_cuda_apply_rope
q38_qsa_cuda_index_scores q38_qsa_cuda_gather_attention
q38_topk_cuda q38_argmax_cuda
q38_cuda_gdn_project q38_cuda_gdn_conv q38_cuda_gdn_conv_update
q38_cuda_gdn_conv_silu q38_cuda_gdn_conv_silu_fused
q38_cuda_gdn_split_qkv q38_cuda_gdn_repeat_key_heads
q38_cuda_gdn_recurrence q38_cuda_gdn_recurrence_reset
q38_cuda_timing_init q38_cuda_timing_destroy q38_cuda_timing_begin
q38_cuda_timing_end q38_cuda_timing_record_launch
q38_cuda_timing_record_allocation
```

**Classification:** CUDA compute is `PROD_HOT`; observers, timing wrappers,
stage context, route counters, and plugin setters are `PROD_SUPPORT` or
diagnostic candidates. The final public runtime should not export backend
selection setters.

### MoE

```text
q38_moe_weights_validate q38_moe_bind_layer q38_moe_expert_slice
q38_moe_cuda_router q38_moe_cuda_route
q38_moe_cuda_expert_q2_workspace q38_moe_cuda_expert_q2
q38_moe_cuda_q2_gate_up q38_moe_cuda_q2_gate_up_candidate
q38_moe_cuda_q2_down
q38_moe_cuda_expert_q4_workspace q38_moe_cuda_expert_q4
q38_moe_cuda_q4_gate_up q38_moe_cuda_q4_down
q38_moe_cuda_accumulate_weighted q38_moe_cuda_shared_f32
```

**Classification:** CUDA MoE symbols are `PROD_HOT`. The host binding/slice
symbols are `PROD_SUPPORT` candidates for merge into weights. The
`q2_gate_up_candidate` name is a direct CLEAN-06 rename/delete candidate once
Reference 0 dispatch is confirmed.

### PLE

```text
q38_ple_choose_gather_mode q38_ple_store_bind q38_ple_store_bind_gguf
q38_ple_store_row_range q38_ple_store_read_row q38_ple_store_read_rows
q38_ple_store_read_rows_mode q38_ple_prefetch_rows
q38_ple_scheduler_create q38_ple_scheduler_destroy
q38_ple_scheduler_reset q38_ple_scheduler_submit
q38_ple_scheduler_wait q38_ple_scheduler_get_stats
q38_ple_hash_config_validate q38_ple_ngram_ids_ref
q38_ple_decode_row_ref q38_ple_grouped_norm_inplace q38_ple_forward_ref
```

**Classification:** store/scheduler symbols are `PROD_HOT` or
`PROD_SUPPORT`; `_ref` symbols are `REFERENCE_ORACLE` and must move out of
the production link. The direct/parallel gather policy requires a separate
call-graph check before removing either mode.

### Reference, replay, and profiler leakage

```text
q38_gdn_ref_state_elements q38_gdn_ref_reset q38_gdn_ref_step
q38_gdn_ref_run q38_gdn_ref_repeat_key_heads
q38_gr_read q38_gr_write q38_gr_collapse
q38_qsa_weights_validate q38_qsa_state_init q38_qsa_state_append
q38_qsa_state_reset q38_qsa_state_destroy q38_qsa_state_clone
q38_profile_init q38_profile_destroy q38_profile_get
q38_profile_record_launch q38_profile_record_sync
q38_profile_record_allocation q38_profile_record_runtime_allocation
q38_profile_set_token_count q38_profile_record_cuda_telemetry
q38_profile_json q38_profile_qsa_trace q38_profile_boundary_trace
q38_profile_stage_trace q38_profile_cuda_init q38_profile_cuda_destroy
q38_profile_cuda_begin q38_profile_cuda_end q38_profile_nvtx_push
q38_profile_nvtx_pop
q38_replay_snapshot_save q38_replay_snapshot_load
q38_replay_restore_and_replay q38_replay_trace_open
q38_replay_trace_close q38_replay_boundary_trace q38_replay_stage_trace
q38_memory_tracker_init q38_memory_track_alloc q38_memory_track_free
q38_memory_capture q38_memory_snapshot_json
q38_model_residency_init q38_model_residency_destroy
q38_model_residency_reset q38_model_residency_account_tensor
q38_resident_arena_init q38_resident_arena_reserve q38_resident_arena_reset
```

**Classification:** reference/oracle and replay symbols are
`REFERENCE_ORACLE` unless a canonical test proves they are required. Full
state clone/snapshot and invasive profile APIs are not production hot-path
requirements. Residency counters required by Reference 0 remain
`PROD_SUPPORT`, but diagnostic serialization should not be linked into the
minimal runtime.

## Callers and address-taken uses

The following callback edges are confirmed by source inspection:

| Callback/API | Production use | Test/diagnostic use | Initial classification |
|---|---|---|---|
| `q38_forward_cuda_matvec_backend` | installed by `q38_session_eval_timed` | probes and profile runners | PROD_HOT, then make implicit |
| `q38_forward_cuda_matrix_backend` | installed by session/decode wrappers | profile runners | PROD_HOT, then make implicit |
| `q38_forward_cuda_matrix_batch_backend` | installed by session/decode wrappers and chunked prefill | canonical benchmark | PROD_HOT, then make implicit |
| `q38_forward_cuda_expert_backend` | installed by session/decode wrappers | worker/probes | PROD_HOT, then make implicit |
| `q38_forward_cuda_moe_layer_q2_backend` | installed by session/decode wrappers | worker/cold-warm tests | PROD_HOT, then make implicit |
| `q38_forward_cuda_qsa_qkv_backend` | installed by `q38_session_eval_timed` | QSA attribution/direct tests | PROD_HOT, then make implicit |
| `q38_forward_cuda_set_qsa_candidate` | not installed by canonical runtime | M7 worker/cold-warm | TEST_ONLY / delete plugin setter |
| stage/telemetry observers | optional session diagnostics | canonical benchmark/profilers | PROD_SUPPORT only as lightweight counters |
| `q38_forward_full` | direct old probes and M8 tests | many historical tests | REFERENCE_ORACLE / TEST_ONLY unless a live caller remains |

Address-taken callback usage is therefore a first-class reason that simple
textual caller search is insufficient. The CLEAN-04/05 implementation must
replace these edges before deleting the old declarations.

## Legacy and forbidden namespace scan

The static scan found the following violations of the final target:

| Pattern | Findings | Action |
|---|---|---|
| `ds4_*` | `ds4_ssd.c/h`, `cuda/mmq/ds4_*`, donor comments and stubs | delete, rename, or isolate under explicit third-party boundary |
| `DeepSeek` / `GLM` | donor comments and `to_be_deleted` files | remove from production source; history is sufficient |
| `Metal` / `ROCm` | `to_be_deleted/speed-bench` and donor tests | delete with parked tree |
| `to_be_deleted/` | donor implementation, old server/tests, old quant tools | remove after deciding whether quant block code is needed |
| `candidate` | QSA candidate plugin and Q2 MoE candidate wrapper | canonicalize or delete after dispatch proof |
| `legacy` | counters and old path names in CUDA instrumentation | remove from production namespace when no longer needed |
| milestone target names | extensive M0-M9 Makefile graph | remove from final Makefile |

## Test and artifact policy

The current repository has hundreds of milestone tests, archived one-off
probes, generated binaries, and historical artifacts. CLEAN-02 must classify
them before deletion:

- `KEEP_CANONICAL`: canonical session/Reference 0 tests;
- `KEEP_FIXTURE`: byte-level or layer-local tests that do not load the full
  model;
- `ONE_OFF_DIAGNOSTIC`: archive or delete;
- `DUPLICATE`: remove after canonical coverage is identified;
- `OBSOLETE`: remove from the build and source tree.

No test-only caller should keep a production symbol alive unless it is promoted
to a canonical fixture/reference test. Reference tests must link reference
implementations from `tests/reference/`, never from the production `q38`
binary.

## Consolidation sequence

The requested sequence remains safe and is now grounded in this inventory:

1. CLEAN-01: this audit, no production code changes.
2. CLEAN-02: classify/delete dead tests, probes, and artifacts.
3. CLEAN-03: remove donor tree and `ds4` leftovers.
4. CLEAN-04: collapse forward/decode wrappers into one token graph.
5. CLEAN-05: remove callback backend propagation.
6. CLEAN-06: consolidate MoE and canonicalize production Q2/Q4 kernels.
7. CLEAN-07: consolidate QSA/state/reference code.
8. CLEAN-08: merge runtime/session implementation.
9. CLEAN-09: rewrite the Makefile to the final targets.
10. CLEAN-10: move CUDA sources under `cuda/`, rewrite README and live docs.
11. CLEAN-11: one final canonical Reference 0 validation load.

During CLEAN-01 through CLEAN-10, use only static inspection, compilation,
linking, unit tests, and existing fixtures. Do not launch another full-model
load until the final cleanup candidate is ready.

## Reproducibility commands

The audit can be regenerated without model inference:

```sh
find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \)
wc -l <source-file>
nm -C --defined-only q38
objdump -t q38
rg 'ds4_|DeepSeek|GLM|Metal|ROCm' --glob '*.{c,h,cu,cuh}' .
rg 'q38_forward_full|q38_decode_step|q38_session_eval|q38_session_prefill' .
```

