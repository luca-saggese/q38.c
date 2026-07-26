# ROCm GLM Distributed Change Validation

Use this playbook for every ROCm GLM distributed performance change. It keeps
speed measurements separate from numerical and model-quality checks, and it
changes one variable at a time.

The commands below describe the current two-Strix-Halo test topology:

- coordinator: `192.168.100.2`, layers `0:37`
- worker: `192.168.100.1`, layers `38:output`
- model: `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf`
- image: `docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4`
- distributed prefill: 256-token chunks, window 2

Change those constants deliberately if the hardware or model changes, and
record the new values with the results.

Unless a command is explicitly labeled for the worker, run it on the
coordinator. The coordinator checkout used below is `~/ds4/ds4`.

## Rules

1. Build baseline and candidate from the same commit and image.
2. Keep the model file, SHA-256, layer split, prompt, context allocation,
   chunk/window, generation length, power state, and background load fixed.
3. Hold previously accepted optimizations constant. Toggle only the change
   under test.
4. Save worker and coordinator logs plus benchmark CSV files.
5. Test performance and numerical closeness separately. A coherent sampled
   answer is not a numerical comparison, and a matching argmax is not enough.
6. Run the baseline and candidate at least once as a smoke test. Before
   declaring a small improvement real or making a path default, repeat the
   complete pair. Treat differences below roughly 2% as noise until repeated.
7. Do not compare `ds4-bench --show-output` text with chat answers. The former
   is a raw continuation of the prompt file; use the server API for chat.

Record the code revision and model hash before each series:

```sh
git -C ~/ds4/ds4 rev-parse HEAD
sha256sum ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
```

## Build gate

The ROCm/gfx1151 compile check does not require a GPU:

```sh
podman run --rm --userns=keep-id \
  -v ~/ds4/ds4:/workspace:Z \
  -w /workspace \
  docker.io/rocm/dev-ubuntu-24.04:7.2.1-complete \
  make strix-halo ROCM_ARCH=gfx1151
```

Require a successful warning-free build and run `git diff --check`.

## One-build kernel flag matrix

The following ROCm paths can be varied independently in one image for
attribution and interaction checks:

| Path | Environment variable | Default | Intended effect |
| --- | --- | --- | --- |
| 64 KiB shared input | `DS4_ROCM_Q8_DECODE_SHAREDX_64K=1` | Off | Let one-token Q8 projections with a 16,384-element input reuse the input through LDS. |
| Wave-parallel value projection | `DS4_ROCM_GLM_VALUE_PROJECT_WAVE_DECODE=0` | On | Give each one-token GLM value-projection output row a 32-lane wave instead of a serial thread. Set `=0` only for a serial-row correctness baseline. |
| Selected attention GEMM | `DS4_ROCM_GLM_SELECTED_ATTN_GEMM=1` | Off | Gather per-token selected KV rows and use FP16 strided-batched BLAS after the 2,048-row indexer boundary. |

Apply any opt-in or fallback override to **both** worker and coordinator. For
the Podman worker, add it as `-e NAME=value`; for the coordinator, put
`NAME=value \` before `ds4-bench` or `ds4-server`.

Use this order so one deployment answers both attribution and interaction:

1. `DS4_ROCM_GLM_VALUE_PROJECT_WAVE_DECODE=0` for the serial-row baseline;
2. no value-projection override for the production wave path;
3. production wave path plus `DS4_ROCM_Q8_DECODE_SHAREDX_64K=1`;
4. `DS4_ROCM_GLM_SELECTED_ATTN_GEMM=1`;
5. both opt-in flags with the production wave path.

The expected activation messages are:

```text
Q8 one-token shared-input kernel enabled through 64 KiB LDS
GLM one-token value projection using wave-parallel Q8 rows
GLM selected indexed prefill using fp16 hipBLAS strided-batched attention GEMMs
```

Absence of a message means the tested workload did not reach that path. The
64 KiB path also reports and automatically falls back if the launch is not
supported by the device. The wave-parallel value projection changes FP32
summation order; validate its layer output and frontier logits even when its
generated text looks coherent.

The normal 2,048-token benchmark below exercises the decode flag, but it does
**not** exercise selected attention: 2,048 visible rows still use the dense
causal path. Measure the prompt-processing cliff with a live 256-token suffix
curve:

```sh
RUN_NAME=selected-baseline

DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 4096 \
  --step-incr 256 \
  --ctx-alloc 4097 \
  --gen-tokens 0 \
  --csv "/tmp/glm-${RUN_NAME}.csv" \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081 \
  2>&1 | tee "/tmp/glm-coordinator-${RUN_NAME}.log"
```

Run it once without selected attention and once with
`DS4_ROCM_GLM_SELECTED_ATTN_GEMM=1` on both nodes. The first 2,048-token row is
the causal control; the later 256-token suffix rows exercise selected
attention. The selected GEMM experiment accepts 64–256 tokens and uses about
580 MiB of temporary workspace at 256 tokens with 2,048 selected rows.

For a layer-local selected-attention comparison, adapt the graph-dump command
below to `--ctx-start 2304 --ctx-max 2304 --ctx-alloc 2433` and set
`DS4_ROCM_GRAPH_DUMP_POS=2048`. For an end-to-end comparison, dump frontier
logits at 2,304 tokens and compare `frontier_002304.logits.json`. A 512-token
dump cannot validate this path.

## Matched baseline and candidate

Write down the single environment setting, option, or code path being tested.
Run the commands below once without it for the baseline, then again with only
that setting changed for the candidate. If the setting affects both halves of
the model, apply it to both worker and coordinator.

Keep the accepted FP16 causal-attention GEMM enabled in both runs:

```text
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1
```

### Worker

Run on `192.168.100.1`. Change `RUN_NAME` to `baseline` or `candidate`, and add
the one candidate environment setting only for the candidate run.

```sh
RUN_NAME=baseline

podman run --rm -it \
  --name "ds4-worker-${RUN_NAME}" \
  -e DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
  -e DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add keep-groups \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt label=disable \
  --userns=keep-id \
  --network=host \
  -v /mnt/storage/ds4:/models:ro \
  docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4 \
  ds4-server \
  -m /models/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --ctx 2177 \
  --host 0.0.0.0 \
  --port 8000 \
  --role worker \
  --layers 38:output \
  --coordinator 192.168.100.2 8081 \
  2>&1 | tee "/tmp/glm-worker-${RUN_NAME}.log"
```

### Coordinator performance run

Set `RUN_NAME` to match the worker. Add the one candidate environment setting
before `ds4-bench` only for the candidate run.

```sh
RUN_NAME=baseline

DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 2048 \
  --ctx-alloc 2177 \
  --gen-tokens 128 \
  --show-output \
  --csv "/tmp/glm-${RUN_NAME}.csv" \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081 \
  2>&1 | tee "/tmp/glm-coordinator-${RUN_NAME}.log"
```

Compare `prefill_tps`, `gen_tps`, and `gen_first_ms`; do not infer a win from
elapsed startup or model-copy time.

## Layer-local numerical comparison

This catches wrong layouts, strides, masking, RoPE, scaling, and result
scattering near the operation that changed. Use workers configured like the
performance runs.

Run the command once as `baseline`, restart the worker with the candidate
configuration, then run it as `candidate`:

```sh
RUN_NAME=baseline
mkdir -p /tmp/glm-validation

DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_ROCM_GRAPH_DUMP_PREFIX="/tmp/glm-validation/${RUN_NAME}" \
DS4_ROCM_GRAPH_DUMP_NONINVASIVE=1 \
DS4_ROCM_GRAPH_DUMP_LAYER=0 \
DS4_ROCM_GRAPH_DUMP_POS=0 \
DS4_ROCM_GRAPH_DUMP_NAME=glm_indexed_attn_out \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 256 \
  --ctx-max 256 \
  --ctx-alloc 641 \
  --gen-tokens 0 \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081
```

Compare the dumps:

```sh
python3 ~/ds4/ds4/speed-bench/compare_glm_validation.py tensor \
  /tmp/glm-validation/baseline_glm_indexed_attn_out-0_pos0.bin \
  /tmp/glm-validation/candidate_glm_indexed_attn_out-0_pos0.bin \
  --hidden 6144
```

Any non-finite value is a failure. For a single early layer, cosine should be
very close to 1. As a diagnostic heuristic, investigate cosine below `0.9999`
or relative RMSE above `0.01`; these are not universal model-quality limits.

## Frontier-logit comparison

Layer-local agreement can still accumulate into meaningful final-logit drift.
Collect complete vocabulary logits at a fixed 512-token frontier. Run once as
`baseline`, restart the worker with the candidate configuration, and run again
as `candidate`.

```sh
RUN_NAME=baseline
mkdir -p "/tmp/glm-logits-${RUN_NAME}"

DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 512 \
  --ctx-max 512 \
  --ctx-alloc 641 \
  --gen-tokens 0 \
  --dump-frontier-logits-dir "/tmp/glm-logits-${RUN_NAME}" \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081
```

Compare the dumps:

```sh
python3 ~/ds4/ds4/speed-bench/compare_glm_validation.py logits \
  /tmp/glm-logits-baseline/frontier_000512.logits.json \
  /tmp/glm-logits-candidate/frontier_000512.logits.json
```

Interpret the result as a group:

- non-finite logits are always a failure;
- centered cosine and relative RMSE measure distribution shape without an
  irrelevant constant logit shift;
- KL divergence measures probability-distribution movement;
- top-10/top-50 overlap and the reference top-token rank show whether ordering
  changed near the useful head of the distribution;
- matching top-1 is encouraging, but does not by itself prove equivalence;
- a top-1 change is less concerning when the reference margin is tiny.

There is no universal threshold that turns one frontier into a model-quality
proof. Investigate a material regression from the last accepted path and use
the official GLM continuation scorer before making an approximation the
unconditional release default.

## API smoke test

After tensor and logit checks pass, run a server and test the actual chat
template. This catches routing, KV replay, and API integration failures:

```sh
wget --timeout=900 --tries=1 -qO- \
  --header='Content-Type: application/json' \
  --post-data='{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with exactly: hello"}],"thinking":{"type":"disabled"},"temperature":0,"max_tokens":16}' \
  http://localhost:8000/v1/chat/completions |
jq -r '.choices[0].message.content // .'
```

Expect a coherent response, normally `hello`. This is a smoke test, not a
replacement for tensor/logit comparison or the 100-case GLM continuation gate
documented in `QA_BEFORE_RELEASES.md`.

## Decision record

For every candidate, retain:

- commit and model SHA-256;
- exact baseline and candidate environment variables;
- coordinator and worker logs;
- baseline and candidate CSV rows;
- layer-local tensor comparison;
- frontier-logit comparison;
- API smoke response;
- decision: reject, keep diagnostic, retain opt-in, or make default.

If a candidate is faster but its drift is unexplained, keep it diagnostic or
reject it. Explain and validate correctness before promoting performance code.
