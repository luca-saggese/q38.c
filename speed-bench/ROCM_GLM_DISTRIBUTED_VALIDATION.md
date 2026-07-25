# ROCm GLM Distributed Change Validation

Use this playbook for every ROCm GLM distributed performance change.  It keeps
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
coordinator.  The coordinator checkout used below is `~/ds4/ds4`.

## Rules

1. Build baseline and candidate from the same commit and image.
2. Keep the model file, SHA-256, layer split, prompt, context allocation,
   chunk/window, generation length, power state, and background load fixed.
3. Hold previously accepted optimizations constant.  Toggle only the change
   under test.
4. Save worker and coordinator logs plus benchmark CSV files.
5. Test performance and numerical closeness separately.  A coherent sampled
   answer is not a numerical comparison, and a matching argmax is not enough.
6. Run the baseline and candidate at least once as a smoke test.  Before
   declaring a small improvement real or making a path default, repeat the
   complete pair.  Treat differences below roughly 2% as noise until repeated.
7. Do not compare `ds4-bench --show-output` text with chat answers.  The former
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

## Performance protocol

The standard development point is a fresh 2048-token prefill followed by 128
greedy generation tokens.  Both nodes keep the accepted FP16 causal-attention
GEMM enabled.  The candidate additionally enables the change under test.

For the F16 compact-cache experiment, the isolated variable is:

```text
baseline:  DS4_ROCM_GLM_COMPACT_CACHE_F16 unset
candidate: DS4_ROCM_GLM_COMPACT_CACHE_F16=1
```

### Baseline worker

Run on `192.168.100.1`:

```sh
podman run --rm -it \
  --name ds4-worker-cache-f32 \
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
  2>&1 | tee /tmp/glm-worker-cache-f32.log
```

### Baseline coordinator

Run on `192.168.100.2`:

```sh
env -u DS4_ROCM_GLM_COMPACT_CACHE_F16 \
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
  --csv /tmp/glm-cache-f32.csv \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081 \
  2>&1 | tee /tmp/glm-coordinator-cache-f32.log
```

### Candidate worker

Stop the baseline worker, then run on `192.168.100.1`:

```sh
podman run --rm -it \
  --name ds4-worker-cache-f16 \
  -e DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
  -e DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
  -e DS4_ROCM_GLM_COMPACT_CACHE_F16=1 \
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
  2>&1 | tee /tmp/glm-worker-cache-f16.log
```

### Candidate coordinator

Run on `192.168.100.2`:

```sh
DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_ROCM_GLM_COMPACT_CACHE_F16=1 \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 2048 \
  --ctx-alloc 2177 \
  --gen-tokens 128 \
  --show-output \
  --csv /tmp/glm-cache-f16.csv \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081 \
  2>&1 | tee /tmp/glm-coordinator-cache-f16.log
```

Confirm that the baseline logs report a compact `f32` cache and the candidate
logs report `f16`.  The candidate compact KV allocation should be approximately
half the baseline size.  Compare `prefill_tps`, `gen_tps`, and
`gen_first_ms`; do not infer a win from elapsed startup or model-copy time.

## Layer-local numerical comparison

This check catches wrong layouts, strides, masking, RoPE, scaling, and result
scattering near the operation that changed.  It is stronger than comparing
generated prose.

Use workers configured like the performance workers, but set `--ctx 641`.
Run the baseline first:

```sh
mkdir -p /tmp/glm-cache-attn

env -u DS4_ROCM_GLM_COMPACT_CACHE_F16 \
DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_ROCM_GRAPH_DUMP_PREFIX=/tmp/glm-cache-attn/baseline \
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

Restart the worker with the candidate F16-cache environment and run:

```sh
DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_ROCM_GLM_COMPACT_CACHE_F16=1 \
DS4_ROCM_GRAPH_DUMP_PREFIX=/tmp/glm-cache-attn/candidate \
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
  /tmp/glm-cache-attn/baseline_glm_indexed_attn_out-0_pos0.bin \
  /tmp/glm-cache-attn/candidate_glm_indexed_attn_out-0_pos0.bin \
  --hidden 6144
```

Any non-finite value is a failure.  For a single early layer, cosine should be
very close to 1.  As a diagnostic heuristic, investigate cosine below `0.9999`
or relative RMSE above `0.01`; these are not universal model-quality limits.

## Frontier-logit comparison

Layer-local agreement can still accumulate into meaningful final-logit drift.
Collect complete vocabulary logits at a fixed 512-token frontier.  Use workers
with `--ctx 641`.

Baseline:

```sh
mkdir -p /tmp/glm-cache-logits-f32

env -u DS4_ROCM_GLM_COMPACT_CACHE_F16 \
DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 512 \
  --ctx-max 512 \
  --ctx-alloc 641 \
  --gen-tokens 0 \
  --dump-frontier-logits-dir /tmp/glm-cache-logits-f32 \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081
```

Candidate:

```sh
mkdir -p /tmp/glm-cache-logits-f16

DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=1 \
DS4_ROCM_GLM_COMPACT_CACHE_F16=1 \
DS4_BENCH_DISABLE_SNAPSHOT=1 \
ds4-bench \
  --prompt-file ~/ds4/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 512 \
  --ctx-max 512 \
  --ctx-alloc 641 \
  --gen-tokens 0 \
  --dump-frontier-logits-dir /tmp/glm-cache-logits-f16 \
  -m ~/ds4/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  --dist-prefill-chunk 256 \
  --dist-prefill-window 2 \
  --role coordinator \
  --layers 0:37 \
  --listen 192.168.100.2 8081
```

Compare them:

```sh
python3 ~/ds4/ds4/speed-bench/compare_glm_validation.py logits \
  /tmp/glm-cache-logits-f32/frontier_000512.logits.json \
  /tmp/glm-cache-logits-f16/frontier_000512.logits.json
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
proof.  Investigate a material regression from the last accepted path and use
the official GLM continuation scorer before making an approximation the
unconditional release default.

## API smoke test

After tensor and logit checks pass, run a server and test the actual chat
template.  This catches routing, KV replay, and API integration failures:

```sh
wget --timeout=900 --tries=1 -qO- \
  --header='Content-Type: application/json' \
  --post-data='{"model":"glm-5.2","messages":[{"role":"user","content":"Reply with exactly: hello"}],"thinking":{"type":"disabled"},"temperature":0,"max_tokens":16}' \
  http://localhost:8000/v1/chat/completions |
jq -r '.choices[0].message.content // .'
```

Expect a coherent response, normally `hello`.  This is a smoke test, not a
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
reject it.  Explain and validate correctness before promoting performance code.
