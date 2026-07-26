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

## Publish and deploy the test image

The image is built by
`kyuz0/strix-halo-ds4-toolbox/.github/workflows/build_and_publish.yml`.
Its `glm-rocm-7.2.4` Dockerfile clones
`https://github.com/kyuz0/ds4.git`, branch
`fix/rocm-distributed-glm`, so push the exact ds4 commit before dispatching the
workflow. The toolbox workflow itself runs from `main`.

Dispatch only the GLM ROCm image and follow the run to completion:

```sh
gh workflow run build_and_publish.yml \
  --repo kyuz0/strix-halo-ds4-toolbox \
  --ref main \
  -f backends=glm-rocm-7.2.4

gh run list \
  --repo kyuz0/strix-halo-ds4-toolbox \
  --workflow build_and_publish.yml \
  --limit 1

gh run watch RUN_ID \
  --repo kyuz0/strix-halo-ds4-toolbox \
  --exit-status
```

The workflow publishes both an immutable timestamped tag and the channel tag
used by this playbook:

```text
docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4
```

Pull the channel tag on both hosts before testing. Do not assume a mutable tag
changed merely because `podman pull` succeeded; compare image IDs:

```sh
ssh fw1 podman pull \
  docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4
ssh fw2 podman pull \
  docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4

ssh fw1 podman image inspect \
  docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4 \
  --format '{{.Id}} {{.Created}}'
ssh fw2 podman image inspect \
  docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4 \
  --format '{{.Id}} {{.Created}}'
```

The IDs must match. The worker and coordinator examples below invoke the image
directly with Podman; entering a Toolbx shell is not required. Host `/tmp`
results must be mounted explicitly when the command writes inside the
container, for example:

```sh
mkdir -p /tmp/glm-validation

podman run --rm \
  --security-opt label=disable \
  -v /tmp/glm-validation:/results \
  IMAGE COMMAND --csv /results/run.csv
```

Keep logs outside the container with `2>&1 | tee /tmp/name.log`. Use distinct
container names and result directories for baseline and candidate, and stop
the worker after its coordinator run completes.

## One-build kernel flag matrix

The following ROCm paths can be varied independently in one image for
attribution and interaction checks:

| Path | Environment variable | Default | Intended effect |
| --- | --- | --- | --- |
| 64 KiB shared input | `DS4_ROCM_Q8_DECODE_SHAREDX_64K=0` | On | Let one-token Q8 projections with a 16,384-element input reuse the input through LDS. Set `=0` for the warp-row fallback. |
| Wave-parallel value projection | `DS4_ROCM_GLM_VALUE_PROJECT_WAVE_DECODE=0` | On | Give each one-token GLM value-projection output row a 32-lane wave instead of a serial thread. Set `=0` only for a serial-row correctness baseline. |
| Causal attention GEMM | `DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=0` | On | Use FP16 BLAS for contiguous causal attention during initial prefill. Set `=0` only for the scalar-kernel fallback. |
| Selected attention GEMM | `DS4_ROCM_GLM_SELECTED_ATTN_GEMM=0` | On | Gather per-token selected KV rows and use FP16 strided-batched BLAS after the 2,048-row indexer boundary. Set `=0` only for the scalar-kernel fallback. |
| Selected-attention head tile | `DS4_ROCM_GLM_SELECTED_ATTN_HEAD_TILE=1\|2\|4\|8\|16\|32\|64` | `16` | Process several attention heads as BLAS matrix columns. Tile 1 is the bit-exact untiled baseline; tiles 32 and 64 remain experimental because they showed accumulated logit drift. |
| Selected-attention phase profile | `DS4_ROCM_GLM_SELECTED_ATTN_PROFILE=1` | Off | Print HIP-event phase totals for the first selected-attention invocation in each process. Diagnostic only. |

Apply any opt-in or fallback override to **both** worker and coordinator. For
Podman, add it as `-e NAME=value`; for a host binary, put `NAME=value \`
before `ds4-bench` or `ds4-server`.

Use this order so one deployment answers both attribution and interaction:

1. `DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=0` for the causal-attention scalar fallback;
2. no causal-attention override for the production causal GEMM path;
3. `DS4_ROCM_GLM_VALUE_PROJECT_WAVE_DECODE=0` for the serial-row baseline;
4. no value-projection override for the production wave path;
5. production wave path plus `DS4_ROCM_Q8_DECODE_SHAREDX_64K=1`;
6. `DS4_ROCM_GLM_SELECTED_ATTN_GEMM=0` for the selected-attention scalar
   fallback;
7. the production selected-attention GEMM path, with the shared-input
   experiment either off or on as required.

The expected activation messages are:

```text
Q8 one-token shared-input kernel enabled through 64 KiB LDS
GLM one-token value projection using wave-parallel Q8 rows
GLM causal indexed prefill using fp16 hipBLAS attention GEMMs
GLM selected indexed prefill using fp16 hipBLAS strided-batched attention GEMMs
ROCm GLM selected attention using head tile N
ROCm GLM selected attention profile tokens=... selected=... heads=... tile=...
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

Run it once with `DS4_ROCM_GLM_SELECTED_ATTN_GEMM=0` on both nodes for the
scalar-kernel baseline and once without the override for the production GEMM
path. The first 2,048-token row is the causal control; the later 256-token
suffix rows exercise selected attention. The selected GEMM path accepts
64–256 tokens and uses about 580 MiB of temporary workspace at 256 tokens with
2,048 selected rows at head tile 1. Tiles 2, 4, 8, 16, 32, and 64 use
approximately 584, 591, 606, 637, 697, and 818 MiB respectively.

The phase profiler intentionally synchronizes at each measured boundary, so
never use its wall-clock benchmark row as a performance result. It claims only
the first selected-attention invocation per process. Run profiling separately
on worker and coordinator, then disable it for all matched performance and
correctness runs.

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

The accepted FP16 causal-attention GEMM is enabled by default. Leave it without
an override in ordinary baseline/candidate comparisons. When validating the
causal GEMM itself, use `DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=0` on both nodes for
the scalar-kernel reference and no override for the production candidate.

### Worker

Run on `192.168.100.1`. Change `RUN_NAME` to `baseline` or `candidate`, and add
the one candidate environment setting only for the candidate run.

```sh
RUN_NAME=baseline

podman run --rm \
  --name "ds4-worker-${RUN_NAME}" \
  -e DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
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
as another `-e NAME=value` only for the candidate run.

```sh
RUN_NAME=baseline
mkdir -p "/tmp/glm-${RUN_NAME}"

podman run --rm \
  --name "ds4-coordinator-${RUN_NAME}" \
  -e DS4_GLM_MEMORY_GUARD_RESERVE_GB=8 \
  -e DS4_BENCH_DISABLE_SNAPSHOT=1 \
  --device /dev/dri \
  --device /dev/kfd \
  --group-add keep-groups \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt label=disable \
  --userns=keep-id \
  --network=host \
  -v /home/kyuz0/ds4:/models:ro \
  -v "/tmp/glm-${RUN_NAME}:/results" \
  docker.io/kyuz0/strix-halo-ds4-toolbox:glm-rocm-7.2.4 \
  ds4-bench \
  --prompt-file /models/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 2048 \
  --ctx-alloc 2177 \
  --gen-tokens 128 \
  --show-output \
  --csv /results/run.csv \
  -m /models/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
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

## Selected-attention promotion record

On 2026-07-26, the selected-attention GEMM was compared with its scalar-kernel
fallback on the topology at the top of this document. Both runs used commit
`34b3862`, FP16 causal-attention GEMMs, a 2,304-token prompt, 256-token
distributed chunks, and a 2,433-token allocation. The only changed setting was
`DS4_ROCM_GLM_SELECTED_ATTN_GEMM=0` versus `=1`, on both nodes.

The graph dump at layer 0 and position 2,048 produced:

```text
all-token relative RMSE: 5.9441e-05
all-token cosine:        0.999999998234
final-token cosine:      0.999999997577
non-finite pairs:        0
```

The complete 154,880-value frontier-logit comparison produced:

```text
KL(reference || candidate): 0.0664153
top-1:                     3956 -> 3956
top-10 overlap:            10/10
top-50 overlap:            46/50
centered cosine:           0.9921320
```

The dump-free performance pair improved from `19.11` to `27.48` prompt
tokens/s (`+43.8%`). The instrumented pair measured `18.85` versus `27.49`
tokens/s; do not compare either instrumented row with a normal benchmark
because serializing the full logit vector is included in the final chunk.

Decision: make the selected-attention GEMM the ROCm GLM default and preserve
`DS4_ROCM_GLM_SELECTED_ATTN_GEMM=0` as the diagnostic rollback. The narrower
candidate top-1 margin (`1.2262` to `0.3320`) remains important context even
though the selected token and top-10 set matched.

The default-on change was published from ds4 commit `ea2e766` by toolbox
GitHub Actions run
[`30214076099`](https://github.com/kyuz0/strix-halo-ds4-toolbox/actions/runs/30214076099).
After pulling the channel tag, both hosts reported:

```text
image ID: 379e27ad0439970bf72e04ac63fe2711e3471468f2171bad757ea03bca52d72c
digest:   sha256:221bc3b031f5600094e40a1b0202fbd7f7057f31ab4854836bf97a5420dbc36f
created:  2026-07-26 18:13:38 UTC
```

A no-override post-deployment benchmark activated both the selected-attention
GEMM and wave-parallel one-token value projection and measured:

```text
ctx=2304 prefill=27.45 t/s generation=2.30 t/s first-token=434.913 ms
```

The long API smoke used a 2,482-token chat prompt and 32 generated tokens. It
activated selected attention, completed prefill in `102.769 s`, decoded at
`2.29 t/s`, and returned a coherent description of the supplied passage as the
introduction to Manzoni's *I promessi sposi*.

## Causal-attention default deployment record

The causal-attention GEMM became default-on in ds4 commit `dd837f0`, published
by toolbox GitHub Actions run
[`30215053603`](https://github.com/kyuz0/strix-halo-ds4-toolbox/actions/runs/30215053603).
Both hosts pulled and verified:

```text
image ID: 6f424ab78c427808d7d7cee3b4a0ea01b232223a0ca525ebfb033042dec4c4a0
digest:   sha256:9bda0b1d4c45c08bc06caffa90262a964d935534353595b199eab98ce9e6c5cb
created:  2026-07-26 18:41:10 UTC
```

The post-deployment benchmark used no kernel environment overrides, layers
`0:37` plus `38:output`, a 2,433-token allocation, 128 generated tokens, and
the production distributed prefill settings:

```text
--dist-prefill-chunk 256
--dist-prefill-window 2
```

Chunk 256 is intentional. Earlier matched 2,048-token runs measured
`50.81 t/s` with chunk 256 versus `48.15 t/s` with chunk 512.

The production curve measured:

```text
ctx   prefill tokens   prefill t/s   generation t/s   first token
2048  2048             50.55         2.30             439.299 ms
2304   256              5.22         2.30             434.724 ms
```

The second row is the incremental selected-attention suffix at an existing
2,048-token frontier, not full-prompt throughput. It exposes the remaining
post-2,048 prefill cliff. A separate cold 2,304-token full-prompt run measured:

```text
prefill=27.46 t/s generation=2.30 t/s first-token=439.220 ms
```

All three default activation messages were present: causal attention GEMM,
selected attention GEMM, and wave-parallel one-token value projection. Both
128-token raw continuations were coherent. Preserve
`DS4_ROCM_GLM_CAUSAL_ATTN_GEMM=0` as the independent scalar-kernel rollback.

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
