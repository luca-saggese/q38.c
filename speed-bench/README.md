## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
uv run python speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` for an environment-gated MXFP4
experiment, or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.

### Campaign reports

Final campaign narratives (not operational handoffs):

- `ds4_m5_fusion_port_results.md`: M5 Max ds4f-q2 decode-fusion promotion,
  rejected attempts, mechanism/eligibility, and rollback contract.
- `ds4_pre_m5_q2_decode_port_results.md`: pre-M5 port of those exact decode
  paths on M3 Ultra (IQ2XXS/Q2), including aggregate rollback.
- M3 Ultra resident-MXFP4 rounds 1–15: tables and manifest below (no separate
  narrative file; the CSV/JSON set is the durable record).

Session restart material and superseded agent handoffs stay under ignored
`speed-bench/local-runs/` and are not part of the review surface.

### M3 Ultra MXFP4 improvement history

The August 2026 exact-MXFP4 optimization campaign is preserved in four
chart-ready tables plus a metadata manifest:

- `m3_ultra_mxfp4_history.csv`: promoted cumulative checkpoints, medians,
  min/max ranges, original-baseline gains, and per-checkpoint contributions.
- `m3_ultra_mxfp4_runs.csv`: all accepted raw end-to-end runs and source-file
  SHA-256 values, including the second baseline run.
- `m3_ultra_mxfp4_controlled.csv`: accepted same-engine forward and inverse
  A/B results with exact-logit coverage.
- `m3_ultra_mxfp4_attempts.csv`: rejected candidates, instrumented stage
  probes, and the SSD-streaming correctness pair.
- `m3_ultra_mxfp4_history.json`: benchmark contract, Git provenance, chart
  column mapping, exactness summary, and known provenance limitations.

The attempts table retains the historical column name `speed_csv_sha256`.
It hashes the retained primary artifact: normally a speed CSV, but for some
controlled A/B rows it is the raw comparison log. The `notes` field identifies
the artifact type and coverage.

The immutable denominator is the first 2,048-token baseline at commit
`4893e0c`: 580.08 prefill tok/s, 35.27 decode tok/s, 31.801 ms first-token
latency, and 36.13 steady decode tok/s. All `*_gain_pct` columns compare to
that run. `gen_first_reduction_pct` is positive when latency improves, while
`gen_first_change_pct` retains the conventional negative-is-faster sign used
in benchmark reports. The `*_step_*` columns capture what each promoted
checkpoint added relative to the immediately preceding checkpoint.

Rounds 1 and 2 were measured as intermediate working-tree states and were
consolidated with Round 3 in commit `c683fb9`; the manifest records the exact
rollback environment needed to reconstruct each variant from that commit.
Rounds 4, 5, and 6 are standalone commits (`3658bd6`, `ee5c0f9`, and
`c0431e3`) with three exact official replicas apiece. Rounds 7, 8, and 9 are
standalone commits (`6b01de1`, `2424d29`, and `b0d2f02`) with three exact
official replicas apiece. Round 7 extends the R6 decode pipeline fast lookup
to the `mul_mv_ext` (nsg+nxpsg) family; Round 8 memoizes the flash attention
pad/blk decode pipeline lookups; Round 9 memoizes the flash attention batched
prefill pipeline lookup. Their promoted defaults target the shared pre-M5
Apple Silicon path used by M1, M2, M3, and M4, but campaign performance was
measured only on M3 Ultra; M1, M2, and M4 were not performance-benchmarked,
and M5 remains on its separate path. New rollback switches use
`DS4_METAL_DISABLE_PRE_M5_*`; historical `DS4_METAL_DISABLE_M3_*` names remain
compatibility aliases and are preserved in the manifest as provenance.

Rounds 7-9 are host-side pipeline-lookup optimizations and are mathematically
identical to the prior checkpoint (all 27 retained frontier artifacts across
rounds 1-9 share the canonical SHA
`6f7edd6d7319d48270e3a5d34eb31f9f8f957ad7f54f4067afa9157c47708179`). Some
decode step deltas across R7-R9 reflect cross-run machine drift rather than
causal change; the controlled same-session A/B evidence (inverse-disable) is
unanimous and is recorded in the controlled table. Prefill improved
monotonically through R9 (+11.09% cumulative versus the immutable baseline).

Rounds 10, 11, and 12 are standalone commits (`3fb202a`, `c21ec43`, and
`6711f76`) with three exact official replicas apiece. Round 10 flushes a
second decode command buffer after layer 16 in the eligible pre-M5 resident
window (pos 2048-2815), closing the GPU idle bubble between the first split
buffer and the tail buffer (+3.0% controlled decode, bit-exact by
construction). Round 11 dispatches the MXFP4 routed pair-swiglu and sum6
decode kernels with one simdgroup per threadgroup (nsg=1), a bit-exact
row-to-simdgroup remap (+0.4% controlled decode). Round 12 widens the second
split window to pos 2048-3327 (+0.2/+1.0% controlled decode). All 36 retained
frontier artifacts across rounds 1-12 share the canonical SHA above. Round
12's official medians read below Round 11 because the machine drifted during
that session (prefill, which the decode-only change cannot affect, moved down
in the same window, and the schedule is byte-identical to Round 11 below pos
2816); the controlled same-session evidence is unanimous and recorded in the
controlled table. Decode is +5.56% steady cumulative versus the immutable
baseline through Round 12.

Rounds 13, 14, and 15 are standalone commits (`0efef8d`, `5c4f816`, and
`f357446`) with three exact official replicas apiece. Round 13 gives the
nsg=1 MXFP4 pair-SwiGLU and sum6 decode pipelines a truthful
threadgroup-width-multiple contract. Round 14 specializes the one-token,
six-route, world-1 pair-SwiGLU routing preamble, and Round 15 applies the same
fixed-routing principle to sum6 while retaining the shared accumulation
helper and exact slot/update order. Their guarded defaults target the shared
pre-M5 M1-M4 resident MXFP4 path, with
`DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_TG_MULTIPLE`,
`DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_FIXED_ROUTE_PAIR`, and
`DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_DECODE_FIXED_ROUTE_SUM6` rollbacks.
M5 remains separate.

All twelve required R13-R15 controlled decode processes were bit-exact and
directionally unanimous. R15 reaches 38.62 decode tok/s and 38.93 steady
decode tok/s, cumulative gains of +9.498% and +7.750% over the immutable
`4893e0c` baseline. All nine new frontier artifacts and all nine new raw
logits payloads match the canonical hashes. Prefill cannot select these
single-token-only specializations; official prefill medians remain flat
between R13 and R15. Performance was measured only on M3 Ultra; M1, M2, and
M4 were not performance-benchmarked.

Historical campaign note (Round 15 tip only): repository-wide `make test` at
that checkpoint retained 17 golden/model-behavior assertion failures.
Disabling all new Round 4-9 defaults reproduced the same 17 failures; the
Metal-kernel suite and tensor-equivalence suite passed exactly, and the Round
15 normalized assertion fingerprint is byte-identical to the R13 campaign
baseline. The manifest records the evidence hashes. The current branch tip
reports a green `make test`; treat the 17-failure count as historical
provenance, not the present suite status.

Use `sequence` as the chart x-axis, the normalized improvement columns for a
shared percentage scale, and the controlled table for contribution labels.
Rejected and diagnostic rows must not be mixed into the promoted line.
