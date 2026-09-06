# GR optimization changelog

This log records only GR subsystem changes. Each entry must include the
candidate name, the exact isolated command, correctness gates, and timing
delta. Full-chain results must remain separate from isolated GR results.

## 2026-09-06 — GR-BASELINE

- Added `tests/gr/GR_CONTRACT.md` with the production formula and call-chain
  classification.
- Organized the existing GR tests under `tests/gr/`.
- Added an independent production-semantics CPU oracle.
- Added early/middle/late compact GR fixture extraction from the production
  BF16 tensor families.
- Added the isolated CUDA benchmark with 100 warmups and 1000 measurements.
- Added baseline artifact:
  `artifacts/perf/subsystems/gr_reference.json`.
- **Optimization:** none.
- **Production code changed for performance:** none.

### Isolated baseline

| Fixture | Layer | Kernel median | End-to-end median | Kernel p95 |
|---|---:|---:|---:|---:|
| early | 0 | 1520.38 us | 1524.81 us | 1559.20 us |
| middle | 23 | 1469.18 us | 1473.71 us | 1509.73 us |
| late | 47 | 1470.75 us | 1475.26 us | 1484.86 us |

Each call currently launches five CUDA kernels. The measurements above are
baseline observations of the existing non-fused test implementation, not a
claim about the canonical full-chain GR bucket.

### Baseline decomposition

The stage medians below are diagnostic CUDA-event spans and are not summed
with the end-to-end wall measurement:

| Fixture | Normalize | Down projection | Up projection | Branch merge | Injection |
|---|---:|---:|---:|---:|---:|
| early | 79.87 us | 1066.91 us | 65.54 us | 4.10 us | 288.77 us |
| middle | 32.77 us | 1075.04 us | 63.55 us | 4.10 us | 288.77 us |
| late | 33.15 us | 1075.04 us | 65.54 us | 4.10 us | 284.67 us |

The dominant isolated stage is the down projection at approximately
71–74% of the measured GR kernel span. This is an observation only; no
kernel or mathematical optimization has been applied.

## 2026-09-06 — GR-C1

- **Hypothesis:** decode-time GR projections were routed through the generic
  `matrix_batch_kernel` even when `token_count == 1`.
- **Production change:** BF16 GR projection stages
  `gr_read_down`, `gr_read_up`, and `gr_write_inject` now use the existing
  cooperative BF16 matvec kernel. All other matrix-batch calls retain the
  generic fallback.
- **Files changed:** `cuda/q38_forward_cuda.cu`,
  `cuda/q38_cuda_primitives.cu`, `q38_cuda_primitives.h`.
- **Isolated command:** `make gr-c1-bench`.
- **Artifact:** `artifacts/perf/subsystems/gr_c1_projection.json`.
- **Correctness:** all early/middle/late projection fixtures passed; no
  non-finite values; maximum candidate absolute error was `2.06e-4`.
- **Traffic/sync deltas:** timed projection loops keep H2D/D2H outside the
  measurement window; the candidate adds no host synchronization or transfer.

### GR-C1 projection results

| Fixture | Projection | Generic median | Candidate median | Improvement |
|---|---|---:|---:|---:|
| early | down | 1114.46 us | 12.77 us | 98.85% |
| early | up | 39.36 us | 31.17 us | 20.81% |
| early | inject | 231.87 us | 4.51 us | 98.05% |
| middle | down | 1114.53 us | 12.74 us | 98.86% |
| middle | up | 39.33 us | 31.14 us | 20.83% |
| middle | inject | 231.90 us | 4.54 us | 98.04% |
| late | down | 1114.50 us | 12.77 us | 98.85% |
| late | up | 39.36 us | 31.17 us | 20.81% |
| late | inject | 231.87 us | 4.54 us | 98.04% |

The summed projection component improves from approximately `1385.7 us` to
`48.4 us` per fixture. This is a model-free projection result, not a
full-chain speedup claim. The canonical full-chain benchmark and Reference 0
remain unchanged.

## 2026-09-06 — GR-C2 (not promoted)

- **Hypothesis:** the four branch reads paid for separate branch-preparation
  and merge launches even though they share the same normalized input and
  token structure.
- **Isolated command:**
  `./tests/gr/gr_c2_bench tests/fixtures/gr artifacts/perf/subsystems/gr_c2_bundle.json`
- **Measurement:** C1 was remeasured as the baseline with the three
  cooperative BF16 projections already active. C2 fused sigmoid branch
  preparation and branch merge into one standalone launch.
- **Artifact:** `artifacts/perf/subsystems/gr_c2_bundle.json`.
- **Correctness:** early/middle/late all passed; maximum input error was
  `3.28e-5`, maximum updated error was `4.77e-6`, and NaN/Inf was zero.
- **Traffic/sync:** H2D/D2H remained outside the measurement window; C1 and
  C2 both used one final synchronization. C2 reduced launches from 9 to 8.

### C1 bundle breakdown and ordering

The artifact contains per-fixture median and p95 values. The accounting is
exclusive: GPU stage spans, CUDA timeline gaps, and host wait outside the GPU
timeline sum to the measured host wall.

| Rank | Substage | Median |
|---:|---|---:|
| 1 | `gr_read_up` | ~32.0 us |
| 2 | `gr_read_down` | ~13.54 us |
| 3 | `normalization` | ~9.2 us |
| 4 | `CUDA dispatch` | ~8.6 us |
| 5 | `gr_write_inject` | ~5.15 us |
| 6 | `host sync/wait` | ~4.5 us |
| 7 | `low-rank/gate compute` | ~3.3 us |
| 8 | `branch merge` | ~3.3 us |
| 9 | `elementwise activation/gating` | ~3.2 us |
| 10 | `residual/writeback` | ~3.1 us |
| 11 | `branch preparation` | ~3.1 us |

### C2 bundle result

| Fixture | C1 wall median | C2 wall median | Improvement |
|---|---:|---:|---:|
| early | 89.33 us | 84.78 us | 5.09% |
| middle | 88.83 us | 85.04 us | 4.27% |
| late | 88.82 us | 84.70 us | 4.63% |

C2 is **not promoted** because it does not reach the structural-candidate
10% isolated-wall gate. The new dominant remains `gr_read_up`; no production
runtime change was made for C2.

## 2026-09-06 — GR-C3 (promoted)

- **Hypothesis:** `gr_read_up` has shape `[10240, 320]`; the 128-thread
  cooperative BF16 geometry is better matched to this projection than the
  256-thread default.
- **Isolated stage command:**
  `./tests/gr/gr_c3_bench tests/fixtures/gr artifacts/perf/subsystems/gr_c3_up_geometry.json`
- **Bundle command:**
  `make gr-c3-bundle`
- **Artifacts:**
  `artifacts/perf/subsystems/gr_c3_up_geometry.json`,
  `artifacts/perf/subsystems/gr_c3_bundle.json`.
- **Production change:** only the GR `gr_read_up` BF16 dispatch selects
  `q38_cuda_bf16_matvec_configured(..., 128, ...)`; `gr_read_down` and
  `gr_write_inject` remain at 256 threads.
- **Correctness:** all early/middle/late stage and bundle fixtures passed;
  maximum input error was `3.28e-5`, maximum updated error was `4.77e-6`,
  stage maximum absolute error was `1.83e-4`, and NaN/Inf was zero.
- **Traffic/sync:** H2D/D2H remained outside the measurement window. Bundle
  launch count stayed at 9 and explicit synchronization stayed at 1.

### C3 isolated geometry

| Geometry | `gr_read_up` median | Improvement vs 256 |
|---:|---:|---:|
| 128 threads | 20.83 us | 32.96% |
| 256 threads | 31.07 us | baseline |
| 512 threads | 57.66 us | -85.6% |

### C3 bundle result

| Fixture | C1 wall median | C3 wall median | Improvement |
|---|---:|---:|---:|
| early | 89.36 us | 78.59 us | 12.05% |
| middle | 88.86 us | 79.07 us | 11.02% |
| late | 88.94 us | 78.96 us | 11.23% |

C3 is promoted to the production GR dispatch. No full inference, model load,
Reference 0 rerun, or change to MoE/GDN/QSA was performed.

## 2026-09-06 — GR-C4 (promoted in isolated bundle)

- **Baseline:** the post-C3 bundle (`gr_read_up` at 128 threads), not the
  older C1/C2 bundle.
- **Hypothesis:** remove launch/dispatch boundaries between adjacent
  operations inside the GR contract without changing tensor semantics.
- **Command:** `make gr-c4-bench`.
- **Artifact:** `artifacts/perf/subsystems/gr_c4_bundle.json`.
- **Fusion plan:**
  - cooperative normalization + `gr_read_down`;
  - cooperative low-rank SiLU + `gr_read_up`;
  - existing branch-preparation/merge fusion;
  - elementwise injection gate + residual writeback.
- **Launches:** 9 -> 6.
- **Synchronization:** one final host synchronization before and after.
- **Transfers:** H2D/D2H remain outside the timed window and remain zero in
  the artifact.

### C4 result versus post-C3

| Fixture | C3 wall median | C4 wall median | Improvement | C4 p95 |
|---|---:|---:|---:|---:|
| early | 78.59 us | 62.40 us | 20.60% | 63.73 us |
| middle | 78.64 us | 62.29 us | 20.79% | 64.06 us |
| late | 78.59 us | 62.27 us | 20.77% | 63.09 us |

The exclusive breakdown accounted for approximately 100% of the measured
wall on all fixtures. The largest remaining bundle category is the fused
normalization/down-projection stage at approximately 23.5 us; `gr_read_up` is
approximately 17.3 us and is no longer the dominant category.

### C4 correctness

- input `max_abs`: `3.27825546e-5`
- input `max_rel`: `2.13350695e-5`
- input RMSE: `8.96732212e-7`
- updated `max_abs`: `4.76837158e-6`
- updated `max_rel`: `9.59034521e-7`
- updated RMSE: `4.23471231e-7`
- NaN/Inf: `0`

C4 passes the isolated promotion gate on early, middle, and late fixtures.
This is an isolated GR-bundle promotion only: no full-model load, full-chain
benchmark, Reference 0 rerun, MoE/GDN/QSA work, or canonical speedup claim was
performed.

## Candidate entry template

```text
## YYYY-MM-DD — GR-CN

- Hypothesis:
- Files changed:
- Isolated command:
- Correctness:
  - max_abs:
  - max_rel:
  - RMSE:
  - NaN/Inf:
- Traffic/sync deltas:
- Kernel median before/after:
- End-to-end median before/after:
- Relative isolated improvement:
- Full-chain result:
- Promotion:
```
