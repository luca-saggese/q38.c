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
