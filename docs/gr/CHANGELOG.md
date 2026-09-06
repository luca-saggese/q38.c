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
