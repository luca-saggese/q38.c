# GR subsystem

This directory is the normative working record for the Q38 grouped-residual
(GR) subsystem. The isolated implementation and fixture suite lives under
`tests/gr`; this directory records the semantics, benchmark baseline, and
optimization history.

## Frozen baselines

- Q2 Reference 0 full-chain decode remains immutable at **590.93 ms/token**
  and **1.692 tok/s**.
- GR optimization starts from the isolated baseline in
  `artifacts/perf/subsystems/gr_reference.json`.
- GR-C1 has passed the model-free projection gates; no full-chain speedup has
  been claimed or used to modify Reference 0.
- GR-C2 was measured and rejected by the 10% isolated-wall gate.
- GR-C3 has passed the model-free bundle gates and is promoted only for
  `gr_read_up`; no full-chain speedup has been claimed.
- GR-C4 has passed the post-C3 isolated bundle gate: six launches, one host
  synchronization, and approximately 20.7% lower GR bundle wall.

The full-chain Reference 0 benchmark must not be rerun for a GR candidate
until the candidate passes all isolated gates.

## Production execution steps

The production call chain is implemented by `full_gr_read` and
`full_gr_write` in `q38_forward.c`:

1. Receive a float32 residual with four 2560-wide branches.
2. Normalize each branch with grouped RMS and `1 + hc_norm`.
3. Run the 320-wide `input_mix_weight_down` projection.
4. Apply `SiLU(value / 4)`.
5. Run the 10240-wide `input_mix_weight_up` projection.
6. Apply sigmoid gates.
7. Average the four gated branches for GR read.
8. Run the intervening GDN/QSA or MoE block.
9. Run the four-row `block_inject_weight` projection for GR write.
10. Apply `2 * sigmoid(inject / 4)`.
11. Add the scaled 2560-wide block output to each residual branch.

The exact contract, tensor shapes, dtype rules, and implementation
classification are frozen in
[`tests/gr/GR_CONTRACT.md`](../../tests/gr/GR_CONTRACT.md).

## Suite layout

```text
tests/gr/
  GR_CONTRACT.md
  gr_reference.c/.h       independent production-semantics oracle
  gr_extract_fixtures.c   compact BF16-to-F32 fixture extractor
  gr_bench.cu             isolated CUDA baseline benchmark
  gr_c1_bench.cu          generic-vs-cooperative BF16 projection benchmark
  gr_c2_bench.cu          C1/C2/C3/C4 bundle breakdown and candidates
  gr_c3_bench.cu          gr_read_up geometry benchmark
  test_m3_gr_ref.c        historical scalar smoke/golden test
  test_m3_gr_cuda.cu      historical CUDA golden test
  test_m3_gr_binding.c    production tensor binding test
tests/fixtures/gr/
  layer_0_early/
  layer_23_middle/
  layer_47_late/
```

The fixture pack contains the four GR tensor families, residual/block inputs,
expected oracle outputs, and provenance metadata. Captured activations are
used where available; replayed branch/layer inputs are explicitly marked in
`metadata.json` and must not be described as Reference 0 boundary captures.

## Required workflow

For every GR candidate:

```text
edit
  -> make test-gr (or the smallest relevant target)
  -> make gr-fixtures
  -> make gr-bench
  -> make gr-c1-bench when changing projection dispatch
  -> make gr-c2-bench using the existing fixture pack only
  -> make gr-c3-bench using the existing fixture pack only
  -> make gr-c4-bench against the post-C3 baseline
  -> inspect correctness and decomposition
  -> only then consider a full-chain benchmark
```

The isolated benchmark uses 100 warmup iterations and 1000 measured
iterations. The C1 bundle benchmark reports median and p95 for
`gr_read_down`, `gr_read_up`, `gr_write_inject`, normalization, low-rank/gate
compute, branch preparation, branch merge, elementwise activation/gating,
residual/writeback, CUDA dispatch, host sync/wait, memcpy, and other. Its
exclusive accounting explains at least 97% of the measured host wall; H2D and
D2H are explicitly outside the measured window.

`gr-c2-bench` and `gr-c3-bench` do not regenerate fixtures or open the GGUF.
They use only `tests/fixtures/gr/`, so candidate iteration does not load the
model. `gr-c4-bench` uses the same fixture-only policy and measures the
post-C3 128-thread dispatch as its baseline.

## Promotion gates

A candidate may not enter the production path unless:

- early, middle, and late fixtures pass;
- CPU oracle and CUDA output have no NaN/Inf;
- max absolute/relative error and RMSE remain within the recorded contract;
- no fallback is taken;
- no new H2D, D2H, or D2D traffic is introduced;
- no new explicit host synchronization is introduced;
- the isolated GR wall improves by at least 10% for a structural candidate.

The next candidate must be recorded in `CHANGELOG.md` and in a separate
artifact under `artifacts/perf/subsystems/`. Do not overwrite the baseline
artifact.
