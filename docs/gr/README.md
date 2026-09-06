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
  -> inspect correctness and decomposition
  -> only then consider a full-chain benchmark
```

The isolated benchmark uses 100 warmup iterations and 1000 measured
iterations. It reports kernel-only and end-to-end call timing, kernel
launches, explicit host synchronizations, transfer bytes, logical bytes read,
effective read bandwidth, and a diagnostic decomposition of normalization,
down projection, up projection, branch merge, and injection. Decomposition
spans are non-additive diagnostics and must not be mistaken for a second wall
clock.

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
