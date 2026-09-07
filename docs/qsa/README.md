# QSA optimization

The QSA fixture benchmark is run with:

```sh
make bench-qsa
```

It reads `tests/fixtures/qsa/{early,middle,late}` and writes the measured
baseline and candidate data to
`artifacts/perf/subsystems/qsa_reference.json`. No model load or full-chain
benchmark is performed by this target.

The first candidate is QSA-C1, a single-token output-projection matvec. The
candidate is measured against the complete layer, not only the projection
kernel, and is promoted only if the correctness and 10% complete-layer gates
pass on early, middle, and late fixtures. The explicit `qsa_chain_c1` mode
extends this to the normal-decode device chain: Q/K/V, index/compress, cache
state, selection, attention, and output projection stay device-resident with
one input H2D and one output D2H at the layer boundary. The production
callback is used for single-token evaluation; chunked prefill keeps the
reference path.
