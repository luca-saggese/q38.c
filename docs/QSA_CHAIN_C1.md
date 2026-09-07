# QSA_CHAIN_C1

QSA_CHAIN_C1 is the single-token normal-decode QSA device chain.  It keeps
Q/K/V projection results, index compression, the causal QSA cache, selection
workspace, gathered K/V, attention, and the BF16 output projection on the
CUDA stream.  The host-visible layer boundary is one hidden-state H2D upload
and one 2,560-element output D2H download.

The chain uses the existing QSA projection, gather, and attention kernels and
the enqueue-only `q38_cuda_bf16_matvec_device()` for the index and output
projections.  Per-layer cache storage grows geometrically and remains owned by
the CUDA context.  No intermediate QSA transfer or synchronization is
performed; the final stream wait belongs to the layer boundary.

The production callback is installed by the runtime backend for
`token_count == 1`.  Batched prefill and diagnostic state snapshots retain the
reference path, so the chain does not change GDN or its state ownership.

## Fixture gate

`qsa_bench ... qsa_chain_c1` validates layers 3 (early), 23 (middle), and 47
(late).  The measured artifact is
`artifacts/perf/current/qsa_chain_c1.json`.  All three fixtures must have
finite output, exact selected IDs, `max_abs <= 3e-3`, one sync, 10,240 input
bytes, 10,240 output bytes, and at least 10% complete-layer reduction before
the mode is considered promoted.

The fixture gate passed for the current capture.  The canonical 16-token
integration is intentionally a separate promotion step and must only be
recorded after this fixture gate is green.
