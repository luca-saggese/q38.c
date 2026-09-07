# DECODER_LAYER_CHAIN_C1

The decoder-layer C1 entry point is
`q38_forward_cuda_decoder_layer_chain_backend`.  It owns the normal single
token layer island:

```
GR_READ -> GDN/QSA -> GR_WRITE -> GR_READ -> router/grouped-MoE -> GR_WRITE
```

The island uses persistent `device_hidden_a` and `device_hidden_b` ping-pong
buffers.  GR, GDN, QSA, router selection, deterministic grouped Q2 reduction,
and the BF16 shared expert execute on the context stream.  The chain itself
does not perform intermediate transfers or waits; its host boundary is one
hidden upload and one output download followed by the final stream wait.
QSA state can be explicitly seeded after host prefill with
`q38_forward_cuda_load_qsa_state`.

GDN_CHAIN_C1 and QSA_CHAIN_C1 math remain in their existing implementations.
The new device entry points only expose their resident workspaces to the
layer orchestrator.  File-backed PLE remains at the existing layer boundary.

The existing GDN and QSA three-fixture gates remain green after this change.
The full-model canonical integration now completes with the chain enabled.
The 16-token generated IDs are bit-identical to the stable host-boundary
path, with no NaN/Inf, fallback, or PLE stall.

The current measured chain run is functionally GREEN but does not pass the
performance promotion gate yet: its 16-token decode median was 151.020
ms/token versus 130.207 ms/token for the stable path.  The chain remains
enabled for correctness validation, but it must not be presented as a
performance promotion until a complete-layer fixture and a representative
microbenchmark demonstrate the required improvement.
