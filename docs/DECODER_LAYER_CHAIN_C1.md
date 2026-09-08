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
The full-model canonical integration completed in a short run with the chain
enabled and produced IDs identical to the stable host-boundary path.  A
longer run exposed an inter-token accumulation/regression: latency increased
substantially and the process could appear to stop near the final token.
Therefore the full-layer chain is currently disabled in production while the
stable host-boundary path remains enabled.

The measured chain run was functionally correct for 16 tokens but did not pass
the performance or long-run stability gates: its 16-token decode median was
151.020 ms/token versus 130.207 ms/token for the stable path.  It must not be
re-enabled until the accumulation source is isolated and a long-generation
benchmark demonstrates stable per-token latency.
