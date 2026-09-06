# M6 CUDA diagnostic forward

`tests/m6_real_forward_gpu` is a separate diagnostic executable.  It emits
the same `q38-m6-full-trace-v4` JSON schema as `tests/m6_real_forward`, but
installs strict CUDA row and matrix matvec backends around the unchanged
scalar graph.  BF16, Q8, F32, and Q2 rows are evaluated on the device using
the validated M3 projection and M2/M6 matvec primitives; graph sequencing,
state handling, and diagnostics remain identical.  A backend refusal is fatal
rather than silently returning to scalar execution.  This is deliberately a
correctness path, not an M7 fusion or tuning path.

Build and run the primitive gates with:

```sh
make m6-gpu-forward
```

The CUDA runner records per-stage backend/scalar row counts and host elapsed
time in the `stage_usage` field of its trace.  The matrix backend is also used
for router and ordinary projection matrices, and the Q2 routed-expert backend
avoids the previous per-row gate/up/down fallback.

The independent GGUF oracle can be exercised with, for example:

```sh
make m6-autoregressive-oracle M6_ORACLE_TOKENS=1 M6_ORACLE_DEVICE=cuda
```

It recomputes each causal prefix and writes
`artifacts/m6/autoregressive_oracle.json`; this is intentionally an oracle,
not a production cache implementation.

The phase-7 GPU-only validation ladder (BF16/Q8/Q2, GR, GDN, PLE, QSA, and
MoE) is available as:

```sh
make m6-gpu-phase7
```

The 1/4/48-layer full-model comparison is not reported by this target when a
required CUDA stage is unavailable; incomplete traces are rejected rather
than recorded as evidence.

On a machine with the runtime GGUF and scalar trace, `make m6-gpu-progressive`
also writes `artifacts/m6/gpu_real_forward_trace.json` and checks the
1-layer, 4-layer, and 48-layer checkpoints (layers 0, 3, and 47).
