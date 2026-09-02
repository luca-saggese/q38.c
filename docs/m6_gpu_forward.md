# M6 CUDA diagnostic forward

`tests/m6_real_forward_gpu` is a separate diagnostic executable.  It emits
the same `q38-m6-full-trace-v4` JSON schema as `tests/m6_real_forward`, but
installs a CUDA row-matvec backend around the unchanged scalar graph.  BF16,
Q8, F32, and Q2 rows are evaluated on the device using the validated M3
projection and M2/M6 matvec primitives; graph sequencing, state handling,
and diagnostics remain identical.  The very wide vocabulary head intentionally
falls back to the scalar matvec to avoid hundreds of thousands of synchronizing
diagnostic launches.  This is deliberately a correctness path, not an M7
fusion or tuning path.

Build and run the primitive gates with:

```sh
make m6-gpu-forward
```

On a machine with the runtime GGUF and scalar trace, `make m6-gpu-progressive`
also writes `artifacts/m6/gpu_real_forward_trace.json` and checks the
1-layer, 4-layer, and 48-layer checkpoints (layers 0, 3, and 47).
