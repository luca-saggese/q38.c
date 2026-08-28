# q38

A native inference runtime skeleton for **Qwen3.8-Flash-Next**, targeting the
**NVIDIA DGX Spark** (GB10 / Grace Blackwell / SM 12.1 / 128 GB unified
coherent LPDDR5x memory / Linux aarch64 / CUDA).

`q38` is at **M0**: it is a fork-and-prune of [ds4](https://github.com/antirez/ds4)
reduced to a minimal, single-target runtime skeleton. There is **no inference
path yet** — M0 provides platform probing, GGUF inspection, tensor inventory,
and a memory-plan dry run only.

## Scope

- **Target only:** DGX Spark GB10, SM 12.1, CUDA. Anything else is refused
  explicitly, never silently degraded.
- **No** Metal, ROCm, CPU, distributed, or tensor-parallel backends.
- **No** DeepSeek / GLM / DSpark / MTP / vision model-family bindings.

## Build

```
make spark
```

This produces:

- `./q38`
- `./tests/test_platform`
- `./tests/test_gguf`
- `./tests/test_memory`

The build prints the selected CUDA arch and refuses to link if it cannot
generate code for the target device. The authoritative driver/runtime/toolkit
versions are recorded in `BASELINE.md`.

## Usage

```
./q38 --platform
./q38 --inspect model.gguf
./q38 --list-tensors model.gguf
./q38 --memory-plan model.gguf
```

Add `--json` for deterministic machine-readable output.

## Acceptance

```
make clean
make spark
make m0-acceptance
```

See `implementations steps/M0_implementation_spec.md` for the definition of
done, the commit plan, and the M0 test matrix.

## Repository layout

- `q38.h`, `q38_cuda.h` — narrow public API.
- `q38_platform.{c,h}`, `q38_cuda.cu` — platform guard (GB10 / SM 12.1).
- `q38_gguf.{c,h}` — GGUF v3 parser core (isolated from model families).
- `q38_memory.{c,h}` — memory telemetry.
- `q38.c` — inspection CLI.
- `tests/` — M0 test suite (`test_platform`, `test_gguf`, `test_memory`).
- `ds4_ssd.{c,h}`, `cuda/mmq/` — parked, out of the M0 link.
- `to_be_deleted/` — donor sources parked for deletion review.
- `BASELINE.md`, `THIRD_PARTY_NOTES.md` — donor freeze and provenance.

## License

MIT. See `LICENSE`.
