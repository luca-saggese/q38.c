# q38

A native inference runtime skeleton for **Qwen3.8-Flash-Next**, targeting the
**NVIDIA DGX Spark** (GB10 / Grace Blackwell / SM 12.1 / 128 GB unified
coherent LPDDR5x memory / Linux aarch64 / CUDA).

`q38` is the native Qwen3.8-Flash-Next runtime for the DGX Spark production
path. The canonical execution flow is `q38_runtime -> q38_session ->
q38_session_prefill/q38_session_eval`; legacy forward runners are not part of
the benchmark contract.

## Scope

- **Target only:** DGX Spark GB10, SM 12.1, CUDA. Anything else is refused
  explicitly, never silently degraded.
- **No** Metal, ROCm, CPU, distributed, or tensor-parallel backends.
- **No** DeepSeek / GLM / DSpark / MTP / vision model-family bindings.
- Q2 conversion retains only `third_party/gguf-tools/quants.{c,h}` for the
  conversion utility and fixture test; those sources are not linked into
  production inference.

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
- `q38_platform.{c,h}`, `cuda/q38_cuda.cu` — platform guard (GB10 / SM 12.1).
- `q38_gguf.{c,h}` — GGUF v3 parser core (isolated from model families).
- `q38_memory.{c,h}` — memory telemetry.
- `q38_tokenizer.{c,h}` — native byte-level BPE tokenizer loaded directly from
  the local tokenizer files. It provides encode/decode, frozen chat-template
  rendering, structured text/image/video content markers, and special-token
  handling. Python is used only by the golden-vector tools.
- `q38.c` — inspection CLI.
- `tests/` — runtime, canonical benchmark, and fixture test suites.
- `third_party/gguf-tools/quants.{c,h}` — retained conversion-only donor pair.
- `BASELINE.md`, `THIRD_PARTY_NOTES.md` — donor freeze and provenance.

## M3 reference boundaries

The Python tokenizer remains an oracle for frozen token IDs and chat-template
behavior in the golden-vector tools; it is not linked into the native runtime.
The q38 GGUF ABI is
the original Qwen3.8 tensor layout as emitted by the q38 converter. Existing
metadata records `q38.quant_manifest`, `q38.source_revision`,
`q38.down_proj_layout`, and the runtime-only/exclusion flags; llama.cpp is a
semantic reference, not a binary-layout requirement.

## License

MIT. See `LICENSE`.
