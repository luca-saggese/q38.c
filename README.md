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
./q38 -m model.gguf -p "Explain why the sky is blue." --max-tokens 128
```

Add `--json` for deterministic machine-readable output.

## Server and HTTP client

The direct `q38` binary remains the canonical local inference and benchmark
path. The resident server loads the model once and keeps the runtime/session
alive:

```
make q38-server
./q38-server --model model.gguf --tokenizer /path/to/tokenizer \
  --host 127.0.0.1 --port 8000
```

`q38-server-mock` is the model-free protocol test server. It is used for
HTTP/parser/SSE tests and never loads a GGUF. `q38-cli` is an HTTP-only
localhost client; it does not link CUDA, GGUF, or the Q38 runtime:

```
./q38-cli
./q38-cli --no-autostart --host 127.0.0.1 --port 8000
```

The server exposes the OpenAI Completions, Chat Completions, and Responses
APIs plus Anthropic Messages, streaming SSE, tools, reasoning fields,
cancellation, and image request parsing. Unsupported Q38 runtime capabilities
return explicit errors instead of silently selecting a donor model path.

## Directional steering

Q38 directional steering uses a raw little-endian float32 file containing
exactly `48 x 2560` normalized direction values. It is loaded once into the
resident runtime and can be enabled on the direct CLI or resident server:

```
./q38 -m model.gguf --dir-steering-file direction.bin \
  --dir-steering-ffn 1.0 --dir-steering-attn 0.0 -p "..."
./q38-server --model model.gguf --tokenizer /path/to/tokenizer \
  --dir-steering-file direction.bin --dir-steering-ffn 1.0
```

The server accepts a request-local `q38.steering` override. Overrides are
cleared after every request and cannot contaminate the next session:

```json
{"q38":{"steering":{"ffn":-0.5,"attn":0.0}}}
```

Reference 0 remains steering-disabled; steering experiments must use separate
artifacts and must not be compared with Reference 0 as a performance claim.

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
