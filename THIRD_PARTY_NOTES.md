# THIRD_PARTY_NOTES

Inventory of third-party code carried into the `q38` (Qwen3.8-Flash-Next /
DGX Spark) prototype. This file is maintained alongside `BASELINE.md` so that
licensing and vendoring provenance remain auditable after the M0 prune.

## Vendored: quantization block writer

Directory: `third_party/gguf-tools/`

The retained `quants.c`/`quants.h` pair is used only by the Q2 conversion
utility and its quantization fixture test. It is not linked into the
production runtime. The rest of the former donor tree is intentionally
removed because it has no dependency from the current q38 build.

## Removed: llama.cpp mmq CUDA kernels

The former `cuda/mmq/` donor tree was removed during the 2026-09 cleanup. It
was not referenced by the current q38 runtime or test targets.

## Vendored: linenoise

Files: `linenoise.c`, `linenoise.h`

| Field   | Value                                   |
|---------|-----------------------------------------|
| Source  | antirez/linenoise                       |
| License | BSD-2-Clause (see header in files)      |

> NOTE (M0): linenoise is only used by the interactive CLI/agent frontends.
> The minimal M0 CLI (`q38.c`) does not link it. Keep in tree but out of the
> M0 link unless a future requirement re-adds an interactive mode.

## License of this project

`LICENSE` — MIT, copyright "2026 The ds4.c authors" and "2023-2026 The ggml
authors".
