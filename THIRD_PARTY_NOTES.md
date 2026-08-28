# THIRD_PARTY_NOTES

Inventory of third-party code carried into the `q38` (Qwen3.8-Flash-Next /
DGX Spark) prototype. This file is maintained alongside `BASELINE.md` so that
licensing and vendoring provenance remain auditable after the M0 prune.

## Vendored: llama.cpp mmq CUDA kernels

Directory: `cuda/mmq/`

See `cuda/mmq/VENDOR.md` for the full file inventory and patch notes. Summary:

| Field         | Value                                              |
|---------------|----------------------------------------------------|
| Source        | https://github.com/ggml-org/llama.cpp              |
| Commit        | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`         |
| Commit date   | 2026-05-14                                         |
| License       | MIT (copyright "2023-2026 The ggml authors")       |
| Compatibility | Compatible with the ds4/q38 MIT license            |

> NOTE (M0): these kernels are a prefill/quantized-matmul tier. M0 removes
> the DeepSeek/GLM model-family bindings but keeps the kernels as generic
> quantized-matmul primitives only after a function-by-function audit. Their
> retention in the final M0 link is gated on that audit; if they cannot be
> kept without dragging in graph/model assumptions, they are parked.

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
