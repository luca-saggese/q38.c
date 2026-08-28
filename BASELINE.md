# BASELINE — M0 donor freeze

This file records the frozen upstream commit from which the Qwen3.8-Flash-Next
/DGX Spark prototype (working name `q38`) is forked. It exists to make the
donor state reproducible and to separate "what came from ds4" from "what we
changed".

## Donor

| Field             | Value                              |
|-------------------|------------------------------------|
| Upstream project  | ds4                                |
| Upstream URL      | https://github.com/antirez/ds4     |
| Remote name       | `upstream`                         |
| Frozen commit     | `c1d4597`                          |
| Commit title      | `qa: update DGX Spark host addresses` |
| Commit date       | 2026-08-23 15:26:19 +0200          |
| Frozen on         | 2026-08-28                          |

## Destination

| Field             | Value                                  |
|-------------------|----------------------------------------|
| Product name      | `q38` (provisional)                    |
| Repository        | https://github.com/luca-saggese/q38.c  |
| Remote name       | `origin`                               |
| Branch            | `qwen38-spark-proto`                   |

## Toolchain baseline (recorded on the donor machine, macOS host)

These are the *host* values captured at freeze time. The authoritative
driver/runtime/toolkit values for the DGX Spark target must be captured on the
real device and recorded below in the "Spark target" section (M0 spec §12
UNKNOWN).

| Field             | Value                                     |
|-------------------|-------------------------------------------|
| Host OS           | macOS (Darwin)                            |
| Host `cc`         | Apple clang 16.0.0 (clang-1600.0.26.6)    |
| `nvcc` on host    | not installed (build happens on Spark)    |

## Spark target (UNKNOWN — to be filled on device)

| Field               | Value |
|---------------------|-------|
| DGX OS version      | UNKNOWN |
| CUDA driver version | UNKNOWN |
| CUDA runtime version| UNKNOWN |
| CUDA toolkit version| UNKNOWN |
| nvcc version        | UNKNOWN |
| Compute capability  | SM 12.1 (to be verified at runtime) |

## Scope note

M0 is a fork-and-prune: the donor code remains identifiable, but the product
does not carry forward ds4 compatibility decisions (DeepSeek/GLM/DSpark/MTP,
Metal, ROCm, CPU, distributed, tensor parallelism). See
`implementations steps/M0_implementation_spec.md`.
