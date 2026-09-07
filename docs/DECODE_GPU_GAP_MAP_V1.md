# PERF-V2A — Decode-Only GPU Gap Map

## Scope

This is a diagnostic-only follow-up to PERF-V2. The trace used one model
resident process, one warmup/prefill path, and four measured decode tokens.
Startup and prefill were excluded from all decode-only statistics. The trace is
`artifacts/perf/current/nsys_perf_v2a/q2_decode_4.nsys-rep`.

Compile-time NVTX ranges were added for token, layer, GR read/mixer/GR write,
router, MoE, PLE, LM head, argmax, H2D, D2H, D2D, and host waits. The ranges
are observational and do not add CUDA synchronization.

Correctness stayed GREEN: generated IDs were
`271, 248068, 198, 760, 1156`, all values were finite, and fallback was false.

## Decode-only reconciliation

The token envelope is the `TOKEN` forward range plus its following `ARGMAX`
range. Mean values over the four measured tokens are:

| Component | ms/token |
|---|---:|
| Token wall envelope | 131.778 |
| GPU kernel active | 62.752 |
| GPU memcpy active | 1.453 |
| GPU device-operation union | 64.204 |
| GPU idle inside token envelope | 67.574 |
| CPU-only before first GPU operation | 0.123 |
| CPU-only after last GPU operation | 0.008 |
| **Reconciled total** | **131.778** |

The residual is zero. Inter-token idle was only `0.159 ms` on average, so
the dominant idle budget is inside the token rather than between tokens.

The q2 observer independently recorded 739 real CUDA synchronizations per
token, `12,311,808` H2D bytes, and `8,843,520` D2H bytes. Nsight's decode-only
timeline measured `12,929,088` H2D bytes and `9,974,020` D2H bytes. The
difference is instrumentation scope: q2 reports the backend's logical
transfers, while Nsight reports all CUDA memcpy activity.

## Large-gap attribution

There were approximately 734 gaps larger than 20 us per token, totaling
`68.588 ms/token` by kernel-to-kernel gap measurement. The union-based idle
number above is the non-overlapping figure and is the authoritative value.

Every classified large gap overlapped an NVTX `HOST_WAIT_STREAM` or
`HOST_WAIT_EVENT` range. CPU-only pre/post windows were negligible. The q2
sync reasons identify the dependency pattern:

- 665 `MATRIX_D2H` waits/token;
- 48 `MOE_GROUPED_D2H` waits/token;
- 12 `QSA_QKV` waits/token;
- 12 `MATRIX_BATCH_D2H` waits/token;
- 2 argmax waits/token.

Thus the GPU idle is predominantly host-wait/D2H dependency time, not a
long CPU-only interval before the first kernel. H2D preparation is present but
does not dominate the classified gap wall.

## Transition boundaries

The largest requested inter-stage boundaries over the four measured tokens
were:

| Transition | Count | Idle wall over trace |
|---|---:|---:|
| GR_WRITE -> GR_READ | 376 | 21.188 ms |
| GR_READ -> MOE via ROUTER | 384 | 19.242 ms |
| MOE -> GR_WRITE | 192 | 12.446 ms |
| MIXER -> GR_WRITE | 192 | 11.388 ms |
| GR_READ -> MIXER | 192 | 10.143 ms |
| LM_HEAD -> ARGMAX | 4 | 0.717 ms |
| token -> token | 3 | 0.477 ms total |

The three boundaries requested for prioritization are therefore
**GR_WRITE -> GR_READ**, **GR_READ -> MOE via ROUTER**, and
**MOE -> GR_WRITE**. They are all marked by host waits and repeated D2H
dependencies.

An important additional observation is that intra-stage gaps are even larger:
`MIXER_GDN -> MIXER_GDN` contributes `123.140 ms` and
`MIXER_QSA -> MIXER_QSA` contributes `49.895 ms` over the four-token trace.
Those are repeated waits between kernels inside a mixer range, not clean
inter-stage transitions.

## Kernel ownership

The decode-only trace contains, per token:

| Kernel | Instances | Owner |
|---|---:|---|
| `matrix_batch_generic_kernel` | 0 | Prefill-only batch path |
| `bf16_matvec_kernel` | 675 | GR, router/shared MoE projections, and token-count-1 LM head |
| `project_kernel` | 38 | GDN mixer |
| `q2_grouped_gate_up_kernel` | 48 | Routed MoE gate/up |
| `q2_grouped_down_private_kernel` | 48 | Routed MoE down |
| `q2_grouped_deterministic_reduce_kernel` | 48 | MoE grouped reduction |
| `argmax_kernel` | 1 | Argmax |

The old PERF-V2 values of **25 matrix-batch kernels/token** and
**257 BF16 matvec kernels/token** were whole-process normalizations:
`675 / 27 = 25` and `6939 / 27 = 257`. They mixed startup/prefill/warmup and
measured work. The decode-only trace disproves that interpretation for the
measured token path: it has **0 matrix-batch kernels/token** and
**675 BF16 matvec kernels/token**.

## Conclusion

The current post-PLE target is not limited by PLE or by token-to-token gaps.
The measurable GPU idle budget is caused by repeated host waits around D2H
boundaries, especially inside GDN/QSA mixers and at the GR/MoE handoff
transitions. No optimization was implemented in PERF-V2A.
