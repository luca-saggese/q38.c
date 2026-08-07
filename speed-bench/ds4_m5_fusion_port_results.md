# DS4 M5 port results: decode fusion promotion (ds4f-q2, M5 Max 128 GiB)

Results of applying `ds4_m5_fusion_port_handoff.md` on this machine.
Model: ds4f-q2 (IQ2XXS experts; the 156 GiB MXFP4 model does not fit
128 GiB RAM). All fusion gates are quant-agnostic shape gates, so the
port applies; the MXFP4-only pre-M5 extras (nsg=1 MoE decode,
fixed-route, pipeline fast lookup, split5/16 schedules) do not engage
with IQ2XXS experts and were not ported.

Code: commit e42d6dc (adds `ds4_gpu_device_is_m5_apple_silicon()` and
admits M5 at the seven gates below; `DS4_METAL_DISABLE_*` rollbacks
unchanged). `make test` fully green on this config.

## Baseline (branch tip caf64d1, fusions off, official command, 3 replicas)

| metric | median | min-max |
|---|---:|---:|
| prefill | 784.13 tok/s | 783.38-784.72 |
| decode | 39.39 tok/s | 39.33-39.48 |
| steady decode | 39.60 tok/s | 39.51-39.63 |
| first token | 28.33 ms | 27.29-28.83 |

Local canonical frontier SHA-256 (ds4f-q2; the MXFP4 canonical
`6f7edd6d...` does not apply):
`eb794f497861d7d9e373665f6e115d8ebe4e17c13c431aa7e318282f16a0f21d`
— stable across every retained replica of every variant.

GPU busy during decode ~98% (DS4_METAL_GPU_BUSY_PROFILE), so the lever
is per-dispatch overhead inside a busy timeline, as on pre-M5.

## Promoted to M5 default (all bit-exact in every retained A/B block)

| feature | forward A/B (2+ processes) | inverse (disable) |
|---|---:|---:|
| F1 HC norm+mix | +0.13% / +0.10% | -0.61% |
| F2 compressor quad projection | +0.20% / +0.17% | -0.42% |
| F3 router + shared gate/up | +0.30% / +0.28% | -1.04% |
| F4 qkv norm + KV RoPE + FP8 store | +0.64% / +0.51% | -0.58% |
| attn inverse-RoPE into FA reduce | +0.28% / +0.14% | -0.29% |
| compressor ratio-4 decode pack | +0.31% / +0.39% | -0.42% |
| router transform+finalize+weights | +0.12% / +0.07% / +0.15% | -0.21% |

## Interleaved same-window official replicas (drift-controlled)

| arm | decode tok/s | steady | prefill | first token |
|---|---|---|---|---|
| all 7 ON | 40.59 / 40.41 / 40.54 (med 40.54) | 40.7 med | 787.6 med | 27.9 ms med |
| all 7 OFF | 39.59/39.55/39.48/39.48 (med 39.51) | 39.7 med | 786.9 med | 28.1 ms med |

Decode **+2.6%** same-window, **+2.9%** vs the baseline median; no
prefill regression. Machine drift between distant windows reached
~10%, so only interleaved/intra-process comparisons were trusted.

## Screened and rejected on M5 (flat or negative, all exact)

- 5/16 decode split schedule: -0.12% (4/0 already ~98% busy on M5)
- compressor exact reduction fusion: -0.16%
- head RMS+RoPE static pipeline: -0.06%
- raw zero attention mask: -0.06%
- output Q8 nr4 + HC sum/norm fusion + HC weights4 (combined): -0.9%
- Q8 decode exact views (alone): -0.6%
