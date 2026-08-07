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


## Round 2: attempt to push decode past 41 tok/s (2026-08-07)

Final revalidated state (committed tree, 3 official replicas):
decode 40.64/40.56/40.63 (median 40.63), steady 40.82, prefill
753-779 (drift window), first token ~27.3-28.1 ms, all frontier
hashes canonical. **41 tok/s was not reached with exact methods.**

Everything below was measured and left OUT of the tree:

- Q8 r4 matvec (DS4_METAL_Q8_MV_ROWS=4): initially appeared to give
  decode 43.7/steady 44.1, but that was row-skipping corruption: the
  q_a/kv pair kernel and the hc_expand4 kernels have compile-time
  NR0=2 geometry and skip rows when the dispatch reports nr0=4.
  CAUTION: the official frontier hash covers only prefill logits;
  decode-token exactness is covered only by the A/B harness
  exact_rows and ds4_test --metal-tensor-equivalence greedy checks.
  With the pair/hc_expand dispatches pinned to nr0=2, r4 singles are
  bit-exact but flat (40.2-40.4 both ways). Reverted.
- IQ2 pair-SwiGLU / Q2_K sum6 nsg=1 and nsg=4 decode variants (new
  pipelines; NSG!=2 needs the collective grid/sign table load in
  kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 scaled as nval=8/NSG):
  bit-exact, flat (nsg1 -0.16%, nsg4 -0.07%). Reverted.
- DS4_METAL_Q8_DECODE_MPP (NAX matmul at n_tok=1): not bit-exact on
  M5 (different accumulation order) - rejected.
- Decode pipeline fast lookup extended to the IQ2XXS/Q2K tape
  (one-line probe): exact, +0.01% - host is not the bottleneck at
  ~98% GPU busy. Reverted.
- Split schedules 2/0, 8/0, 12/0, 4/16, 8/16, 2/16, 6/12: all within
  +/-0.15%; 4/0 remains optimal.
- DS4_METAL_ENABLE_HC_RMS_SCALE_PROJ / GATHERED_KV_STAGE /
  DECODE_NORM_EXACT_VIEWS / F32_DECODE_EXACT_VIEWS / SHARED_KV_PAD /
  UNRETAINED_COMMAND_BUFFERS: all flat (|delta| <= 0.15%).
- DS4_METAL_DISABLE_METAL4: prefill 784 -> 390 (NAX tensor matmuls are
  the prefill driver), decode slightly worse. Keep Metal 4 on.
- MTP speculative decode (Q4K MTP module, --mtp-draft 1..4, greedy
  exact verify): draft 1 ~= baseline (41.6 vs 41.5 t/s CLI short
  prompt), drafts 2-4 progressively slower (28.2, 23.5 t/s). Verify
  cost exceeds acceptance gains on this config.

Structural work that could plausibly beat 41 (out of session scope):
NAX/tensor-op IQ2XXS MoE kernels (prefill routed_moe is 40% of layer
time; decode pair_swiglu is the largest decode matvec), Metal
command-buffer replay for the fixed decode tape, or relaxing the
bit-exactness discipline (e.g. NAX Q8 decode matmuls).


## Round 3: two more exact M5 fusions — decode crosses 41 (steady)

Commits 7a3b9ca and 0d0a78b (ds4f-q2, M5 Max):

1. **q_a/kv pair + quad compressor store merge**: the Q8 q_a/kv pair
   projection and the four F16 compressor projections all read the same
   normalized attention input; one dispatch with two virtual NSG=4
   cohorts per threadgroup for the Q8 range and the verbatim quad
   ranges.  Controlled A/B +0.44%/+0.57%; official interleaved pairs
   +0.40%/+1.48%.
2. **Emit-path compressor finalize merge**: every 4th token, each
   layer's eleven tiny single-row dispatches (norm, rope, FP8
   round-trip + F16 commit for the attention row; norm, rope,
   Hadamard+FP4 QAT for the indexer row; both ratio-4 state shifts)
   become ONE dispatch (22 threadgroups).  Six controlled inverse A/B
   runs all positive (+0.36%..+0.59%), every block bit-exact.

Also measured and reverted: q_b + indexer q_b compound merge (exact,
flat: +0.03% — the indexer chain is idle below top_k=512 at the 2K
benchmark window, and cohort-packing the 24 MB q_b matvec offsets the
launch saving), unpacked cohort variants (flat).

### Final numbers (6 official replicas, drift-controlled window)

| metric | session baseline | final | delta |
|---|---:|---:|---:|
| prefill | 784.13 | 790.2 med | +0.8% |
| decode gen_tps | 39.39 | **40.85 med (40.37-41.05)** | **+3.7%** |
| steady decode | 39.60 | **41.05 med (40.54-41.25)** | **+3.7%** |
| first token | 28.33 ms | 27.0 ms med | -4.7% |

All frontier hashes canonical
(eb794f497861d7d9e373665f6e115d8ebe4e17c13c431aa7e318282f16a0f21d);
full `make test` green.

Remaining unfused emit-path work: the pool pipeline (pack + softmax +
product + 8-lane sum_rows).  The sum_rows 8-thread simd topology is
delicate to replicate exactly; expected value ~+0.2-0.3%.


### Round-3 final tip (09c53a5): 8 official replicas

Adds the M5 admission of the compressor exact softmax+product fusion
(+0.10-0.24% controlled; flat when first screened pre-finalize).

decode gen_tps: 40.38/40.66/40.94/40.95/40.96/40.99/41.01/41.04
    median 40.96 tok/s  (+4.0% vs the 39.39 session baseline)
steady decode:  median 41.19 tok/s (+4.0% vs 39.60)
prefill:        785-788 (no regression)
first token:    ~27.2 ms med
All frontier hashes canonical; focused tests and full make test green.
