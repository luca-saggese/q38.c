# GDN optimization changelog

## S3-C1 - resident single-token projections

- Promoted the real early/middle/late GDN candidate after standalone fixture
  validation.
- Routed decode-time single-token GDN projections through
  `q38_cuda_gdn_project`, while retaining the generic batch kernel for
  multi-token prefill.
- Preserved the existing launch and synchronization boundary: five launches
  and five synchronizations per isolated token.
- Measured approximately 53--55% lower isolated GDN wall time than the
  pre-C1 production path.
- Correctness gates: no NaN/Inf, output `max_abs <= 5e-3`,
  output `max_rel <= 1e-2`, next history `max_abs <= 1e-4`, and
  recurrent state `max_abs <= 1e-5`.

## S3-C2 - persistent device recurrence

- Added a complete single-token CUDA GDN island after the BF16 projections.
- Moved channel-major causal convolution/history update, q/k/v preparation,
  FP32 recurrent arithmetic, state update, recurrent normalization/gating, and
  output projection onto the device.
- Kept recurrent state and convolution history persistent across tokens; the
  timed path performs one final D2H for the output and one stream sync.
- Promoted after early/middle/late fixtures measured approximately 42--43%
  lower wall time than C1: about 646--655 us versus 1136--1138 us.
- Reduced the measured synchronization count from 5 to 1. C2 uses nine
  launches because the former host stages are now explicit device kernels.
- Correctness remains green for final output, complete recurrent state, and
  complete convolution history with zero NaN/Inf.
- Fixed the standalone legacy benchmark workspace to cover both the 2560-wide
  hidden input and the 6144-wide GDN output-projection input. The production,
  C1, and C2 fixture runs now all complete against the same allocation model.
- Added an explicit production state-sync callback for diagnostic trace
  snapshots. It copies the persistent device state/history only at that
  boundary; the measured C2 token path remains one 10,240-byte H2D, one
  10,240-byte D2H, and one stream synchronization.

## S3-C3 - fused non-projection island

- Promoted the real early/middle/late C3 candidate after standalone fixture
  validation.
- Fused causal convolution plus SiLU, Q/K normalization, decay/beta
  preparation, the FP32 recurrent update, recurrent RMS normalization, and
  sigmoid(Z) gating into one value-head CUDA kernel. History writeback remains
  a second kernel so every head reads the previous history consistently.
- Kept all four BF16 projections and the output projection unchanged at their
  measured bandwidth floor. The complete isolated path now uses seven launches
  and one synchronization instead of nine launches and one synchronization.
- Reduced the measured C2 wall from 641--659 us to 567--588 us across the
  early/middle/late fixtures, exceeding the 10% promotion gate.
- Correctness remains green for captured output, independent CPU oracle,
  next recurrent state, and next convolution history; NaN/Inf count is zero.
- Routed the production output projection into a dedicated 2560-element
  workspace instead of aliasing the 6144-element gated input. This removes an
  unsafe in-place BF16 matvec race that could corrupt decode output.
