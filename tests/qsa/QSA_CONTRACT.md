# QSA subsystem contract

S4 measures one-token QSA execution using compact real captures from layers 3,
23, and 47. The fixture is self-contained: it contains the hidden input,
Q/K/V and index projection outputs, QSA state after the current token, selected
IDs, attention output before output projection, expected output, and the QSA
weight tensors required by the CPU oracle and CUDA benchmark.

The CPU oracle in `qsa_reference.c` is independent of the CUDA QSA kernels. It
recomputes the BF16 projections, state normalization, causal selection,
attention, and output projection. Correctness requires exact selected IDs and
finite output/state/attention values; numeric comparisons report max absolute
error, max relative error, and RMSE. The BF16 candidate gate uses
`max_abs <= 3e-3` and `NaN/Inf == 0`; relative error is reported but is not
used as a pass/fail criterion because it is ill-conditioned near zero.
The captured `keys.f32` and `state_main_k.f32` files contain the normalized
and rotary-transformed K representation, so K correctness is gated through
the updated state rather than against the raw projection buffer.

The production baseline is the current single-token split path:

1. CUDA Q/K/V projection with host staging.
2. Host index projection, state update, selection, and attention.
3. Host scalar output projection.

QSA-C1 replaces only step 3 with a resident BF16 device matvec. It adds one
output upload, one output download, one kernel launch, and one synchronization.
It is promotable only when all three fixtures remain correct and the complete
QSA-layer median improves by at least 10%.

The benchmark must not load a GGUF model. Fixture capture is a separate,
one-shot operation performed only when real captures are unavailable.
