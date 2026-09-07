# QSA changelog

## S4 baseline and QSA-C1

- Captured real compact fixtures for layers 3, 23, and 47.
- Added an independent CPU QSA oracle and standalone CUDA benchmark.
- Added device-only Q/K/V projection support for fixture benchmarking.
- Measured the current host scalar output projection and a BF16 device
  single-token output-projection candidate.
- Wired QSA-C1 into the production single-token forward path and promoted it
  as `QSA_OPT_V1` after the three fixture gates passed.
