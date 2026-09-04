# ADR: MTP decision

**Decision: DISABLE for M9.** The repository has no validated MTP proposal,
verification, CUDA path, or Q4 losslessness suite. Enabling it would violate
the exact-greedy requirement and would fabricate acceptance data. Reconsider
only after the reference mapping, target verification loop, and memory cost
are implemented and measured.
