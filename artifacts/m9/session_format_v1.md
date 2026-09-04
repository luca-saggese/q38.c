# Q38 M9 session/checkpoint format v1

The existing `q38_replay_snapshot_save/load` format is a versioned binary
snapshot (`Q38RPLY`, version 1). It serializes the pointer-free session layout,
EOS token, n-gram history, PLE history, GDN recurrent/conv state, GR/workspace
state, QSA main K/V and index-K caches, positions, committed-token counters,
and pending-token descriptors.

PLE/expert caches and CUDA allocations are intentionally not serialized; they
are reconstructible. Load rejects bad magic, unsupported versions, layout
mismatches, cache descriptor mismatches, and truncated payloads. Model/manifest
checksums are not part of this legacy API and remain an M9 follow-up limitation.
