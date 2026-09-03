# ADR-SSD: SSD residency for M7

**Status:** DO NOT ENABLE  
**Scope:** M7 Q2 decode path on GB10 / 128 GiB unified memory

## Evidence

- The committed M7 baseline records a 103,901,247,456-byte mapped model and
  baseline RSS after the one-token profile, but no startup peak or page-fault
  series.
- The validated M7 change reuses the Q2 MoE intermediate workspace. It does
  not add expert residency, streaming, or SSD I/O.
- Expert locality/sensitivity is `unavailable`: no versioned multi-token
  routing corpus is available.
- Long-context and startup gates are explicitly `blocked`/`not-run` in the
  M7 acceptance artifacts.

## Decisions

**PLE:** retain the existing mmap/residency policy. No measured memory saving
or throughput benefit justifies an SSD scheduler.

**Routed experts:** retain the existing Q2 file-backed/mapped policy. SSD
streaming is not enabled without locality, page-fault, and peak-memory data.
Projected Q4 needs are outside this M7 decision.

**Decision: DO NOT ENABLE.** Revisit only after the missing measurements are
collected; this ADR intentionally makes no performance or memory claim for
SSD residency.
