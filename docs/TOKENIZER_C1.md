# TOKENIZER-C1 — Remove `jstr()` O(N²) Suffix Scan

## Scope

TOKENIZER-C1 changes only the JSON string parser used while loading the Hugging
Face tokenizer. `jstr()` now receives the tokenizer document end, finds the
current JSON string's closing quote while honoring escapes, and allocates from
that bounded span. Ownership, string copies, hash-table construction, merge
loading, and tokenizer semantics are unchanged. No model was loaded.

## Frozen baseline

`artifacts/perf/current/tokenizer_baseline_v0.json` records one cold-ish process
and five warm processes before C1. The old profile measured 248,077 `strlen()`
calls and accumulated 2,344,777,597,427 requested allocation bytes because each
vocabulary string allocated from its current JSON position to the document NUL.

## C1 result

| Metric | Baseline cold-ish | C1 cold-ish | Baseline warm mean | C1 warm mean |
|---|---:|---:|---:|---:|
| Total wall (ms) | 43,495.562 | 84.661 | 30,711.632 | 82.557 |
| Vocab string allocation (ms) | 43,363.479 | 15.091 | 30,585.439 | 11.328 |
| Hash build (ms) | 86.540 | 32.362 | 80.564 | 25.736 |
| Merge index build (ms) | 7.029 | 6.840 | 7.093 | 7.120 |

The C1 wall reduction is 99.805% for the cold-ish process and 99.731% for
the warm mean. `strlen_calls` is zero in the C1 profile. The requested
allocation total falls to 32,140,939 bytes while malloc/calloc/realloc/strdup
counts and copied string bytes remain unchanged.

## Correctness

The existing deterministic fixture remains GREEN for ASCII, whitespace,
punctuation, UTF-8, CJK, emoji, special tokens, and long input. Vocabulary
count remains 248,044 and merge count remains 247,504. The bounded parser
preserves escaped quotes, backslashes, `\uXXXX`, surrogate pairs, UTF-8, and
empty strings.

## Promotion

C1 passes the correctness gate and exceeds the required 10% startup improvement
by a wide margin. No C2 redesign, arena allocation, binary cache, hash-table
change, or model-load benchmark is included.
