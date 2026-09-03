#!/usr/bin/env python3
"""Compare deterministic generation records without inventing quality results.

The runner accepts JSONL records produced by an external/native evaluator:
each record must contain an ``id`` and a ``tokens`` list.  This keeps corpus
execution separate from scoring and makes missing calibration data explicit.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            identifier = item.get("id")
            tokens = item.get("tokens")
            if not isinstance(identifier, str) or not isinstance(tokens, list):
                raise ValueError(f"{path}:{number}: expected id and tokens")
            if identifier in result:
                raise ValueError(f"{path}:{number}: duplicate id {identifier}")
            result[identifier] = tokens
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus")
    args = parser.parse_args()
    baseline = records(args.baseline)
    candidate = records(args.candidate)
    ids = sorted(set(baseline) | set(candidate))
    missing_baseline = sorted(set(ids) - set(baseline))
    missing_candidate = sorted(set(ids) - set(candidate))
    changed = [
        identifier for identifier in ids
        if identifier in baseline and identifier in candidate
        and baseline[identifier] != candidate[identifier]
    ]
    token_changes = sum(
        sum(a != b for a, b in zip(baseline[i], candidate[i])) +
        abs(len(baseline[i]) - len(candidate[i]))
        for i in changed
    )
    output = {
        "format": "q38-m8-eval-v1",
        "status": "pass" if not missing_baseline and not missing_candidate else "fail",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "records": len(ids),
        "exact_matches": len(ids) - len(changed) - len(missing_baseline) -
        len(missing_candidate),
        "changed_records": len(changed),
        "token_changes": token_changes,
        "missing_baseline": missing_baseline,
        "missing_candidate": missing_candidate,
        "corpus": {
            "status": "not-provided" if not args.corpus else "provided",
            "sha256": sha256(args.corpus) if args.corpus else None,
        },
        "quality_metrics": "not-computed",
        "reason": "Token comparison is not perplexity, KL, NLL, or task quality.",
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
