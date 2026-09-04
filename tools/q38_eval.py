#!/usr/bin/env python3
"""Compare deterministic runtime/reference records without inventing results.

The preferred record schema is one JSON object per sequence containing
``id``, ``logits``, ``router_top10`` and ``qsa_selected``.  These records are
produced by the native runtime and the independent reference runner; this
script only scores paired observations.  The older ``tokens``-only schema is
kept for greedy-generation comparisons, but it cannot produce numeric quality
metrics.
"""

import argparse
import hashlib
import json
import math
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


def _finite_vector(value, field, identifier):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{identifier}: {field} must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{identifier}: {field}[{index}] is not finite")
        result.append(float(item))
    return result


def _ids(value, field, identifier):
    if not isinstance(value, list):
        raise ValueError(f"{identifier}: {field} must be a list")
    result = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("id", item.get("expert"))
        if not isinstance(item, int) or item < 0:
            raise ValueError(f"{identifier}: {field} contains an invalid ID")
        result.append(item)
    return result


def _nested_ids(value, field, identifier):
    if not isinstance(value, list):
        raise ValueError(f"{identifier}: {field} must be a list")
    return [_ids(item, f"{field}[{index}]", identifier)
            for index, item in enumerate(value)]


def _top_k(values, k):
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def _jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _log_softmax(values):
    maximum = max(values)
    normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return [value - normalizer for value in values]


def _quality_metrics(baseline, candidate, top_k):
    pairs = []
    for identifier in sorted(set(baseline) & set(candidate)):
        left, right = baseline[identifier], candidate[identifier]
        if not all(key in left for key in
                   ("logits", "router_top10", "qsa_selected")):
            raise ValueError(f"{identifier}: baseline quality fields missing")
        if not all(key in right for key in
                   ("logits", "router_top10", "qsa_selected")):
            raise ValueError(f"{identifier}: candidate quality fields missing")
        left_logits = _finite_vector(left["logits"], "logits", identifier)
        right_logits = _finite_vector(right["logits"], "logits", identifier)
        if len(left_logits) != len(right_logits):
            raise ValueError(f"{identifier}: logits length mismatch")
        left_routes = _nested_ids(left["router_top10"], "router_top10",
                                  identifier)
        right_routes = _nested_ids(right["router_top10"], "router_top10",
                                   identifier)
        left_qsa = _nested_ids(left["qsa_selected"], "qsa_selected",
                               identifier)
        right_qsa = _nested_ids(right["qsa_selected"], "qsa_selected",
                                identifier)
        if len(left_routes) != len(right_routes):
            raise ValueError(f"{identifier}: router layer count mismatch")
        if len(left_qsa) != len(right_qsa):
            raise ValueError(f"{identifier}: QSA layer count mismatch")
        if any(len(layer) != 10 for layer in left_routes + right_routes):
            raise ValueError(f"{identifier}: router_top10 must contain ten IDs")
        pairs.append((identifier, left_logits, right_logits,
                      left_routes, right_routes, left_qsa, right_qsa))
    if not pairs:
        raise ValueError("no paired quality records")

    logit_abs = []
    logit_squared = []
    logit_max = 0.0
    kl_forward = []
    kl_reverse = []
    argmax_equal = 0
    topk_overlaps = []
    router_overlaps = []
    router_exact = []
    qsa_overlaps = []
    qsa_exact = []
    for _identifier, left, right, left_routes, right_routes, left_qsa, right_qsa in pairs:
        errors = [a - b for a, b in zip(left, right)]
        logit_abs.extend(abs(error) for error in errors)
        logit_squared.extend(error * error for error in errors)
        logit_max = max(logit_max, max((abs(error) for error in errors),
                                       default=0.0))
        left_logp, right_logp = _log_softmax(left), _log_softmax(right)
        kl_forward.append(sum(math.exp(a) * (a - b)
                              for a, b in zip(left_logp, right_logp)))
        kl_reverse.append(sum(math.exp(b) * (b - a)
                              for a, b in zip(right_logp, left_logp)))
        left_argmax = _top_k(left, 1)[0]
        right_argmax = _top_k(right, 1)[0]
        argmax_equal += left_argmax == right_argmax
        k = min(top_k, len(left))
        topk_overlaps.append(len(set(_top_k(left, k)) &
                                set(_top_k(right, k))) / k)
        for left_layer, right_layer in zip(left_routes, right_routes):
            router_overlaps.append(_jaccard(left_layer, right_layer))
            router_exact.append(left_layer == right_layer)
        for left_layer, right_layer in zip(left_qsa, right_qsa):
            qsa_overlaps.append(_jaccard(left_layer, right_layer))
            qsa_exact.append(left_layer == right_layer)

    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "records": len(pairs),
        "argmax_agreement": {
            "count": argmax_equal,
            "total": len(pairs),
            "rate": argmax_equal / len(pairs),
        },
        "top_k_overlap": {
            "k": top_k,
            "mean_recall": mean(topk_overlaps),
        },
        "logit_error": {
            "mae": mean(logit_abs),
            "rmse": math.sqrt(mean(logit_squared)),
            "max_abs": logit_max,
            "elements": len(logit_abs),
        },
        "kl": {
            "reference_to_candidate": mean(kl_forward),
            "candidate_to_reference": mean(kl_reverse),
            "symmetric_mean": mean([(a + b) / 2
                                   for a, b in zip(kl_forward, kl_reverse)]),
        },
        "router_top10_stability": {
            "comparisons": len(router_overlaps),
            "mean_jaccard": mean(router_overlaps),
            "exact_rate": mean(router_exact),
        },
        "qsa_selected_id_stability": {
            "comparisons": len(qsa_overlaps),
            "mean_jaccard": mean(qsa_overlaps),
            "exact_rate": mean(qsa_exact),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-records", type=int, default=32)
    args = parser.parse_args()
    if args.top_k < 1 or args.min_records < 1:
        raise SystemExit("--top-k and --min-records must be positive")
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
    quality_ready = all(
        isinstance(baseline.get(identifier), dict) and
        isinstance(candidate.get(identifier), dict) and
        all(field in baseline[identifier] and field in candidate[identifier]
            for field in ("logits", "router_top10", "qsa_selected"))
        for identifier in ids if identifier in baseline and identifier in candidate
    )
    metrics = None
    metrics_reason = None
    if quality_ready and len(ids) >= args.min_records and not missing_baseline and not missing_candidate:
        try:
            metrics = _quality_metrics(baseline, candidate, args.top_k)
        except ValueError as exc:
            metrics_reason = str(exc)
    else:
        metrics_reason = (
            "paired full logits/router_top10/qsa_selected fields are required "
            f"for at least {args.min_records} records"
        )
    output = {
        "format": "q38-m8-eval-v2",
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
        "quality_metrics": metrics,
        "quality_metrics_status": "measured" if metrics is not None else
        "not-computed",
        "quality_metrics_reason": metrics_reason,
        "minimum_quality_records": args.min_records,
        "reason": (
            "Metrics are computed only from paired finite vectors and exact "
            "runtime/reference route and QSA observations."
        ),
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
