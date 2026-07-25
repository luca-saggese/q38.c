#!/usr/bin/env python3
"""Compare ROCm GLM validation dumps without third-party dependencies."""

import argparse
import array
import heapq
import json
import math
from pathlib import Path


def vector_metrics(reference, candidate):
    if len(reference) != len(candidate):
        raise SystemExit(
            f"length mismatch: reference={len(reference)} "
            f"candidate={len(candidate)}"
        )
    if not reference:
        raise SystemExit("input vectors are empty")

    nonfinite = sum(
        not (math.isfinite(a) and math.isfinite(b))
        for a, b in zip(reference, candidate)
    )
    diff2 = sum((a - b) ** 2 for a, b in zip(reference, candidate))
    ref2 = sum(a * a for a in reference)
    cand2 = sum(b * b for b in candidate)
    dot = sum(a * b for a, b in zip(reference, candidate))

    return {
        "values": len(reference),
        "nonfinite_pairs": nonfinite,
        "max_abs": max(abs(a - b) for a, b in zip(reference, candidate)),
        "rmse": math.sqrt(diff2 / len(reference)),
        "rel_rmse": math.sqrt(diff2 / ref2) if ref2 else math.inf,
        "cosine": dot / math.sqrt(ref2 * cand2)
        if ref2 and cand2
        else math.nan,
    }


def print_metrics(metrics, indent="  "):
    print(f"{indent}values:          {metrics['values']}")
    print(f"{indent}nonfinite_pairs: {metrics['nonfinite_pairs']}")
    print(f"{indent}max_abs:        {metrics['max_abs']:.12g}")
    print(f"{indent}rmse:           {metrics['rmse']:.12g}")
    print(f"{indent}rel_rmse:       {metrics['rel_rmse']:.12g}")
    print(f"{indent}cosine:         {metrics['cosine']:.12g}")


def load_f32(path):
    values = array.array("f")
    with path.open("rb") as fp:
        values.frombytes(fp.read())
    return values


def compare_tensor(args):
    reference = load_f32(args.reference)
    candidate = load_f32(args.candidate)
    if args.hidden <= 0:
        raise SystemExit("--hidden must be positive")
    if len(reference) % args.hidden:
        raise SystemExit(
            f"reference contains {len(reference)} values, which is not "
            f"divisible by hidden size {args.hidden}"
        )

    print(f"tokens: {len(reference) // args.hidden}")
    print("all tokens:")
    print_metrics(vector_metrics(reference, candidate))
    print("final token:")
    print_metrics(
        vector_metrics(reference[-args.hidden :], candidate[-args.hidden :])
    )


def load_logits(path):
    with path.open(encoding="utf-8") as fp:
        obj = json.load(fp)
    logits = obj.get("logits")
    if not isinstance(logits, list):
        raise SystemExit(f"{path}: missing logits array")
    return logits


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def compare_logits(args):
    reference = load_logits(args.reference)
    candidate = load_logits(args.candidate)
    metrics = vector_metrics(reference, candidate)
    count = len(reference)

    ref_mean = sum(reference) / count
    cand_mean = sum(candidate) / count
    ref_centered = [value - ref_mean for value in reference]
    cand_centered = [value - cand_mean for value in candidate]
    centered = vector_metrics(ref_centered, cand_centered)

    ref_top = heapq.nlargest(50, range(count), key=reference.__getitem__)
    cand_top = heapq.nlargest(50, range(count), key=candidate.__getitem__)
    ref_top1 = ref_top[0]
    cand_top1 = cand_top[0]
    ref_margin = reference[ref_top[0]] - reference[ref_top[1]]
    cand_margin = candidate[cand_top[0]] - candidate[cand_top[1]]
    ref_top1_rank = 1 + sum(
        value > candidate[ref_top1] for value in candidate
    )

    ref_lse = logsumexp(reference)
    cand_lse = logsumexp(candidate)
    kl_divergence = sum(
        math.exp(a - ref_lse)
        * ((a - ref_lse) - (b - cand_lse))
        for a, b in zip(reference, candidate)
    )

    print("raw logits:")
    print_metrics(metrics)
    print(f"  mean_shift:     {cand_mean - ref_mean:.12g}")
    print("centered logits:")
    print_metrics(centered)
    print(f"KL(reference || candidate): {kl_divergence:.12g}")
    print(f"top1 IDs:                  {ref_top1} -> {cand_top1}")
    print(f"reference top1 rank:       {ref_top1_rank}")
    print(f"top1 margins:              {ref_margin:.12g} -> {cand_margin:.12g}")
    print(
        "top10 overlap:             "
        f"{len(set(ref_top[:10]) & set(cand_top[:10]))}/10"
    )
    print(
        "top50 overlap:             "
        f"{len(set(ref_top) & set(cand_top))}/50"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare ds4 ROCm GLM tensor or frontier-logit dumps."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    tensor = subparsers.add_parser(
        "tensor", help="compare two raw F32 graph tensor dumps"
    )
    tensor.add_argument("reference", type=Path)
    tensor.add_argument("candidate", type=Path)
    tensor.add_argument(
        "--hidden",
        type=int,
        default=6144,
        help="hidden width used to report the final token (default: 6144)",
    )
    tensor.set_defaults(func=compare_tensor)

    logits = subparsers.add_parser(
        "logits", help="compare two ds4-bench frontier-logit JSON files"
    )
    logits.add_argument("reference", type=Path)
    logits.add_argument("candidate", type=Path)
    logits.set_defaults(func=compare_logits)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
