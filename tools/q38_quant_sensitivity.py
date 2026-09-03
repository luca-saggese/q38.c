#!/usr/bin/env python3
"""Create class-isolated M8 quantization experiments from a frozen manifest.

This tool only describes recipes and computes deterministic storage deltas.  It
does not turn proxy deltas into quality claims; quality/perplexity fields stay
explicitly not-run until an evaluator is supplied a fixed corpus.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


BLOCK_INFO = {
    "Q2_K": (256, 84),
    "IQ2_XXS": (256, 66),
    "Q4_K": (256, 144),
    "Q8_0": (32, 34),
}
QUANT_TYPES = set(BLOCK_INFO) | {"BF16"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_bytes(tensor, quant_type, fallback=None, layout_transform=None):
    shape = list(tensor["shape"])
    if layout_transform == "transpose_last_two_axes":
        shape[-1], shape[-2] = shape[-2], shape[-1]
    target = quant_type
    if target in BLOCK_INFO and (len(shape) < 2 or
                                 shape[-1] % BLOCK_INFO[target][0]):
        target = fallback or tensor["source_dtype"]
    elements = 1
    for dimension in shape:
        elements *= dimension
    if target == "BF16":
        return elements * 2, target
    if target == "I64":
        return elements * 8, target
    if target not in BLOCK_INFO:
        raise ValueError(f"{tensor['name']}: unsupported quant type {target}")
    block, block_bytes = BLOCK_INFO[target]
    return (elements // shape[-1]) * (shape[-1] // block) * block_bytes, target


def recipe_changes(recipe):
    changes = {
        "R0": {},
        "R1": {"routed_gate_up": "Q4_K", "routed_down": "Q4_K"},
        "R2": {"routed_gate_up": "Q4_K", "routed_down": "Q4_K",
               "ple": "Q4_K"},
        "R3": {"routed_gate_up": "Q4_K", "routed_down": "Q4_K"},
        "R4": {"routed_gate_up": "Q4_K", "routed_down": "Q4_K",
               "shared_expert": "Q8_0", "shared_expert_gate": "Q8_0"},
        "R5": {"routed_gate_up": "Q4_K", "routed_down": "Q4_K",
               "embedding": "Q8_0", "output": "Q8_0"},
    }
    if recipe not in changes:
        raise ValueError(f"unsupported recipe {recipe}")
    return changes[recipe]


def rule_key(rule):
    identifier = rule["id"]
    if identifier in {"routed_gate_up_q2_k", "routed_gate_up_q4_k"}:
        return "routed_gate_up"
    if identifier in {"routed_down_q2_k", "routed_down_q4_k"}:
        return "routed_down"
    if identifier == "ple_q8_0":
        return "ple"
    if identifier == "shared_expert_bf16":
        return "shared_expert"
    if identifier == "shared_expert_gate_bf16":
        return "shared_expert_gate"
    if identifier == "embedding_bf16":
        return "embedding"
    if identifier == "output_bf16":
        return "output"
    return None


def make_manifest(base, recipe):
    manifest = copy.deepcopy(base)
    changes = recipe_changes(recipe)
    for rule in manifest["rules"]:
        key = rule_key(rule)
        if key in changes:
            rule["quant_type"] = changes[key]
            if changes[key] == "Q4_K" and key == "ple":
                rule["fallback_quant_type"] = "BF16"
            rationale = rule.get("rationale", "")
            rule["rationale"] = f"M8 {recipe} isolated ablation: {rationale}"
    manifest["m8_recipe"] = recipe
    # Keep the v2 schema valid: metadata is intentionally outside emitted
    # manifests and is returned in the result record instead.
    manifest.pop("m8_recipe", None)
    return manifest


def evaluate(classes, manifest):
    rules = [(rule, re.compile(rule["pattern"])) for rule in manifest["rules"]]
    totals = {}
    unmatched = []
    for tensor in classes["tensors"]:
        if not tensor.get("included_runtime", True):
            continue
        matches = [(rule, pattern) for rule, pattern in rules
                   if rule["class"] == tensor["class"]
                   and pattern.search(tensor["name"])]
        if len(matches) != 1:
            unmatched.append(tensor["name"])
            continue
        rule = matches[0][0]
        source_type = tensor["source_dtype"]
        target = rule["quant_type"] if source_type == "BF16" else source_type
        bytes_count, effective = tensor_bytes(
            tensor, target, rule.get("fallback_quant_type"),
            rule.get("layout_transform"))
        item = totals.setdefault(tensor["class"], {
            "tensors": 0, "source_bytes": 0, "quantized_bytes": 0,
            "effective_types": {},
        })
        item["tensors"] += 1
        item["source_bytes"] += tensor["bytes_source"]
        item["quantized_bytes"] += bytes_count
        item["effective_types"][effective] = (
            item["effective_types"].get(effective, 0) + 1
        )
    if unmatched:
        raise ValueError(f"manifest did not match runtime tensors: {unmatched[:3]}")
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--recipe", choices=("R0", "R1", "R2", "R3", "R4", "R5"),
                        required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--emit-manifest")
    args = parser.parse_args()

    classes = load(args.classes)
    baseline = load(args.baseline_manifest)
    manifest = make_manifest(baseline, args.recipe)
    if args.emit_manifest:
        Path(args.emit_manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    totals = evaluate(classes, manifest)
    result = {
        "format": "q38-m8-sensitivity-v1",
        "recipe": args.recipe,
        "baseline_manifest": str(Path(args.baseline_manifest)),
        "baseline_manifest_sha256": hashlib.sha256(
            Path(args.baseline_manifest).read_bytes()).hexdigest(),
        "manifest_sha256": digest(manifest),
        "class_isolation": {
            "changed_classes": sorted(
                key for key, value in recipe_changes(args.recipe).items()
                if value
            ),
            "quality": "not-run",
            "reason": "No fixed M8 calibration evaluator/corpus was executed.",
        },
        "storage": totals,
        "status": "descriptor_generated",
    }
    if args.recipe == "R2":
        result["status"] = "descriptor_generated"
        result["notes"] = [
            "PLE Q4_K is shape-incompatible for 160-wide rows and falls back to BF16.",
            "A dedicated PLE Q4 row format remains an M8-C06 task.",
        ]
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
