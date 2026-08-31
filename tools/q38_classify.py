#!/usr/bin/env python3
"""Classify source tensor names; unknown names are a hard error."""

import argparse
import json
import re


RULES = (
    ("vision", re.compile(r"(^|\.)(visual|vision)(\.|$)")),
    ("mtp", re.compile(r"(^|\.)(mtp|mtp_layers?)(\.|$)")),
    ("embedding", re.compile(r"(^|\.)(embed_tokens|token_embedding|wte)(\.|$)")),
    ("output", re.compile(r"(^|\.)(lm_head|output)(\.|$)")),
    ("norm", re.compile(r"(^|\.)(norm|input_layernorm|post_attention_layernorm)(\.|$)")),
    ("router", re.compile(r"(^|\.)(router|gate)(\.|$)")),
    ("shared_expert", re.compile(r"(^|\.)(shared_expert)(\.|$)")),
    ("routed_expert", re.compile(r"(^|\.)(experts?)(\.|$)")),
    ("ple", re.compile(r"(^|\.)(ple|ngram)(\.|$)")),
    ("gdn", re.compile(r"(^|\.)(gdn|linear_attn|mamba)(\.|$)")),
    ("qsa", re.compile(r"(^|\.)(qsa|self_attn|attention)(\.|$)")),
    ("gr", re.compile(r"(^|\.)(gated_residual|residual)(\.|$)")),
)


def classify(name):
    matches = [label for label, pattern in RULES if pattern.search(name)]
    if len(matches) != 1:
        raise ValueError(f"{name}: expected exactly one class, got {matches}")
    label = matches[0]
    layer_match = re.search(r"(?:layers?|blk)\.(\d+)", name)
    layer = int(layer_match.group(1)) if layer_match else None
    role = name.rsplit(".", 1)[-1]
    return label, layer, role


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.inventory, encoding="utf-8") as stream:
        source = json.load(stream)

    classified = []
    for tensor in source["tensors"]:
        label, layer, role = classify(tensor["name"])
        item = dict(tensor)
        item.update(
            {
                "class": label,
                "layer": layer,
                "role": role,
                "included_runtime": label not in ("vision", "mtp"),
                "quant_rule": None,
            }
        )
        classified.append(item)

    report = {
        "format": "q38_tensor_classes_v1",
        "tensor_count": len(classified),
        "unclassified_count": 0,
        "tensors": classified,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
