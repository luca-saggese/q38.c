#!/usr/bin/env python3
"""Check manifest block alignment against source shapes and declared transforms."""

import argparse
import json
import re


BLOCKS = {"Q2_K": 256, "IQ2_XXS": 256, "Q4_K": 256, "Q8_0": 32}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory")
    parser.add_argument("manifest")
    args = parser.parse_args()
    tensors = json.load(open(args.inventory, encoding="utf-8"))["tensors"]
    rules = json.load(open(args.manifest, encoding="utf-8"))["rules"]
    checked = 0
    for rule in rules:
        block = BLOCKS.get(rule["quant_type"])
        if block is None:
            continue
        for tensor in tensors:
            if tensor["class"] != rule["class"] or not re.search(
                rule["pattern"], tensor["name"]
            ):
                continue
            shape = tensor["shape"]
            if rule.get("layout_transform") == "transpose_last_two_axes":
                shape = shape[:-2] + [shape[-1], shape[-2]]
            if not shape or shape[-1] % block:
                if rule.get("fallback_quant_type"):
                    continue
                raise ValueError(
                    f"{tensor['name']}: final dimension {shape[-1]} "
                    f"is not divisible by {block}"
                )
            checked += 1
    print(f"quant block alignment passed: {checked} tensors")


if __name__ == "__main__":
    main()
