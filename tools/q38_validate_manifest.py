#!/usr/bin/env python3
"""Validate strict manifest coverage and exact match counts."""

import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory")
    parser.add_argument("manifest")
    args = parser.parse_args()

    with open(args.inventory, encoding="utf-8") as stream:
        inventory = json.load(stream)["tensors"]
    with open(args.manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)

    matches = {}
    for rule in manifest["rules"]:
        selected = [
            tensor
            for tensor in inventory
            if tensor["class"] == rule["class"]
            and re.search(rule["pattern"], tensor["name"])
        ]
        count = len(selected)
        if not (
            rule["expected_min_matches"]
            <= count
            <= rule["expected_max_matches"]
        ):
            raise ValueError(
                f"{rule['id']}: expected "
                f"{rule['expected_min_matches']}..{rule['expected_max_matches']}, "
                f"got {count}"
            )
        for tensor in selected:
            name = tensor["name"]
            if name in matches:
                raise ValueError(
                    f"{name}: matched by both {matches[name]} and {rule['id']}"
                )
            matches[name] = rule["id"]

    if len(matches) != len(inventory):
        missing = sorted(
            tensor["name"] for tensor in inventory if tensor["name"] not in matches
        )
        raise ValueError(
            f"{len(missing)} tensors are not covered; first: {missing[:5]}"
        )
    print(f"manifest validation passed: {len(matches)} tensors")


if __name__ == "__main__":
    main()
