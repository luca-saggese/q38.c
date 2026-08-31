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

    policy = manifest.get("expert_policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("default"), dict):
        raise ValueError("expert_policy.default is required")
    required_projection = {"gate", "up", "down"}
    if set(policy["default"]) != required_projection:
        raise ValueError("expert_policy.default must define gate/up/down")
    seen_experts = set()
    for override in policy.get("overrides", []):
        experts = override.get("experts", [])
        if not experts or len(set(experts)) != len(experts):
            raise ValueError("expert override experts must be non-empty and unique")
        if any(not isinstance(expert, int) or not 0 <= expert < 512
               for expert in experts):
            raise ValueError("expert override id outside 0..511")
        if seen_experts.intersection(experts):
            raise ValueError("expert override entries overlap")
        seen_experts.update(experts)
        if set(override) != {"experts", "gate", "up", "down"}:
            raise ValueError("expert override must define gate/up/down")

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
