#!/usr/bin/env python3
"""Audit the M3 reference freeze without filling in unpublished semantics."""

import argparse
import json
from pathlib import Path


REQUIRED_SECTIONS = (
    "## Reference identity",
    "## Frozen shapes and layer schedule",
    "## GDN equations",
    "## Gated Residual equations",
    "## Explicit unresolved reference semantics",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    text = Path(args.doc).read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    config = json.loads((Path(args.model_dir) / "config.json").read_text())
    text_config = config["text_config"]
    if text_config["hc_count"] != 4 or text_config["hc_lowrank"] != 320:
        missing.append("verified GR dimensions")
    if text_config["linear_num_key_heads"] != 16 or \
            text_config["linear_num_value_heads"] != 48:
        missing.append("verified GDN head dimensions")
    if text_config["linear_conv_kernel_dim"] != 4:
        missing.append("verified convolution dimension")
    if missing:
        raise SystemExit("M3-C00 audit missing: " + ", ".join(missing))
    Path(args.output).write_text(
        json.dumps(
            {
                "gate": "M3-C00",
                "status": "pass",
                "reference_audit": "complete",
                "unknown_blockers": [
                    "fused projection slice order",
                    "16-to-48 head mapping",
                    "serialized recurrent-state layout",
                    "exact GR tensor assignment",
                    "runnable Qwen3.8 forward reference",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("m3_c00_audit: reference freeze checklist passed; blockers recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
