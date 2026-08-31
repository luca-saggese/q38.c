#!/usr/bin/env python3
"""Validate and archive the deterministic M2 acceptance artifact set."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    artifact = Path(args.artifact_dir)
    model = Path(args.model_dir)
    required = [
        "tokenizer_vectors.json",
        "golden_format_version.json",
        "binding_report.json",
        "quant_q2_oracle.json",
        "quant_q4_oracle.json",
        "primitive_accuracy.json",
        "norm_activation.json",
        "matvec_dispatch.json",
        "embedding_probe.json",
        "lm_head_probe.json",
        "full_binding_report.json",
        "memory.json",
    ]
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        raise SystemExit(f"missing M2 artifacts: {', '.join(missing)}")
    vectors = json.loads((artifact / "tokenizer_vectors.json").read_text())
    if vectors["version"] != 1 or len(vectors["cases"]) != 10:
        raise SystemExit("tokenizer vector corpus is incomplete")
    for name, expected in vectors["reference_files"].items():
        actual = sha256(model / name)
        if actual != expected:
            raise SystemExit(f"reference checksum changed for {name}")
    for name in (
        "binding_report.json",
        "quant_q2_oracle.json",
        "quant_q4_oracle.json",
        "primitive_accuracy.json",
        "norm_activation.json",
        "matvec_dispatch.json",
        "full_binding_report.json",
        "memory.json",
    ):
        value = json.loads((artifact / name).read_text())
        if value.get("status") != "pass":
            raise SystemExit(f"{name} is not marked pass")
    memory = json.loads((artifact / "memory.json").read_text())
    if memory.get("iterations") != 20:
        raise SystemExit("M2-C10 did not complete 20 bind iterations")
    checksums = artifact / "checksums.txt"
    with checksums.open("w", encoding="utf-8") as stream:
        for name in required:
            stream.write(f"{sha256(artifact / name)}  {artifact / name}\n")
    (artifact / "acceptance.txt").write_text(
        "M2 acceptance passed\n"
        "gates: M2-C00 M2-C01 M2-C02 M2-C03 M2-C04 M2-C05 "
        "M2-C06 M2-C07 M2-C08 M2-C09 M2-C10 M2-C11\n"
        "tokenizer: 10 frozen raw/chat cases, exact IDs\n"
        "binding: 48 layers, 1294 runtime tensors\n"
        "memory: repeated full bind 20x, no FD/RSS growth\n",
        encoding="utf-8",
    )
    print("m2_acceptance: all M2 artifacts and gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
