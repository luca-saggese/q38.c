#!/usr/bin/env python3
"""Frozen Python tokenizer oracle used by the repository-compatible C bridge."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(model_dir: str):
    root = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    template = root / "chat_template.jinja"
    if template.is_file():
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model_dir)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "encode":
                ids = tokenizer.encode(
                    request.get("text", ""),
                    add_special_tokens=bool(request.get("add_special_tokens", False)),
                )
                response = {"ids": ids}
            elif operation == "chat":
                options = {
                    "add_generation_prompt": bool(
                        request.get("add_generation_prompt", False)
                    ),
                    "add_special_tokens": False,
                }
                for key in (
                    "enable_thinking",
                    "preserve_thinking",
                    "reasoning_effort",
                ):
                    if key in request:
                        options[key] = request[key]
                if "tools" in request:
                    options["tools"] = request["tools"]
                result = tokenizer.apply_chat_template(
                    request["messages"], tokenize=True, **options
                )
                ids = result if isinstance(result, list) else result["input_ids"]
                response = {"ids": ids}
            else:
                raise ValueError(f"unknown tokenizer operation: {operation!r}")
        except Exception as exc:  # protocol errors are returned, not swallowed
            response = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
