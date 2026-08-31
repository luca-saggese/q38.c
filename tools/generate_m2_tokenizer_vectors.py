#!/usr/bin/env python3
"""Generate versioned tokenizer parity vectors from the frozen local files."""

import argparse
import hashlib
import json
from pathlib import Path

from q38_tokenizer_ref import load_tokenizer


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(tokenizer, text: str):
    return tokenizer.encode(text, add_special_tokens=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.model_dir)
    tokenizer = load_tokenizer(args.model_dir)

    raw_cases = [
        {"name": "ascii_plain", "kind": "text", "text": "Hello, world!\nSecond line."},
        {
            "name": "whitespace_newline_punctuation",
            "kind": "text",
            "text": "  leading\tspaces\nline  ",
        },
        {
            "name": "unicode_combining_cjk_emoji",
            "kind": "text",
            "text": "Caffè déjà vu — 你好世界 🙂 e\u0301",
        },
        {
            "name": "long_repetitive_boundary",
            "kind": "text",
            "text": ("abcabcabcabcabcabcabcabcabcabc " * 8),
        },
        {
            "name": "special_markers",
            "kind": "text",
            "text": "<|im_start|>assistant\n<think>\n",
        },
        {
            "name": "tool_like_markup",
            "kind": "text",
            "text": '{"name":"lookup","arguments":{"q":"<x>&"}}\n<tool_call>\n</tool_call>',
        },
        {
            "name": "chat_system_user_no_thinking",
            "kind": "chat",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hello?"},
            ],
            "options": {"add_generation_prompt": True, "enable_thinking": False},
        },
        {
            "name": "chat_empty_system_no_thinking",
            "kind": "chat",
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Ciao"},
            ],
            "options": {"add_generation_prompt": True, "enable_thinking": False},
        },
        {
            "name": "chat_assistant_default_reasoning",
            "kind": "chat",
            "messages": [
                {"role": "user", "content": "Compute 2+2."},
                {"role": "assistant", "content": "4"},
            ],
            "options": {"add_generation_prompt": False, "enable_thinking": True},
        },
        {
            "name": "chat_tool_response",
            "kind": "chat",
            "messages": [
                {"role": "user", "content": "Call lookup."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"q": "abc"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": '{"ok":true}'},
            ],
            "options": {"add_generation_prompt": False, "enable_thinking": False},
        },
    ]

    cases = []
    for case in raw_cases:
        case = dict(case)
        if case["kind"] == "text":
            case["ids"] = encode(tokenizer, case["text"])
        else:
            result = tokenizer.apply_chat_template(
                case["messages"],
                tokenize=True,
                add_special_tokens=False,
                **case["options"],
            )
            case["ids"] = result if isinstance(result, list) else result["input_ids"]
            rendered = tokenizer.apply_chat_template(
                case["messages"], tokenize=False, **case["options"]
            )
            case["rendered"] = rendered
        cases.append(case)

    output = {
        "format": "q38-tokenizer-vectors",
        "version": 1,
        "model_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "tokenizer_class": type(tokenizer).__name__,
        "add_special_tokens": False,
        "reference_files": {
            name: digest(root / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "merges.txt",
            )
        },
        "cases": cases,
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
