#!/usr/bin/env python3
"""Validate normalized multimodal sarcasm JSONL."""

from __future__ import annotations

import argparse
import json

from pace_plus_cli import validate_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--require-inline-reference", action="store_true")
    args = parser.parse_args()
    result = validate_data(args.data, require_reference=args.require_inline_reference)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
