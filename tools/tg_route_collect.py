#!/usr/bin/env python3
"""Issue one streamed decode request so a server started with
``SGLANG_DECODE_ROUTE_DUMP`` records per-token decode routing.

  python3 tools/tg_route_collect.py --prompts-json ... --context 2048 --decode 32
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--prompts-class", default="reasoning")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30001/generate"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())
    ids = prompts["classes"][args.prompts_class]["cells"][str(args.context)]["prompt_ids"]
    body = json.dumps(
        {
            "input_ids": ids,
            "stream": True,
            "sampling_params": {
                "max_new_tokens": args.decode,
                "temperature": 0,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(args.url, body, {"Content-Type": "application/json"})
    count = 0
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for line in response:
            if line.startswith(b"data:"):
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                count = json.loads(payload)["meta_info"]["completion_tokens"]
    print(f"  collected routing for {count} decode tokens")


if __name__ == "__main__":
    main()
