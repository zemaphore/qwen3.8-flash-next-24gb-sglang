#!/usr/bin/env python3
"""RC1 GPU-only hybrid prefix-cache probe.

Sends exact-token multi-turn shapes to a running SGLang server and records the
server-reported cached prefix length for each request. Each scenario uses its
own unique prefix marker so a retained prefix from one scenario cannot make
another scenario's cold step look warm.

Expected cached length for an appended/branched step is the page-aligned token
length of the shared prefix, taken from that prefix's own measured
``prompt_tokens``. Cold steps must report 0.

Usage:
  python3 tools/rc1_cache_probe.py --url http://127.0.0.1:30001/generate \
      --out docs/logs/raw/rc1_3090_2026-09-16/probe.json [--max-new 16]

No server-side state is changed; the server must already be running.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PAGE = 64  # compressed-QSA page size; prefix cache shares page-aligned prefixes


def repo_block(marker: str) -> str:
    return "".join(
        f"# {marker} module {i}\ndef handler_{i}(request):\n"
        f"    payload = request.get('payload', {{}})\n"
        f"    return {{'id': {i}, 'ok': True, 'size': len(payload)}}\n\n"
        for i in range(120)
    )


def base_prefix(marker: str) -> str:
    return (
        "You are an autonomous coding agent working in a repository. "
        "Follow the user's instruction, use tools when needed, and keep edits scoped.\n\n"
        f"Repository files for task {marker}:\n" + repo_block(marker)
    )


def call(url: str, text: str, max_new: int, timeout: float = 900.0) -> dict:
    body = json.dumps(
        {
            "text": text,
            "sampling_params": {
                "max_new_tokens": max_new,
                "temperature": 0,
                "ignore_eos": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    m = out.get("meta_info", {})
    return {
        "prompt_tokens": m.get("prompt_tokens"),
        "cached_tokens": m.get("cached_tokens", 0),
        "completion_tokens": m.get("completion_tokens"),
        "wall_s": round(dt, 3),
        "text": out.get("text", ""),
    }


def floor_page(n: int) -> int:
    return (int(n) // PAGE) * PAGE


def run_scenario(url, max_new, marker, steps):
    """steps: list of (label, text, shared_prefix_text_or_None)."""
    rows = []
    token_len = {}
    for label, text, shared in steps:
        rec = call(url, text, max_new)
        rec["step"] = label
        rec["text_prompt"] = text
        expected = 0 if shared is None else floor_page(token_len[shared])
        rec["expected_cached"] = expected
        rec["cached_ok"] = rec["cached_tokens"] == expected
        rec["finite"] = bool(rec["text"]) and rec["completion_tokens"] not in (None, 0)
        token_len[text] = rec["prompt_tokens"] or 0
        rows.append(rec)
    return {"marker": marker, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30001/generate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    nonce = format(int(time.time()), "x")
    scenarios = []

    for marker, kind in [(f"repeated-{nonce}", "repeat"), (f"append-{nonce}", "append"),
                         (f"branch-{nonce}", "branch"), (f"returnaba-{nonce}", "return")]:
        base = base_prefix(marker)
        t1 = f"\n\nTool result 1 for {marker}: tests pass; 42 files changed.\n"
        t2 = f"\n\nTool result 2 for {marker}: lint clean; no type errors.\n"
        a = f"\n\nAssistant branch A for {marker}: refactor the handler.\n"
        b = f"\n\nAssistant branch B for {marker}: add a test first.\n"

        if kind == "repeat":
            steps = [("cold", base, None), ("repeat", base, base)]
        elif kind == "append":
            steps = [
                ("base_cold", base, None),
                ("plus_t1", base + t1, base),
                ("plus_t1_t2", base + t1 + t2, base + t1),
            ]
        elif kind == "branch":
            steps = [
                ("base_cold", base, None),
                ("branch_A", base + a, base),
                ("branch_B", base + b, base),
                ("branch_A_repeat", base + a, base + a),
            ]
        else:  # return A -> B -> A
            conv_a = base + a + t1
            conv_b = base + b + t2
            steps = [
                ("base_cold", base, None),
                ("A", conv_a, base),
                ("B", conv_b, base),
                ("A_return", conv_a, conv_a),
            ]
        scenarios.append(run_scenario(args.url, args.max_new, marker, steps))

    checks = []
    for sc in scenarios:
        for row in sc["rows"]:
            checks.append(
                {
                    "scenario": sc["marker"],
                    "step": row["step"],
                    "prompt_tokens": row["prompt_tokens"],
                    "cached_tokens": row["cached_tokens"],
                    "expected_cached": row["expected_cached"],
                    "cached_ok": row["cached_ok"],
                    "finite": row["finite"],
                    "wall_s": row["wall_s"],
                }
            )

    # Output consistency: the same full token sequence reached cold vs via a
    # cache hit must produce the same greedy continuation.
    reuse_outputs = []
    for sc in scenarios:
        by_text = {}
        for row in sc["rows"]:
            by_text.setdefault(row["text_prompt"], []).append(row)
        for _, group in by_text.items():
            if len(group) < 2:
                continue
            first = group[0]
            for later in group[1:]:
                reuse_outputs.append(
                    {
                        "scenario": sc["marker"],
                        "cold_step": first["step"],
                        "reuse_step": later["step"],
                        "cold_cached": first["cached_tokens"],
                        "reuse_cached": later["cached_tokens"],
                        "same_output": first["text"] == later["text"],
                        "cold_text_sha": __import__("hashlib")
                        .sha256(first["text"].encode())
                        .hexdigest()[:16],
                        "reuse_text_sha": __import__("hashlib")
                        .sha256(later["text"].encode())
                        .hexdigest()[:16],
                    }
                )

    ev = {
        "url": args.url,
        "page": PAGE,
        "scenarios": scenarios,
        "checks": checks,
        "reuse_outputs": reuse_outputs,
        "all_cached_ok": all(c["cached_ok"] for c in checks),
        "all_finite": all(c["finite"] for c in checks),
        "all_reuse_same_output": all(r["same_output"] for r in reuse_outputs),
    }
    with open(args.out, "w") as f:
        json.dump(ev, f, indent=2)
    print(json.dumps(checks, indent=2))
    print("all_cached_ok:", ev["all_cached_ok"], "all_finite:", ev["all_finite"])


if __name__ == "__main__":
    main()
