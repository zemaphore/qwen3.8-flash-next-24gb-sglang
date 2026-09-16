#!/usr/bin/env python3
"""TG0 sustained decode-path numerical/state check.

Bulk input-suffix logprobs and greedy text similarity cannot verify that the
*incremental* decode path carries recurrent, PLE and KV state correctly.  This
tool compares the logprob of each greedily generated token against the logprob
the same token receives when the whole history is recomputed in one pass:

  1. incremental: generate K tokens from a fixed prompt P with ``return_logprob``
     (the server advances one decode step at a time, recording each step's
     logprob and chosen token);
  2. re-prefill: teacher-force ``P + generated`` in a single request and read the
     logprob of every generated position.

If the recurrent/KV/PLE state is preserved across incremental steps the two
per-position logprobs agree within numerical tolerance.  A chosen prompt length
lets the generated window cross the 8,192-token tiered-KV ring boundary.

Coverage is deliberately explicit: this checks one sequence on this
configuration and does not independently verify every layer or the sampling
kernel.  It reports measured drift rather than asserting equivalence.

  python3 tools/tg_numcheck.py --prompts-json ... --prompts-class code \\
      --prefix-tokens 8180 --generate 32 --out .../numcheck-ring8192.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:30001/generate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return data[0] if isinstance(data, list) else data


def extract_output_logprobs(meta: dict) -> tuple[list[float], list[int]]:
    entries = meta.get("output_token_logprobs") or []
    values: list[float] = []
    ids: list[int] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry:
            values.append(float(entry[0]) if entry[0] is not None else float("nan"))
            if len(entry) > 1 and entry[1] is not None:
                ids.append(int(entry[1]))
    return values, ids


def generated_ids(data: dict, meta: dict, expected: int) -> list[int]:
    _, ids = extract_output_logprobs(meta)
    if len(ids) == expected:
        return ids
    for key in ("output_ids", "output_token_ids"):
        candidate = meta.get(key)
        if isinstance(candidate, list) and len(candidate) == expected:
            return [int(x) for x in candidate]
    raise RuntimeError(
        f"could not recover {expected} generated token ids from meta_info keys "
        f"{sorted(meta.keys())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--prompts-class", default="code")
    parser.add_argument("--source-context", type=int, default=32768)
    parser.add_argument("--prefix-tokens", type=int, default=8180)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())
    corpus = prompts["classes"][args.prompts_class]["cells"][str(args.source_context)][
        "prompt_ids"
    ]
    prefix = corpus[: args.prefix_tokens]
    if len(prefix) != args.prefix_tokens:
        raise SystemExit("source context shorter than requested prefix")
    crossing = args.prefix_tokens + args.generate

    # 1. Incremental decode.
    incremental = post(
        args.url,
        {
            "input_ids": prefix,
            "sampling_params": {
                "max_new_tokens": args.generate,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
        },
        args.timeout,
    )
    inc_meta = incremental["meta_info"]
    inc_logprobs, _ = extract_output_logprobs(inc_meta)
    ids = generated_ids(incremental, inc_meta, args.generate)
    if len(inc_logprobs) != args.generate:
        raise RuntimeError(
            f"incremental pass returned {len(inc_logprobs)} logprobs, expected {args.generate}"
        )

    # 2. Teacher-forced re-prefill of the full history.
    forced = post(
        args.url,
        {
            "input_ids": prefix + ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": len(prefix) - 1,
        },
        args.timeout,
    )
    forced_meta = forced["meta_info"]
    raw = [entry[0] for entry in forced_meta["input_token_logprobs"]]
    expected = args.generate + 1
    if len(raw) != expected:
        raise RuntimeError(
            f"forced pass returned {len(raw)} input logprobs, expected {expected} (truncated)"
        )
    forced_logprobs = [float(v) for v in raw[1:]]
    if not all(math.isfinite(v) for v in forced_logprobs):
        raise RuntimeError("forced pass returned a non-finite logprob")

    deltas = [abs(a - b) for a, b in zip(inc_logprobs, forced_logprobs)]
    worst = max(range(len(deltas)), key=lambda i: deltas[i])
    payload = {
        "schema": 1,
        "captured_utc": utc_now(),
        "prompts_class": args.prompts_class,
        "prefix_tokens": args.prefix_tokens,
        "generate_tokens": args.generate,
        "crossing": crossing,
        "crosses_ring_8192": args.prefix_tokens <= 8192 <= (crossing - 1),
        "generated_ids": ids,
        "incremental_logprobs": inc_logprobs,
        "reprefill_logprobs": forced_logprobs,
        "deltas": deltas,
        "max_delta": deltas[worst],
        "max_delta_index": worst,
        "mean_delta": statistics.fmean(deltas),
        "finite": all(math.isfinite(v) for v in deltas),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"  prefix={args.prefix_tokens} gen={args.generate} crossing={crossing} "
        f"ring_cross={payload['crosses_ring_8192']} "
        f"max|dlogprob|={payload['max_delta']:.6f} (idx {worst}) "
        f"mean={payload['mean_delta']:.6f}"
    )
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
