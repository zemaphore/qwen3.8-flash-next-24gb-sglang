#!/usr/bin/env python3
"""Long-prompt teacher-forced logprob oracle.

The existing short oracles (``tools/logprob_diff.py``) use three prompts that
tokenize to a few hundred tokens.  They never reach the 4,565-token canonical
extend, the M=4565 INT2 MoE config, or the PP14 cross-layer prefetch path that is
gated at 2,048 tokens.  This tool scores a *fixed continuation* after longer
fixed prompts, so numerically equivalent kernels agree within run-to-run noise
while a wrong kernel/config moves the logprobs much further.

Prompts (all built deterministically, actual server token counts are recorded):

  canonical   same construction as ``tools/bench_agentic.py --tokens 4096``,
              one 4,608-token extend (actual 4,565 tokens)
  long-2chunk instruction plus a held-out code corpus sized to span two 4,608
              chunks plus a tail
  long-4chunk instruction plus the same corpus sized to span four 4,608 chunks
              plus a tail

Continuations come from ``tools/greedy/oa.json`` (fixed token ids), so every arm
scores an identical suffix.

  python3 long_logprob_oracle.py save  NAME
  python3 long_logprob_oracle.py check NAME     # max/mean |dlogprob| vs NAME
  python3 long_logprob_oracle.py compare A B    # symmetric: B vs A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30001/generate")
D = REPO / "tools" / "logprob_long"
MODEL = os.environ.get(
    "MODEL", "/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang"
)
CONT = json.loads((REPO / "tools" / "greedy" / "oa.json").read_text())[2][:150]


def tokenize(text: str) -> list[int]:
    """Tokenize locally; requesting logprobs for a whole long prompt OOMs the
    server's logprob processor, so ids are computed here and only a short forced
    suffix is scored server-side."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return tokenizer(text, add_special_tokens=False)["input_ids"]

# Same instruction text as bench_agentic.py; kept literal so the oracle does not
# depend on that benchmark's import side effects.
INSTRUCTION = """You are reviewing a performance patch for an LLM serving engine.
Read the supplied diff, trace the affected execution paths, and identify concrete
correctness bugs, performance regressions, missing edge cases, and useful tests.
Give file-specific recommendations and prioritize issues that affect production.

PATCH TO REVIEW
===============
"""


def _cyclic(source: str, chars: int) -> str:
    if len(source) < chars:
        source = (source + "\n") * (chars // max(1, len(source)) + 1)
    return source[:chars]


def build_prompts() -> dict[str, str]:
    canonical_src = (REPO / "sglang" / "qwen4exp-serving-73a255206f.patch").read_text(
        encoding="utf-8", errors="replace"
    )
    heldout = "\n\n".join(
        (REPO / name).read_text(encoding="utf-8", errors="replace")
        for name in ("scripts/05_quantize.py", "scripts/phase1.py")
    )
    canonical_chars = int(4096 * 3.5) - len(INSTRUCTION)
    two_chunk_chars = int(11_500 * 3.5)
    four_chunk_chars = int(21_000 * 3.5)
    return {
        "canonical": INSTRUCTION + _cyclic(canonical_src, canonical_chars),
        "long-2chunk": INSTRUCTION + _cyclic(heldout, two_chunk_chars),
        "long-4chunk": INSTRUCTION + _cyclic(heldout, four_chunk_chars),
    }


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        data = json.load(response)
    return data[0] if isinstance(data, list) else data


def server_prompt_tokens(text: str) -> int:
    """Cheap server-side count (no logprobs) to confirm local tokenization."""
    data = post(
        {
            "text": text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
    )
    return int(data["meta_info"]["prompt_tokens"])


def forced_logprobs(ids: list[int], cont: list[int]) -> tuple[list[float], int]:
    data = post(
        {
            "input_ids": ids + cont,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": len(ids) - 1,
        }
    )
    meta = data["meta_info"]
    logprobs = [t[0] for t in meta["input_token_logprobs"]]
    logprobs = [value for value in logprobs if value is not None]
    return logprobs[-len(cont):], int(meta["prompt_tokens"])


def collect(cached: dict | None = None) -> dict:
    prompts = build_prompts()
    result: dict[str, dict] = {}
    for name, text in prompts.items():
        if cached and name in cached and cached[name].get("prompt_ids"):
            ids = cached[name]["prompt_ids"]
            tokens = cached[name]["prompt_tokens"]
            server_tokens = None
        else:
            ids = tokenize(text)
            tokens = len(ids)
            server_tokens = server_prompt_tokens(text)
        logprobs, forced_prompt_tokens = forced_logprobs(ids, CONT)
        result[name] = {
            "prompt_tokens": tokens,
            "server_prompt_tokens": server_tokens,
            "forced_prompt_tokens": forced_prompt_tokens,
            "prompt_chars": len(text),
            "continuation_tokens": len(CONT),
            "forced_tokens": len(logprobs),
            "prompt_ids": ids,
            "logprobs": logprobs,
        }
    return result


def stats(reference: dict, current: dict) -> dict:
    """Compare two collected payloads prompt by prompt."""
    report = {}
    for name in reference:
        if name not in current:
            report[name] = {"error": "missing in current"}
            continue
        ref = reference[name]["logprobs"]
        cur = current[name]["logprobs"]
        n = min(len(ref), len(cur))
        deltas = [abs(a - b) for a, b in zip(ref[:n], cur[:n])]
        report[name] = {
            "prompt_tokens": current[name]["prompt_tokens"],
            "compared_tokens": n,
            "max": max(deltas) if deltas else 0.0,
            "mean": (sum(deltas) / n) if n else 0.0,
        }
    all_max = max((v.get("max", 0.0) for v in report.values() if "max" in v), default=0.0)
    total = sum(v.get("mean", 0.0) * v.get("compared_tokens", 0) for v in report.values())
    count = sum(v.get("compared_tokens", 0) for v in report.values() if "max" in v)
    report["_overall"] = {"max": all_max, "mean": (total / count) if count else 0.0}
    return report


def print_report(label: str, report: dict) -> None:
    print(f"  comparison {label}")
    for name, row in report.items():
        if name == "_overall":
            continue
        if "error" in row:
            print(f"    {name:14s} {row['error']}")
        else:
            print(
                f"    {name:14s} prompt={row['prompt_tokens']:6d}  "
                f"max |dlogprob| {row['max']:.6f}  mean {row['mean']:.6f}  "
                f"over {row['compared_tokens']} forced tokens"
            )
    overall = report["_overall"]
    print(f"    {'OVERALL':14s} max {overall['max']:.6f}  mean {overall['mean']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for command in ("save", "check"):
        p = sub.add_parser(command)
        p.add_argument("name")
    p = sub.add_parser("compare")
    p.add_argument("reference")
    p.add_argument("current")
    args = parser.parse_args()

    D.mkdir(parents=True, exist_ok=True)
    path = D / f"{args.name}.json" if hasattr(args, "name") and args.name else None

    if args.cmd == "save":
        cached = json.loads(path.read_text()) if path.exists() else None
        payload = collect(cached)
        path.write_text(json.dumps(payload, indent=1))
        sizes = {name: (row["prompt_tokens"], row["forced_tokens"]) for name, row in payload.items()}
        print(f"  saved {path}  (prompt_tokens, forced_tokens) = {sizes}")
        return

    if args.cmd == "check":
        reference = json.loads(path.read_text())
        cached = {name: {"prompt_ids": row["prompt_ids"], "prompt_tokens": row["prompt_tokens"]}
                  for name, row in reference.items()}
        current = collect(cached)
        (D / f"{args.name}.last.json").write_text(json.dumps(current, indent=1))
        report = stats(reference, current)
        print_report(f"{args.name} vs fresh run", report)
        sys.exit(0 if report["_overall"]["max"] < 0.05 else 1)

    if args.cmd == "compare":
        a = json.loads((D / f"{args.reference}.json").read_text())
        b = json.loads((D / f"{args.current}.json").read_text())
        print_report(f"{args.current} vs {args.reference}", stats(a, b))
        return


if __name__ == "__main__":
    main()
