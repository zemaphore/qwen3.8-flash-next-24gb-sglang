#!/usr/bin/env python3
"""Measure one deterministic code-review prompt against a running SGLang server.

The standard ``bench_speed.py`` corpus is deliberately tiny and repetitive.  This
benchmark uses a real repository patch plus a coding-agent instruction so elastic
MoE runs see the expert diversity of source-code work.  ``--tokens`` controls the
approximate input size; the table reports the actual server-tokenized size.

  SGLANG_URL=http://127.0.0.1:30001/generate \
    python3 tools/bench_agentic.py --tokens 4096 --decode-tokens 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench_speed import stream


INSTRUCTION = """You are reviewing a performance patch for an LLM serving engine.
Read the supplied diff, trace the affected execution paths, and identify concrete
correctness bugs, performance regressions, missing edge cases, and useful tests.
Give file-specific recommendations and prioritize issues that affect production.

PATCH TO REVIEW
===============
"""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=repo / "sglang" / "qwen4exp-serving-73a255206f.patch",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096])
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=3.5,
        help="Initial deterministic size estimate; actual prompt tokens are reported",
    )
    args = parser.parse_args()

    source = args.corpus.read_text(encoding="utf-8", errors="replace")
    print(f"  {'target':>8s} {'actual':>8s} {'prefill s':>10s} {'prefill t/s':>12s} {'decode t/s':>11s}")
    print("  " + "-" * 57)
    for target in args.tokens:
        chars = max(1, int(target * args.chars_per_token) - len(INSTRUCTION))
        if len(source) < chars:
            source = (source + "\n") * (chars // max(1, len(source)) + 1)
        text = INSTRUCTION + source[:chars]
        prompt, started, times, counts = stream(text, args.decode_tokens)
        decode_seconds = (times[-1] - times[0]) / (counts[-1] - counts[0])
        prefill_seconds = max(times[0] - started - decode_seconds, 1e-3)
        print(
            f"  {target:8d} {prompt:8d} {prefill_seconds:10.2f} "
            f"{prompt / prefill_seconds:12.0f} {1 / decode_seconds:11.1f}"
        )


if __name__ == "__main__":
    main()
