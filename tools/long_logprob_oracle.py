#!/usr/bin/env python3
"""Long-prompt teacher-forced logprob oracle (hardened).

The existing short oracles (``tools/logprob_diff.py``) use three prompts that
tokenize to a few hundred tokens.  They never reach the 4,565-token canonical
extend, the M=4565 INT2 MoE config, or the PP14 cross-layer prefetch path that is
gated at 2,048 tokens.  This tool scores a *fixed continuation* after longer
fixed prompts.  A numerically equivalent kernel agrees within run-to-run noise;
a broken one moves the logprobs much further.

Prompts (all built deterministically, actual server token counts are recorded):

  canonical   same construction as ``tools/bench_agentic.py --tokens 4096``,
              one 4,608-token extend (actual 4,565 tokens)
  long-2chunk instruction plus a held-out code corpus sized to span two 4,608
              chunks plus a tail
  long-4chunk instruction plus the same corpus sized to span four 4,608 chunks
              plus a tail

Continuations come from ``tools/greedy/oa.json`` (fixed token ids), so every arm
scores an identical suffix.

Hardening added for TG0:

* collected rows must carry identical ``prompt_ids`` and ``continuation_ids``,
  exactly ``len(continuation_ids)`` finite logprobs, and self-consistent
  metadata (``forced_prompt_tokens == prompt_tokens + continuation_tokens``);
* :func:`stats` raises on mismatch instead of silently comparing truncated
  arrays, and returns per-token deltas;
* measurement/reporting is separated from pass/fail: ``check`` and ``compare``
  report only, and fail only when the caller supplies ``--fail-above``.  The old
  hard-coded 0.05 gate is gone because it is below the observed unchanged-server
  drift (max |dlogprob| ~1.78 on early forced tokens).

  python3 long_logprob_oracle.py save  NAME
  python3 long_logprob_oracle.py check NAME [--fail-above T]
  python3 long_logprob_oracle.py compare REFERENCE CURRENT [--fail-above T]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


class OracleValidationError(ValueError):
    """Raised when a collected or stored oracle payload is not usable."""


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


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
    raw = [entry[0] for entry in meta["input_token_logprobs"]]
    # logprob_start_len = len(ids)-1 returns the last prompt position plus every
    # continuation position.  Require exactly that shape; do not drop entries and
    # silently shift positions.
    expected = len(cont) + 1
    if len(raw) != expected:
        raise OracleValidationError(
            f"expected {expected} input logprobs from the forced request, got {len(raw)} "
            "(truncated result)"
        )
    continuation = raw[1:]
    if len(continuation) != len(cont):
        raise OracleValidationError("continuation logprob count mismatch")
    if not all(_finite(value) for value in continuation):
        bad = [i for i, value in enumerate(continuation) if not _finite(value)]
        raise OracleValidationError(
            f"non-finite logprob at continuation positions {bad[:8]} (invalid result)"
        )
    return continuation, int(meta["prompt_tokens"])


def validate_row(
    name: str,
    row: dict,
    expect_ids: list[int] | None = None,
    expect_cont_ids: list[int] | None = None,
) -> None:
    """Raise OracleValidationError naming every problem in one collected row."""
    if not isinstance(row, dict):
        raise OracleValidationError(f"{name}: row is not an object")
    problems: list[str] = []
    for key in ("prompt_tokens", "prompt_ids", "logprobs", "continuation_tokens"):
        if key not in row:
            problems.append(f"missing {key}")
    if problems:
        raise OracleValidationError(f"{name}: " + "; ".join(problems))

    ids = row["prompt_ids"]
    logprobs = row["logprobs"]
    cont_ids = row.get("continuation_ids", CONT)

    if not ids:
        problems.append("empty prompt_ids")
    if expect_ids is not None and list(ids) != list(expect_ids):
        problems.append("prompt_ids differ from reference")
    if expect_cont_ids is not None and list(cont_ids) != list(expect_cont_ids):
        problems.append("continuation_ids differ from reference")
    if row["continuation_tokens"] != len(cont_ids):
        problems.append(
            f"continuation_tokens={row['continuation_tokens']} != "
            f"len(continuation_ids)={len(cont_ids)}"
        )
    if len(logprobs) != len(cont_ids):
        problems.append(
            f"logprob array has {len(logprobs)} entries, expected {len(cont_ids)} "
            "(truncated or mismatched result)"
        )
    else:
        bad = [i for i, value in enumerate(logprobs) if not _finite(value)]
        if bad:
            problems.append(f"non-finite logprob at positions {bad[:8]} (invalid result)")

    prompt_tokens = row["prompt_tokens"]
    forced = row.get("forced_prompt_tokens")
    if forced is not None and forced != prompt_tokens + len(cont_ids):
        problems.append(
            f"forced_prompt_tokens={forced} != prompt_tokens + continuation "
            f"({prompt_tokens + len(cont_ids)})"
        )
    server_prompt = row.get("server_prompt_tokens")
    if server_prompt is not None and server_prompt != prompt_tokens:
        problems.append(
            f"server_prompt_tokens={server_prompt} != prompt_tokens={prompt_tokens}"
        )
    if problems:
        raise OracleValidationError(f"{name}: " + "; ".join(problems))


def validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or not payload:
        raise OracleValidationError("payload is not a non-empty object")
    for name, row in payload.items():
        validate_row(name, row)


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
            "continuation_ids": list(CONT),
            "forced_tokens": len(logprobs),
            "prompt_ids": ids,
            "logprobs": logprobs,
        }
        validate_row(name, result[name])
    return result


def stats(reference: dict, current: dict) -> dict:
    """Compare two collected payloads prompt by prompt.

    Raises OracleValidationError if either payload is unusable or if the current
    payload does not score the reference's exact prompts and continuation.
    """
    validate_payload(reference)
    validate_payload(current)
    report: dict = {}
    for name in reference:
        if name not in current:
            raise OracleValidationError(f"{name}: missing in current payload")
        ref_row = reference[name]
        cur_row = current[name]
        validate_row(
            name,
            cur_row,
            expect_ids=ref_row["prompt_ids"],
            expect_cont_ids=ref_row.get("continuation_ids", CONT),
        )
        if cur_row["prompt_tokens"] != ref_row["prompt_tokens"]:
            raise OracleValidationError(
                f"{name}: prompt_tokens {cur_row['prompt_tokens']} != reference "
                f"{ref_row['prompt_tokens']} (mismatched input)"
            )
        ref = ref_row["logprobs"]
        cur = cur_row["logprobs"]
        deltas = [abs(a - b) for a, b in zip(ref, cur)]
        worst = max(range(len(deltas)), key=lambda i: deltas[i])
        report[name] = {
            "prompt_tokens": cur_row["prompt_tokens"],
            "compared_tokens": len(deltas),
            "max": max(deltas),
            "mean": sum(deltas) / len(deltas),
            "max_index": worst,
            "max_at_index": deltas[worst],
            "per_token_deltas": deltas,
            "reference_logprobs": ref,
            "current_logprobs": cur,
        }
    all_max = max((v["max"] for v in report.values()), default=0.0)
    total = sum(v["mean"] * v["compared_tokens"] for v in report.values())
    count = sum(v["compared_tokens"] for v in report.values())
    report["_overall"] = {"max": all_max, "mean": (total / count) if count else 0.0}
    return report


def print_report(label: str, report: dict) -> None:
    print(f"  comparison {label}")
    for name, row in report.items():
        if name == "_overall":
            continue
        print(
            f"    {name:14s} prompt={row['prompt_tokens']:6d}  "
            f"max |dlogprob| {row['max']:.6f} (idx {row['max_index']})  "
            f"mean {row['mean']:.6f}  over {row['compared_tokens']} forced tokens"
        )
    overall = report["_overall"]
    print(f"    {'OVERALL':14s} max {overall['max']:.6f}  mean {overall['mean']:.6f}")


def write_detail(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for command in ("save", "check"):
        p = sub.add_parser(command)
        p.add_argument("name")
        p.add_argument(
            "--fail-above",
            type=float,
            default=None,
            help="exit 1 when overall max |dlogprob| exceeds this (policy is caller-owned)",
        )
    p = sub.add_parser("compare")
    p.add_argument("reference")
    p.add_argument("current")
    p.add_argument("--fail-above", type=float, default=None)
    args = parser.parse_args()

    D.mkdir(parents=True, exist_ok=True)
    path = D / f"{args.name}.json" if getattr(args, "name", None) else None

    try:
        if args.cmd == "save":
            cached = json.loads(path.read_text()) if path.exists() else None
            payload = collect(cached)
            path.write_text(json.dumps(payload, indent=1))
            sizes = {
                name: (row["prompt_tokens"], row["forced_tokens"])
                for name, row in payload.items()
            }
            print(f"  saved {path}  (prompt_tokens, forced_tokens) = {sizes}")
            return

        if args.cmd == "check":
            reference = json.loads(path.read_text())
            cached = {
                name: {"prompt_ids": row["prompt_ids"], "prompt_tokens": row["prompt_tokens"]}
                for name, row in reference.items()
            }
            current = collect(cached)
            (D / f"{args.name}.last.json").write_text(json.dumps(current, indent=1))
            report = stats(reference, current)
            write_detail(D / f"{args.name}.check.json", report)
            print_report(f"{args.name} vs fresh run", report)
        elif args.cmd == "compare":
            a = json.loads((D / f"{args.reference}.json").read_text())
            b = json.loads((D / f"{args.current}.json").read_text())
            report = stats(a, b)
            write_detail(D / f"{args.current}.compare.json", report)
            print_report(f"{args.current} vs {args.reference}", report)
    except OracleValidationError as exc:
        print(f"  ORACLE VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.fail_above is not None and args.fail_above and report["_overall"]["max"] > args.fail_above:
        print(
            f"  POLICY FAIL: overall max {report['_overall']['max']:.6f} > "
            f"--fail-above {args.fail_above}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
