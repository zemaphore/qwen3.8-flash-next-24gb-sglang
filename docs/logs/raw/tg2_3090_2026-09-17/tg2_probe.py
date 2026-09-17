#!/usr/bin/env python3
"""TG2 compact single-stream probe against a running 3090 server.

Predeclared: code @ 2048 input ids and reasoning @ 32768 input ids from the
frozen TG0 prompt set; 128 generated tokens, greedy, EOS ignored; one excluded
warmup then two measured requests per prompt. Records TG excluding TTFT,
generated count, TTFT and total time per request.

  python3 tg2_probe.py --boot-label baseline --out-dir ./baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
from tg_baseline import run_sample, summarize_sample  # noqa: E402

PROMPTS = REPO / "docs/logs/raw/tg0_3090_2026-09-16/prompts/prompts.json"
URL = "http://127.0.0.1:30001/generate"
# (class, input length) fixed order
ORDER = (("code", 2048), ("reasoning", 32768))
DECODE = 128
WARMUPS = 1
MEASURED = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-label", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--decode-tokens", type=int, default=DECODE)
    ap.add_argument("--warmups", type=int, default=WARMUPS)
    ap.add_argument("--measured", type=int, default=MEASURED)
    a = ap.parse_args()
    prompts = json.loads(PROMPTS.read_text())
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"boot_label": a.boot_label, "cells": {}}
    for cls, length in ORDER:
        ids = prompts["classes"][cls]["cells"][str(length)]["prompt_ids"]
        cell = out / f"{cls}-{length}"
        cell.mkdir(parents=True, exist_ok=True)
        recs = []
        for n in range(a.warmups + a.measured):
            warmup = n < a.warmups
            raw, gpu = run_sample(a.url, ids, a.decode_tokens, 3600)
            rec = summarize_sample(raw, a.decode_tokens, gpu, warmup, n)
            rec["class"] = cls
            rec["requested_input_tokens"] = length
            (cell / f"sample-{n:02d}.json").write_text(json.dumps(rec, indent=1) + "\n")
            recs.append(rec)
            print(
                f"  {a.boot_label} {cls:9s} {length:6d} #{n} "
                f"{'warmup' if warmup else 'measure':7s} {rec.get('status')} "
                f"tg={rec.get('tg_tokens_per_s')} "
                f"gen={rec.get('completion_tokens')} "
                f"ttft={rec.get('ttft_seconds')} "
                f"wall={rec.get('wall_seconds')}",
                flush=True,
            )
        meas = [r for r in recs if not r.get("warmup") and r.get("status") == "ok"]
        tgs = [r["tg_tokens_per_s"] for r in meas]
        summary["cells"][f"{cls}-{length}"] = {
            "actual_prompt_tokens": meas[0]["prompt_tokens"] if meas else None,
            "tg_values": tgs,
            "tg_mean": sum(tgs) / len(tgs) if tgs else None,
            "generated_tokens": [r["completion_tokens"] for r in meas],
            "ttft_seconds": [r["ttft_seconds"] for r in meas],
            "wall_seconds": [r["wall_seconds"] for r in meas],
            "gpu_free_min_mib": [
                r.get("gpu", {}).get("memory_free_mib", {}).get("min") for r in meas
            ],
        }
        # drop bulky per-event arrays from the summary (kept in sample files)
        for r in recs:
            r.pop("times_monotonic", None)
            r.pop("completion_counts", None)
    summary["captured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
