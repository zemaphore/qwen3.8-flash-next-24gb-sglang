#!/usr/bin/env python3
"""TG3b timing on the fresh held-out prompts. 512 greedy tokens, EOS ignored,
one excluded warmup + three measured per prompt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
from tg_baseline import run_sample, summarize_sample  # noqa: E402

ORDER = ("code_held_fresh", "reasoning_held_fresh")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--boot-label", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30001/generate")
    ap.add_argument("--decode", type=int, default=512)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--measured", type=int, default=3)
    a = ap.parse_args()
    prompts = json.loads(a.prompts.read_text())
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {"boot_label": a.boot_label, "cells": {}}
    for key in ORDER:
        ids = prompts["sets"][key]["prompt_ids"]
        cell = out / key
        cell.mkdir(parents=True, exist_ok=True)
        recs = []
        for n in range(a.warmups + a.measured):
            warmup = n < a.warmups
            raw, gpu = run_sample(a.url, ids, a.decode, 3600)
            rec = summarize_sample(raw, a.decode, gpu, warmup, n)
            rec["set"] = key
            (cell / f"sample-{n:02d}.json").write_text(json.dumps(rec, indent=1) + "\n")
            recs.append(rec)
            print(f"  {a.boot_label} {key:18s} #{n} "
                  f"{'warmup' if warmup else 'measure':7s} {rec.get('status')} "
                  f"tg={rec.get('tg_tokens_per_s')} gen={rec.get('completion_tokens')} "
                  f"ttft={rec.get('ttft_seconds')}", flush=True)
        meas = [r for r in recs if not r.get("warmup") and r.get("status") == "ok"]
        tgs = [r["tg_tokens_per_s"] for r in meas]
        summary["cells"][key] = {
            "prompt_tokens": meas[0]["prompt_tokens"] if meas else None,
            "tg_values": tgs, "tg_mean": sum(tgs) / len(tgs) if tgs else None,
            "generated_tokens": [r["completion_tokens"] for r in meas],
            "ttft_seconds": [r["ttft_seconds"] for r in meas],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
