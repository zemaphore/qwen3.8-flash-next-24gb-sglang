#!/usr/bin/env python3
"""Freeze the TG0 prompt set (prose / reasoning / code) as exact token IDs.

The plan requires prompt text *and* token IDs to be frozen and hashed before any
timing sample.  Building the inputs from ``input_ids`` (not text) makes the
actual server prompt count exactly the predeclared length; the sentinel request
in the timing harness re-confirms it server-side.

Classes are built from natural repository material.  Where one corpus is shorter
than the largest target it is cycled; the repetition is recorded explicitly in
the manifest (``repeated`` and ``cycles``).

  python3 tools/tg_prompts.py --out docs/logs/raw/tg0_3090_2026-09-16/prompts/prompts.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = os.environ.get(
    "MODEL", "/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang"
)

LENGTHS = (2048, 32768, 131072)
CAPACITY_INPUT = 257456

SOURCES = {
    "prose": (
        "README.md",
        "CHANGELOG.md",
        "docs/README.md",
        "docs/WRITEUP.md",
        "docs/MODELCARD.md",
        "docs/HISTORY.md",
        "docs/CAMPAIGN.md",
        "docs/TIMELINE.md",
    ),
    "reasoning": (
        "docs/DECODE_PERF_PLAN.md",
        "docs/ELASTIC_MEMORY.md",
        "docs/KV_INT8_PLAN.md",
        "docs/KV_TIERS_PLAN.md",
        "docs/KV_PAGED_PREFIX_PLAN.md",
        "docs/SPEC_NGRAM_PLAN.md",
        "docs/PP_PERF_PLAN_3090.md",
        "docs/TG_PERF_PLAN_3090.md",
    ),
    "code": (
        "sglang/qwen4exp-serving-73a255206f.patch",
        "scripts/01_inspect_model.py",
        "scripts/02_prepare_source.py",
        "scripts/03_split_ple.py",
        "scripts/04_recipe.py",
        "scripts/05_quantize.py",
        "scripts/06_subset_tuning.py",
        "scripts/budget.py",
        "scripts/phase1.py",
        "scripts/requant_int8.py",
        "tools/bench_agentic.py",
        "tools/bench_expert_gather.py",
        "tools/bench_speed.py",
        "tools/capture_pp5b_arm.py",
        "tools/depth_sweep.py",
        "tools/expert_freq.py",
        "tools/long_logprob_oracle.py",
        "tools/pp5_presence.py",
        "tools/snapshot_server.py",
        "tools/tune_moe_int2.py",
        "tools/tune_moe_int2_real.py",
    ),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_text(names: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    parts, hashes = [], {}
    for name in names:
        path = REPO / name
        raw = path.read_text(encoding="utf-8", errors="replace")
        hashes[name] = sha256_text(raw)
        parts.append(raw)
    return "\n\n".join(parts), hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capacity-input", type=int, default=CAPACITY_INPUT)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    classes = {}
    for name, names in SOURCES.items():
        text, hashes = source_text(names)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise SystemExit(f"class {name} tokenized to zero ids")
        cells = {}
        for length in LENGTHS:
            cycles = 0
            full = list(ids)
            while len(full) < length:
                full.extend(ids)
                cycles += 1
            prefix = full[:length]
            cells[str(length)] = {
                "prompt_tokens": len(prefix),
                "prompt_ids": prefix,
                "prompt_ids_sha256": hashlib.sha256(
                    json.dumps(prefix).encode()
                ).hexdigest(),
                "repeated": cycles > 0,
                "cycles": cycles,
            }
        capacity_cycles = 0
        capacity = list(ids)
        while len(capacity) < args.capacity_input:
            capacity.extend(ids)
            capacity_cycles += 1
        capacity = capacity[: args.capacity_input]
        classes[name] = {
            "sources": list(names),
            "source_sha256": hashes,
            "corpus_tokens": len(ids),
            "corpus_sha256": sha256_text(text),
            "repeated_in_cells": any(c["repeated"] for c in cells.values()),
            "cells": cells,
            "capacity": {
                "prompt_tokens": len(capacity),
                "prompt_ids": capacity,
                "prompt_ids_sha256": hashlib.sha256(
                    json.dumps(capacity).encode()
                ).hexdigest(),
                "repeated": capacity_cycles > 0,
                "cycles": capacity_cycles,
            },
        }
        print(
            f"  {name:10s} corpus={len(ids):7d} tokens  "
            f"2K/32K/128K repeats="
            f"{cells['2048']['cycles']}/{cells['32768']['cycles']}/{cells['131072']['cycles']}  "
            f"capacity cycles={capacity_cycles}"
        )

    payload = {
        "schema": 1,
        "tokenizer": MODEL,
        "lengths": list(LENGTHS),
        "capacity_input": args.capacity_input,
        "classes": classes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
