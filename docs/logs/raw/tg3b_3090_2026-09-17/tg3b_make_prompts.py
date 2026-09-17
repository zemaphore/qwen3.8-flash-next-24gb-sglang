#!/usr/bin/env python3
"""TG3b: freeze calibration (2 code + 2 reasoning) and a FRESH held-out prompt
per class BEFORE any capture. Slices are disjoint from the old TG3 slices
([0:2048], [4096:6144]) and from each other.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MODEL = os.environ.get(
    "MODEL", "/mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang"
)
SOURCES = {
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
}
CAL_SLICES = [(0, 2048), (8192, 10240)]
HELD_SLICE = (16384, 18432)


def corpus(name: str, tok) -> list[int]:
    text = "\n\n".join((REPO / n).read_text(encoding="utf-8", errors="replace")
                        for n in SOURCES[name])
    return tok(text, add_special_tokens=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    payload = {"model": MODEL, "cal_slices": CAL_SLICES,
               "held_slice": list(HELD_SLICE), "sets": {}}
    for name in ("code", "reasoning"):
        ids = corpus(name, tok)
        for i, (lo, hi) in enumerate(CAL_SLICES, 1):
            seq = ids[lo:hi]
            assert len(seq) == hi - lo
            key = f"{name}_cal{i}"
            payload["sets"][key] = {"class": name, "role": "cal",
                                    "prompt_ids": seq,
                                    "sha256": hashlib.sha256(json.dumps(seq).encode()).hexdigest()}
        lo, hi = HELD_SLICE
        seq = ids[lo:hi]
        key = f"{name}_held_fresh"
        payload["sets"][key] = {"class": name, "role": "held",
                                "prompt_ids": seq,
                                "sha256": hashlib.sha256(json.dumps(seq).encode()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {a.out}: " + ", ".join(payload["sets"]))


if __name__ == "__main__":
    main()
