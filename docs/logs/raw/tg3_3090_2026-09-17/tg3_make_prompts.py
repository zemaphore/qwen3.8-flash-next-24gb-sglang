#!/usr/bin/env python3
"""TG3: freeze calibration and held-out prompt ids BEFORE looking at routing.

Calibration: first 2048 tokens of the code corpus and of the reasoning corpus.
Held-out: tokens [4096:6144] of the same corpora (disjoint from calibration).
Uses the same SOURCES as tools/tg_prompts.py.
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


def corpus(name: str, tok) -> list[int]:
    text = "\n\n".join((REPO / n).read_text(encoding="utf-8", errors="replace")
                        for n in SOURCES[name])
    return tok(text, add_special_tokens=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--gap", type=int, default=4096)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    payload = {"model": MODEL, "length": a.len, "gap": a.gap, "sets": {}}
    for name in ("code", "reasoning"):
        ids = corpus(name, tok)
        cal = ids[0:a.len]
        held = ids[a.gap:a.gap + a.len]
        assert len(cal) == a.len and len(held) == a.len
        for tag, seq in (("cal", cal), ("held", held)):
            key = f"{name}_{tag}"
            payload["sets"][key] = {
                "class": name,
                "prompt_ids": seq,
                "sha256": hashlib.sha256(json.dumps(seq).encode()).hexdigest(),
            }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {a.out}: " + ", ".join(payload["sets"]))


if __name__ == "__main__":
    main()
