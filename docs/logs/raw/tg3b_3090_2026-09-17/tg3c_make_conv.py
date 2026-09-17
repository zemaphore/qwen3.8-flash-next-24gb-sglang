#!/usr/bin/env python3
"""TG3b acceptance: freeze two conversations (code, reasoning).

Each starts at ~32K input tokens with two appended fixed follow-up turns.
Input histories are identical across arms and never include generated answers;
turn N input = base + follow1 + ... (prefix grows, so prefix reuse applies).
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
BASE = 32768
SOURCES = {
    "code": (
        "sglang/qwen4exp-serving-73a255206f.patch",
        "scripts/01_inspect_model.py", "scripts/02_prepare_source.py",
        "scripts/03_split_ple.py", "scripts/04_recipe.py", "scripts/05_quantize.py",
        "scripts/06_subset_tuning.py", "scripts/budget.py", "scripts/phase1.py",
        "scripts/requant_int8.py", "tools/bench_agentic.py",
        "tools/bench_expert_gather.py", "tools/bench_speed.py",
        "tools/capture_pp5b_arm.py", "tools/depth_sweep.py", "tools/expert_freq.py",
        "tools/long_logprob_oracle.py", "tools/pp5_presence.py",
        "tools/snapshot_server.py", "tools/tune_moe_int2.py",
        "tools/tune_moe_int2_real.py",
    ),
    "reasoning": (
        "docs/DECODE_PERF_PLAN.md", "docs/ELASTIC_MEMORY.md",
        "docs/KV_INT8_PLAN.md", "docs/KV_TIERS_PLAN.md",
        "docs/KV_PAGED_PREFIX_PLAN.md", "docs/SPEC_NGRAM_PLAN.md",
        "docs/PP_PERF_PLAN_3090.md", "docs/TG_PERF_PLAN_3090.md",
    ),
}
FOLLOW1 = {
    "code": "\n\nFollow-up: revise the implementation above to handle the "
            "boundary cases, and briefly explain what changed and why.",
    "reasoning": "\n\nFollow-up: extend the analysis above with the main "
                 "trade-offs and the conditions under which the conclusion "
                 "would change.",
}
FOLLOW2 = {
    "code": "\n\nFollow-up: now summarise the resulting behaviour in three "
            "concise bullet points.",
    "reasoning": "\n\nFollow-up: now give a short final recommendation in "
                 "three concise bullet points.",
}


def corpus(name, tok):
    text = "\n\n".join((REPO / n).read_text(encoding="utf-8", errors="replace")
                        for n in SOURCES[name])
    return tok(text, add_special_tokens=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=int, default=BASE)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    payload = {"model": MODEL, "base_tokens": a.base, "decode_tokens": 256,
               "conversations": {}}
    for name in ("code", "reasoning"):
        base = corpus(name, tok)[: a.base]
        assert len(base) == a.base
        f1 = tok(FOLLOW1[name], add_special_tokens=False)["input_ids"]
        f2 = tok(FOLLOW2[name], add_special_tokens=False)["input_ids"]
        turns = [list(base), list(base) + f1, list(base) + f1 + f2]
        payload["conversations"][name] = {
            "base_sha256": hashlib.sha256(json.dumps(base).encode()).hexdigest(),
            "follow1_tokens": len(f1), "follow2_tokens": len(f2),
            "turn_prompt_tokens": [len(t) for t in turns],
            "turns": turns,
        }
        print(f"  {name}: base={len(base)} f1={len(f1)} f2={len(f2)} "
              f"turns={[len(t) for t in turns]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
