#!/usr/bin/env python3
"""Enable the accepted PP14 cross-layer cold-row prefetch in the 3090 launcher.

The launcher defaults the feature on for prefill chunks of at least 2,048
tokens.  Both values remain shell-overridable: set
``SGLANG_MOE_COLD_PREFETCH=0`` for the old selected-row path, or adjust
``SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS`` for a targeted experiment.

Usage:
  python3 patches/enable_moe_cross_layer_prefetch.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

ANCHOR = (
    '      SGLANG_MOE_GATHER_DMA_BATCH="${SGLANG_MOE_GATHER_DMA_BATCH:-1}" \\\n'
)
BLOCK = ANCHOR + (
    '      SGLANG_MOE_COLD_PREFETCH="${SGLANG_MOE_COLD_PREFETCH:-1}" \\\n'
    '      SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS="${SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS:-2048}" \\\n'
)


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def state() -> tuple[bool, bool]:
    source = read()
    applied = source.count(BLOCK) == 1
    clean = source.count(ANCHOR) == 1 and not applied
    return clean, applied


def check() -> None:
    clean, applied = state()
    status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {LAUNCHER}: PP14 cross-layer cold-row prefetch")
    if status == "MISMATCH":
        raise SystemExit(1)


def transform(reverse: bool = False) -> None:
    clean, applied = state()
    if reverse and clean:
        print("  already clean")
        return
    if not reverse and applied:
        print("  already applied")
        return
    if reverse and not applied or not reverse and not clean:
        raise RuntimeError("launcher anchor mismatch")
    source = read()
    old, new = (BLOCK, ANCHOR) if reverse else (ANCHOR, BLOCK)
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(source.replace(old, new, 1))
    print("  reverted" if reverse else "  applied")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": transform, "revert": lambda: transform(True)}[command]()
