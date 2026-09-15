#!/usr/bin/env python3
"""Set the RTX 3090 expert row-gather tile to its tuned 2048-byte value."""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

BEFORE = """      SGLANG_MOE_EXPERT_STREAM=1 \\
      SGLANG_MOE_CONFIG_NEAREST_E="${SGLANG_MOE_CONFIG_NEAREST_E:-1}" \\
"""

AFTER = """      SGLANG_MOE_EXPERT_STREAM=1 \\
      SGLANG_MOE_GATHER_BLOCK="${SGLANG_MOE_GATHER_BLOCK:-2048}" \\
      SGLANG_MOE_CONFIG_NEAREST_E="${SGLANG_MOE_CONFIG_NEAREST_E:-1}" \\
"""


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def state() -> tuple[bool, bool]:
    text = read()
    return BEFORE in text, AFTER in text


def check() -> None:
    clean, applied = state()
    status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {LAUNCHER}: RTX 3090 expert gather tile")


def replace(old: str, new: str) -> None:
    text = read()
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one launcher anchor, found {text.count(old)}")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(old, new, 1))


def apply() -> None:
    clean, applied = state()
    if applied:
        print("  already applied")
        return
    if not clean:
        raise RuntimeError("launcher anchor mismatch")
    replace(BEFORE, AFTER)
    print("  applied (next launcher start defaults to 2048-byte gather tiles)")


def revert() -> None:
    clean, applied = state()
    if clean:
        print("  already clean")
        return
    if not applied:
        raise RuntimeError("launcher anchor mismatch")
    replace(AFTER, BEFORE)
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
