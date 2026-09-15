#!/usr/bin/env python3
"""Enable the opt-in bulk PLE read path in the RTX 3090 launcher.

This edits configuration only; it never starts or stops the server.

Usage:
  python3 enable_ple_bulk_pread.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

BEFORE = """      SGLANG_QWEN4_PLE_MMAP="$PLE" \\
      SGLANG_VLM_CACHE_SIZE_MB=0 \\
"""

AFTER = """      SGLANG_QWEN4_PLE_MMAP="$PLE" \\
      SGLANG_QWEN4_PLE_BULK_PREAD="${SGLANG_QWEN4_PLE_BULK_PREAD:-1}" \\
      SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE="${SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE:-2048}" \\
      SGLANG_QWEN4_PLE_WORKERS="${SGLANG_QWEN4_PLE_WORKERS:-16}" \\
      SGLANG_VLM_CACHE_SIZE_MB=0 \\
"""


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def state() -> tuple[bool, bool]:
    text = read()
    return BEFORE in text, AFTER in text


def check() -> None:
    clean, applied = state()
    # A later PP1b launcher patch inserts its bounded-cache controls between
    # the bulk-pread variables and VLM_CACHE_SIZE_MB.
    layered = "SGLANG_QWEN4_PLE_RECENT_CACHE_MB" in read()
    status = "APPLIED" if applied or layered else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {LAUNCHER}: bulk PLE pread environment")


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
    print("  applied (next launcher start will enable bulk PLE pread)")


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
