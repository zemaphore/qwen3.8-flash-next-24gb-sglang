#!/usr/bin/env python3
"""Enable the bounded recent PLE row cache in the RTX 3090 launcher.

This configuration patch layers on top of ``enable_ple_bulk_pread.py`` and
does not start or stop the server.

Usage:
  python3 patches/enable_ple_recent_row_cache.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

BEFORE = """      SGLANG_QWEN4_PLE_WORKERS="${SGLANG_QWEN4_PLE_WORKERS:-16}" \\
      SGLANG_VLM_CACHE_SIZE_MB=0 \\
"""

AFTER = """      SGLANG_QWEN4_PLE_WORKERS="${SGLANG_QWEN4_PLE_WORKERS:-16}" \\
      SGLANG_QWEN4_PLE_RECENT_CACHE_MB="${SGLANG_QWEN4_PLE_RECENT_CACHE_MB:-128}" \\
      SGLANG_QWEN4_PLE_PROFILE="${SGLANG_QWEN4_PLE_PROFILE:-0}" \\
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
    status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {LAUNCHER}: recent PLE row cache environment")


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
    print("  applied (next launcher start defaults to 128 MiB recent PLE cache)")


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

