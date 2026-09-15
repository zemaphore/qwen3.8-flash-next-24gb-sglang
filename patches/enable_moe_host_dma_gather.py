#!/usr/bin/env python3
"""Enable PP7 host-row DMA staging on the RTX 3090 launcher."""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

LINE = ('      SGLANG_MOE_GATHER_DMA="${SGLANG_MOE_GATHER_DMA:-1}" \\\n')
ANCHOR = ('      SGLANG_MOE_GATHER_BLOCK="${SGLANG_MOE_GATHER_BLOCK:-2048}" \\\n')


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def check() -> None:
    text = read()
    line_count = text.count(LINE)
    status = "APPLIED" if line_count == 1 else (
        "clean" if line_count == 0 and ANCHOR in text else "MISMATCH"
    )
    print(f"  {status:<8} {LAUNCHER}: host-row DMA staging environment")
    if status == "MISMATCH":
        raise SystemExit(1)


def apply() -> None:
    text = read()
    if text.count(LINE) == 1:
        print("  already applied")
        return
    if LINE in text:
        raise RuntimeError("duplicate DMA environment lines")
    if text.count(ANCHOR) != 1:
        raise RuntimeError("expected exactly one launcher anchor")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(ANCHOR, ANCHOR + LINE, 1))
    print("  applied (next launcher start stages host rows by DMA)")


def revert() -> None:
    text = read()
    if LINE not in text:
        print("  already clean")
        return
    if text.count(LINE) != 1:
        raise RuntimeError("expected exactly one DMA environment line")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(LINE, "", 1))
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
