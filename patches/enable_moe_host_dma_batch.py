#!/usr/bin/env python3
"""Enable the accepted PP11 DMA batch path in the 3090 launcher."""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")
LINE = ('      SGLANG_MOE_GATHER_DMA_BATCH='
        '"${SGLANG_MOE_GATHER_DMA_BATCH:-1}" \\\n')
OLD_LINE = ('      SGLANG_MOE_GATHER_DMA_BATCH='
            '"${SGLANG_MOE_GATHER_DMA_BATCH:-0}" \\\n')
ANCHOR = '      SGLANG_MOE_GATHER_DMA="${SGLANG_MOE_GATHER_DMA:-1}" \\\n'


def read():
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def check():
    text = read()
    count = text.count(LINE)
    old_count = text.count(OLD_LINE)
    status = (
        "APPLIED" if count == 1 and old_count == 0 else
        "clean" if count == 0 and old_count == 0 and ANCHOR in text else
        "OLD-OFF" if count == 0 and old_count == 1 else
        "MISMATCH"
    )
    print(f"  {status:<8} {LAUNCHER}: PP11 DMA batch environment")
    if status in ("OLD-OFF", "MISMATCH"):
        raise SystemExit(1)


def apply():
    text = read()
    count = text.count(LINE)
    old_count = text.count(OLD_LINE)
    if count == 1 and old_count == 0:
        print("  already applied")
        return
    if count or old_count > 1:
        raise RuntimeError("duplicate DMA batch environment lines")
    if old_count == 1:
        with open(LAUNCHER, "w", encoding="utf-8") as output:
            output.write(text.replace(OLD_LINE, LINE, 1))
        print("  migrated PP11 DMA batching default from off to on")
        return
    if text.count(ANCHOR) != 1:
        raise RuntimeError("expected exactly one launcher anchor")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(ANCHOR, ANCHOR + LINE, 1))
    print("  applied (PP11 DMA batching enabled by default)")


def revert():
    text = read()
    count = text.count(LINE)
    old_count = text.count(OLD_LINE)
    if count == 0 and old_count == 0:
        print("  already clean")
        return
    if count + old_count != 1:
        raise RuntimeError("expected exactly one DMA batch environment line")
    found = LINE if count else OLD_LINE
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(found, "", 1))
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
