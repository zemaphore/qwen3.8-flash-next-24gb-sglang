#!/usr/bin/env python3
"""Make the expert-stream row-gather tile configurable.

The RTX 3090 model-free sweep favored 2048 bytes over the original 1024-byte
tile for the two large INT2 qweight rows.  This patch keeps 1024 as the source
default; the 3090 launcher opts into 2048 separately.

Usage:
  SGLANG=/root/sglang python3 patches/moe_gather_block.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")),
    "python/sglang",
)
TARGET = os.path.join(SG, "srt/layers/moe/expert_stream.py")

BEFORE_CONST = """# Work without deduplication up to this many ids (avoids the device sync).
_NO_DEDUP_LIMIT = 64
_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}
"""

AFTER_CONST = """# Work without deduplication up to this many ids (avoids the device sync).
_NO_DEDUP_LIMIT = 64
_GATHER_BLOCK = int(os.environ.get(\"SGLANG_MOE_GATHER_BLOCK\", \"1024\"))
if _GATHER_BLOCK not in (512, 1024, 2048, 4096, 8192):
    raise ValueError(
        \"SGLANG_MOE_GATHER_BLOCK must be one of 512, 1024, 2048, 4096, 8192\"
    )
_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}
"""

BEFORE_BLOCK = """                BLOCK = 1024
                _gather_rows_tab_kernel"""
AFTER_BLOCK = """                BLOCK = _GATHER_BLOCK
                _gather_rows_tab_kernel"""

BEFORE_HOST_BLOCK = """                BLOCK = 1024
                _gather_rows_kernel"""
AFTER_HOST_BLOCK = """                BLOCK = _GATHER_BLOCK
                _gather_rows_kernel"""

EDITS = [
    (BEFORE_CONST, AFTER_CONST),
    (BEFORE_BLOCK, AFTER_BLOCK),
    (BEFORE_HOST_BLOCK, AFTER_HOST_BLOCK),
]


def read() -> str:
    with open(TARGET, encoding="utf-8") as source:
        return source.read()


def states() -> list[tuple[bool, bool]]:
    text = read()
    return [(before in text, after in text) for before, after in EDITS]


def check() -> None:
    values = states()
    if all(applied and not clean for clean, applied in values):
        status = "APPLIED"
    elif all(clean and not applied for clean, applied in values):
        status = "clean"
    else:
        status = "MISMATCH"
    print(f"  {status:<8} {os.path.relpath(TARGET, SG)}: configurable gather tile")


def apply() -> None:
    values = states()
    if all(applied and not clean for clean, applied in values):
        print("  already applied")
        return
    if not all(clean and not applied for clean, applied in values):
        check()
        raise RuntimeError("patch anchor mismatch")
    text = read()
    for before, after in EDITS:
        if text.count(before) != 1:
            raise RuntimeError(
                f"expected exactly one patch anchor, found {text.count(before)}"
            )
        text = text.replace(before, after, 1)
    with open(TARGET, "w", encoding="utf-8") as output:
        output.write(text)
    print("  applied (SGLANG_MOE_GATHER_BLOCK controls expert row-gather tile)")


def revert() -> None:
    values = states()
    if all(clean and not applied for clean, applied in values):
        print("  already clean")
        return
    if not all(applied and not clean for clean, applied in values):
        check()
        raise RuntimeError("patch anchor mismatch")
    text = read()
    for before, after in reversed(EDITS):
        text = text.replace(after, before, 1)
    with open(TARGET, "w", encoding="utf-8") as output:
        output.write(text)
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()

