#!/usr/bin/env python3
"""Set the RTX 3090 PP-oriented chunk and expert-residency defaults."""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

OLD_FILL = 'SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-2048}"'
NEW_FILL = 'SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-99999}"'
OLD_CHUNK = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-1024}"'
NEW_CHUNK = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-2048}"'


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def state() -> tuple[bool, bool]:
    text = read()
    clean = OLD_FILL in text and OLD_CHUNK in text
    applied = NEW_FILL in text and NEW_CHUNK in text
    return clean, applied


def check() -> None:
    clean, applied = state()
    status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {LAUNCHER}: RTX 3090 PP chunk/residency defaults")


def replace(old_fill: str, new_fill: str, old_chunk: str, new_chunk: str) -> None:
    text = read()
    if text.count(old_fill) != 1 or text.count(old_chunk) != 1:
        raise RuntimeError("expected exactly one fill and chunk launcher anchor")
    text = text.replace(old_fill, new_fill, 1).replace(old_chunk, new_chunk, 1)
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text)


def apply() -> None:
    clean, applied = state()
    if applied:
        print("  already applied")
        return
    if not clean:
        raise RuntimeError("launcher anchor mismatch")
    replace(OLD_FILL, NEW_FILL, OLD_CHUNK, NEW_CHUNK)
    print("  applied (next launcher start defaults to chunk 2048 and S184)")


def revert() -> None:
    clean, applied = state()
    if clean:
        print("  already clean")
        return
    if not applied:
        raise RuntimeError("launcher anchor mismatch")
    replace(NEW_FILL, OLD_FILL, NEW_CHUNK, OLD_CHUNK)
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
