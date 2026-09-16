#!/usr/bin/env python3
"""Set the RTX 3090 PP-oriented chunk and expert-residency defaults.

States form a ladder over two accepted changes:

  clean     : pre-PP3 (fill 2048, chunk 1024)
  mid       : PP3 accepted (fill 99999, chunk 2048)
  applied   : PP12 accepted (fill 99999, chunk 4608)

``apply`` moves one step up (clean -> applied, or mid -> applied); ``revert``
moves one step down (applied -> mid, or mid -> clean). Both use the launcher's
environment fallbacks, so explicit ``SGLANG_3090_CHUNKED_PREFILL_SIZE`` /
``SGLANG_MOE_ELASTIC_FILL_MB`` values still override at start time.
"""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

OLD_FILL = 'SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-2048}"'
NEW_FILL = 'SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-99999}"'
OLD_CHUNK = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-1024}"'
MID_CHUNK = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-2048}"'
NEW_CHUNK = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-4608}"'


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def state() -> str:
    text = read()
    if OLD_FILL in text and OLD_CHUNK in text:
        return "clean"
    if NEW_FILL in text and MID_CHUNK in text:
        return "mid"
    if NEW_FILL in text and NEW_CHUNK in text:
        return "applied"
    return "mismatch"


def check() -> None:
    current = state()
    label = {"clean": "clean", "mid": "MID", "applied": "APPLIED"}.get(
        current, "MISMATCH"
    )
    print(f"  {label:<8} {LAUNCHER}: RTX 3090 PP chunk/residency defaults")
    if current == "mismatch":
        raise SystemExit(1)


def write(old: str, new: str) -> None:
    text = read()
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one launcher anchor {old!r}")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(old, new, 1))


def apply() -> None:
    current = state()
    if current == "applied":
        print("  already applied")
    elif current == "mid":
        write(MID_CHUNK, NEW_CHUNK)
        print("  applied (next launcher start defaults to chunk 4608 and S184)")
    elif current == "clean":
        write(OLD_FILL, NEW_FILL)
        write(OLD_CHUNK, NEW_CHUNK)
        print("  applied (next launcher start defaults to chunk 4608 and S184)")
    else:
        raise RuntimeError("launcher anchor mismatch")


def revert() -> None:
    current = state()
    if current == "clean":
        print("  already clean")
    elif current == "applied":
        write(NEW_CHUNK, MID_CHUNK)
        print("  reverted (back to the PP3 chunk 2048 default)")
    elif current == "mid":
        write(NEW_FILL, OLD_FILL)
        write(MID_CHUNK, OLD_CHUNK)
        print("  reverted (back to the pre-PP3 chunk 1024 default)")
    else:
        raise RuntimeError("launcher anchor mismatch")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
