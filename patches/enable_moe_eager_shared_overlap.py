#!/usr/bin/env python3
"""Pass the opt-in eager shared/router overlap threshold through the launcher.

State detection is line-based (the presence of the environment line itself),
not adjacency-based: later pass-through patches insert sibling env lines and
must not invalidate this check.
"""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

LINE = ('      SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS='
        '"${SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS:-0}" \\\n')
ANCHOR = '      SGLANG_MOE_CONFIG_NEAREST_E="${SGLANG_MOE_CONFIG_NEAREST_E:-1}" \\\n'


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def check() -> None:
    text = read()
    if LINE in text:
        status = "APPLIED"
    elif ANCHOR in text:
        status = "clean"
    else:
        status = "MISMATCH"
    print(f"  {status:<8} {LAUNCHER}: eager shared/router overlap pass-through")


def apply() -> None:
    text = read()
    if LINE in text:
        print("  already applied")
        return
    if text.count(ANCHOR) != 1:
        raise RuntimeError("expected exactly one launcher anchor")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(ANCHOR, LINE + ANCHOR, 1))
    print("  applied (threshold pass-through present; default 0 = disabled)")


def revert() -> None:
    text = read()
    if LINE not in text:
        print("  already clean")
        return
    if text.count(LINE) != 1:
        raise RuntimeError("expected exactly one overlap line")
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(LINE, "", 1))
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
