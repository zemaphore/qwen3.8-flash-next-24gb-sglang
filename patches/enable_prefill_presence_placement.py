#!/usr/bin/env python3
"""Set the accepted PP5b prefill-presence placement on the RTX 3090 launcher.

The launcher passes ``SGLANG_MOE_PLACEMENT`` through, so the accepted
``assets/expert_presence_code.pt`` candidate becomes the default while an
explicit environment value can still select ``assets/expert_freq.pt`` (routing
mass) or any other placement file for an A/B.
"""

from __future__ import annotations

import os
import sys


LAUNCHER = os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh")

MASS_LINE = (
    '      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-$ASSETS/expert_freq.pt}" \\\n'
)
PRESENCE_LINE = (
    '      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-$ASSETS/expert_presence_code.pt}" \\\n'
)
OLD_MASS_LINE = '      SGLANG_MOE_PLACEMENT="$ASSETS/expert_freq.pt" \\\n'


def read() -> str:
    with open(LAUNCHER, encoding="utf-8") as source:
        return source.read()


def state() -> tuple[bool, bool]:
    text = read()
    applied = text.count(PRESENCE_LINE) == 1 and text.count(MASS_LINE) == 0
    clean = (
        text.count(PRESENCE_LINE) == 0
        and text.count(MASS_LINE) + text.count(OLD_MASS_LINE) == 1
    )
    return clean, applied


def check() -> None:
    clean, applied = state()
    status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {LAUNCHER}: PP5b prefill-presence placement default")
    if status == "MISMATCH":
        raise SystemExit(1)


def apply() -> None:
    clean, applied = state()
    if applied:
        print("  already applied")
        return
    if not clean:
        raise RuntimeError("launcher placement anchor mismatch")
    text = read()
    if text.count(OLD_MASS_LINE) == 1:
        with open(LAUNCHER, "w", encoding="utf-8") as output:
            output.write(text.replace(OLD_MASS_LINE, PRESENCE_LINE, 1))
    else:
        with open(LAUNCHER, "w", encoding="utf-8") as output:
            output.write(text.replace(MASS_LINE, PRESENCE_LINE, 1))
    print("  applied (next launcher start defaults to the PP5b presence placement)")


def revert() -> None:
    clean, applied = state()
    if clean:
        print("  already clean")
        return
    if not applied:
        raise RuntimeError("launcher placement anchor mismatch")
    text = read()
    with open(LAUNCHER, "w", encoding="utf-8") as output:
        output.write(text.replace(PRESENCE_LINE, MASS_LINE, 1))
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
