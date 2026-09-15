#!/usr/bin/env python3
"""Make Qwen MoE shared/router dual-stream execution opt-in for eager prefill.

The upstream path already overlaps the independent shared-expert MLP and routed
experts during CUDA-graph capture.  This patch extends that existing path to
ordinary extend batches up to an environment-controlled token count.  The
source default is zero, so behavior is unchanged unless explicitly enabled.

Usage:
  SGLANG=/root/sglang python3 patches/moe_eager_shared_overlap.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")),
    "python/sglang",
)
TARGET = os.path.join(SG, "srt/models/qwen2_moe.py")

BEFORE_IMPORT = """import logging
from contextlib import nullcontext
"""
AFTER_IMPORT = """import logging
import os
from contextlib import nullcontext
"""

BEFORE_CONST = """_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

logger = logging.getLogger(__name__)
"""
AFTER_CONST = """_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
_SHARED_EXPERT_EAGER_ALT_STREAM_MAX_TOKENS = int(
    os.environ.get("SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS", "0")
)
if _SHARED_EXPERT_EAGER_ALT_STREAM_MAX_TOKENS < 0:
    raise ValueError("SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS must be non-negative")

logger = logging.getLogger(__name__)
"""

BEFORE_GATE = """        elif (
            self.alt_stream is not None
            and get_is_capture_mode()
            and not torch.compiler.is_compiling()
        ):
"""
AFTER_GATE = """        elif (
            self.alt_stream is not None
            and not torch.compiler.is_compiling()
            and (
                get_is_capture_mode()
                or (
                    _SHARED_EXPERT_EAGER_ALT_STREAM_MAX_TOKENS > 0
                    and forward_batch is not None
                    and forward_batch.forward_mode.is_extend()
                    and num_tokens
                    <= _SHARED_EXPERT_EAGER_ALT_STREAM_MAX_TOKENS
                )
            )
        ):
"""

EDITS = [
    (BEFORE_IMPORT, AFTER_IMPORT),
    (BEFORE_CONST, AFTER_CONST),
    (BEFORE_GATE, AFTER_GATE),
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
    print(f"  {status:<8} {os.path.relpath(TARGET, SG)}: eager shared/router overlap")


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
    print("  applied (set SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS to opt in)")


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
