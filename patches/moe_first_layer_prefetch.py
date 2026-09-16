#!/usr/bin/env python3
"""Prefetch the first MoE layer's cold rows at the start of a prefill forward.

PP14 copies layer L+1's fixed cold rows while layer L computes, so only layer 0
keeps the old selected-row path: the PP14 trace attributes 18.4 ms of pinned
HtoD to layer 0 alone, serialized before its fused MoE.  The cold-row set does
not depend on routing, so layer 0's copy can start before its attention and be
hidden the same way layers 1..47 already are.

The feature requires PP14 and is gated by
``SGLANG_MOE_PREFETCH_FIRST_LAYER`` (default 0, opt-in) plus the existing
``SGLANG_MOE_COLD_PREFETCH`` / ``SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS``.  It adds
no memory: the copy targets the existing shared prefetch cache.  It is a no-op
until the layer streamers exist (the first forward of a fresh server).

Status: the 10-sample bracketed A/B measured 2703.1 +/- 17.1 versus 2679.8 +/-
23.5 (+0.87%, t=2.26), but pooling the 5-sample capture and the 10-sample arms
gives only +0.3%, inside the capture noise band.  It is therefore opt-in rather
than a launcher default; see docs/logs/pp15_real_routing_moe_3090_2026-09-16.md.

Usage:
  SGLANG=/root/sglang python3 patches/moe_first_layer_prefetch.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang"))
STREAM = os.path.join(SG, "python/sglang/srt/layers/moe/expert_stream.py")
QWEN4 = os.path.join(SG, "python/sglang/srt/models/qwen4_exp.py")

BEFORE_CONST = '''def enabled() -> bool:
    return os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"
'''

AFTER_CONST = '''def enabled() -> bool:
    return os.environ.get("SGLANG_MOE_EXPERT_STREAM") == "1"


_FIRST_LAYER_PREFETCH = os.environ.get(
    "SGLANG_MOE_PREFETCH_FIRST_LAYER", "0"
) == "1"


def prefetch_first_layer(num_tokens: int) -> None:
    """Start the first streamer's routing-independent cold-row copy early."""
    if not _FIRST_LAYER_PREFETCH or not _COLD_PREFETCH:
        return
    if num_tokens < _COLD_PREFETCH_MIN_TOKENS:
        return
    if not _PREFETCH_STREAMERS:
        return
    first = min(_PREFETCH_STREAMERS)
    _PREFETCH_STREAMERS[first]._prefetch_all_cold()
'''

BEFORE_HOOK = '''        residual = None
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
'''

AFTER_HOOK = '''        residual = None
        aux_hidden_states = []
        if expert_stream.enabled():
            expert_stream.prefetch_first_layer(int(hidden_states.shape[0]))
        for i in range(self.start_layer, self.end_layer):
'''

# The model module does not import expert_stream, so add one import alongside
# an existing sglang import that PP14 does not touch.
BEFORE_IMPORT = '''from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
'''

AFTER_IMPORT = '''from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe import expert_stream
'''

EDITS = [
    (STREAM, BEFORE_CONST, AFTER_CONST),
    (QWEN4, BEFORE_HOOK, AFTER_HOOK),
    (QWEN4, BEFORE_IMPORT, AFTER_IMPORT),
]


def read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def states():
    values = []
    for path, before, after in EDITS:
        source = read(path)
        masked = source.replace(after, "\x00")
        values.append((masked.count(before) == 1, source.count(after) == 1))
    return values


def check():
    values = states()
    for (path, before, _), (clean, applied) in zip(EDITS, values):
        status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
        print(f"  {status:<8} {os.path.relpath(path, SG)}: {before.splitlines()[0][:60]}")
    if not (all(applied and not clean for clean, applied in values)
            or all(clean and not applied for clean, applied in values)):
        raise SystemExit(1)


def transform(reverse=False):
    values = states()
    wanted = all(a and not c for c, a in values) if reverse else all(c and not a for c, a in values)
    already = all(c and not a for c, a in values) if reverse else all(a and not c for c, a in values)
    if already:
        print("  already clean" if reverse else "  already applied")
        return
    if not wanted:
        check()
        raise RuntimeError("patch state mismatch")
    changed = {}
    edits = reversed(EDITS) if reverse else EDITS
    for path, before, after in edits:
        if path not in changed:
            changed[path] = read(path)
        old, new = (after, before) if reverse else (before, after)
        if changed[path].count(old) != 1:
            raise RuntimeError(f"expected one anchor in {path}, found {changed[path].count(old)}")
        changed[path] = changed[path].replace(old, new, 1)
    for path, source in changed.items():
        with open(path, "w", encoding="utf-8") as output:
            output.write(source)
    print("  reverted" if reverse else "  applied (set SGLANG_MOE_PREFETCH_FIRST_LAYER=1 to enable)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": transform, "revert": lambda: transform(True)}[command]()
