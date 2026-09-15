#!/usr/bin/env python3
"""Opt-in prefill routing recorder for PP5 placement studies.

When SGLANG_PREFILL_ROUTE_DUMP names a directory, each prefill forward
(M > 1) records its per-layer topk ids; the flush happens on the last MoE
layer and writes one file per forward: a list of (layer_id, ids int16
[M, top_k]) CPU tensors.  Decode forwards and idle passes are untouched.
The recorder is environment-gated and adds no work when the variable is
unset.

Usage:
  SGLANG=/root/sglang python3 patches/prefill_route_dump.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")),
    "python/sglang",
)
TARGET = os.path.join(SG, "srt/layers/quantization/moe_wna16.py")

BEFORE = """        _el = getattr(type(self), "_ELASTIC", None)
        if _el is not None:
            _el.poll(int(dispatch_output.hidden_states.shape[0]), int(getattr(layer, "layer_id", 0)))
"""

AFTER = """        _el = getattr(type(self), "_ELASTIC", None)
        if _el is not None:
            _el.poll(int(dispatch_output.hidden_states.shape[0]), int(getattr(layer, "layer_id", 0)))
        _prd = os.environ.get("SGLANG_PREFILL_ROUTE_DUMP")
        if _prd:
            _prefill_route_dump(_prd, layer, dispatch_output.topk_output,
                                int(dispatch_output.hidden_states.shape[0]))
"""

HELPER_ANCHOR = "class MoeWNA16Method(FusedMoEMethodBase):"
HELPER = '''_PRD = {"pending": [], "seq": 0}


def _prefill_route_dump(path, layer, topk, m):
    """Record prefill-chunk routing ids; sync only at the per-forward flush."""
    lid = int(getattr(layer, "layer_id", -1))
    if m <= 1:
        _PRD["pending"].clear()
        return
    _PRD["pending"].append((lid, topk.topk_ids.to(torch.int16).clone()))
    n_moe = int(getattr(layer, "num_layers", 48) or 48)
    if lid != n_moe - 1 or len(_PRD["pending"]) < n_moe:
        return
    os.makedirs(path, exist_ok=True)
    out = [(l, i.cpu()) for l, i in _PRD["pending"]]
    torch.save(out, os.path.join(path, f"prouting_{_PRD['seq']:05d}.pt"))
    _PRD["seq"] += 1
    _PRD["pending"].clear()


'''

EDITS = [
    (BEFORE, AFTER),
    (HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
]


def read() -> str:
    with open(TARGET, encoding="utf-8") as source:
        return source.read()


def states() -> list[tuple[bool, bool]]:
    text = read()
    values = []
    for before, after in EDITS:
        masked = text.replace(after, "\x00")
        values.append((before in masked, after in text))
    return values


def check() -> None:
    values = states()
    if all(applied and not clean for clean, applied in values):
        status = "APPLIED"
    elif all(clean and not applied for clean, applied in values):
        status = "clean"
    else:
        status = "MISMATCH"
    print(f"  {status:<8} {os.path.relpath(TARGET, SG)}: prefill route dump")


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
    print("  applied (SGLANG_PREFILL_ROUTE_DUMP records prefill routing per forward)")


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
