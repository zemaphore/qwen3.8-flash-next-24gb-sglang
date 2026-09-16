#!/usr/bin/env python3
"""Opt-in *decode* routing recorder (recorded negative result).

TG0 needs per-token decode selections to separate resident/host coverage from
gate-weight mass.  This patch was written to capture them, but it is **not
usable on the accepted stack**: the decode path is replayed by the breakable
decode CUDA graph, so Python in ``MoeWNA16Method.apply`` runs only during graph
*capture*, never during replay.  Recording there observes capture-time routing
(one fixed input), and an unconditional CPU copy inside capture aborts the
capture itself (``Cannot copy between CPU and CUDA tensors during CUDA graph
capture``).  It is retained as the TG0 negative result and left capture-safe: it
does nothing unless ``SGLANG_DECODE_ROUTE_DUMP`` names a directory *and* the
decode graph is disabled, which the accepted launcher does not do.

The supported in-graph mechanism is SGLang's ``enable_return_routed_experts``
capturer, which TG0 did not run because it requires a diagnostic server-arg
change and return plumbing.  See the TG0 report's validation limitations.

This is temporary diagnostic instrumentation; revert it before accepted
measurements:

  SGLANG=/root/sglang python3 patches/decode_route_dump.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys

SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")),
    "python/sglang",
)
TARGET = os.path.join(SG, "srt/layers/quantization/moe_wna16.py")

BEFORE = """        if (
            getattr(layer, "_b_n_contig", False)
            and self.quant_config.weight_bits == 2
            and dispatch_output.hidden_states.shape[0] <= 16
            and os.environ.get("SGLANG_MOE_GEMV", "1") == "1"
        ):
            return self._apply_gemv(layer, dispatch_output)
"""

AFTER = """        _drd = os.environ.get("SGLANG_DECODE_ROUTE_DUMP")
        if _drd and dispatch_output.hidden_states.shape[0] <= 16:
            _decode_route_dump(_drd, layer, dispatch_output.topk_output)
        if (
            getattr(layer, "_b_n_contig", False)
            and self.quant_config.weight_bits == 2
            and dispatch_output.hidden_states.shape[0] <= 16
            and os.environ.get("SGLANG_MOE_GEMV", "1") == "1"
        ):
            return self._apply_gemv(layer, dispatch_output)
"""

HELPER_ANCHOR = "class MoeWNA16Method(FusedMoEMethodBase):"
HELPER = '''_DRD = {"pending": [], "seq": 0}


def _decode_route_dump(path, layer, topk):
    """Record one eager decode forward's per-layer topk ids and gate weights.

    Skipped during graph capture: the CPU copy would abort capture, and the
    graph replays without running this Python anyway.
    """
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode():
        return
    lid = int(getattr(layer, "layer_id", -1))
    ids = topk.topk_ids.detach().to(torch.int16).cpu()
    weights = topk.topk_weights.detach().to(torch.float32).cpu()
    _DRD["pending"].append((lid, ids, weights))
    n_moe = int(getattr(layer, "num_layers", 48) or 48)
    if lid != n_moe - 1 or len(_DRD["pending"]) < n_moe:
        return
    os.makedirs(path, exist_ok=True)
    torch.save(_DRD["pending"], os.path.join(path, f"drouting_{_DRD['seq']:05d}.pt"))
    _DRD["seq"] += 1
    _DRD["pending"].clear()


'''

EDITS = [
    (BEFORE, AFTER),
    (HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
]


def read() -> str:
    with open(TARGET, encoding="utf-8") as source:
        return source.read()


def states():
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
    print(f"  {status:<8} {os.path.relpath(TARGET, SG)}: decode route dump")


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
    print("  applied (SGLANG_DECODE_ROUTE_DUMP records per-token decode routing)")


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
