#!/usr/bin/env python3
"""PP15: skip the data-dependent top-k dedup for prefetched MoE layers.

PP14 stages every cold expert row into a device cache and then compacts only the
selected rows through the tabular gather.  Before that it must call
``torch.unique(topk_ids, return_inverse=True)``: the output size is
data-dependent, so it forces a device synchronization per layer.  A canonical
PP14 profile attributes ~922 ms of CPU wall to ``aten::_unique2`` (and ~906 ms to
the ``cudaStreamSynchronize`` it triggers) inside the 1743 ms extend span.

Once PP14 has prefetched the whole cold set, every expert row is addressable on
the GPU (resident rows in the resident tensor, cold rows in the cache).  The
``_prefetch_tabs`` table already maps *all* expert ids to those addresses, so the
fused MoE can read a full ``E``-row staging buffer with the router's original
ids.  That removes ``torch.unique``, the inverse renumbering, and the per-layer
sync from layers 1..47.  The per-token expert set and summation order are
unchanged, so the fused result is identical.

This is default-off (``SGLANG_MOE_PREFETCH_FULL_TABLE=1``) and requires PP14 to
be applied.  Layer 0 and sub-threshold chunks keep the selected-row path.

Usage:
  SGLANG=/root/sglang python3 patches/moe_prefetch_full_table.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang"))
STREAM = os.path.join(SG, "python/sglang/srt/layers/moe/expert_stream.py")

BEFORE_CONST = '''_PREFETCH_STREAMERS = {}
_PREFETCH_LOGGED = False
'''

AFTER_CONST = '''_PREFETCH_STREAMERS = {}
_PREFETCH_LOGGED = False
_FULL_TABLE = os.environ.get("SGLANG_MOE_PREFETCH_FULL_TABLE", "0") == "1"
'''

BEFORE_SHORTCUT = '''        flat = topk_ids.reshape(-1)
        n = flat.numel()
'''

AFTER_SHORTCUT = '''        flat = topk_ids.reshape(-1)
        if (
            _FULL_TABLE
            and _COLD_PREFETCH
            and _GATHER_DMA
            and self._prefetch_tabs
        ):
            _ft_placed = getattr(self.layer, "_placed", None)
            if (
                _ft_placed is not None
                and _PREFETCH_LAYER == self.lid
                and all(nm in _ft_placed.get("home", {}) for nm in self.names)
            ):
                return _prefetched_full_gather(self, _ft_placed, topk_ids)
        n = flat.numel()
'''

BEFORE_TAIL = '''        if use_batch:
            current.wait_stream(copy_stream)
        return new_ids, out
'''

AFTER_TAIL = '''        if use_batch:
            current.wait_stream(copy_stream)
        return new_ids, out


def _prefetched_full_gather(self, placed, topk_ids):
    """Gather every expert id into a full staging table; keep the router ids.

    ``self._prefetch_tabs`` already maps resident ids to the resident tensor and
    prefetched cold ids into the reusable device cache.  Copying ``arange(E)``
    through it produces a complete ``E``-row table indexed by expert id, so the
    fused kernel consumes the original ``topk_ids`` directly.  No data-dependent
    size is computed, so no device synchronization is required.
    """
    global _PREFETCH_LAYER, _PREFETCH_RELEASE
    current = torch.cuda.current_stream(self.device)
    current.wait_event(_PREFETCH_READY)
    experts = int(placed["E"])
    all_ids = _cached_arange(experts, self.device, torch.int64)
    block = _GATHER_BLOCK
    out: Dict[str, torch.Tensor] = {}
    for name in self.names:
        proto = placed["proto"][name]
        buf = _staging(
            name, experts, experts, tuple(proto.shape[1:]),
            proto.dtype, self.device,
        )
        row_bytes = proto[0].numel() * proto.element_size()
        _gather_rows_tab_kernel[(experts, triton.cdiv(row_bytes, block))](
            self._prefetch_tabs[name], all_ids, buf.view(torch.uint8),
            row_bytes, BLOCK=block,
        )
        out[name] = buf
    release = torch.cuda.Event()
    release.record(current)
    _PREFETCH_RELEASE = release
    _PREFETCH_LAYER = None
    return topk_ids, out
'''

EDITS = [
    (BEFORE_CONST, AFTER_CONST),
    (BEFORE_SHORTCUT, AFTER_SHORTCUT),
    (BEFORE_TAIL, AFTER_TAIL),
]


def read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def states():
    text = read(STREAM)
    values = []
    for before, after in EDITS:
        masked = text.replace(after, "\x00")
        values.append((masked.count(before) == 1, text.count(after) == 1))
    return values


def check():
    values = states()
    for (before, _), (clean, applied) in zip(EDITS, values):
        status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
        print(f"  {status:<8} {os.path.relpath(STREAM, SG)}: {before.splitlines()[0][:60]}")
    if not (all(c and not a for c, a in values) or all(a and not c for c, a in values)):
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
    text = read(STREAM)
    edits = reversed(EDITS) if reverse else EDITS
    for before, after in edits:
        old, new = (after, before) if reverse else (before, after)
        if text.count(old) != 1:
            raise RuntimeError(f"expected one anchor, found {text.count(old)}")
        text = text.replace(old, new, 1)
    with open(STREAM, "w", encoding="utf-8") as output:
        output.write(text)
    print("  reverted" if reverse else "  applied (set SGLANG_MOE_PREFETCH_FULL_TABLE=1 to enable)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": transform, "revert": lambda: transform(True)}[command]()
