#!/usr/bin/env python3
"""PP7 host-row DMA staging: copy cold expert rows with cudaMemcpyAsync.

The PP7 hybrid bench showed the low-token direct-GEMV variant cannot win:
resident-expert execution aside, both the gather kernel and the pointer-table
GEMV read pinned host memory through the SMs at ~7.2-7.5 GB/s in every tile
geometry tried (2048-8192 B tiles, int32 vectors, row-per-CTA), while the DMA
engine sustains ~22 GB/s for the same scattered pinned rows (the big-buffer
cudaMemcpy ceiling on this box is ~23 GB/s).  So stage exactly as today - one
kernel plus one copy per row - but move the host rows through the DMA engine.

The split is by current elastic residency: resident rows keep going through
the existing tab-kernel into the staging prefix (same bytes, GPU reads are
already fast), host rows are DMA'd one pinned slot at a time into the staging
tail.  Values are bit-identical; only the staging row ORDER changes, and the
top-k renumbering follows through a position lookup table.  The fused kernel
sees the same bytes per (renumbered) expert id - exactly equivalent by
construction.

Requires the elastic placement's per-expert home slots (expert_elastic.py
gains a "home" map in _placed).  Layers without it (plain placement, qzeros
kinds) keep the existing gather path untouched.

Env: SGLANG_MOE_GATHER_DMA=1 enables it; the 3090 launcher opts in.

Usage:
  SGLANG=/root/sglang python3 patches/moe_host_dma_gather.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang"))
STREAM = os.path.join(SG, "python/sglang/srt/layers/moe/expert_stream.py")
ELASTIC = os.path.join(SG, "python/sglang/srt/layers/moe/expert_elastic.py")

# ------------------------------------------------------------------ stream
BEFORE_DMA_CONST = """_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}
"""

AFTER_DMA_CONST = """_ARANGE_CACHE: Dict[Tuple, torch.Tensor] = {}
_GATHER_DMA = os.environ.get("SGLANG_MOE_GATHER_DMA", "0") == "1"
"""

BEFORE_DISPATCH = """        out: Dict[str, torch.Tensor] = {}
        placed = getattr(self.layer, "_placed", None)
        for name in self.names:
"""

AFTER_DISPATCH = """        out: Dict[str, torch.Tensor] = {}
        placed = getattr(self.layer, "_placed", None)
        if placed is not None and _GATHER_DMA \\
                and all(nm in placed.get("home", {}) for nm in self.names):
            return self._gather_dma(placed, uniq, inverse, topk_ids)
        for name in self.names:
"""

BEFORE_TAIL = """            out[name] = buf
        return inverse.reshape(topk_ids.shape).to(topk_ids.dtype), out
"""

AFTER_TAIL = '''            out[name] = buf
        return inverse.reshape(topk_ids.shape).to(topk_ids.dtype), out

    def _gather_dma(self, placed, uniq, inverse, topk_ids):
        """Resident rows via one tab-kernel launch; host rows via DMA copy_.

        Staging layout: resident rows first (in uniq order), then host rows.
        The returned ids are renumbered onto that layout through a position
        lookup; bytes per id are identical to the pure-kernel path, so the
        fused MoE result is bit-exact.
        """
        home = placed["home"]
        h0 = home[self.names[0]]
        u = uniq.tolist()
        kr = 0
        lut = torch.empty(len(u), dtype=torch.int32)
        for i, e in enumerate(u):
            if h0[e] is None:
                kr += 1
        r = h = 0
        h = kr
        for i, e in enumerate(u):
            if h0[e] is None:
                lut[i] = r
                r += 1
            else:
                lut[i] = h
                h += 1
        dev = self.device
        res_ids = torch.tensor([e for e in u if h0[e] is None],
                               dtype=torch.int64, device=dev) if kr else None
        host = [e for e in u if h0[e] is not None]
        lut = lut.to(dev)
        new_ids = lut.index_select(0, inverse.to(torch.int64)) \\
                     .reshape(topk_ids.shape).to(topk_ids.dtype)
        out: Dict[str, torch.Tensor] = {}
        BLOCK = _GATHER_BLOCK
        for name in self.names:
            proto = placed["proto"][name]
            buf = _staging(name, len(u), placed["E"], tuple(proto.shape[1:]),
                           proto.dtype, dev)
            if kr:
                row_bytes = proto[0].numel() * proto.element_size()
                _gather_rows_tab_kernel[(kr, triton.cdiv(row_bytes, BLOCK))](
                    placed["addr"][name], res_ids,
                    buf.view(torch.uint8)[:kr], row_bytes, BLOCK=BLOCK)
            hm = home[name]
            for j, e in enumerate(host):
                t, slot = hm[e]
                buf[kr + j].copy_(t[slot], non_blocking=True)
            out[name] = buf
        return new_ids, out
'''

# ------------------------------------------------------------------ elastic
BEFORE_PLACED = """        layer._placed = {"addr": addr, "proto": proto, "E": E, "S": S, "keep": []}
"""

AFTER_PLACED = """        layer._placed = {"addr": addr, "proto": proto, "E": E, "S": S, "keep": [],
                         "home": {n: k.home for n, k in st.kinds.items()}}
"""

EDITS = [
    (STREAM, BEFORE_DMA_CONST, AFTER_DMA_CONST),
    (STREAM, BEFORE_DISPATCH, AFTER_DISPATCH),
    (STREAM, BEFORE_TAIL, AFTER_TAIL),
    (ELASTIC, BEFORE_PLACED, AFTER_PLACED),
]


def state(path: str, before: str, after: str) -> tuple[bool, bool]:
    with open(path, encoding="utf-8") as source:
        text = source.read()
    return before in text, after in text


def check() -> None:
    ok = True
    for path, before, after in EDITS:
        clean, applied = state(path, before, after)
        status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
        ok = ok and (applied or clean)
        print(f"  {status:<8} {os.path.relpath(path, SG)}: "
              f"{before.splitlines()[0].strip()[:60]}")
    if not ok:
        raise SystemExit(1)


def apply() -> None:
    for path, before, after in EDITS:
        clean, applied = state(path, before, after)
        if applied:
            continue
        if not clean:
            raise RuntimeError(f"anchor mismatch in {path}")
        with open(path, encoding="utf-8") as source:
            text = source.read()
        if text.count(before) != 1:
            raise RuntimeError(f"expected one anchor in {path}, found "
                               f"{text.count(before)}")
        with open(path, "w", encoding="utf-8") as out:
            out.write(text.replace(before, after, 1))
    print("  applied (set SGLANG_MOE_GATHER_DMA=1 to enable)")


def revert() -> None:
    for path, before, after in reversed(EDITS):
        clean, applied = state(path, before, after)
        if clean:
            continue
        if not applied:
            raise RuntimeError(f"anchor mismatch in {path}")
        with open(path, encoding="utf-8") as source:
            text = source.read()
        if text.count(after) != 1:
            raise RuntimeError(f"expected one block in {path}")
        with open(path, "w", encoding="utf-8") as out:
            out.write(text.replace(after, before, 1))
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
