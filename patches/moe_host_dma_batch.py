#!/usr/bin/env python3
"""PP11: batch scattered pinned expert-row copies with cudaMemcpyBatchAsync.

Applies on top of ``moe_host_dma_gather.py``.  The source feature is opt-in via
``SGLANG_MOE_GATHER_DMA_BATCH=1``; the measured 3090 launcher defaults it on.
It preserves the PP7 unique set,
staging layout, bytes, and top-k renumbering; only the host submission path is
changed from one Python/Tensor operation per row to one CUDA batch per tensor
kind.  A dedicated non-default stream is required because CUDA rejects the
legacy NULL stream for this API.  One event bridge per layer preserves staging
buffer lifetime and permits resident-row gathers to overlap the host DMA.

Usage:
  SGLANG=/root/sglang python3 patches/moe_host_dma_batch.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang"))
STREAM = os.path.join(SG, "python/sglang/srt/layers/moe/expert_stream.py")
ELASTIC = os.path.join(SG, "python/sglang/srt/layers/moe/expert_elastic.py")

BEFORE_IMPORT = """import os
from typing import Dict, List, Tuple
"""

AFTER_IMPORT = """import ctypes
import os
from typing import Dict, List, Tuple
"""

BEFORE_CONST = '''_GATHER_DMA = os.environ.get("SGLANG_MOE_GATHER_DMA", "0") == "1"
'''

AFTER_CONST = '''_GATHER_DMA = os.environ.get("SGLANG_MOE_GATHER_DMA", "0") == "1"
_GATHER_DMA_BATCH = os.environ.get("SGLANG_MOE_GATHER_DMA_BATCH", "0") == "1"
_DMA_BATCH_STREAM = None
_DMA_BATCH_FN = None
_DMA_BATCH_USABLE = True


class _CudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _CudaMemLocation),
        ("dstLocHint", _CudaMemLocation),
        ("flags", ctypes.c_uint),
    ]
'''

HELPER_ANCHOR = "class ExpertStreamer:"

HELPER = '''def _dma_batch_stream(device):
    global _DMA_BATCH_STREAM
    if _DMA_BATCH_STREAM is None:
        _DMA_BATCH_STREAM = torch.cuda.Stream(device=device)
    return _DMA_BATCH_STREAM


def _dma_batch_fn():
    global _DMA_BATCH_FN
    if _DMA_BATCH_FN is None:
        fn = ctypes.CDLL(None).cudaMemcpyBatchAsync
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(_CudaMemcpyAttributes),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
        _DMA_BATCH_FN = fn
    return _DMA_BATCH_FN


def _dma_batch_submit(placed, name, hm, host, buf, dst_start, row_bytes, stream):
    """Submit independent stable-pinned-source copies; False requests fallback."""
    global _DMA_BATCH_USABLE
    if not _DMA_BATCH_USABLE:
        return False
    cache = placed.setdefault("_dma_src_addr", {})
    src_all = cache.get(name)
    if src_all is None:
        src_all = []
        for slot in hm:
            if slot is None:
                src_all.append(0)
            else:
                tensor, row = slot
                src_all.append(tensor.data_ptr() + row * row_bytes)
        cache[name] = src_all
    src_values = [src_all[e] for e in host]
    if not all(src_values):
        raise RuntimeError(f"DMA batch source map disagrees with residency for {name}")
    count = len(src_values)
    srcs = (ctypes.c_void_p * count)(*src_values)
    base = buf.data_ptr() + dst_start * row_bytes
    dsts = (ctypes.c_void_p * count)(
        *(base + j * row_bytes for j in range(count))
    )
    sizes = (ctypes.c_size_t * count)(*[row_bytes] * count)
    attrs = _CudaMemcpyAttributes(
        3,                         # stable pinned rows: access may outlive API call
        _CudaMemLocation(2, 0),    # host
        _CudaMemLocation(1, torch.cuda.current_device()),
        1,                         # prefer overlap with compute
    )
    attrs_idx = (ctypes.c_size_t * 1)(0)
    err = _dma_batch_fn()(
        dsts, srcs, sizes, count, ctypes.byref(attrs), attrs_idx, 1,
        ctypes.c_void_p(stream.cuda_stream),
    )
    if err in (36, 801):           # newer driver required / not supported
        ctypes.CDLL(None).cudaGetLastError()
        _DMA_BATCH_USABLE = False
        logger.warning("cudaMemcpyBatchAsync unavailable (cudaError=%d); using copy_ loop", err)
        return False
    if err:
        raise RuntimeError(f"cudaMemcpyBatchAsync failed with cudaError={err}")
    return True


'''

BEFORE_GATHER = '''        out: Dict[str, torch.Tensor] = {}
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

AFTER_GATHER = '''        out: Dict[str, torch.Tensor] = {}
        BLOCK = _GATHER_BLOCK
        use_batch = _GATHER_DMA_BATCH and bool(host)
        current = torch.cuda.current_stream(dev) if use_batch else None
        copy_stream = _dma_batch_stream(dev) if use_batch else None
        if use_batch:
            copy_stream.wait_stream(current)
        for name in self.names:
            proto = placed["proto"][name]
            buf = _staging(name, len(u), placed["E"], tuple(proto.shape[1:]),
                           proto.dtype, dev)
            row_bytes = proto[0].numel() * proto.element_size()
            if kr:
                _gather_rows_tab_kernel[(kr, triton.cdiv(row_bytes, BLOCK))](
                    placed["addr"][name], res_ids,
                    buf.view(torch.uint8)[:kr], row_bytes, BLOCK=BLOCK)
            hm = home[name]
            submitted = False
            if use_batch:
                with torch.cuda.stream(copy_stream):
                    submitted = _dma_batch_submit(
                        placed, name, hm, host, buf, kr, row_bytes, copy_stream
                    )
            if not submitted:
                for j, e in enumerate(host):
                    t, slot = hm[e]
                    buf[kr + j].copy_(t[slot], non_blocking=True)
            out[name] = buf
        if use_batch:
            current.wait_stream(copy_stream)
        return new_ids, out
'''

BEFORE_INVALIDATE = '''        st.layer._placed["S"] = S_new
        st.S = S_new
'''

AFTER_INVALIDATE = '''        st.layer._placed["S"] = S_new
        st.layer._placed.pop("_dma_src_addr", None)
        st.S = S_new
'''

EDITS = [
    (STREAM, BEFORE_IMPORT, AFTER_IMPORT),
    (STREAM, BEFORE_CONST, AFTER_CONST),
    (STREAM, HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
    (STREAM, BEFORE_GATHER, AFTER_GATHER),
    (ELASTIC, BEFORE_INVALIDATE, AFTER_INVALIDATE),
]


def read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def states():
    values = []
    for path, before, after in EDITS:
        text = read(path)
        masked = text.replace(after, "\x00")
        values.append((masked.count(before) == 1, text.count(after) == 1))
    return values


def all_applied(values):
    return all(applied and not clean for clean, applied in values)


def all_clean(values):
    return all(clean and not applied for clean, applied in values)


def check():
    values = states()
    for (path, before, _), (clean, applied) in zip(EDITS, values):
        status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
        print(f"  {status:<8} {os.path.relpath(path, SG)}: {before.splitlines()[0][:60]}")
    if not (all_applied(values) or all_clean(values)):
        raise SystemExit(1)


def transform(reverse=False):
    values = states()
    wanted = all_applied(values) if reverse else all_clean(values)
    already = all_clean(values) if reverse else all_applied(values)
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
            raise RuntimeError(f"expected one anchor in {path}")
        changed[path] = changed[path].replace(old, new, 1)
    for path, text in changed.items():
        with open(path, "w", encoding="utf-8") as output:
            output.write(text)
    print("  reverted" if reverse else "  applied (set SGLANG_MOE_GATHER_DMA_BATCH=1 to enable)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": transform, "revert": lambda: transform(True)}[command]()
