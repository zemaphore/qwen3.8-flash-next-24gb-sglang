#!/usr/bin/env python3
"""PP14: overlap next-layer cold expert H2D with current-layer MoE compute.

The accepted PP12 stack gathers selected cold rows from pinned host memory into
the fused-MoE staging buffer immediately before each layer.  Large prefills
select nearly every cold row, leaving that H2D almost serialized with compute.

This opt-in patch adds one reusable full-expert device cache.  After layer L's
fused MoE is queued, all fixed cold rows for layer L+1 are copied into the
cache on a dedicated CUDA stream.  Layer L+1 waits for that copy, then uses the
existing table-gather kernel to compact resident and cached rows into the
ordinary staging layout.  A release event makes reuse safe; double buffering
is unnecessary because the cache is no longer read after compaction.

The feature is default-off and requires the existing elastic placement and
batched-DMA stack.  Prefetch defaults to prompts/chunks of at least 2,048
tokens so short interactive requests retain the selected-row path.  Enable it
for an experiment with:

  SGLANG_MOE_COLD_PREFETCH=1 /root/quant/serve-3090.sh

Usage:
  SGLANG=/root/sglang python3 patches/moe_cross_layer_prefetch.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang"))
STREAM = os.path.join(SG, "python/sglang/srt/layers/moe/expert_stream.py")
WNA16 = os.path.join(SG, "python/sglang/srt/layers/quantization/moe_wna16.py")

BEFORE_CONST = '''_DMA_BATCH_USABLE = True
'''

AFTER_CONST = '''_DMA_BATCH_USABLE = True
_COLD_PREFETCH = os.environ.get("SGLANG_MOE_COLD_PREFETCH", "0") == "1"
_COLD_PREFETCH_MIN_TOKENS = int(
    os.environ.get("SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS", "2048")
)
if _COLD_PREFETCH_MIN_TOKENS < 17:
    raise ValueError("SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS must be at least 17")
_PREFETCH_STREAM = None
_PREFETCH_CACHE: Dict[Tuple, torch.Tensor] = {}
_PREFETCH_READY = None
_PREFETCH_RELEASE = None
_PREFETCH_LAYER = None
_PREFETCH_STREAMERS = {}
_PREFETCH_LOGGED = False
'''

HELPER_ANCHOR = "class ExpertStreamer:"

HELPER = '''def _cold_prefetch_stream(device):
    global _PREFETCH_STREAM
    if _PREFETCH_STREAM is None:
        _PREFETCH_STREAM = torch.cuda.Stream(device=device)
    return _PREFETCH_STREAM


def _cold_prefetch_cache(name, max_k, rest, dtype, device):
    """One full-size cache per tensor kind, reused by every MoE layer."""
    global _PREFETCH_LOGGED
    key = (name, dtype, str(device), tuple(rest))
    buf = _PREFETCH_CACHE.get(key)
    if buf is None:
        buf = torch.empty((max_k,) + tuple(rest), dtype=dtype, device=device)
        _PREFETCH_CACHE[key] = buf
        logger.info(
            "MoE cold-prefetch cache %s: %s, %.0f MB",
            name, tuple(buf.shape), buf.numel() * buf.element_size() / 1e6,
        )
        if not _PREFETCH_LOGGED:
            _PREFETCH_LOGGED = True
            logger.info(
                "MoE cross-layer cold-row prefetch active at >=%d tokens",
                _COLD_PREFETCH_MIN_TOKENS,
            )
    return buf


'''

BEFORE_INIT = '''    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self.names: List[str] = []
        self.device = torch.device("cuda")
'''

AFTER_INIT = '''    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self.lid = int(getattr(layer, "layer_id", -1))
        self.names: List[str] = []
        self.device = torch.device("cuda")
        self._prefetch_cold = None
        self._prefetch_tabs = {}
'''

BEFORE_INIT_END = '''            logger.info(
                "MoE expert streaming active: %d experts, source on %s, "
                "tensors %s",
                t.shape[0], t.device, ",".join(self.names),
            )

    def gather(self, topk_ids: torch.Tensor
'''

AFTER_INIT_END = '''            logger.info(
                "MoE expert streaming active: %d experts, source on %s, "
                "tensors %s",
                t.shape[0], t.device, ",".join(self.names),
            )

        if _COLD_PREFETCH and getattr(layer, "_placed", None) is not None:
            _PREFETCH_STREAMERS[self.lid] = self

    def gather(self, topk_ids: torch.Tensor
'''

BEFORE_DISPATCH = '''        out: Dict[str, torch.Tensor] = {}
        placed = getattr(self.layer, "_placed", None)
        if placed is not None and _GATHER_DMA \\
                and all(nm in placed.get("home", {}) for nm in self.names):
            return self._gather_dma(placed, uniq, inverse, topk_ids)
'''

AFTER_DISPATCH = '''        out: Dict[str, torch.Tensor] = {}
        placed = getattr(self.layer, "_placed", None)
        if placed is not None and _GATHER_DMA \\
                and all(nm in placed.get("home", {}) for nm in self.names):
            if _COLD_PREFETCH and _PREFETCH_LAYER == self.lid:
                return self._gather_prefetched(placed, uniq, inverse, topk_ids)
            return self._gather_dma(placed, uniq, inverse, topk_ids)
'''

METHOD_ANCHOR = '''    def _gather_dma(self, placed, uniq, inverse, topk_ids):
'''

METHODS = '''    def _ensure_prefetch_tables(self, placed, cold):
        """Build expert-id address tables pointing cold ids into the cache."""
        cold = tuple(cold)
        if self._prefetch_cold == cold and len(self._prefetch_tabs) == len(self.names):
            return
        cold_ids = torch.tensor(cold, dtype=torch.int64, device=self.device)
        tabs = {}
        for name in self.names:
            proto = placed["proto"][name]
            cache = _cold_prefetch_cache(
                name, placed["E"], tuple(proto.shape[1:]), proto.dtype, self.device
            )
            row_bytes = proto[0].numel() * proto.element_size()
            tab = placed["addr"][name].clone()
            if cold:
                offsets = torch.arange(
                    len(cold), dtype=torch.int64, device=self.device
                ) * row_bytes
                tab[cold_ids] = offsets + cache.data_ptr()
            tabs[name] = tab
        self._prefetch_cold = cold
        self._prefetch_tabs = tabs

    def _prefetch_all_cold(self):
        """Queue this layer's fixed cold rows into the shared device cache."""
        global _PREFETCH_LAYER, _PREFETCH_READY
        placed = getattr(self.layer, "_placed", None)
        if placed is None or not self.names:
            return
        home = placed["home"]
        cold = [e for e, slot in enumerate(home[self.names[0]]) if slot is not None]
        for name in self.names[1:]:
            other = [e for e, slot in enumerate(home[name]) if slot is not None]
            if other != cold:
                raise RuntimeError("cold-prefetch residency differs across tensor kinds")
        self._ensure_prefetch_tables(placed, cold)
        stream = _cold_prefetch_stream(self.device)
        if _PREFETCH_RELEASE is not None:
            stream.wait_event(_PREFETCH_RELEASE)
        with torch.cuda.stream(stream):
            for name in self.names:
                proto = placed["proto"][name]
                cache = _cold_prefetch_cache(
                    name, placed["E"], tuple(proto.shape[1:]),
                    proto.dtype, self.device,
                )
                row_bytes = proto[0].numel() * proto.element_size()
                submitted = bool(cold) and _GATHER_DMA_BATCH and _dma_batch_submit(
                    placed, name, home[name], cold, cache, 0, row_bytes, stream
                )
                if cold and not submitted:
                    for j, e in enumerate(cold):
                        tensor, slot = home[name][e]
                        cache[j].copy_(tensor[slot], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        _PREFETCH_READY = ready
        _PREFETCH_LAYER = self.lid

    def prefetch_next(self):
        """Schedule layer L+1 only after layer L's fused MoE has been queued."""
        if (
            not _COLD_PREFETCH
            or getattr(self, "_last_m", 0) < _COLD_PREFETCH_MIN_TOKENS
        ):
            return
        nxt = _PREFETCH_STREAMERS.get(self.lid + 1)
        if nxt is not None:
            nxt._prefetch_all_cold()

    def _gather_prefetched(self, placed, uniq, inverse, topk_ids):
        """Compact resident and already-prefetched cold rows on the GPU."""
        global _PREFETCH_LAYER, _PREFETCH_RELEASE
        if _PREFETCH_READY is None or not self._prefetch_tabs:
            return self._gather_dma(placed, uniq, inverse, topk_ids)
        current = torch.cuda.current_stream(self.device)
        current.wait_event(_PREFETCH_READY)
        k = int(uniq.numel())
        out: Dict[str, torch.Tensor] = {}
        BLOCK = _GATHER_BLOCK
        for name in self.names:
            proto = placed["proto"][name]
            buf = _staging(
                name, k, placed["E"], tuple(proto.shape[1:]),
                proto.dtype, self.device,
            )
            row_bytes = proto[0].numel() * proto.element_size()
            _gather_rows_tab_kernel[(k, triton.cdiv(row_bytes, BLOCK))](
                self._prefetch_tabs[name], uniq, buf.view(torch.uint8),
                row_bytes, BLOCK=BLOCK,
            )
            out[name] = buf
        release = torch.cuda.Event()
        release.record(current)
        _PREFETCH_RELEASE = release
        _PREFETCH_LAYER = None
        new_ids = inverse.reshape(topk_ids.shape).to(topk_ids.dtype)
        return new_ids, out

'''

BEFORE_GATHER_START = '''    def gather(self, topk_ids: torch.Tensor
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat = topk_ids.reshape(-1)
'''

AFTER_GATHER_START = '''    def gather(self, topk_ids: torch.Tensor
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._last_m = int(topk_ids.shape[0])
        if _COLD_PREFETCH and self._last_m >= _COLD_PREFETCH_MIN_TOKENS:
            placed = getattr(self.layer, "_placed", None)
            if placed is not None:
                for name in self.names:
                    proto = placed["proto"][name]
                    _cold_prefetch_cache(
                        name, placed["E"], tuple(proto.shape[1:]),
                        proto.dtype, self.device,
                    )
        flat = topk_ids.reshape(-1)
'''

BEFORE_RUN = '''        return self.runner.run(dispatch_output, quant_info)

    def _maybe_expert_streamer(self, layer):
'''

AFTER_RUN = '''        result = self.runner.run(dispatch_output, quant_info)
        streamer.prefetch_next()
        return result

    def _maybe_expert_streamer(self, layer):
'''

EDITS = [
    (STREAM, BEFORE_CONST, AFTER_CONST),
    (STREAM, HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
    (STREAM, BEFORE_INIT, AFTER_INIT),
    (STREAM, BEFORE_INIT_END, AFTER_INIT_END),
    (STREAM, BEFORE_GATHER_START, AFTER_GATHER_START),
    (STREAM, BEFORE_DISPATCH, AFTER_DISPATCH),
    (STREAM, METHOD_ANCHOR, METHODS + METHOD_ANCHOR),
    (WNA16, BEFORE_RUN, AFTER_RUN),
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
    for path, source in changed.items():
        with open(path, "w", encoding="utf-8") as output:
            output.write(source)
    action = "reverted" if reverse else "applied (set SGLANG_MOE_COLD_PREFETCH=1 to enable)"
    print(f"  {action}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": transform, "revert": lambda: transform(True)}[command]()
