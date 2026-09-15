#!/usr/bin/env python3
"""Bounded recent-row cache for the mmap-backed Qwen4 PLE table.

This patch layers on top of ``ple_bulk_pread.py``.  High-entropy prefill chunks
still use bounded parallel pread, but their deduplicated FP8 rows are retained
in a small exact-match cache.  Repeating the same chunk then avoids both random
page faults and thousands of tiny pread syscalls.  Decode remains unchanged.

Environment:
  SGLANG_QWEN4_PLE_RECENT_CACHE_MB=0    disabled (default)
  SGLANG_QWEN4_PLE_RECENT_CACHE_MB=128  bounded retained row payload
  SGLANG_QWEN4_PLE_PROFILE=1            per-prefill gather diagnostics

Usage:
  SGLANG=/root/sglang python3 patches/ple_recent_row_cache.py --check
  SGLANG=/root/sglang python3 patches/ple_recent_row_cache.py apply
  SGLANG=/root/sglang python3 patches/ple_recent_row_cache.py revert
"""

from __future__ import annotations

import os
import sys


SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")),
    "python/sglang",
)
QWEN4 = os.path.join(SG, "srt/models/qwen4_exp.py")

BEFORE_IMPORT = """import math
import os
from contextlib import nullcontext
"""

AFTER_IMPORT = """import math
import os
import time
from contextlib import nullcontext
"""

BEFORE_INIT = """        self._row_bytes = self._dim * itemsize

        scale = torch.tensor([float(meta[\"weight_scale\"])], dtype=torch.bfloat16)
"""

AFTER_INIT = """        self._row_bytes = self._dim * itemsize
        self._recent_cache_limit = max(
            0, int(os.environ.get(\"SGLANG_QWEN4_PLE_RECENT_CACHE_MB\", \"0\"))
        ) * 1024 * 1024
        self._recent_row_cache = []
        self._recent_cache_bytes = 0
        self._profile_ple = os.environ.get(\"SGLANG_QWEN4_PLE_PROFILE\") == \"1\"

        scale = torch.tensor([float(meta[\"weight_scale\"])], dtype=torch.bfloat16)
"""

BEFORE_GATHER = """        # Decode always uses parallel pread.  For prefill, retain mmap when
        # the row working set is small (the repeated-prompt case is then served
        # almost entirely from the page cache) and expose NVMe queue depth for
        # high-entropy chunks.  Deduplication is on CPU, after the unavoidable
        # ids D2H, so querying the unique count does not synchronize the GPU.
        unique, inverse = torch.unique(local, return_inverse=True)
        bulk_pread = os.environ.get(\"SGLANG_QWEN4_PLE_BULK_PREAD\") == \"1\"
        min_unique = int(os.environ.get(
            \"SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE\", \"2048\"
        ))
        use_pread = local.numel() <= 64 or (
            bulk_pread and unique.numel() >= min_unique
        )
        if use_pread:
            rb = self._row_bytes

            def _rd(r):
                data = os.pread(self._fd, rb, int(r) * rb)
                if len(data) != rb:
                    raise RuntimeError(f\"short PLE read: {len(data)} != {rb}\")
                return data

            buf = b\"\".join(self._pool.map(_rd, unique.tolist()))
            unique_rows = torch.frombuffer(
                bytearray(buf), dtype=torch.uint8
            ).view(-1, rb)
            rows = unique_rows.index_select(0, inverse)
        else:
            rows = torch.from_numpy(self._mm[local.numpy()])      # uint8, (N, dim*b)

        # Large random working sets have no useful reuse and otherwise evict
        # the pinned expert working set.  A small repeated set stays cached.
        if unique.numel() >= min_unique:
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:                                  # pragma: no cover
                pass
"""

AFTER_GATHER = """        # Decode always uses parallel pread.  For prefill, retain mmap when
        # the row working set is small and expose NVMe queue depth for
        # high-entropy chunks.  A bounded exact recent-row cache makes repeated
        # chunks independent of kernel page-cache eviction heuristics.
        gather_started = time.perf_counter()
        unique, inverse = torch.unique(local, return_inverse=True)
        bulk_pread = os.environ.get(\"SGLANG_QWEN4_PLE_BULK_PREAD\") == \"1\"
        min_unique = int(os.environ.get(
            \"SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE\", \"2048\"
        ))
        is_decode_sized = local.numel() <= 64
        use_pread = is_decode_sized or (
            bulk_pread and unique.numel() >= min_unique
        )
        path = \"mmap\"
        if use_pread:
            rb = self._row_bytes
            unique_rows = None
            if not is_decode_sized and self._recent_cache_limit:
                for cached_ids, cached_rows, _ in reversed(self._recent_row_cache):
                    if cached_ids.numel() == unique.numel() and torch.equal(
                        cached_ids, unique
                    ):
                        unique_rows = cached_rows
                        path = \"recent\"
                        break

            if unique_rows is None:
                path = \"pread\"

                def _rd(r):
                    data = os.pread(self._fd, rb, int(r) * rb)
                    if len(data) != rb:
                        raise RuntimeError(f\"short PLE read: {len(data)} != {rb}\")
                    return data

                buf = b\"\".join(self._pool.map(_rd, unique.tolist()))
                unique_rows = torch.frombuffer(
                    bytearray(buf), dtype=torch.uint8
                ).view(-1, rb)

                if not is_decode_sized and self._recent_cache_limit:
                    entry_bytes = unique.numel() * (
                        rb + unique.element_size()
                    )
                    if entry_bytes <= self._recent_cache_limit:
                        while (
                            self._recent_row_cache
                            and self._recent_cache_bytes + entry_bytes
                            > self._recent_cache_limit
                        ):
                            _, _, evicted_bytes = self._recent_row_cache.pop(0)
                            self._recent_cache_bytes -= evicted_bytes
                        self._recent_row_cache.append(
                            (unique.clone(), unique_rows, entry_bytes)
                        )
                        self._recent_cache_bytes += entry_bytes

            rows = unique_rows.index_select(0, inverse)
        else:
            rows = torch.from_numpy(self._mm[local.numpy()])      # uint8, (N, dim*b)

        # Large random working sets have no useful implicit mmap reuse and can
        # otherwise evict the pinned expert working set.  Explicit recent rows
        # remain valid after this advisory kernel-cache drop.
        if unique.numel() >= min_unique:
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:                                  # pragma: no cover
                pass
        if self._profile_ple and not is_decode_sized:
            logger.info(
                \"Qwen4 PLE profile: lookups=%d unique=%d path=%s \"
                \"recent_entries=%d recent_mib=%.1f cpu_gather_ms=%.1f\",
                local.numel(),
                unique.numel(),
                path,
                len(self._recent_row_cache),
                self._recent_cache_bytes / (1024 * 1024),
                (time.perf_counter() - gather_started) * 1000,
            )
"""

EDITS = [
    (BEFORE_IMPORT, AFTER_IMPORT),
    (BEFORE_INIT, AFTER_INIT),
    (BEFORE_GATHER, AFTER_GATHER),
]


def read() -> str:
    with open(QWEN4, encoding="utf-8") as source:
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
    print(f"  {status:<8} {os.path.relpath(QWEN4, SG)}: recent PLE row cache")


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
    with open(QWEN4, "w", encoding="utf-8") as output:
        output.write(text)
    print("  applied (opt-in bounded recent PLE row cache)")


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
    with open(QWEN4, "w", encoding="utf-8") as output:
        output.write(text)
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
