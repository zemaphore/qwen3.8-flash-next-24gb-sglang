#!/usr/bin/env python3
"""Opt-in parallel PLE reads for high-entropy prefill chunks.

The existing mmap path faults random 4 KiB pages synchronously.  On the 3090
host, a 1024-token/16-head replay measured 3.1-3.8 s cold through mmap versus
0.42-0.48 s with bounded parallel pread.  Warm mmap remained much faster, so
this patch keeps mmap for small repeated working sets and uses pread only when
the number of unique rows crosses a configurable threshold.

Environment:
  SGLANG_QWEN4_PLE_BULK_PREAD=1
  SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE=2048

Usage:
  SGLANG=/root/sglang python3 ple_bulk_pread.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")),
    "python/sglang",
)
QWEN4 = f"{SG}/srt/models/qwen4_exp.py"

BEFORE = """        if local.numel() <= 64:
            rb = self._row_bytes
            def _rd(r):
                return os.pread(self._fd, rb, int(r) * rb)
            buf = b"".join(self._pool.map(_rd, local.tolist()))
            rows = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(-1, rb)
        else:
            rows = torch.from_numpy(self._mm[local.numpy()])      # uint8, (N, dim*b)
            if local.numel() >= 512:
                try:
                    os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:                              # pragma: no cover
                    pass
"""

AFTER = """        # Decode always uses parallel pread.  For prefill, retain mmap when
        # the row working set is small (the repeated-prompt case is then served
        # almost entirely from the page cache) and expose NVMe queue depth for
        # high-entropy chunks.  Deduplication is on CPU, after the unavoidable
        # ids D2H, so querying the unique count does not synchronize the GPU.
        unique, inverse = torch.unique(local, return_inverse=True)
        bulk_pread = os.environ.get("SGLANG_QWEN4_PLE_BULK_PREAD") == "1"
        min_unique = int(os.environ.get(
            "SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE", "2048"
        ))
        use_pread = local.numel() <= 64 or (
            bulk_pread and unique.numel() >= min_unique
        )
        if use_pread:
            rb = self._row_bytes

            def _rd(r):
                data = os.pread(self._fd, rb, int(r) * rb)
                if len(data) != rb:
                    raise RuntimeError(f"short PLE read: {len(data)} != {rb}")
                return data

            buf = b"".join(self._pool.map(_rd, unique.tolist()))
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


def state() -> tuple[bool, bool]:
    with open(QWEN4, encoding="utf-8") as source:
        text = source.read()
    return BEFORE in text, AFTER in text


def check() -> None:
    clean, applied = state()
    # PP1b layers a bounded recent-row cache on this exact bulk-pread path, so
    # the literal AFTER anchor is intentionally no longer present in that state.
    layered = "SGLANG_QWEN4_PLE_RECENT_CACHE_MB" in open(
        QWEN4, encoding="utf-8"
    ).read()
    status = "APPLIED" if applied or layered else ("clean" if clean else "MISMATCH")
    print(f"  {status:<8} {os.path.relpath(QWEN4, SG)}: bulk PLE pread")


def replace(old: str, new: str) -> None:
    with open(QWEN4, encoding="utf-8") as source:
        text = source.read()
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch anchor, found {text.count(old)}")
    with open(QWEN4, "w", encoding="utf-8") as output:
        output.write(text.replace(old, new, 1))


def apply() -> None:
    clean, applied = state()
    if applied:
        print("  already applied")
        return
    if not clean:
        raise RuntimeError("patch anchor mismatch")
    replace(BEFORE, AFTER)
    print("  applied (opt-in bulk PLE pread)")


def revert() -> None:
    clean, applied = state()
    if clean:
        print("  already clean")
        return
    if not applied:
        raise RuntimeError("patch anchor mismatch")
    replace(AFTER, BEFORE)
    print("  reverted")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": apply, "revert": revert}[command]()
