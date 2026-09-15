#!/usr/bin/env python3
"""Benchmark the Qwen3.8 PLE storage path without loading the model.

The production prefill path performs NumPy advanced indexing into an mmap.
Random page faults from that operation may expose much less NVMe queue depth
than the O_DIRECT probe in ``nvme_probe.py``.  This tool compares that exact
CPU gather with parallel 160-byte ``pread`` calls over the same row IDs.

The first pass is made cold with POSIX_FADV_DONTNEED.  The immediately repeated
mmap pass reports the best-case page-cache behavior.  No GPU is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def drop_file_cache(fd: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def digest_rows(rows: np.ndarray) -> bytes:
    return hashlib.blake2b(rows, digest_size=16).digest()


def mmap_gather(path: Path, ids: np.ndarray, rows: int, row_bytes: int):
    fd = os.open(path, os.O_RDONLY)
    try:
        drop_file_cache(fd)
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, row_bytes))
        try:
            if hasattr(mm._mmap, "madvise"):
                import mmap

                mm._mmap.madvise(mmap.MADV_RANDOM)
            t0 = time.perf_counter()
            cold = mm[ids]
            cold_s = time.perf_counter() - t0

            # Force a consumer-visible reduction so an accidental lazy view
            # cannot make either pass appear free.
            cold_digest = digest_rows(cold)
            t0 = time.perf_counter()
            warm = mm[ids]
            warm_s = time.perf_counter() - t0
            warm_digest = digest_rows(warm)
        finally:
            del mm
    finally:
        os.close(fd)
    if cold_digest != warm_digest:
        raise RuntimeError("mmap cold/warm gathers returned different data")
    return cold_s, warm_s, cold_digest


def parallel_pread(
    path: Path, ids: np.ndarray, row_bytes: int, workers: int
) -> tuple[float, int]:
    fd = os.open(path, os.O_RDONLY)
    try:
        drop_file_cache(fd)

        def read_row(row: int) -> bytes:
            data = os.pread(fd, row_bytes, int(row) * row_bytes)
            if len(data) != row_bytes:
                raise RuntimeError(f"short PLE read: {len(data)} != {row_bytes}")
            return data

        t0 = time.perf_counter()
        unique, inverse = np.unique(ids, return_inverse=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(read_row, unique, chunksize=32))
        unique_rows = np.frombuffer(b"".join(values), dtype=np.uint8).reshape(
            -1, row_bytes
        )
        gathered = unique_rows[inverse]
        elapsed = time.perf_counter() - t0
    finally:
        os.close(fd)
    return elapsed, digest_rows(gathered), unique.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ple-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--working-set",
        type=int,
        default=0,
        help="sample from this many random rows; 0 makes every lookup independent",
    )
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    with (args.ple_dir / "ple.json").open() as src:
        meta = json.load(src)
    path = args.ple_dir / meta["file"]
    rows = int(meta["rows"])
    itemsize = 1 if meta["dtype"] == "F8_E4M3" else 2
    row_bytes = int(meta["dim"]) * itemsize
    lookups = args.tokens * args.heads
    rng = np.random.default_rng(args.seed)
    if args.working_set:
        population = rng.integers(
            0, rows, size=args.working_set, dtype=np.int64
        )
        ids = rng.choice(population, size=lookups, replace=True)
    else:
        ids = rng.integers(0, rows, size=lookups, dtype=np.int64)

    cold_s, warm_s, mmap_digest = mmap_gather(path, ids, rows, row_bytes)
    pread_s, pread_digest, unique_rows = parallel_pread(
        path, ids, row_bytes, args.workers
    )
    if mmap_digest != pread_digest:
        raise RuntimeError("mmap and deduplicated pread data differ")

    touched_pages = np.unique((ids * row_bytes) // 4096).size
    payload_mb = lookups * row_bytes / 1e6
    print(f"file: {path} ({path.stat().st_size / 1e9:.2f} GB)")
    print(
        f"shape: {args.tokens} tokens x {args.heads} heads = {lookups:,} rows, "
        f"{unique_rows:,} unique, {touched_pages:,} start pages, "
        f"{payload_mb:.2f} MB payload"
    )
    print(f"mmap cold:       {cold_s * 1e3:9.1f} ms  ({lookups / cold_s:,.0f} rows/s)")
    print(f"mmap warm:       {warm_s * 1e3:9.1f} ms  ({lookups / warm_s:,.0f} rows/s)")
    print(
        f"pread qd={args.workers:<2}:    {pread_s * 1e3:9.1f} ms  "
        f"({lookups / pread_s:,.0f} rows/s)"
    )


if __name__ == "__main__":
    main()
