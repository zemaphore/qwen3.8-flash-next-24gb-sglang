#!/usr/bin/env python3
"""PP11 offline benchmark for scattered pinned-row HtoD submission paths.

Compares the production-style Python ``Tensor.copy_`` loop with precomputed
source views and CUDA 12.8+'s ``cudaMemcpyBatchAsync``.  No model is loaded.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import torch


class CudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", CudaMemLocation),
        ("dstLocHint", CudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


def batch_function(dst_ptrs, src_ptrs, sizes, stream_handle, prefer_overlap=False):
    fn = ctypes.CDLL(None).cudaMemcpyBatchAsync
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(CudaMemcpyAttributes),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    attrs = CudaMemcpyAttributes(
        3,                       # cudaMemcpySrcAccessOrderAny (stable pinned rows)
        CudaMemLocation(2, 0),  # cudaMemLocationTypeHost
        CudaMemLocation(1, torch.cuda.current_device()),
        int(prefer_overlap),
    )
    attrs_idx = (ctypes.c_size_t * 1)(0)
    stream = ctypes.c_void_p(stream_handle)

    def submit():
        err = fn(
            dst_ptrs,
            src_ptrs,
            sizes,
            len(sizes),
            ctypes.byref(attrs),
            attrs_idx,
            1,
            stream,
        )
        if err:
            raise RuntimeError(f"cudaMemcpyBatchAsync failed with cudaError={err}")

    return submit


def measure(fn, reps):
    fn()
    torch.cuda.synchronize()
    gpu_ms = []
    submit_ms = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        fn()
        submit_ms.append((time.perf_counter() - t0) * 1e3)
        end.record()
        end.synchronize()
        gpu_ms.append(start.elapsed_time(end))
    return statistics.median(gpu_ms), statistics.median(submit_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--rows", type=int, default=258)
    ap.add_argument("--row-bytes", type=int, default=819200)
    ap.add_argument("--reps", type=int, default=9)
    args = ap.parse_args()

    if args.row_bytes % 4:
        raise SystemExit("--row-bytes must be divisible by four")
    words = args.row_bytes // 4
    host = torch.empty((args.experts, words), dtype=torch.int32, pin_memory=True)
    host.zero_()
    host[:, 0] = torch.arange(args.experts, dtype=torch.int32)
    ids = torch.randperm(args.experts)[: args.rows].tolist()
    current_dst = torch.empty((args.rows, words), dtype=torch.int32, device="cuda")
    views_dst = torch.empty_like(current_dst)
    batch_dst = torch.empty_like(current_dst)
    overlap_dst = torch.empty_like(current_dst)
    views = [host[e] for e in ids]

    def current_loop():
        for j, expert in enumerate(ids):
            current_dst[j].copy_(host[expert], non_blocking=True)

    def views_loop():
        for j, source in enumerate(views):
            views_dst[j].copy_(source, non_blocking=True)

    count = args.rows
    src_ptrs = (ctypes.c_void_p * count)(
        *(host.data_ptr() + expert * args.row_bytes for expert in ids)
    )
    sizes = (ctypes.c_size_t * count)(*[args.row_bytes] * count)

    def ptrs(dst):
        return (ctypes.c_void_p * count)(
            *(dst.data_ptr() + j * args.row_bytes for j in range(count))
        )

    copy_stream = torch.cuda.Stream()
    batch = batch_function(
        ptrs(batch_dst), src_ptrs, sizes, copy_stream.cuda_stream
    )
    overlap = batch_function(
        ptrs(overlap_dst), src_ptrs, sizes, copy_stream.cuda_stream,
        prefer_overlap=True,
    )

    def bridged(submit):
        current = torch.cuda.current_stream()
        copy_stream.wait_stream(current)
        with torch.cuda.stream(copy_stream):
            submit()
        current.wait_stream(copy_stream)

    def batch_submit():
        bridged(batch)

    def overlap_submit():
        bridged(overlap)

    variants = [
        ("tensor-select loop", current_loop),
        ("precomputed-view loop", views_loop),
        ("cudaMemcpyBatchAsync", batch_submit),
        ("batch prefer-overlap", overlap_submit),
    ]
    print(
        f"rows={args.rows} row_bytes={args.row_bytes} "
        f"total={args.rows * args.row_bytes / 1e6:.1f} MB"
    )
    reference = None
    for name, fn in variants:
        gpu_ms, cpu_ms = measure(fn, args.reps)
        torch.cuda.synchronize()
        out = {
            "tensor-select loop": current_dst,
            "precomputed-view loop": views_dst,
            "cudaMemcpyBatchAsync": batch_dst,
            "batch prefer-overlap": overlap_dst,
        }[name]
        if reference is None:
            reference = out.clone()
        if not torch.equal(out, reference):
            raise RuntimeError(f"copy mismatch: {name}")
        gbps = args.rows * args.row_bytes / gpu_ms / 1e6
        print(f"{name:<24} GPU {gpu_ms:7.2f} ms {gbps:5.1f} GB/s  CPU {cpu_ms:7.3f} ms")


if __name__ == "__main__":
    main()
