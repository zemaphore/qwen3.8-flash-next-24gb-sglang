#!/usr/bin/env python3
"""Tune the expert row-gather tile without loading the model.

The benchmark recreates the elastic S=224 address table with a mixture of GPU
and pinned-host rows and uses the exact production Triton kernel.  It reports
kernel time only; end-to-end acceptance still belongs to ``bench_agentic.py``.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import triton

from sglang.srt.layers.moe.expert_stream import _gather_rows_tab_kernel


ROW_BYTES = {
    "w13_qweight": 160 * 1280 * 4,
    "w2_qweight": 40 * 2560 * 4,
    "w13_scales": 20 * 1280 * 2,
    "w2_scales": 5 * 2560 * 2,
}


def make_address_table(experts: int, resident: int, row_bytes: int):
    gpu_rows = torch.empty((resident, row_bytes), dtype=torch.uint8, device="cuda")
    host_rows = torch.empty(
        (experts - resident, row_bytes), dtype=torch.uint8, pin_memory=True
    )
    gpu_rows.zero_()
    host_rows.zero_()
    addresses = torch.empty(experts, dtype=torch.int64)
    for expert in range(experts):
        if expert < resident:
            addresses[expert] = gpu_rows.data_ptr() + expert * row_bytes
        else:
            addresses[expert] = (
                host_rows.data_ptr() + (expert - resident) * row_bytes
            )
    return addresses.cuda(), gpu_rows, host_rows


def selected_ids(experts: int, resident: int, count: int, hot_fraction: float):
    hot = min(resident, int(round(count * hot_fraction)))
    cold = min(experts - resident, count - hot)
    hot = min(resident, count - cold)
    ids = torch.cat(
        [
            torch.arange(hot, dtype=torch.int64),
            torch.arange(resident, resident + cold, dtype=torch.int64),
        ]
    )
    if ids.numel() != count:
        raise ValueError(
            f"cannot select {count} unique experts with hot_fraction={hot_fraction}"
        )
    return ids.cuda()


def bench_one(
    table: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    row_bytes: int,
    block: int,
    repeats: int,
) -> tuple[float, float]:
    grid = (ids.numel(), triton.cdiv(row_bytes, block))
    _gather_rows_tab_kernel[grid](table, ids, output, row_bytes, BLOCK=block)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _gather_rows_tab_kernel[grid](table, ids, output, row_bytes, BLOCK=block)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values), statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--resident", type=int, default=224)
    parser.add_argument("--selected", type=int, nargs="+", default=[288, 352, 432])
    parser.add_argument("--hot-fraction", type=float, default=0.60)
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(
        f"device={torch.cuda.get_device_name()} E={args.experts} S={args.resident} "
        f"hot_fraction={args.hot_fraction:.2f} repeats={args.repeats}"
    )
    print(
        f"{'tensor':<14} {'rows':>5} {'block':>6} {'grid_y':>6} "
        f"{'median ms':>10} {'mean ms':>9} {'logical GB/s':>12}"
    )
    for name, row_bytes in ROW_BYTES.items():
        table, gpu_rows, host_rows = make_address_table(
            args.experts, args.resident, row_bytes
        )
        max_selected = max(args.selected)
        output = torch.empty(
            (max_selected, row_bytes), dtype=torch.uint8, device="cuda"
        )
        for count in args.selected:
            ids = selected_ids(
                args.experts,
                args.resident,
                count,
                args.hot_fraction,
            )
            view = output[:count]
            for block in args.blocks:
                median_ms, mean_ms = bench_one(
                    table, ids, view, row_bytes, block, args.repeats
                )
                gbps = count * row_bytes / (median_ms * 1e6)
                print(
                    f"{name:<14} {count:5d} {block:6d} "
                    f"{triton.cdiv(row_bytes, block):6d} {median_ms:10.3f} "
                    f"{mean_ms:9.3f} {gbps:12.2f}"
                )
        del output, table, gpu_rows, host_rows
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
