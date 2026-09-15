#!/usr/bin/env python3
"""Stream a Chrome trace and aggregate duration without loading it into RAM."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path


def trace_events(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8") as source:
        buf = ""
        pos = 0
        in_events = False
        eof = False
        while True:
            if pos > 1 << 20:
                buf = buf[pos:]
                pos = 0
            if not eof and len(buf) - pos < 1 << 20:
                chunk = source.read(1 << 20)
                if chunk:
                    buf += chunk
                else:
                    eof = True

            if not in_events:
                marker = buf.find('"traceEvents"', pos)
                if marker < 0:
                    if eof:
                        raise ValueError("traceEvents array not found")
                    pos = max(0, len(buf) - 32)
                    continue
                start = buf.find("[", marker)
                if start < 0:
                    if eof:
                        raise ValueError("traceEvents array is incomplete")
                    pos = marker
                    continue
                pos = start + 1
                in_events = True

            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos >= len(buf):
                    break
                try:
                    event, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                pos = end
                yield event

            if eof:
                raise ValueError("unterminated traceEvents array")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="case-insensitive name substring; may be repeated",
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--sample-args",
        help="print arguments for the first matching duration event",
    )
    parser.add_argument(
        "--group-input-dims",
        action="store_true",
        help="split operator names by their recorded input dimensions",
    )
    parser.add_argument(
        "--group-grid",
        action="store_true",
        help="split CUDA kernel names by their recorded launch grid",
    )
    parser.add_argument(
        "--group-grid-y",
        action="store_true",
        help="split CUDA kernel names by only the second grid dimension",
    )
    parser.add_argument(
        "--within",
        help=(
            "only include events whose start timestamp is inside the union of "
            "duration spans with this case-insensitive name substring"
        ),
    )
    args = parser.parse_args()

    names = defaultdict(lambda: [0, 0.0, set()])
    process_names = {}
    intervals = []
    if args.within:
        needle = args.within.lower()
        for event in trace_events(args.trace):
            if event.get("ph") == "M" and event.get("name") == "process_name":
                process_names[event.get("pid")] = event.get("args", {}).get(
                    "name", ""
                )
                continue
            name = event.get("name")
            timestamp = event.get("ts")
            duration = event.get("dur")
            if (
                isinstance(name, str)
                and needle in name.lower()
                and timestamp is not None
                and duration is not None
            ):
                intervals.append((float(timestamp), float(timestamp) + float(duration)))
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        intervals = merged
        if not intervals:
            raise ValueError(f"no duration spans matched --within={args.within!r}")
    interval_starts = [start for start, _ in intervals]

    event_count = 0
    sampled = None
    for event in trace_events(args.trace):
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_names[event.get("pid")] = event.get("args", {}).get("name", "")
            continue
        duration = event.get("dur")
        name = event.get("name")
        if duration is None or not isinstance(name, str):
            continue
        if intervals:
            timestamp = event.get("ts")
            if timestamp is None:
                continue
            index = bisect_right(interval_starts, float(timestamp)) - 1
            if index < 0 or float(timestamp) >= intervals[index][1]:
                continue
        event_count += 1
        if (
            sampled is None
            and args.sample_args
            and args.sample_args.lower() in name.lower()
        ):
            sampled = event
        if args.group_input_dims:
            dims = event.get("args", {}).get("Input Dims")
            if dims:
                name = f"{name} input_dims={dims}"
        if args.group_grid:
            grid = event.get("args", {}).get("grid")
            if grid:
                name = f"{name} grid={grid}"
        if args.group_grid_y:
            grid = event.get("args", {}).get("grid")
            if grid and len(grid) > 1:
                name = f"{name} grid_y={grid[1]}"
        row = names[name]
        row[0] += 1
        row[1] += float(duration)
        row[2].add(event.get("pid"))

    def domain(pids: set) -> str:
        labels = {process_names.get(pid, str(pid)) for pid in pids}
        return ",".join(sorted(labels))

    lowered = [item.lower() for item in args.match]
    rows = []
    for name, (count, duration_us, pids) in names.items():
        if lowered and not any(item in name.lower() for item in lowered):
            continue
        rows.append((duration_us, count, domain(pids), name))
    rows.sort(reverse=True)

    interval_note = f"; within spans: {len(intervals):,}" if intervals else ""
    print(
        f"duration events: {event_count:,}; distinct names: {len(names):,}"
        f"{interval_note}"
    )
    if sampled is not None:
        print("sample event:")
        print(json.dumps(sampled, indent=2)[:12000])
    print(f"{'total ms':>12} {'count':>8} {'mean us':>10}  {'domain':<20} name")
    for duration_us, count, where, name in rows[: args.top]:
        print(
            f"{duration_us / 1000:12.3f} {count:8d} "
            f"{duration_us / count:10.2f}  {where[:20]:<20} {name}"
        )


if __name__ == "__main__":
    main()
