#!/usr/bin/env python3
"""TG0 decode-window profiler: capture and attribute one warmed decode window.

The accepted server replays a *breakable* decode CUDA graph, so a single decode
token appears in the Chrome trace as several ``step[DECODE bs=1]`` graph
segments, not one span per token.  This tool therefore:

* warms the target prompt, then asks the live server for a stage-filtered
  ``decode`` capture (``/start_profile``, ``profile_by_stage``, no stack and no
  recorded shapes) while a normal streamed request runs;
* measures the same request unprofiled to quantify profiler overhead;
* attributes kernel duration to families (expert GEMV / host reads, dense and
  Marlin, attention/KV, GDN, PLE/graph, router+sampling, other) and reports the
  summed duration, the union, and the idle gap separately.

Nothing in the serving path is instrumented; the endpoint is the stock one.

  python3 tools/tg_profile.py --prompts-json ... --out-dir ... \\
      --context 2048 --num-steps 20 --label short
  python3 tools/tg_profile.py analyze --trace TRACE.json.gz --tokens 20 --out out.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:30001"

FAMILY_RULES = (
    ("expert_gemv", ("_moe_gemv_int2_tab", "moe_gemv_int2")),
    ("expert_reads_dma", ("memcpy32", "htod", "dtoh")),
    ("dense_marlin", ("marlin", "gemm", "gemvx", "cutlass", "ampere", "wmma")),
    ("attention_kv", (
        "flash", "fa2", "attention", "qsa", "_kv", "kv_tiered", "ring_owner",
        "block_indices", "sparse",
    )),
    ("gdn", (
        "gated_delta", "conv1d", "qkvzba", "recurrent", "gdn", "_fused_qk_rmsnorm",
    )),
    ("ple_graph", ("_hc_mix", "hc_combine", "gemma_rmsnorm", "ple")),
    ("router_sampling", (
        "router", "fast_topk", "sigmoid", "_fa2_valid", "argmax", "sampling",
        "fillfunctor",
    )),
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def post_json(url: str, path: str, payload: dict | None = None, timeout: int = 60) -> str:
    """POST and return the raw response body.

    SGLang's ``/start_profile`` and ``/stop_profile`` reply with a short plain
    text confirmation (``Start profiling.``), not JSON, so the body is returned
    verbatim rather than parsed.
    """
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    request = urllib.request.Request(
        url.rstrip("/") + path, data, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode(errors="replace")


def stream_request(url: str, ids: list[int], decode_tokens: int, timeout: int) -> dict:
    body = json.dumps(
        {
            "input_ids": ids,
            "stream": True,
            "sampling_params": {
                "max_new_tokens": decode_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/generate", body, {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    times: list[float] = []
    counts: list[int] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            events_meta = event["meta_info"]
            times.append(time.monotonic())
            counts.append(int(events_meta["completion_tokens"]))
    ended = time.monotonic()
    return {
        "wall_seconds": ended - started,
        "completion_tokens": counts[-1] if counts else 0,
        "stream_events": len(times),
        "times": times,
        "counts": counts,
    }


def classify(name: str) -> str:
    lowered = name.lower()
    for family, needles in FAMILY_RULES:
        if any(needle in lowered for needle in needles):
            return family
    return "other"


def iter_events(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        data = json.load(source)
    for event in data.get("traceEvents", []):
        yield event


def union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for a, b in intervals[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    total += end - start
    return total


def analyze_trace(trace: Path, tokens: int) -> dict:
    families: dict[str, dict] = {}
    kernel_intervals: list[tuple[float, float]] = []
    dma_intervals: list[tuple[float, float]] = []
    global_min = None
    global_max = None
    decode_spans = []
    for event in iter_events(trace):
        name = event.get("name")
        ts = event.get("ts")
        dur = event.get("dur")
        if isinstance(ts, (int, float)) and isinstance(dur, (int, float)):
            global_min = ts if global_min is None else min(global_min, ts)
            global_max = ts + dur if global_max is None else max(global_max, ts + dur)
        if event.get("cat") == "gpu_user_annotation" and name == "step[DECODE bs=1]":
            decode_spans.append((ts, ts + dur))
        if event.get("cat") != "kernel" or dur is None or not isinstance(name, str):
            continue
        family = classify(name)
        row = families.setdefault(family, {"sum_us": 0.0, "count": 0, "names": {}})
        row["sum_us"] += float(dur)
        row["count"] += 1
        row["names"][name] = row["names"].get(name, 0.0) + float(dur)
        kernel_intervals.append((float(ts), float(ts) + float(dur)))
    # DMA events live under the gpu_memcpy category.
    for event in iter_events(trace):
        if event.get("cat") == "gpu_memcpy" and isinstance(event.get("ts"), (int, float)):
            dma_intervals.append((event["ts"], event["ts"] + event["dur"]))
    trace_wall_us = (global_max - global_min) if global_min is not None else 0.0
    kernel_union_us = union(list(kernel_intervals))
    report = {
        "trace": str(trace),
        "tokens": tokens,
        "trace_wall_ms": trace_wall_us / 1000.0,
        "trace_wall_ms_per_token": (trace_wall_us / 1000.0 / tokens) if tokens else None,
        "decode_graph_segments": len(decode_spans),
        "kernel_sum_ms": sum(v["sum_us"] for v in families.values()) / 1000.0,
        "kernel_sum_ms_per_token": (
            sum(v["sum_us"] for v in families.values()) / 1000.0 / tokens
        )
        if tokens
        else None,
        "kernel_union_ms": kernel_union_us / 1000.0,
        "kernel_union_ms_per_token": (kernel_union_us / 1000.0 / tokens) if tokens else None,
        "dma_union_ms": union(list(dma_intervals)) / 1000.0,
        "compute_dma_overlap_ms": (
            union(list(kernel_intervals)) + union(list(dma_intervals)) - union(list(kernel_intervals) + list(dma_intervals))
        )
        / 1000.0,
        "families": {
            family: {
                "count": row["count"],
                "sum_ms": row["sum_us"] / 1000.0,
                "sum_ms_per_token": (row["sum_us"] / 1000.0 / tokens) if tokens else None,
                "top": sorted(
                    ((n, d / 1000.0) for n, d in row["names"].items()),
                    key=lambda kv: -kv[1],
                )[:5],
            }
            for family, row in sorted(
                families.items(), key=lambda kv: -kv[1]["sum_us"]
            )
        },
    }
    if tokens:
        report["idle_ms_per_token"] = max(
            0.0, report["trace_wall_ms_per_token"] - report["kernel_union_ms_per_token"]
        )
    return report


def print_report(report: dict) -> None:
    tokens = report["tokens"]
    print(f"  trace {report['trace']}")
    print(
        f"  tokens={tokens}  graph_segments={report['decode_graph_segments']}  "
        f"wall={report['trace_wall_ms']:.2f} ms ({report['trace_wall_ms_per_token']:.2f} ms/tok)  "
        f"kernel_union={report['kernel_union_ms_per_token']:.2f} ms/tok  "
        f"idle={report.get('idle_ms_per_token', 0):.2f} ms/tok"
    )
    print(f"  {'family':18s} {'count':>6s} {'sum ms':>9s} {'ms/tok':>9s}")
    for family, row in report["families"].items():
        print(
            f"  {family:18s} {row['count']:6d} {row['sum_ms']:9.3f} "
            f"{row['sum_ms_per_token']:9.3f}"
        )


def steady_ms_per_token(record: dict, window: int | None = None) -> float:
    """Steady-state ms/token from the streamed window, excluding TTFT/prefill.

    ``window`` limits the measurement to the first ``window`` token increments,
    matching the profiled span when profiling auto-stops before the request ends.
    """
    times = record.get("times") or []
    counts = record.get("counts") or []
    if len(times) >= 2 and counts and counts[-1] > counts[0]:
        last = len(times) - 1 if window is None else min(window, len(times) - 1)
        delta = counts[last] - counts[0]
        if delta > 0:
            return (times[last] - times[0]) * 1000.0 / delta
    return record["wall_seconds"] * 1000.0 / max(1, record["completion_tokens"])


def capture(args: argparse.Namespace) -> None:
    prompts = json.loads(args.prompts_json.read_text())
    ids = prompts["classes"][args.prompts_class]["cells"][str(args.context)]["prompt_ids"]
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile_dir = out / f"profile-{args.label}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Unprofiled matched request first: its steady-state ms/token is the
    # overhead baseline for the profiled run.
    stream_request(args.url, ids, args.margin + args.num_steps, args.timeout)
    unprofiled = stream_request(args.url, ids, args.margin + args.num_steps, args.timeout)
    unprofiled_ms = steady_ms_per_token(unprofiled, args.num_steps)

    post_json(
        args.url,
        "/start_profile",
        {
            "output_dir": str(profile_dir),
            "profile_by_stage": True,
            "profile_stages": ["decode"],
            "num_steps": args.num_steps,
            "activities": ["CPU", "GPU"],
            "with_stack": False,
            "record_shapes": False,
        },
    )
    profiled = stream_request(args.url, ids, args.num_steps + args.margin, args.timeout)
    profiled_ms = steady_ms_per_token(profiled, args.num_steps)

    deadline = time.time() + 120
    trace = None
    while time.time() < deadline:
        traces = sorted(profile_dir.glob("*-DECODE.trace.json.gz")) or sorted(
            profile_dir.glob("*.trace.json.gz")
        )
        if traces:
            trace = traces[0]
            break
        time.sleep(2)
    if trace is None:
        raise SystemExit(f"no trace appeared in {profile_dir}")

    report = analyze_trace(trace, args.num_steps)
    report.update(
        {
            "label": args.label,
            "context_requested": args.context,
            "prompts_class": args.prompts_class,
            "prompt_tokens": len(ids),
            "unprofiled_ms_per_token": unprofiled_ms,
            "client_profiled_ms_per_token": profiled_ms,
            "captured_utc": utc_now(),
        }
    )
    # The client-observed stream is distorted while the profiler is active (the
    # detokenizer starves and flushes late), so overhead is measured from the
    # trace-internal per-token wall against the matched unprofiled request.
    report["profiled_ms_per_token"] = report["trace_wall_ms_per_token"]
    report["overhead_pct"] = (
        (report["trace_wall_ms_per_token"] - unprofiled_ms) / unprofiled_ms * 100
    )
    (out / f"profile-{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print_report(report)
    print(
        f"  overhead (trace wall vs unprofiled): {unprofiled_ms:.2f} -> "
        f"{report['trace_wall_ms_per_token']:.2f} ms/tok ({report['overhead_pct']:+.1f}%); "
        f"client-observed during profile {profiled_ms:.2f} ms/tok"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    cap = sub.add_parser("capture")
    cap.add_argument("--prompts-json", type=Path, required=True)
    cap.add_argument("--prompts-class", default="reasoning")
    cap.add_argument("--context", type=int, default=2048)
    cap.add_argument("--out-dir", type=Path, required=True)
    cap.add_argument("--label", required=True)
    cap.add_argument("--num-steps", type=int, default=20)
    cap.add_argument("--margin", type=int, default=8)
    cap.add_argument("--url", default=os.environ.get("SGLANG_BASE_URL", DEFAULT_URL))
    cap.add_argument("--timeout", type=int, default=1800)
    an = sub.add_parser("analyze")
    an.add_argument("--trace", type=Path, required=True)
    an.add_argument("--tokens", type=int, required=True)
    an.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args)
    elif args.cmd == "analyze":
        report = analyze_trace(args.trace, args.tokens)
        print_report(report)
        if args.out:
            args.out.write_text(json.dumps(report, indent=2) + "\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
