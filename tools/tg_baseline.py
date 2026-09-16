#!/usr/bin/env python3
"""TG0 single-stream token-generation timing harness.

Runs the predeclared TG0 matrix against an already-running SGLang server.  It
does not start or stop the server; use it once per boot and keep each boot's
output directory separate (``--boot-label``).

Policy (predeclared, not chosen from results):

* input classes prose/reasoning/code at exact 2,048 / 32,768 / 131,072 ids from
  ``tools/tg_prompts.py``;
* 512 generated tokens per request, greedy (temperature 0), EOS ignored;
* one explicitly excluded warmup then N measured requests per cell;
* cells are visited in a balanced order that rotates the class order per depth;
* no cache flush and no clock change between cells;
* every stream event's monotonic timestamp and cumulative token count is kept.

  python3 tools/tg_baseline.py --prompts-json ... --out-dir ... \
      --boot-label boot1 --matrix full --measured 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:30001/generate"

# Balanced boot-1 order: rotate class order at each depth, then interleave so a
# full class is not measured back to back.
FULL_ORDER = (
    ("prose", 2048),
    ("reasoning", 2048),
    ("code", 2048),
    ("code", 32768),
    ("prose", 32768),
    ("reasoning", 32768),
    ("reasoning", 131072),
    ("code", 131072),
    ("prose", 131072),
)
SHORT_ORDER = (
    ("prose", 2048),
    ("reasoning", 2048),
    ("code", 2048),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_gpu() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pstate,temperature.gpu,power.draw,memory.used,"
                "memory.free,utilization.gpu,clocks.sm,clocks.mem,clocks.gr",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout.strip()
        values = [item.strip() for item in out.split(",")]
        keys = (
            "pstate",
            "temperature_c",
            "power_w",
            "memory_used_mib",
            "memory_free_mib",
            "util_pct",
            "clock_sm_mhz",
            "clock_mem_mhz",
            "clock_gr_mhz",
        )
        record: dict = {}
        for key, value in zip(keys, values):
            if key == "pstate":
                record[key] = value
            else:
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = None
        return record
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


class GpuMonitor:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.samples.append({"t": time.monotonic(), **sample_gpu()})

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        def col(key: str) -> list[float]:
            return [
                float(s[key])
                for s in self.samples
                if isinstance(s.get(key), (int, float))
            ]

        def agg(key: str) -> dict | None:
            values = col(key)
            if not values:
                return None
            return {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }

        return {
            "n": len(self.samples),
            "temperature_c": agg("temperature_c"),
            "power_w": agg("power_w"),
            "util_pct": agg("util_pct"),
            "clock_sm_mhz": agg("clock_sm_mhz"),
            "clock_mem_mhz": agg("clock_mem_mhz"),
            "memory_free_mib": {"min": min(col("memory_free_mib")) if col("memory_free_mib") else None},
        }


def stream_once(url: str, ids: list[int], decode_tokens: int, timeout: int) -> dict:
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
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    started = time.monotonic()
    started_utc = utc_now()
    times: list[float] = []
    counts: list[int] = []
    text_parts: list[str] = []
    prompt_tokens = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "meta_info" not in event:
                raise RuntimeError(f"server error: {json.dumps(event)[:400]}")
            meta = event["meta_info"]
            times.append(time.monotonic())
            counts.append(int(meta["completion_tokens"]))
            prompt_tokens = int(meta["prompt_tokens"])
            piece = event.get("text")
            if piece:
                text_parts.append(piece)
    ended = time.monotonic()
    return {
        "started_utc": started_utc,
        "started_monotonic": started,
        "wall_seconds": ended - started,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": counts[-1] if counts else 0,
        "stream_events": len(times),
        "times_monotonic": times,
        "completion_counts": counts,
        "text": "".join(text_parts),
        "ended_utc": utc_now(),
    }


def summarize_sample(
    sample: dict, decode_tokens: int, gpu: dict, warmup: bool, index: int
) -> dict:
    times = sample.pop("times_monotonic")
    counts = sample.pop("completion_counts")
    text = sample.pop("text")
    started = sample.pop("started_monotonic")
    out: dict = {
        **sample,
        "warmup": warmup,
        "sample_index": index,
        "requested_decode_tokens": decode_tokens,
        "gpu": gpu,
        "output_text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
        "output_text": text,
    }
    if not times or len(times) < 2:
        out["status"] = "too_few_events"
        out["stream_events"] = len(times)
        return out
    out["ttft_seconds"] = times[0] - started
    out["status"] = "ok"
    deltas = [c2 - c1 for c1, c2 in zip(counts, counts[1:])]
    intervals = [
        (t2 - t1) / delta
        for t1, t2, delta in zip(times, times[1:], deltas)
        if delta > 0
    ]
    token_delta = counts[-1] - counts[0]
    elapsed = times[-1] - times[0]
    if token_delta <= 0 or elapsed <= 0:
        out["status"] = "no_token_progress"
        return out
    out["tg_tokens_per_s"] = token_delta / elapsed
    out["decode_seconds_per_token"] = elapsed / token_delta
    out["individual_token_timing"] = len(deltas) == token_delta and all(
        d == 1 for d in deltas
    )
    out["event_intervals_ms"] = {
        "n": len(intervals),
        "median": statistics.median(intervals) * 1000 if intervals else None,
        "p95": (
            sorted(intervals)[max(0, int(round(0.95 * len(intervals))) - 1)] * 1000
            if intervals
            else None
        ),
        "mean": (sum(intervals) / len(intervals) * 1000) if intervals else None,
    }
    return out


def run_sample(url: str, ids: list[int], decode_tokens: int, timeout: int) -> tuple[dict, dict]:
    monitor = GpuMonitor()
    with monitor:
        raw = stream_once(url, ids, decode_tokens, timeout)
    return raw, monitor.summary()


def freeze_state(label: str) -> dict:
    status = Path(os.environ.get("SGLANG_MOE_ELASTIC_CTL", "/root/quant/elastic.ctl") + ".status")
    server_log = Path(
        os.environ.get(
            "TG0_SERVER_LOG",
            "/root/qwen3.8-flash-next-24gb-sglang/docs/logs/raw/tg0_3090_2026-09-16/server.log",
        )
    )
    return {
        "label": label,
        "captured_utc": utc_now(),
        "gpu": sample_gpu(),
        "elastic_status": read_text(status),
        "server_log_tail": tail(server_log),
    }


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def tail(path: Path, n: int = 20) -> str | None:
    data = read_text(path)
    if data is None:
        return None
    return "\n".join(data.splitlines()[-n:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--boot-label", required=True)
    parser.add_argument("--matrix", choices=("full", "short"), default="full")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())
    order = FULL_ORDER if args.matrix == "full" else SHORT_ORDER
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": 1,
        "boot_label": args.boot_label,
        "started_utc": utc_now(),
        "matrix": args.matrix,
        "warmups": args.warmups,
        "measured": args.measured,
        "decode_tokens": args.decode_tokens,
        "url": args.url,
        "prompts_json": str(args.prompts_json),
        "prompts_json_sha256": sha256(args.prompts_json),
        "predeclared_order": order,
        "frozen_input_state": freeze_state("before"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summary: dict = {"boot_label": args.boot_label, "cells": {}, "capacity": None}

    for class_name, length in order:
        ids = prompts["classes"][class_name]["cells"][str(length)]["prompt_ids"]
        assert len(ids) == length, (class_name, length, len(ids))
        cell_dir = out / f"{class_name}-{length}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for number in range(args.warmups + args.measured):
            warmup = number < args.warmups
            raw, gpu = run_sample(args.url, ids, args.decode_tokens, args.timeout)
            record = summarize_sample(raw, args.decode_tokens, gpu, warmup, number)
            record["class"] = class_name
            record["requested_input_tokens"] = length
            (cell_dir / f"sample-{number:02d}.json").write_text(
                json.dumps(record, indent=1) + "\n"
            )
            samples.append(record)
            status = record.get("status")
            tg = record.get("tg_tokens_per_s")
            print(
                f"  {args.boot_label} {class_name:9s} {length:7d} "
                f"#{number} {'warmup' if warmup else 'measure':7s} "
                f"{status:16s} tg={tg if tg is None else round(tg, 2)} "
                f"events={record.get('stream_events')} "
                f"ptok={record.get('prompt_tokens')} "
                f"free_min={record.get('gpu', {}).get('memory_free_mib', {}).get('min')}",
                flush=True,
            )
        measured = [s for s in samples if not s.get("warmup") and s.get("status") == "ok"]
        tgs = [s["tg_tokens_per_s"] for s in measured]
        med_lat = [
            s["event_intervals_ms"]["median"]
            for s in measured
            if s.get("event_intervals_ms", {}).get("median") is not None
        ]
        p95_lat = [
            s["event_intervals_ms"]["p95"]
            for s in measured
            if s.get("event_intervals_ms", {}).get("p95") is not None
        ]
        summary["cells"][f"{class_name}-{length}"] = {
            "class": class_name,
            "requested_input_tokens": length,
            "actual_prompt_tokens": measured[0]["prompt_tokens"] if measured else None,
            "n_measured_ok": len(tgs),
            "n_samples": len(samples),
            "tg_tokens_per_s": {
                "values": tgs,
                "mean": (sum(tgs) / len(tgs)) if tgs else None,
                "sd": (statistics.pstdev(tgs) if len(tgs) > 1 else 0.0) if tgs else None,
            },
            "per_request_throughput_tokens_per_s": tgs,
            "median_token_latency_ms": {
                "values": med_lat,
                "median_of_medians": statistics.median(med_lat) if med_lat else None,
            },
            "p95_token_latency_ms": {
                "values": p95_lat,
                "median_of_p95": statistics.median(p95_lat) if p95_lat else None,
            },
            "individual_token_timing": [s.get("individual_token_timing") for s in measured],
            "ttft_seconds": [s.get("ttft_seconds") for s in measured],
        }

    if args.capacity:
        ids = prompts["classes"]["code"]["capacity"]["prompt_ids"]
        assert len(ids) == prompts["capacity_input"]
        cell_dir = out / "capacity-smoke"
        cell_dir.mkdir(parents=True, exist_ok=True)
        raw, gpu = run_sample(args.url, ids, args.decode_tokens, args.timeout)
        record = summarize_sample(raw, args.decode_tokens, gpu, False, 0)
        record["class"] = "code"
        record["requested_input_tokens"] = len(ids)
        (cell_dir / "sample-00.json").write_text(json.dumps(record, indent=1) + "\n")
        summary["capacity"] = record
        print(
            f"  {args.boot_label} capacity {len(ids)} status={record.get('status')} "
            f"tg={record.get('tg_tokens_per_s')} prompt_tokens={record.get('prompt_tokens')} "
            f"free_min={gpu.get('memory_free_mib', {}).get('min')}",
            flush=True,
        )

    summary["ended_utc"] = utc_now()
    summary["frozen_input_state_after"] = freeze_state("after")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
