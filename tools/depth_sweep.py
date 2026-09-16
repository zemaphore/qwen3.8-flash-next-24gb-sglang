#!/usr/bin/env python3
"""Reproducible prefill depth sweep against a running SGLang server.

Streams one deterministic technical-document prompt per requested depth, records
the *actual* server token counts (not the requested labels), TTFT, prefill and
decode throughput, minimum free VRAM, and any failure/retraction.  The prompt is
built from a fixed corpus and a pre-declared chars-per-token estimate; every row
reports the real ``prompt_tokens`` the server tokenized.

  SGLANG_URL=http://127.0.0.1:30001/generate \
    python3 tools/depth_sweep.py --depths 8192 32768 65536 131072 262144 \
      --decode-tokens 16 --out docs/logs/raw/.../depth-sweep.json

Warmup/exclusion policy (decided before measurement): the launcher's own warmup
is not part of this sweep; each depth is issued exactly once, in increasing order.
No sample is dropped silently.  A request that fails is recorded with its error.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO / "docs" / "CAMPAIGN.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip().splitlines()[0]
    return int(out)


def build_prompt(corpus: str, target_tokens: int, chars_per_token: float) -> str:
    chars = max(1, int(target_tokens * chars_per_token))
    if len(corpus) < chars:
        corpus = (corpus + "\n") * (chars // max(1, len(corpus)) + 1)
    return corpus[:chars].rstrip() + "\nSummarize the memory tiers in one sentence:"


def stream(url: str, text: str, decode_tokens: int, timeout: int) -> dict:
    body = json.dumps(
        {
            "text": text,
            "stream": True,
            "sampling_params": {
                "max_new_tokens": decode_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        }
    ).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    started = time.time()
    times: list[float] = []
    completion: list[int] = []
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "meta_info" not in event:
                raise RuntimeError(f"server error: {json.dumps(event)[:300]}")
            meta = event["meta_info"]
            times.append(time.time())
            completion.append(meta["completion_tokens"])
            prompt_tokens = meta["prompt_tokens"]
    return {
        "started_utc": utc_now(),
        "wall_seconds": time.time() - started,
        "ttft_seconds": (times[0] - started) if times else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion[-1] if completion else 0,
        "stream_events": len(times),
        "times": times,
        "completion": completion,
    }


def summarize(row: dict) -> dict:
    times = row.pop("times")
    completion = row.pop("completion")
    if row.get("ttft_seconds") is None or len(times) < 2:
        row["prefill_seconds"] = None
        row["prefill_tokens_per_s"] = None
        row["decode_tokens_per_s"] = None
        return row
    decode_seconds = (times[-1] - times[0]) / max(1, completion[-1] - completion[0])
    prefill_seconds = max(row["ttft_seconds"] - decode_seconds, 1e-3)
    row["decode_seconds_per_token"] = decode_seconds
    row["prefill_seconds"] = prefill_seconds
    row["prefill_tokens_per_s"] = row["prompt_tokens"] / prefill_seconds
    row["decode_tokens_per_s"] = 1 / decode_seconds
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30001/generate"))
    parser.add_argument("--depths", type=int, nargs="+", required=True)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--chars-per-token", type=float, default=3.5)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    corpus = args.corpus.read_text(encoding="utf-8", errors="replace")
    sweep_started = utc_now()
    print(f"  {'requested':>9s} {'actual':>8s} {'prefill s':>9s} {'prefill t/s':>11s} "
          f"{'decode t/s':>10s} {'min free MiB':>12s} {'status':>10s}")
    print("  " + "-" * 78)

    rows = []
    for depth in args.depths:
        before_free = free_mib()
        min_free = [before_free]
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(0.25):
                try:
                    min_free[0] = min(min_free[0], free_mib())
                except Exception:
                    pass

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        record = {"requested_tokens": depth, "free_mib_before": before_free}
        try:
            raw = stream(args.url, build_prompt(corpus, depth, args.chars_per_token),
                         args.decode_tokens, args.timeout)
            record.update(summarize(raw))
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - record any retraction/failure
            body = ""
            if hasattr(exc, "read"):
                body = exc.read()[:400].decode(errors="replace")
            record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_body": body,
                }
            )
        stop.set()
        watcher.join(timeout=5)
        record["free_mib_min"] = min_free[0]
        record["completed_utc"] = utc_now()
        rows.append(record)
        prefill = record.get("prefill_tokens_per_s")
        decode = record.get("decode_tokens_per_s")
        print(
            f"  {depth:9d} {record.get('prompt_tokens', 0):8d} "
            f"{(record.get('prefill_seconds') or 0):9.2f} "
            f"{(prefill or 0):11.0f} {(decode or 0):10.1f} "
            f"{record['free_mib_min']:12d} {record['status']:>10s}"
        )

    payload = {"schema": 1, "started_utc": sweep_started,
               "corpus": str(args.corpus), "decode_tokens": args.decode_tokens, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
