#!/usr/bin/env python3
"""TG3b acceptance runner: 2 conversations x 3 turns x 256 tokens per arm.

Per arm: one short excluded warmup, then per conversation a cache flush
(POST /flush_cache) so each conversation starts cold; the three turns are run
sequentially on the same server so shared prefixes are reused.

  python3 tg3c_accept.py --conv conv.json --arm baseline --out-dir acc_baseline
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:30001"


def flush(timeout: int = 120) -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/flush_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception as exc:  # noqa: BLE001
        print(f"    flush_cache error: {type(exc).__name__}: {exc}")
        return False


def stream_turn(ids: list[int], decode: int, timeout: int) -> dict:
    body = json.dumps({
        "input_ids": ids, "stream": True,
        "sampling_params": {"max_new_tokens": decode, "temperature": 0,
                            "ignore_eos": True},
    }).encode()
    req = urllib.request.Request(f"{BASE_URL}/generate", body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    times, counts = [], []
    ptok = ctok = cached = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            if "meta_info" not in ev:
                raise RuntimeError(f"server error: {json.dumps(ev)[:300]}")
            m = ev["meta_info"]
            times.append(time.monotonic())
            counts.append(int(m["completion_tokens"]))
            ptok = int(m["prompt_tokens"])
            if cached is None:
                cached = int(m.get("cached_tokens", -1))
    t1 = time.monotonic()
    out = {"prompt_tokens": ptok, "cached_tokens": cached,
           "completion_tokens": counts[-1] if counts else 0,
           "wall_seconds": t1 - t0, "events": len(times)}
    if len(times) >= 2 and counts[-1] > counts[0]:
        out["ttft_seconds"] = times[0] - t0
        out["decode_seconds"] = times[-1] - times[0]
        out["tg_tokens_per_s"] = (counts[-1] - counts[0]) / out["decode_seconds"]
    else:
        out["status"] = "too_few_events"
    out["status"] = out.get("status", "ok")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    conv = json.loads(a.conv.read_text())
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # one short excluded warmup, then flush so conversations start cold
    _ = stream_turn([1, 2, 3, 4], 4, a.timeout)
    flush()

    summary = {"arm": a.arm, "conversations": {}}
    for name in ("code", "reasoning"):
        spec = conv["conversations"][name]
        flushed = flush()
        turns = []
        for i, ids in enumerate(spec["turns"], 1):
            rec = stream_turn(ids, a.decode, a.timeout)
            rec["turn"] = i
            rec["requested_prompt_tokens"] = len(ids)
            turns.append(rec)
            print(f"  {a.arm} {name:9s} turn{i} prompt={rec['prompt_tokens']} "
                  f"cached={rec['cached_tokens']} gen={rec['completion_tokens']} "
                  f"ttft={rec.get('ttft_seconds')} wall={rec['wall_seconds']:.2f} "
                  f"tg={rec.get('tg_tokens_per_s')}", flush=True)
        summary["conversations"][name] = {
            "flushed_before": flushed,
            "turn_prompt_tokens": [t["prompt_tokens"] for t in turns],
            "total_wall_seconds": sum(t["wall_seconds"] for t in turns),
            "total_decode_seconds": sum(t.get("decode_seconds", 0.0) for t in turns),
            "turns": turns,
        }
        print(f"    {name}: total_wall={summary['conversations'][name]['total_wall_seconds']:.2f}s "
              f"total_decode={summary['conversations'][name]['total_decode_seconds']:.2f}s")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
