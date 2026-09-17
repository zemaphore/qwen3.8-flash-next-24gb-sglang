#!/usr/bin/env python3
"""TG3: capture per-decode-token routed experts via enable_return_routed_experts.

Sends calibration and held-out prompts with return_routed_experts=true and
routed_experts_start_len=prompt_len, so the returned rows are exactly the
decode-step forward positions (prefill rows excluded). Saves the raw int32
tensor [rows, 48, 10] per prompt plus decode-step variation checks.

  python3 tg3_capture.py --prompts prompts.json --out-dir cap --decode 96
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "http://127.0.0.1:30001/generate"
LAYERS = 48
TOPK = 10


def one(url: str, ids: list[int], decode: int, timeout: int) -> dict:
    prompt_len = len(ids)
    body = json.dumps({
        "input_ids": ids,
        "sampling_params": {
            "max_new_tokens": decode,
            "temperature": 0,
            "ignore_eos": True,
        },
        "return_routed_experts": True,
        "routed_experts_start_len": prompt_len,
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read())
    if "meta_info" not in obj or "routed_experts" not in obj["meta_info"]:
        raise RuntimeError("no routed_experts in meta_info: " + json.dumps(obj)[:400])
    raw = base64.b64decode(obj["meta_info"]["routed_experts"])
    arr = np.frombuffer(raw, dtype=np.int32)
    if arr.size % (LAYERS * TOPK) != 0:
        raise RuntimeError(f"unexpected routed_experts size {arr.size}")
    arr = arr.reshape(-1, LAYERS, TOPK)
    meta = obj["meta_info"]
    distinct = len({tuple(row.flatten().tolist()) for row in arr})
    return {
        "prompt_tokens": int(meta["prompt_tokens"]),
        "completion_tokens": int(meta["completion_tokens"]),
        "cached_tokens": int(meta.get("cached_tokens", -1)),
        "rows": int(arr.shape[0]),
        "distinct_rows": distinct,
        "rows_equal_capture_constant": distinct == 1,
        "wall_seconds": time.monotonic() - t0,
        "output_ids": list(obj.get("output_ids") or []),
        "ids": arr.astype(np.int16),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--decode", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    prompts = json.loads(a.prompts.read_text())
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sets = a.sets or list(prompts["sets"])
    manifest = {"url": a.url, "decode": a.decode, "sets": {}}
    for key in sets:
        spec = prompts["sets"][key]
        rec = one(a.url, spec["prompt_ids"], a.decode, a.timeout)
        ids = rec.pop("ids")
        out_ids = rec.pop("output_ids")
        np.save(out / f"{key}.npy", ids)
        np.save(out / f"{key}.out_ids.npy", np.asarray(out_ids, dtype=np.int32))
        rec["class"] = spec["class"]
        rec["prompt_ids_sha256"] = spec["sha256"]
        rec["output_ids_len"] = len(out_ids)
        # decode rows exclude the prefill-produced first token: row j corresponds
        # to the forward that produced output_ids[j + 1]; expected rows = N - 1.
        rec["expected_rows"] = a.decode - 1
        rec["row_token_alignment_ok"] = (
            rec["rows"] == a.decode - 1 and len(out_ids) == a.decode
        )
        manifest["sets"][key] = rec
        print(f"  {key:14s} class={spec['class']:9s} prompt={rec['prompt_tokens']} "
              f"gen={rec['completion_tokens']} rows={rec['rows']} "
              f"out_ids={rec['output_ids_len']} rows_ok={rec['row_token_alignment_ok']} "
              f"distinct_rows={rec['distinct_rows']} "
              f"capture_constant={rec['rows_equal_capture_constant']} "
              f"wall={rec['wall_seconds']:.1f}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
