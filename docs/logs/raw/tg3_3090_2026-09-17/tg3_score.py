#!/usr/bin/env python3
"""TG3: rank a static top-184-per-layer candidate by decode selection COUNT and
score it offline against the current presence asset on held-out decode sets.

Cold = selected expert not in that layer's resident top-184.
Bytes are ESTIMATES (selection count x per-expert qweight+scales bytes).

  python3 tg3_score.py --cap-dir cap --presence assets/expert_presence_code.pt \
      --out candidate_selection.pt --report score.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
LAYERS = 48
TOPK = 10
S = 184
HIDDEN, INTER = 2560, 640
BYTES_PER_EXPERT = (
    (HIDDEN // 16) * (2 * INTER) * 4 + (HIDDEN // 128) * (2 * INTER) * 2
    + (INTER // 16) * HIDDEN * 4 + (INTER // 128) * HIDDEN * 2
)


def counts(path: Path) -> np.ndarray:
    arr = np.load(path)  # [rows,48,10]
    c = np.zeros((LAYERS, 512), dtype=np.int64)
    for layer in range(LAYERS):
        np.add.at(c[layer], arr[:, layer, :].reshape(-1).astype(np.int64), 1)
    return arr, c


def resident(order: np.ndarray) -> np.ndarray:
    m = np.zeros((LAYERS, 512), dtype=bool)
    for layer in range(LAYERS):
        m[layer, order[layer]] = True
    return m


def score(ids: np.ndarray, res: np.ndarray) -> dict:
    rows = ids.shape[0]
    total = int(rows) * LAYERS * TOPK
    cold = 0
    per_layer_cold = np.zeros(LAYERS, dtype=np.int64)
    for layer in range(LAYERS):
        sel = ids[:, layer, :].reshape(-1).astype(np.int64)
        c = int((~res[layer][sel]).sum())
        cold += c
        per_layer_cold[layer] = c
    return {
        "rows": int(rows),
        "tokens": int(rows),
        "total_selections": total,
        "cold_selections": cold,
        "cold_per_token": cold / rows if rows else None,
        "resident_selection_fraction": (total - cold) / total if total else None,
        "estimated_host_traffic_mb_per_token": cold * BYTES_PER_EXPERT / rows / 1e6
        if rows else None,
        "per_layer_cold": per_layer_cold.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-dir", type=Path, required=True)
    ap.add_argument("--presence", type=Path,
                    default=REPO / "assets/expert_presence_code.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--s", type=int, default=S)
    a = ap.parse_args()
    cap = a.cap_dir

    cal = {k: counts(cap / f"{k}.npy") for k in ("code_cal", "reasoning_cal")}
    cal_count = cal["code_cal"][1] + cal["reasoning_cal"][1]
    cand_order = np.argsort(-cal_count, axis=1)[:, :a.s]

    asset = torch.load(a.presence, map_location="cpu", weights_only=True)
    pres_order = torch.argsort(asset["mass"].float(), dim=1, descending=True)[
        :, :a.s].numpy()

    torch.save(
        {"mass": torch.from_numpy(cal_count.astype(np.float32)),
         "count": torch.from_numpy(cal_count),
         "source": "tg3 decode selection counts (calibration sets)"},
        a.out,
    )

    held = {}
    for k in ("code_held", "reasoning_held"):
        ids = np.load(cap / f"{k}.npy")
        held[k] = {
            "candidate": score(ids, resident(cand_order)),
            "presence": score(ids, resident(pres_order)),
        }

    ov = {}
    for who in ("candidate", "presence"):
        tc = sum(held[k][who]["cold_selections"] for k in held)
        tt = sum(held[k][who]["tokens"] for k in held)
        ttok = sum(held[k][who]["total_selections"] for k in held)
        ov[who] = {
            "cold_selections": tc,
            "tokens": tt,
            "cold_per_token": tc / tt,
            "resident_selection_fraction": (ttok - tc) / ttok,
            "estimated_host_traffic_mb_per_token": tc * BYTES_PER_EXPERT / tt / 1e6,
        }
    cold_ratio = ov["candidate"]["cold_per_token"] / ov["presence"]["cold_per_token"]
    per_work = {
        k: {
            "candidate_cold_per_token": held[k]["candidate"]["cold_per_token"],
            "presence_cold_per_token": held[k]["presence"]["cold_per_token"],
            "ratio": held[k]["candidate"]["cold_per_token"]
            / held[k]["presence"]["cold_per_token"],
        }
        for k in held
    }
    no_work_worse = all(v["ratio"] <= 1.0 for v in per_work.values())
    qualifies = (cold_ratio <= 0.85) and no_work_worse
    payload = {
        "s": a.s,
        "bytes_per_expert": BYTES_PER_EXPERT,
        "calibration": {k: {"rows": cal[k][0].shape[0]} for k in cal},
        "held": held,
        "overall": ov,
        "cold_ratio_candidate_over_presence": cold_ratio,
        "cold_reduction_pct": 100 * (1 - cold_ratio),
        "per_workload": per_work,
        "no_workload_worse": no_work_worse,
        "qualifies_timing": bool(qualifies),
        "estimate_note": "host traffic is an estimate from selection counts x per-expert bytes",
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("cold_ratio_candidate_over_presence", "cold_reduction_pct",
                       "no_workload_worse", "qualifies_timing")}, indent=2))
    print("overall candidate:", json.dumps(ov["candidate"]))
    print("overall presence: ", json.dumps(ov["presence"]))
    print("per-workload:", json.dumps(per_work, indent=1))
    print(f"wrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
