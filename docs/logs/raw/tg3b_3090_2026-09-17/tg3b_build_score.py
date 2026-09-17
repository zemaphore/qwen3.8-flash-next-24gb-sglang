#!/usr/bin/env python3
"""TG3b: build one static top-184 asset from calibration decode selection counts
(equal weight per calibration prompt) and score it against presence on the SAME
fresh held-out traces.

Gate: candidate cold <= 0.85 * presence cold overall AND in the later window,
with neither held-out workload worsening overall or in either window.

  python3 tg3b_build_score.py --cap-dir cap --presence assets/expert_presence_code.pt \
      --candidate-out asset.pt --report report.json --split 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
LAYERS, TOPK, S = 48, 10, 184
HIDDEN, INTER = 2560, 640
BYTES = (
    (HIDDEN // 16) * (2 * INTER) * 4 + (HIDDEN // 128) * (2 * INTER) * 2
    + (INTER // 16) * HIDDEN * 4 + (INTER // 128) * HIDDEN * 2
)
CAL = ("code_cal1", "code_cal2", "reasoning_cal1", "reasoning_cal2")
HELD = ("code_held_fresh", "reasoning_held_fresh")


def order_of(mass: torch.Tensor) -> np.ndarray:
    return torch.argsort(mass.float(), dim=1, descending=True)[:, :S].numpy()


def resident(order: np.ndarray) -> np.ndarray:
    m = np.zeros((LAYERS, 512), dtype=bool)
    for layer in range(LAYERS):
        m[layer, order[layer]] = True
    return m


def sel_count(ids: np.ndarray) -> np.ndarray:
    c = np.zeros((LAYERS, 512), dtype=np.int64)
    for layer in range(LAYERS):
        np.add.at(c[layer], ids[:, layer, :].reshape(-1).astype(np.int64), 1)
    return c


def score(ids: np.ndarray, res: np.ndarray) -> dict:
    rows = ids.shape[0]
    total = rows * LAYERS * TOPK
    cold = 0
    per_layer = np.zeros(LAYERS, dtype=np.int64)
    for layer in range(LAYERS):
        sel = ids[:, layer, :].reshape(-1).astype(np.int64)
        c = int((~res[layer][sel]).sum())
        cold += c
        per_layer[layer] = c
    return {
        "rows": int(rows), "total_selections": int(total),
        "cold_selections": cold,
        "cold_per_token": cold / rows if rows else None,
        "resident_selection_fraction": (total - cold) / total if total else None,
        "estimated_host_traffic_mb_per_token": cold * BYTES / rows / 1e6 if rows else None,
        "per_layer_cold": per_layer.tolist(),
    }


def ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-dir", type=Path, required=True)
    ap.add_argument("--presence", type=Path,
                    default=REPO / "assets/expert_presence_code.pt")
    ap.add_argument("--candidate-out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--split", type=int, default=128)
    a = ap.parse_args()
    cap = a.cap_dir

    cal_count = None
    per_cal_rows = {}
    for k in CAL:
        ids = np.load(cap / f"{k}.npy")
        per_cal_rows[k] = int(ids.shape[0])
        c = sel_count(ids)
        cal_count = c if cal_count is None else cal_count + c
    torch.save({"mass": torch.from_numpy(cal_count.astype(np.float32)),
                "count": torch.from_numpy(cal_count),
                "source": "tg3b decode selection counts, equal weight per calibration prompt"},
               a.candidate_out)
    cand_res = resident(order_of(torch.from_numpy(cal_count.astype(np.float32))))
    pres_res = resident(order_of(torch.load(a.presence, map_location="cpu",
                                             weights_only=True)["mass"]))
    # sanity: server-exact ordering
    assert np.array_equal(cand_res, resident(torch.argsort(
        torch.from_numpy(cal_count.astype(np.float32)), dim=1, descending=True)[:, :S].numpy()))

    report: dict = {"split": a.split, "s": S, "bytes_per_expert": BYTES,
                    "calibration_rows": per_cal_rows,
                    "held": {}, "gate": {}}
    ok_overall = ok_later = True
    for k in HELD:
        ids = np.load(cap / f"{k}.npy")
        ent = {}
        for who, res in (("presence", pres_res), ("candidate", cand_res)):
            ent[who] = {
                "overall": score(ids, res),
                "first": score(ids[:a.split], res),
                "rest": score(ids[a.split:], res),
            }
        og = ratio(ent["candidate"]["overall"]["cold_per_token"],
                   ent["presence"]["overall"]["cold_per_token"])
        rw = ratio(ent["candidate"]["rest"]["cold_per_token"],
                   ent["presence"]["rest"]["cold_per_token"])
        fw = ratio(ent["candidate"]["first"]["cold_per_token"],
                   ent["presence"]["first"]["cold_per_token"])
        ent["ratio"] = {"overall": og, "first": fw, "rest": rw}
        ent["ok_overall_15"] = og <= 0.85
        ent["ok_rest_15"] = rw <= 0.85
        ent["worsens_overall"] = og > 1.0
        ent["worsens_rest"] = rw > 1.0
        ent["worsens_first"] = fw > 1.0
        report["held"][k] = ent
        ok_overall = ok_overall and ent["ok_overall_15"]
        ok_later = ok_later and ent["ok_rest_15"]

    # overall across both held sets (sum counts)
    def agg(who, win):
        c = sum(report["held"][k][who][win]["cold_selections"] for k in HELD)
        t = sum(report["held"][k][who][win]["rows"] for k in HELD)
        return c / t
    gate_overall = ratio(agg("candidate", "overall"), agg("presence", "overall"))
    gate_rest = ratio(agg("candidate", "rest"), agg("presence", "rest"))
    no_worse = all(
        not any(report["held"][k][w] for w in ("worsens_overall", "worsens_first", "worsens_rest"))
        for k in HELD)
    qualifies = (gate_overall <= 0.85 and gate_rest <= 0.85 and no_worse)
    report["gate"] = {
        "overall_ratio": gate_overall, "overall_reduction_pct": 100 * (1 - gate_overall),
        "rest_ratio": gate_rest, "rest_reduction_pct": 100 * (1 - gate_rest),
        "no_workload_worse": no_worse, "qualifies_timing": bool(qualifies),
        "estimate_note": "host bytes are estimates (cold count x 1,305,600 B/expert)",
    }
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate"], indent=2))
    for k in HELD:
        e = report["held"][k]
        print(f"  {k}: overall {e['ratio']['overall']:.3f} first {e['ratio']['first']:.3f} "
              f"rest {e['ratio']['rest']:.3f} | cand cold/tok "
              f"overall={e['candidate']['overall']['cold_per_token']:.1f} "
              f"rest={e['candidate']['rest']['cold_per_token']:.1f} | pres "
              f"overall={e['presence']['overall']['cold_per_token']:.1f} "
              f"rest={e['presence']['rest']['cold_per_token']:.1f}")
    print(f"wrote {a.candidate_out} and {a.report}")


if __name__ == "__main__":
    main()
