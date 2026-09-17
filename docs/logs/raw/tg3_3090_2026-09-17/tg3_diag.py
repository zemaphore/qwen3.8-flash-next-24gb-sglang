#!/usr/bin/env python3
"""TG3 corrected scoring: server-exact torch.argsort ordering + trace diagnostic.

Step 1: rebuild candidate/presence resident sets with
``torch.argsort(mass, dim=1, descending=True)[:, :S]`` (matching
moe_wna16.py:466) and recompute the held-out cold-selection scores.

Step 3: score each captured reasoning trace under BOTH placements, overall,
per layer, and for the first 95 vs the remaining decode steps.

  python3 tg3_diag.py --cap-dir cap --presence assets/expert_presence_code.pt \
      --candidate tg3_candidate_selection.pt --out diag.json \
      [--trace label=file.npy ...]
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
        "rows": int(rows),
        "total_selections": int(total),
        "cold_selections": cold,
        "cold_per_token": cold / rows if rows else None,
        "resident_selection_fraction": (total - cold) / total if total else None,
        "estimated_host_traffic_mb_per_token": cold * BYTES / rows / 1e6 if rows else None,
        "per_layer_cold": per_layer.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-dir", type=Path)
    ap.add_argument("--presence", type=Path, default=REPO / "assets/expert_presence_code.pt")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace", action="append", default=[], help="label=path.npy")
    ap.add_argument("--split", type=int, default=95)
    a = ap.parse_args()

    pres = torch.load(a.presence, map_location="cpu", weights_only=True)["mass"]
    cand_mass = torch.load(a.candidate, map_location="cpu", weights_only=True)["mass"]
    pres_res = resident(order_of(pres))
    cand_res = resident(order_of(cand_mass))
    out: dict = {"bytes_per_expert": BYTES, "s": S,
                 "ordering": "torch.argsort(mass, dim=1, descending=True)[:, :S]"}

    if a.cap_dir:
        cap = a.cap_dir
        cal = sel_count(np.load(cap / "code_cal.npy")) + sel_count(np.load(cap / "reasoning_cal.npy"))
        # candidate order is over calibration selection counts; ties follow
        # torch.argsort exactly, as the server does for the asset mass tensor.
        cand_res_cal = resident(order_of(torch.from_numpy(cal.astype(np.float32))))
        # sanity: order_of(counts) must equal torch.argsort on counts directly
        assert np.array_equal(cand_res_cal, resident(
            torch.argsort(torch.from_numpy(cal.astype(np.float32)), dim=1, descending=True
                          )[:, :S].numpy()))
        held = {}
        for k in ("code_held", "reasoning_held"):
            ids = np.load(cap / f"{k}.npy")
            held[k] = {"candidate": score(ids, cand_res_cal),
                       "presence": score(ids, pres_res)}
        ov = {}
        for who, res in (("candidate", cand_res_cal), ("presence", pres_res)):
            tc = sum(held[k][who]["cold_selections"] for k in held)
            tt = sum(held[k][who]["rows"] for k in held)
            ttok = sum(held[k][who]["total_selections"] for k in held)
            ov[who] = {"cold_selections": tc, "tokens": tt,
                       "cold_per_token": tc / tt,
                       "resident_selection_fraction": (ttok - tc) / ttok,
                       "estimated_host_traffic_mb_per_token": tc * BYTES / tt / 1e6}
        ratio = ov["candidate"]["cold_per_token"] / ov["presence"]["cold_per_token"]
        out["step1_corrected"] = {
            "held": held, "overall": ov,
            "cold_ratio_candidate_over_presence": ratio,
            "cold_reduction_pct": 100 * (1 - ratio),
            "per_workload": {
                k: {"candidate_cold_per_token": held[k]["candidate"]["cold_per_token"],
                    "presence_cold_per_token": held[k]["presence"]["cold_per_token"],
                    "ratio": held[k]["candidate"]["cold_per_token"]
                    / held[k]["presence"]["cold_per_token"]}
                for k in held},
        }

    for spec in a.trace:
        label, path = spec.split("=", 1)
        ids = np.load(path)
        n = ids.shape[0]
        entry = {"rows": int(n), "split_first": a.split,
                 "by_placement": {}, "by_placement_split": {}}
        for who, res in (("presence", pres_res), ("candidate", cand_res)):
            entry["by_placement"][who] = score(ids, res)
            entry["by_placement_split"][who] = {
                "first": score(ids[:a.split], res) if n >= a.split else None,
                "rest": score(ids[a.split:], res) if n > a.split else None,
            }
        out.setdefault("traces", {})[label] = entry

    a.out.write_text(json.dumps(out, indent=2) + "\n")
    if "step1_corrected" in out:
        s = out["step1_corrected"]
        print("STEP1 corrected (torch.argsort): "
              f"cold_reduction={s['cold_reduction_pct']:.1f}% "
              f"ratio={s['cold_ratio_candidate_over_presence']:.3f}")
        print("  per-workload:", json.dumps(s["per_workload"]))
    for label, e in out.get("traces", {}).items():
        print(f"TRACE {label} rows={e['rows']}")
        for who in ("presence", "candidate"):
            b = e["by_placement"][who]
            fs = e["by_placement_split"][who]["first"]["cold_per_token"]
            rs = e["by_placement_split"][who]["rest"]["cold_per_token"]
            print(f"  {who:9s} cold/tok={b['cold_per_token']:.1f} "
                  f"first95={fs:.1f} rest={rs:.1f} "
                  f"est_MB/tok={b['estimated_host_traffic_mb_per_token']:.1f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
