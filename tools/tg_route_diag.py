#!/usr/bin/env python3
"""Decode routing/residency diagnostics from a ``SGLANG_DECODE_ROUTE_DUMP`` tree.

This is a *separate* diagnostic from the headline timing.  It reads the
per-token decode routing recorded by ``patches/decode_route_dump.py`` and the
accepted placement asset, then reports, per layer and overall:

* actual decode selection counts and resident-vs-host selection coverage (by
  selection count, not by gate weight);
* routed gate-weight mass resident vs host (kept separate from count coverage);
* an estimated host-read traffic in MB/token, using the GEMV's per-expert
  qweight+scales byte sizes.  The decode GEMV reads the selected expert rows in
  place, so every host-resident selection re-reads that row.

The estimate is labelled as such; the tool does not claim measured DMA bytes.

  python3 tools/tg_route_diag.py --dump DIR --placement assets/expert_presence_code.pt \
      --out out.json [--s 184]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

# Config / GEMV layout for Qwen3.8-Flash-Next int2: hidden 2560, moe intermediate
# 640, symmetric INT2 (zero point constant, so qzeros are not read at decode).
HIDDEN = 2560
INTER = 640
W13 = (HIDDEN // 16) * (2 * INTER) * 4 + (HIDDEN // 128) * (2 * INTER) * 2
W2 = (INTER // 16) * HIDDEN * 4 + (INTER // 128) * HIDDEN * 2
BYTES_PER_EXPERT = W13 + W2


def load_dump(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (selection_count [48, 512] long, gate_mass [48, 512] float, tokens)."""
    files = sorted(glob.glob(str(path / "drouting_*.pt")))
    if not files:
        raise SystemExit(f"no drouting_*.pt in {path}")
    layers = 48
    experts = 512
    count = torch.zeros(layers, experts, dtype=torch.long)
    mass = torch.zeros(layers, experts, dtype=torch.float64)
    for name in files:
        for lid, ids, weights in torch.load(name, map_location="cpu", weights_only=True):
            ids = ids.reshape(-1).long()
            weights = weights.reshape(-1).double()
            count[lid].index_add_(0, ids, torch.ones_like(ids))
            mass[lid].index_add_(0, ids, weights)
    return count, mass, len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--placement", type=Path, default=REPO / "assets/expert_presence_code.pt")
    parser.add_argument("--s", type=int, default=184)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    count, mass, tokens = load_dump(args.dump)
    asset = torch.load(args.placement, map_location="cpu", weights_only=True)
    score = asset["mass"].float()
    order = torch.argsort(score, dim=1, descending=True)[:, : args.s]
    resident = torch.zeros_like(count, dtype=torch.bool)
    resident.scatter_(1, order, True)

    total_selections = int(count.sum())
    host_selections = int(count[~resident].sum())
    resident_selections = total_selections - host_selections
    total_mass = float(mass.sum())
    host_mass = float(mass[~resident].sum())

    per_layer = {}
    for layer in range(count.shape[0]):
        sel = int(count[layer].sum())
        host = int(count[layer][~resident[layer]].sum())
        lay_mass = float(mass[layer].sum())
        lay_host_mass = float(mass[layer][~resident[layer]].sum())
        per_layer[str(layer)] = {
            "selections": sel,
            "host_selections": host,
            "resident_selection_fraction": (sel - host) / sel if sel else None,
            "gate_mass": lay_mass,
            "resident_gate_mass_fraction": (lay_mass - lay_host_mass) / lay_mass
            if lay_mass
            else None,
        }

    payload = {
        "schema": 1,
        "dump": str(args.dump),
        "placement": str(args.placement),
        "s": args.s,
        "tokens": tokens,
        "bytes_per_expert": BYTES_PER_EXPERT,
        "overall": {
            "total_selections": total_selections,
            "resident_selections": resident_selections,
            "host_selections": host_selections,
            "resident_selection_fraction": resident_selections / total_selections,
            "selection_process_coverage": args.s / count.shape[1],
            "total_gate_mass": total_mass,
            "host_gate_mass": host_mass,
            "resident_gate_mass_fraction": (total_mass - host_mass) / total_mass,
            "estimated_host_traffic_mb_per_token": (
                host_selections * BYTES_PER_EXPERT / tokens / 1e6
            ),
            "estimated_total_traffic_mb_per_token": (
                total_selections * BYTES_PER_EXPERT / tokens / 1e6
            ),
        },
        "per_layer": per_layer,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["overall"], indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
