#!/usr/bin/env python3
"""PP5: prefill presence vs routing-mass placement analysis.

Ranks the S resident experts of each layer two ways and reports the number
of distinct cold rows a staged prefill chunk must transfer per layer:

  mass      current placement: top-S by routing mass in expert_freq.pt
  presence  top-S by per-chunk presence probability on prefill chunks
  token     top-S by per-chunk token share (presence-weighted volume)

  python3 tools/pp5_presence.py [dump_dir] [expert_freq.pt] [--s 184] [--skip 2]
"""
import argparse
import glob
import os
import sys

import torch


def load_chunks(dump_dir, skip):
    chunks = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "prouting_*.pt"))):
        rec = torch.load(f, weights_only=False)
        per_layer = {}
        for lid, ids in rec:
            per_layer.setdefault(int(lid), []).append(ids.long())
        layers = sorted(per_layer)
        m = sum(t.shape[0] for t in per_layer[layers[0]])
        if m <= int(skip):
            continue
        chunks.append({l: torch.cat(per_layer[l]) for l in layers})
    return chunks


def rank_sets(score, s):
    order = torch.argsort(score, descending=True)
    return [set(order[l, :s].tolist()) for l in range(score.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default="/root/quant/route_dump_prefill")
    ap.add_argument("freq", nargs="?", default="assets/expert_freq.pt")
    ap.add_argument("--s", type=int, default=184)
    ap.add_argument("--skip", type=int, default=2, help="ignore forwards with <= this many tokens")
    args = ap.parse_args()

    freq = torch.load(args.freq, weights_only=False, map_location="cpu")
    mass = freq["mass"]
    chunks = load_chunks(args.dump_dir, args.skip)
    if not chunks:
        sys.exit("no prefill chunks found")
    layers = sorted(chunks[0])
    print(f"{len(chunks)} prefill chunks, layers {layers[0]}..{layers[-1]}, S={args.s}")

    presence = torch.zeros_like(mass)
    tokens = torch.zeros_like(mass)
    routed = []
    for ch in chunks:
        rs = []
        for l in layers:
            ids = ch[l]
            uniq, cnt = ids.unique(return_counts=True)
            presence[l, uniq] += 1.0
            tokens[l, uniq] += cnt.float()
            rs.append(set(uniq.tolist()))
        routed.append(rs)

    n = len(chunks)
    cand = {
        "mass": mass,
        "presence": presence,
        "token": tokens,
    }
    report = {}
    for name, score in cand.items():
        res = rank_sets(score, args.s)
        cover = 1.0 * presence / n
        # expected distinct cold rows per layer-chunk under this static set
        cold = 0
        covered_mass = 0.0
        for chunk_sets, r in zip(routed, res):
            for rs in chunk_sets:
                cold += len(rs - r)
        for l in layers:
            top = torch.argsort(score[l], descending=True)[: args.s]
            covered_mass += mass[l, top].sum().item()
        report[name] = (cold / (n * len(layers)), covered_mass / mass.sum().item())
        pr = (cover > 0).float()
        pr_res = torch.zeros_like(pr)
        res = rank_sets(score, args.s)
        for l in layers:
            idx = torch.tensor(sorted(res[l]))
            pr_res[l, idx] = pr[l, idx]
        miss_p = (cover * (1 - pr_res)).sum(1).mean().item()
        print(f"  {name:<9} distinct cold rows/layer-chunk = {report[name][0]:7.2f}"
              f"   mass cover = {report[name][1]*100:5.1f}%   E[sum p_present*cold] = {miss_p:5.2f}")

    a = report["mass"][0] * len(layers)
    b = report["presence"][0] * len(layers)
    print(f"presence vs mass: {(a - b) / a * 100:+.1f}% total distinct cold rows per 48-layer chunk")
    ov = sum(len(reset & mset) for reset, mset in
             zip(rank_sets(cand["presence"], args.s), rank_sets(mass, args.s)))
    print(f"resident-set overlap presence/mass: {ov}/{args.s*len(layers)}")


if __name__ == "__main__":
    main()
