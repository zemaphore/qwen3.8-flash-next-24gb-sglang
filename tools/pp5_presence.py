#!/usr/bin/env python3
"""PP5: prefill presence vs routing-mass placement analysis.

Ranks the S resident experts of each layer two ways and reports the number
of distinct cold rows a staged prefill chunk must transfer per layer:

  mass      current placement: top-S by routing mass in expert_freq.pt
  presence  top-S by per-chunk presence probability on prefill chunks
  token     top-S by per-chunk token share (presence-weighted volume)

  python3 tools/pp5_presence.py [dump_dir] [expert_freq.pt] \
      [--s 184] [--min-tokens 64] [--write-presence PATH]
"""
import argparse
import glob
import os
import sys

import torch


def load_chunks(dump_dir, min_tokens):
    chunks = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "prouting_*.pt"))):
        rec = torch.load(f, weights_only=False)
        per_layer = {}
        for lid, ids in rec:
            per_layer.setdefault(int(lid), []).append(ids.long())
        layers = sorted(per_layer)
        if not layers:
            continue
        m = sum(t.shape[0] for t in per_layer[layers[0]])
        if m < int(min_tokens):
            continue
        chunks.append({l: torch.cat(per_layer[l]) for l in layers})
    return chunks


def rank_indices(score, s, tiebreak=None):
    """Return per-layer top-s indices, optionally with a stable secondary score."""
    result = []
    for l in range(score.shape[0]):
        if tiebreak is None:
            order = torch.argsort(score[l], descending=True, stable=True)
        else:
            secondary = torch.argsort(tiebreak[l], descending=True, stable=True)
            primary_order = torch.argsort(
                score[l, secondary], descending=True, stable=True
            )
            order = secondary[primary_order]
        result.append(order[:s])
    return result


def rank_sets(score, s, tiebreak=None):
    return [set(v.tolist()) for v in rank_indices(score, s, tiebreak)]


def cold_rows(routed, resident, layers):
    """Mean distinct non-resident rows, matching every layer to its own set."""
    total = sum(
        len(layer_ids - resident[l])
        for chunk_layers in routed
        for l, layer_ids in zip(layers, chunk_layers)
    )
    return total / (len(routed) * len(layers))


def placement_score(primary, mass):
    """Encode primary rank + mass tie-break for ExpertElastic's single score."""
    scale = mass.amax(dim=1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)
    return primary * 2.0 + mass / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default="/root/quant/route_dump_prefill")
    ap.add_argument("freq", nargs="?", default="assets/expert_freq.pt")
    ap.add_argument("--s", type=int, default=184)
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=64,
        help="ignore decode and launcher warmups smaller than this (default: 64)",
    )
    ap.add_argument(
        "--write-presence",
        metavar="PATH",
        help="write a presence-primary, routing-mass-tiebroken placement file",
    )
    args = ap.parse_args()

    freq = torch.load(args.freq, weights_only=False, map_location="cpu")
    mass = freq["mass"]
    chunks = load_chunks(args.dump_dir, args.min_tokens)
    if not chunks:
        sys.exit("no prefill chunks found")
    layers = sorted(chunks[0])
    sizes = {}
    for ch in chunks:
        m = int(ch[layers[0]].shape[0])
        sizes[m] = sizes.get(m, 0) + 1
    print(
        f"{len(chunks)} prefill chunks, sizes={dict(sorted(sizes.items()))}, "
        f"layers {layers[0]}..{layers[-1]}, S={args.s}"
    )

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
        tie = None if name == "mass" else mass
        res = rank_sets(score, args.s, tie)
        cover = 1.0 * presence / n
        # expected distinct cold rows per layer-chunk under this static set
        covered_mass = 0.0
        for l in layers:
            top = rank_indices(score[l:l + 1], args.s,
                               None if tie is None else tie[l:l + 1])[0]
            covered_mass += mass[l, top].sum().item()
        report[name] = (
            cold_rows(routed, res, layers),
            covered_mass / mass.sum().item(),
        )
        pr = (cover > 0).float()
        pr_res = torch.zeros_like(pr)
        res = rank_sets(score, args.s, tie)
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
             zip(rank_sets(cand["presence"], args.s, mass), rank_sets(mass, args.s)))
    print(f"resident-set overlap presence/mass: {ov}/{args.s*len(layers)}")

    if args.write_presence:
        encoded = placement_score(presence, mass)
        torch.save(
            {
                "mass": encoded,
                "presence": presence,
                "routing_mass": mass,
                "token_count": tokens,
                "source_chunks": len(chunks),
                "min_tokens": args.min_tokens,
                "resident_s": args.s,
                "objective": "prefill_presence_then_routing_mass",
            },
            args.write_presence,
        )
        print(f"wrote presence placement: {args.write_presence}")


if __name__ == "__main__":
    main()
