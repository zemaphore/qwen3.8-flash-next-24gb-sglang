#!/usr/bin/env python3
"""RC1 reuse checks against a running hybrid-cache SGLang server.

Committed replacement for the ad-hoc scripts behind the first RC1 evidence
(determinism, logprob reuse-vs-cold, churn) plus the eviction-then-rehit test
that RC1 did not exercise. Every check writes <name>.json and a short
<name>.txt into --out-dir. Uses /flush_cache to force cold runs.

Usage:
  python3 tools/rc1_reuse_checks.py --out-dir docs/logs/raw/rc1_3090_2026-09-16/reuse_checks
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc1_cache_probe import PAGE, base_prefix, floor_page, repo_block  # noqa: E402


def post(url: str, path: str, body: dict | None, timeout: float = 900.0) -> dict:
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip().startswith(b"{") else {"raw": raw.decode()}


def gen(url: str, text: str, max_new: int, logprob: bool = False) -> dict:
    body = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True},
    }
    if logprob:
        body["return_logprob"] = True
    t0 = time.perf_counter()
    out = post(url, "/generate", body)
    m = out.get("meta_info", {})
    rec = {
        "prompt_tokens": m.get("prompt_tokens"),
        "cached_tokens": m.get("cached_tokens", 0),
        "completion_tokens": m.get("completion_tokens"),
        "wall_s": round(time.perf_counter() - t0, 3),
        "text": out.get("text", ""),
    }
    if logprob:
        lp = m.get("output_token_logprobs") or []
        rec["ids"] = [t[1] for t in lp]
        rec["logprobs"] = [t[0] for t in lp]
    return rec


def flush(url: str) -> None:
    post(url, "/flush_cache", None, timeout=120)
    time.sleep(0.5)


def free_vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def compare_logprobs(a: dict, b: dict) -> dict:
    n = min(len(a["ids"]), len(b["ids"]))
    first_div = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
    upto = n if first_div is None else first_div
    maxd = max((abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(upto)), default=0.0)
    return {"ids_match": first_div is None, "first_divergence": first_div,
            "max_abs_dlogprob_matched_prefix": round(maxd, 5), "compared": upto}


def texts(marker: str) -> dict:
    base = base_prefix(marker)
    return {
        "base": base,
        "conv_a": base + f"\n\nAssistant branch A for {marker}: refactor the handler.\n"
        + f"\n\nTool result 1 for {marker}: tests pass; 42 files changed.\n",
    }


def write(out_dir: str, name: str, data: dict, lines: list[str]) -> None:
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(out_dir, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def check_determinism(url, out_dir, nonce, trials, max_new):
    rows, lines = [], [f"determinism: identical prompt cold/warm/warm2, max_new={max_new}"]
    for t in range(trials):
        tx = texts(f"det{t}-{nonce}")
        flush(url)
        c = gen(url, tx["conv_a"], max_new)
        w = gen(url, tx["conv_a"], max_new)
        w2 = gen(url, tx["conv_a"], max_new)
        row = {"trial": t, "cached": [c["cached_tokens"], w["cached_tokens"], w2["cached_tokens"]],
               "c_eq_w": c["text"] == w["text"], "w_eq_w2": w["text"] == w2["text"],
               "texts": [c["text"], w["text"], w2["text"]]}
        rows.append(row)
        lines.append(f"trial {t}: cached c={row['cached'][0]} w={row['cached'][1]} w2={row['cached'][2]}"
                     f"  c==w={row['c_eq_w']} w==w2={row['w_eq_w2']}")
    lines.append(f"c==w {sum(r['c_eq_w'] for r in rows)}/{trials}  w==w2 {sum(r['w_eq_w2'] for r in rows)}/{trials}")
    write(out_dir, "determinism", {"rows": rows}, lines)


def check_logprob_reuse(url, out_dir, nonce, trials, max_new):
    rows = []
    lines = [f"logprob reuse: cold1 vs cold2 (flush between) and cold1 vs reuse (base then conv_a), {max_new} greedy tokens",
             "trial cached coldcold_maxd reuse_maxd ids_cold1==cold2 ids_cold1==reuse text_cold1==cold2 text_cold1==reuse"]
    for t in range(trials):
        tx = texts(f"lp{t}-{nonce}")
        flush(url)
        c1 = gen(url, tx["conv_a"], max_new, logprob=True)
        flush(url)
        c2 = gen(url, tx["conv_a"], max_new, logprob=True)
        flush(url)
        gen(url, tx["base"], max_new)
        r = gen(url, tx["conv_a"], max_new, logprob=True)
        cc, cr = compare_logprobs(c1, c2), compare_logprobs(c1, r)
        row = {"trial": t, "reuse_cached": r["cached_tokens"], "coldcold": cc, "reuse": cr,
               "text_c1_eq_c2": c1["text"] == c2["text"], "text_c1_eq_r": c1["text"] == r["text"],
               "runs": {"cold1": c1, "cold2": c2, "reuse": r}}
        rows.append(row)
        lines.append(f"{t:5d} {r['cached_tokens']:6d} {cc['max_abs_dlogprob_matched_prefix']:13.5f} "
                     f"{cr['max_abs_dlogprob_matched_prefix']:10.5f} {str(cc['ids_match']):>16} {str(cr['ids_match']):>16} "
                     f"{str(row['text_c1_eq_c2']):>17} {str(row['text_c1_eq_r']):>16}")
    write(out_dir, "logprob_reuse", {"rows": rows}, lines)


def check_churn(url, out_dir, nonce, n, max_new):
    rows, lines = [], [f"churn: {n} unique ~6K conversations, each cold then warm; free VRAM sampled"]
    min_free = None
    for i in range(n):
        tx = texts(f"churn{i}-{nonce}")
        c = gen(url, tx["base"], max_new)
        w = gen(url, tx["base"], max_new)
        fv = free_vram_mib()
        min_free = fv if min_free is None or (fv is not None and fv < min_free) else min_free
        row = {"i": i, "prompt_tokens": c["prompt_tokens"], "cold_cached": c["cached_tokens"],
               "warm_cached": w["cached_tokens"], "expected_warm": floor_page(c["prompt_tokens"]),
               "free_vram_mib": fv, "cold_wall_s": c["wall_s"], "warm_wall_s": w["wall_s"]}
        rows.append(row)
        if i % 6 == 0:
            lines.append(f"  i={i:2d} pt={row['prompt_tokens']} cold_cached={row['cold_cached']} "
                         f"warm_cached={row['warm_cached']} free_vram={fv}MiB")
    cold_ok = sum(r["cold_cached"] == 0 for r in rows)
    warm_ok = sum(r["warm_cached"] == r["expected_warm"] for r in rows)
    total = sum(r["prompt_tokens"] + max_new for r in rows)
    lines.append(f"cold_ok={cold_ok}/{n} warm_hits={warm_ok}/{n} retained_tokens~{total} min_free_vram={min_free}MiB")
    write(out_dir, "churn", {"rows": rows, "min_free_vram_mib": min_free, "retained_tokens": total}, lines)


def gen_suffix_logprobs(url: str, text: str, start_len: int, max_new: int) -> dict:
    """Teacher-forced logprobs of prompt tokens from start_len on, plus generated tokens."""
    body = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": start_len,
    }
    t0 = time.perf_counter()
    out = post(url, "/generate", body)
    m = out.get("meta_info", {})
    inp = [t for t in (m.get("input_token_logprobs") or []) if t[0] is not None]
    return {
        "prompt_tokens": m.get("prompt_tokens"), "cached_tokens": m.get("cached_tokens", 0),
        "wall_s": round(time.perf_counter() - t0, 3),
        "suffix_ids": [t[1] for t in inp], "suffix_logprobs": [t[0] for t in inp],
        "ids": [t[1] for t in (m.get("output_token_logprobs") or [])],
        "logprobs": [t[0] for t in (m.get("output_token_logprobs") or [])],
        "text": out.get("text", ""),
    }


def suffix_delta(a: dict, b: dict) -> dict:
    n = min(len(a["suffix_logprobs"]), len(b["suffix_logprobs"]))
    same = a["suffix_ids"][:n] == b["suffix_ids"][:n]
    d = [abs(a["suffix_logprobs"][i] - b["suffix_logprobs"][i]) for i in range(n)]
    return {"tokens": n, "ids_identical": same, "max_abs_dlogprob": round(max(d, default=0.0), 5),
            "mean_abs_dlogprob": round(sum(d) / n, 5) if n else 0.0}


def check_suffix_logprob(url, out_dir, nonce, trials, max_new):
    """Teacher-forced logprobs of the uncached suffix: cold1 vs cold2 vs reuse (base cached)."""
    rows = []
    lines = [f"suffix logprob (teacher-forced, no sampling): conv_a tokens after the page-aligned base prefix",
             "trial start_len suffix_tokens reuse_cached coldcold_max coldcold_mean reuse_max reuse_mean first_gen_dlp_cc first_gen_dlp_reuse"]
    for t in range(trials):
        tx = texts(f"sfx{t}-{nonce}")
        flush(url)
        probe = gen(url, tx["base"], 1)
        start = floor_page(probe["prompt_tokens"])
        flush(url)
        c1 = gen_suffix_logprobs(url, tx["conv_a"], start, max_new)
        flush(url)
        c2 = gen_suffix_logprobs(url, tx["conv_a"], start, max_new)
        flush(url)
        gen(url, tx["base"], max_new)
        r = gen_suffix_logprobs(url, tx["conv_a"], start, max_new)
        cc, cr = suffix_delta(c1, c2), suffix_delta(c1, r)
        fg_cc = round(abs(c1["logprobs"][0] - c2["logprobs"][0]), 5) if c1["ids"][:1] == c2["ids"][:1] else None
        fg_cr = round(abs(c1["logprobs"][0] - r["logprobs"][0]), 5) if c1["ids"][:1] == r["ids"][:1] else None
        rows.append({"trial": t, "start_len": start, "reuse_cached": r["cached_tokens"], "coldcold": cc, "reuse": cr,
                     "first_gen_same_token_dlogprob": {"coldcold": fg_cc, "reuse": fg_cr},
                     "first_gen_ids": [c1["ids"][:1], c2["ids"][:1], r["ids"][:1]],
                     "first_gen_logprobs": [c1["logprobs"][:1], c2["logprobs"][:1], r["logprobs"][:1]],
                     "runs": {"cold1": c1, "cold2": c2, "reuse": r}})
        lines.append(f"{t:5d} {start:9d} {cc['tokens']:13d} {r['cached_tokens']:12d} {cc['max_abs_dlogprob']:12.5f} "
                     f"{cc['mean_abs_dlogprob']:13.5f} {cr['max_abs_dlogprob']:9.5f} {cr['mean_abs_dlogprob']:10.5f} "
                     f"{str(fg_cc):>16} {str(fg_cr):>19}")
    write(out_dir, "suffix_logprob", {"rows": rows}, lines)


def check_mamba_capacity(url, out_dir, nonce, max_new, max_k=10):
    """How many unique conversations can intervene before A loses its GPU hit."""
    rows, lines = [], ["mamba/hot-tier capacity: A cold, k short unique fillers, A again"]
    for k in range(0, max_k + 1):
        tx = texts(f"cap{k}-{nonce}")
        flush(url)
        a0 = gen(url, tx["conv_a"], max_new)
        for i in range(k):
            gen(url, "Filler %d/%d %s\n" % (i, k, nonce) + repo_block(f"c{k}-{i}-{nonce}")[:1500], max_new)
        a1 = gen(url, tx["conv_a"], max_new)
        rows.append({"k": k, "a_prompt_tokens": a0["prompt_tokens"], "expected_full": floor_page(a0["prompt_tokens"]),
                     "a_again_cached": a1["cached_tokens"], "wall_s": a1["wall_s"]})
        lines.append(f"  k={k:2d} intervening -> A cached={a1['cached_tokens']} (full={floor_page(a0['prompt_tokens'])}) wall={a1['wall_s']}s")
    write(out_dir, "mamba_capacity", {"rows": rows}, lines)


def check_growing_session(url, out_dir, nonce, max_new, turns, turn_modules):
    """One linear session: each turn appends the previous output plus a tool result."""
    flush(url)
    text = base_prefix(f"session-{nonce}")
    rows, lines = [], [f"growing session: {turns} turns, ~{turn_modules} repo modules of tool result per turn, max_new={max_new}"]
    prev_prompt = prev_total = None
    for t in range(turns):
        r = gen(url, text, max_new)
        exp_lo = 0 if prev_prompt is None else floor_page(prev_prompt)
        exp_hi = 0 if prev_total is None else floor_page(prev_total)
        ok = exp_lo <= r["cached_tokens"] <= exp_hi if t else r["cached_tokens"] == 0
        fv = free_vram_mib()
        rows.append({"turn": t, "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
                     "uncached": (r["prompt_tokens"] or 0) - r["cached_tokens"], "expected_range": [exp_lo, exp_hi],
                     "hit_ok": ok, "wall_s": r["wall_s"], "free_vram_mib": fv, "completion_tokens": r["completion_tokens"]})
        if t % 5 == 0 or t == turns - 1 or not ok:
            lines.append(f"  turn {t:2d}: prompt={r['prompt_tokens']} cached={r['cached_tokens']} uncached={rows[-1]['uncached']} "
                         f"expected=[{exp_lo},{exp_hi}] ok={ok} wall={r['wall_s']}s free_vram={fv}MiB")
        prev_prompt = r["prompt_tokens"]
        prev_total = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        tool = "".join(
            f"# {nonce} turn{t} module {i}\ndef handler_{t}_{i}(request):\n    return {{'turn': {t}, 'id': {i}}}\n\n"
            for i in range(turn_modules)
        )
        text = text + r["text"] + f"\n\nTool result for turn {t}:\n" + tool + f"\nUser: continue with turn {t + 1}.\n"
    hits = sum(r["hit_ok"] for r in rows)
    lines.append(f"hits_ok={hits}/{turns} final_prompt={rows[-1]['prompt_tokens']} min_free_vram={min(r['free_vram_mib'] or 0 for r in rows)}MiB "
                 f"warm_wall_median={sorted(r['wall_s'] for r in rows[1:])[len(rows) // 2]}s")
    write(out_dir, "growing_session", {"rows": rows}, lines)


def check_evict_rehit(url, out_dir, nonce, max_new, pool_tokens, filler_tokens_target):
    """A cold -> enough unique fillers to exceed the pool -> A again -> A again."""
    tx = texts(f"evict-{nonce}")
    flush(url)
    a0 = gen(url, tx["conv_a"], max_new, logprob=True)
    # mamba-slot-only pressure first: 12 short unique fillers (KV stays far below the pool)
    short = [gen(url, "Short filler %d %s\n" % (i, nonce) + repo_block(f"s{i}-{nonce}")[:2000], max_new)
             for i in range(12)]
    a1 = gen(url, tx["conv_a"], max_new, logprob=True)
    # then KV pressure: ~6K unique fillers until the pool is exceeded
    fillers, filled = [], 0
    while filled < filler_tokens_target:
        f = gen(url, base_prefix(f"fill{len(fillers)}-{nonce}"), max_new)
        fillers.append({"prompt_tokens": f["prompt_tokens"], "cached": f["cached_tokens"], "wall_s": f["wall_s"]})
        filled += (f["prompt_tokens"] or 0) + max_new
    fv = free_vram_mib()
    a2 = gen(url, tx["conv_a"], max_new, logprob=True)
    a3 = gen(url, tx["conv_a"], max_new, logprob=True)
    exp_full = floor_page(a0["prompt_tokens"])
    data = {
        "pool_tokens": pool_tokens, "a_prompt_tokens": a0["prompt_tokens"], "expected_full_hit": exp_full,
        "a0_cold": {k: a0[k] for k in ("cached_tokens", "wall_s", "text")},
        "short_fillers": [{"prompt_tokens": s["prompt_tokens"], "cached": s["cached_tokens"]} for s in short],
        "a1_after_mamba_pressure": {"cached_tokens": a1["cached_tokens"], "wall_s": a1["wall_s"],
                                    "vs_a0": compare_logprobs(a0, a1), "text_eq_a0": a0["text"] == a1["text"]},
        "kv_fillers": fillers, "kv_filler_tokens": filled, "free_vram_mib_after_fill": fv,
        "a2_after_kv_pressure": {"cached_tokens": a2["cached_tokens"], "wall_s": a2["wall_s"],
                                 "vs_a0": compare_logprobs(a0, a2), "text_eq_a0": a0["text"] == a2["text"]},
        "a3_rehit": {"cached_tokens": a3["cached_tokens"], "wall_s": a3["wall_s"],
                     "vs_a0": compare_logprobs(a0, a3), "vs_a2": compare_logprobs(a2, a3),
                     "text_eq_a2": a2["text"] == a3["text"]},
    }
    lines = [
        f"evict/rehit: A={a0['prompt_tokens']} tokens (full hit would be {exp_full}); pool={pool_tokens}",
        f"  a0 cold: cached={a0['cached_tokens']} wall={a0['wall_s']}s",
        f"  12 short fillers ({sum(s['prompt_tokens'] for s in short)} tokens) -> a1 cached={a1['cached_tokens']} "
        f"wall={a1['wall_s']}s ids_match={data['a1_after_mamba_pressure']['vs_a0']['ids_match']} "
        f"maxd={data['a1_after_mamba_pressure']['vs_a0']['max_abs_dlogprob_matched_prefix']} text_eq={data['a1_after_mamba_pressure']['text_eq_a0']}",
        f"  {len(fillers)} x ~6K fillers ({filled} tokens, > pool) free_vram={fv}MiB -> a2 cached={a2['cached_tokens']} "
        f"wall={a2['wall_s']}s ids_match={data['a2_after_kv_pressure']['vs_a0']['ids_match']} "
        f"maxd={data['a2_after_kv_pressure']['vs_a0']['max_abs_dlogprob_matched_prefix']} text_eq={data['a2_after_kv_pressure']['text_eq_a0']}",
        f"  a3 rehit: cached={a3['cached_tokens']} wall={a3['wall_s']}s ids_match_a0={data['a3_rehit']['vs_a0']['ids_match']} "
        f"text_eq_a2={data['a3_rehit']['text_eq_a2']}",
    ]
    write(out_dir, "evict_rehit", data, lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--churn-n", type=int, default=30)
    ap.add_argument("--pool-tokens", type=int, default=131072)
    ap.add_argument("--capacity-max-k", type=int, default=10)
    ap.add_argument("--session-turns", type=int, default=40)
    ap.add_argument("--session-turn-modules", type=int, default=80)
    ap.add_argument("--checks", default="determinism,logprob_reuse,suffix_logprob,mamba_capacity,churn,evict_rehit")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    nonce = format(int(time.time()), "x")
    checks = args.checks.split(",")
    with open(os.path.join(args.out_dir, "run.txt"), "w") as f:
        f.write(f"utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nnonce {nonce}\nargs {vars(args)}\npage {PAGE}\n")
    if "determinism" in checks:
        check_determinism(args.url, args.out_dir, nonce, args.trials, args.max_new)
    if "logprob_reuse" in checks:
        check_logprob_reuse(args.url, args.out_dir, nonce, args.trials, args.max_new)
    if "suffix_logprob" in checks:
        check_suffix_logprob(args.url, args.out_dir, nonce, args.trials, args.max_new)
    if "mamba_capacity" in checks:
        check_mamba_capacity(args.url, args.out_dir, nonce, args.max_new, args.capacity_max_k)
    if "growing_session" in checks:
        check_growing_session(args.url, args.out_dir, nonce, args.max_new, args.session_turns, args.session_turn_modules)
    if "churn" in checks:
        check_churn(args.url, args.out_dir, nonce, args.churn_n, args.max_new)
    if "evict_rehit" in checks:
        check_evict_rehit(args.url, args.out_dir, nonce, args.max_new, args.pool_tokens,
                          filler_tokens_target=int(args.pool_tokens * 1.25))


if __name__ == "__main__":
    main()
