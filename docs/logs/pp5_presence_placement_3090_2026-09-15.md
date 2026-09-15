# PP5 — prefill-aware expert placement (presence vs mass): REJECTED

Date: 2026-09-15. Host: RTX 3090, PP3 baseline (2,048-token chunks, S184,
bulk pread + 128 MiB recent-row cache, 2,048-byte gather tile).

## Method

- `patches/prefill_route_dump.py` (opt-in via `SGLANG_PREFILL_ROUTE_DUMP`):
  records per-layer prefill `topk_ids` for every forward with M > 1, flushed
  on the last MoE layer. Installed and left applied; off unless the variable
  is set, so no serving-path change.
- Restarted the launcher with `SGLANG_PREFILL_ROUTE_DUMP=/root/quant/route_dump_prefill`,
  ran the canonical code prompt (`tools/bench_agentic.py --tokens 4096
  --decode-tokens 256`, 4,565 tokens) three times after the launcher warmup.
  Collected 9 real prefill chunks (2 x 2,048 + 1 tail per request).
- `tools/pp5_presence.py` ranks each layer's top S=184 residents by mass
  (current `assets/expert_freq.pt`), by per-chunk presence count, and by
  per-chunk token count, and counts the distinct cold rows each 2,048-token
  chunk must stage.

## Results

| Ranking | distinct cold rows / layer-chunk | mass cover | E[sum p_present x cold] |
|---|---:|---:|---:|
| mass (current) | 210.09 | 84.3% | 196.52 |
| presence | 209.66 | 43.2% | 174.89 |
| token count | 209.38 | 47.4% | 178.31 |

- Resident-set overlap between presence and mass rankings: 3,869 / 8,832
  (43.8%) — the rankings do disagree strongly.
- Despite that, distinct cold rows improve by only **-0.2%**. Transfers are
  paid per distinct row, so the presence-weighted -11% "expected miss mass"
  does not buy anything.
- Structural cause: a code chunk routes to ~329 distinct experts per layer out
  of 512 (S=184 resident), so every static set leaves ~197-210 cold rows;
  placement is saturated.
- Cold-row token histogram (mass placement, 9 code chunks): 22.0% of staged
  rows carry <= 2 tokens while covering only 1.0% of tokens; 37.4% carry
  <= 5 tokens covering 2.9%. That is wasted full-row staging, not misplaced
  residency.

## Baseline notes

Warmed code PP after this two-restart session: 606/595/599 then
627/635/615 tok/s (means 600 -> 625.7). The first triplet is depressed while
the PLE page cache re-warms after repeated weight loads; the earlier steady
baseline was 634/633/622 (mean 629.7), so the session is unchanged within
noise after re-warm. Take at least four warm runs after a restart-weight-load
before accepting any A/B delta under ~5%.

## Decision

Acceptance gate (fewer distinct cold rows and >= 5% cold code PP) is not
reachable in static form: the placement objective itself cannot move. Keep
mass placement and `expert_freq.pt` unchanged; nothing numerically changed,
so no oracle run is required. Promote PP7 (hybrid cold-expert execution for
low-token-count rows) to NEXT; the 22% <= 2-token row share is its size
estimate.

Raw artifacts on the host: `/root/quant/route_dump_prefill/prouting_*.pt`
(13 MB), analysis output above reproducible with
`python3 tools/pp5_presence.py /root/quant/route_dump_prefill assets/expert_freq.pt --s 184`.
