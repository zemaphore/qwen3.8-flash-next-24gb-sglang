# Step 5d — pressure churn across boots (candidate only)

Date: 2026-09-16 21:44–22:00 UTC. Execution-plan step 5d of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md),
run on the main-based candidate only (the frozen control was not exercised).

Profile: the RC1 pressure reproducer on the main port —
`/root/quant/serve-3090-main-pressure.sh` (16 Mamba slots, chunk 2,048,
262,144 requested pool, R1/R2/R3 on, floor `SGLANG_KV_LAZY_MIN_FREE_MB=64`,
`--mamba-radix-cache-strategy extra_buffer`, port 30011). Effective:
profiled 243,136 → admitted 187,214 (`max_total_num_tokens=187200`).

Reproducer: the RC1-d crash workload — 30 unique ~6.5K conversations, each cold
then warm (~196K retained on the ~187K pool), then evict/rehit with 50 fillers.

## Results

| Boot | churn | retained | min free | R1 | R3 | R2 | errors |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 30/30 cold, 30/30 warm | ~195,860 | 2 MiB | 60 | 56 | 0 | 0 |
| 2 | 30/30 cold, 30/30 warm | ~195,860 | 2 MiB | 14 | 14 | 0 | 0 |

Boot 1 evict/rehit: A 6,597 tokens; a1 hit 6,592 after 12 short fillers; a2 miss
after 50 fillers (333,290 tokens, > pool) with free VRAM 16 MiB; a3 rehit
6,592 in 0.579 s. Server healthy (HTTP 200) after. 0
`OutOfMemory`/`out of memory`/`Traceback` lines in either boot.

The server survived the churn reproducer on both boots: R3 refused commits below
the 64 MiB floor, R1 re-sorted and evicted LRU prefixes (or bypassed the floor
when nothing was evictable), and R2 was never needed (0 aborts). This is the
final R1 v3 / floor-64 behaviour repeated across two boots, which the RB stage
had left as an open item even for the frozen control.

## Evidence

`checks_candidate_pressure/` (boot 1: `churn.*`, `evict_rehit.*`, `server-boot1.log`)
and `checks_candidate_pressure_boot2/` (`churn.*`, `server-boot2.log`); launcher
`serve-3090-main-pressure.sh`. Near-limit capacity is separately covered by the
TG0 redo capacity smoke (257,456 tokens, status ok).
