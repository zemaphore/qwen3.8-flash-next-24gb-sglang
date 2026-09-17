# Promotion — main-based 3090 default (2026-09-17)

Execution-plan step 6 of
[`docs/SGLANG_MAIN_MIGRATION_PLAN.md`](../../../SGLANG_MAIN_MIGRATION_PLAN.md).

## What changed

The default 3090 launcher `/root/quant/serve-3090.sh` now runs the main-based
implementation instead of the frozen `73a255206f` control.

| | Value |
|---|---|
| New default | `/root/quant/serve-3090.sh` = `serve-3090-main.sh`, sha256 `fb72fc9566d68533021c527d9a5bc6f14e20ca3f04f1dc5fb7df20c72f49eccd`, port 30001 |
| Tree / venv | `/root/quant/sglang-main` (`port/main-3090`, HEAD `b114264c01`) / `/root/quant/venv-sglang-main` |
| Profile | radix enabled, 4 Mamba slots, 235,520-token pool, `qwen38-flash-230K`, `int8ring_int4`, S184, R1/R2/R3 on, `--cuda-graph-backend-prefill disabled` |
| Previous default (rollback) | frozen flags-on control, sha256 `d0866e7c…f57447`; local copy `/root/quant/serve-3090-frozen-rollback.sh`; repo snapshot `docs/logs/raw/promotion_3090_2026-09-16/serve-3090-flags.sh` |

Launcher snapshot: `serve-3090-main.sh` in this directory.

The candidate form `/root/quant/serve-3090-main-candidate.sh` (port 30011,
separate `SGLANG_CACHE_DIR` and elastic control) is retained for A/B or
troubleshooting.

## Verified after switching

Promoted boot (port 30001) effective profile:

```
max_total_num_tokens=235520, chunked_prefill_size=4608, max_prefill_tokens=32768,
max_running_requests=1, context_len=235520, available_gpu_mem=4.54 GB
KV lazy backing: token capacity 423488 (profiled) -> 235520 (requested 235520, safety 0.77)
KV tiers: ring R=8192 slots (int8_g64 over int4_g32)
Mamba Cache: max 4 slots; page size 64 (QSA); elastic placement 48 layers at S=184
Load weight end elapsed=185.82 s, mem usage=18.41 GB
```

Near-limit capacity on this profile (one request, before the switch, same
profile on the candidate port): **232,000 input + 512 generated, status ok,
TTFT 147.99 s, TG 27.79 tok/s, 18 MiB minimum free VRAM**
(`capacity-promoted.json`). The request exercised the R3 strict-headroom
refusal at 225,280 tokens → R1 re-sort → nothing evictable → sole-owner bypass
→ ok; 0 `OutOfMemory`/`Traceback` lines.

## Rollback

```bash
# stop the running scope, then restore the frozen launcher and start it
install -m 755 /root/quant/serve-3090-frozen-rollback.sh /root/quant/serve-3090.sh
sha256sum /root/quant/serve-3090.sh     # d0866e7c…f57447
/root/quant/serve-3090.sh
```

Source, environment and launcher roll back together: the frozen tree
`/root/sglang` and venv `/root/quant/venv-sglang` are intact, so no rebuild is
required.

## Status

The promoted main-based server is running on port 30001. Per the plan, the
remaining open items (kernel/state micro-gates, PP14 drift, untested NGRAM) are
recorded in `sglang/README.md` and the series manifest and do not affect the
default profile.
