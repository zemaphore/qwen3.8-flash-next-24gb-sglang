# PP5 — corrected presence placement and PP7 E2E recheck: REJECTED below gate

Date: 2026-09-15. Host: RTX 3090. Stack: accepted PP7 host-row DMA staging,
2,048-token chunks, S184, bulk PLE pread plus 128 MiB recent-row cache. Only
the canonical code benchmark was used; no repeated-sentence benchmark or
`llama-benchy` was run.

## Audit and repair

The original `tools/pp5_presence.py` result was invalid for two independent
reasons:

- `--skip 2` retained the launcher's M=6 and M=3 warmups. The claimed nine
  chunks were actually eleven; their mean was the documented ~329 distinct
  experts/layer-chunk. The nine canonical chunks average 395.16 (426.19 for
  the six full 2,048-token chunks).
- `zip(routed, res)` paired chunks with resident layer sets, then tested all
  48 layers in a chunk against that one set. It did not compare layer L's
  routing with layer L's residents.

The repaired tool defaults to `--min-tokens 64`, reports its admitted size
histogram, performs layer-matched counting, uses stable routing mass as the
secondary key for tied presence counts, and can emit the exact placement
asset. Regression tests cover both the layer-axis error and warmup filtering.

Reproduction:

```bash
/root/quant/venv-sglang/bin/python tools/pp5_presence.py \
  /root/quant/route_dump_prefill assets/expert_freq.pt \
  --s 184 --min-tokens 64 --write-presence assets/expert_presence_code.pt
```

The generated `assets/expert_presence_code.pt` is presence-primary with
routing-mass tie-breaking (SHA-256
`73d8f0df1875f6b61124b4835173dccf56d57792fa91bee8d270fc65b42ddfd8`).

## Corrected static analysis

Nine canonical chunks were admitted: six M=2048 plus three M=469.

| Ranking | cold rows/layer-chunk | routing-mass cover | overlap with mass S184 |
|---|---:|---:|---:|
| mass | 236.68 | 84.3% | 8,832 / 8,832 |
| presence then mass | **211.16** | 66.0% | 6,313 / 8,832 |
| token count then mass | 213.58 | 47.4% | 4,123 / 8,832 |

Presence reduces cold rows by **10.8%** overall. Split by chunk shape, it
reduces full-chunk rows 258.23 -> 242.19 (-6.2%) and tail rows 193.59 ->
149.08 (-23.0%). This overturned the original “objective saturated” premise
and justified an E2E test, but it did not itself satisfy the performance gate.

## Controlled PP7 E2E A/B

The only functional variable was `SGLANG_MOE_PLACEMENT`; DMA staging stayed
enabled. Every arm used one excluded population/cache-warm request followed by
five measured code requests (4,565 actual prompt tokens, 256 decode tokens).
Mass controls bracketed the candidate to expose restart/cache drift.

| Arm | scope | excluded | measured PP tok/s | mean | sample SD |
|---|---|---:|---|---:|---:|
| mass control A | `sglang-1789491753.scope` | 1096 | 1107 / 1125 / 1129 / 1100 / 1105 | 1113.2 | 12.9 |
| presence candidate | `sglang-1789495945.scope` | 767 | 1122 / 1155 / 1161 / 1121 / 1165 | **1144.8** | 21.6 |
| mass control B | `sglang-1789496308.scope` | 785 | 1124 / 1087 / 1112 / 1121 / 1090 | 1106.8 | 17.3 |

Pooled mass mean: **1110.0 tok/s**, sample SD 14.8. Presence is **+3.1%**
versus pooled mass (+2.8% versus control A, +3.4% versus control B). The
direction agrees with the corrected row analysis, but the result is below the
predeclared 5% adoption gate.

Decode did not regress: presence mean 39.94 tok/s versus pooled mass 38.65.
The presence arm had no CUDA fault, OOM, exception, or retraction; post-oracle
free VRAM was 2,941 MiB. The restored mass server has 3,239 MiB free.

## Exactness and regression tests

- `m4_3090_untuned`: `LOGPROB_MAX=0.000000 LOGPROB_MEAN=0.000000`.
- `lp2`: `LOGPROB_MAX=0.168757 LOGPROB_MEAN=0.011632`, identical to PP7 and
  the earlier accepted stack.
- Six CPU-side PP5/patch-state regression tests passed.
- `gemv/test_moe_host_dma_gather.py` passed on the 3090 while the server was
  stopped. It covers mixed, all-host, all-resident, duplicate-id, and simulated
  elastic-resize staging.

## Decision

**REJECTED as the static default:** the measured +3.1% does not clear the 5%
gate. Keep `assets/expert_freq.pt` (routing mass) in the launcher. Preserve the
corrected tool, candidate asset, and tests so this result is reproducible.
PP7 remains accepted and unchanged. Do not start PP11 as part of this recheck.

Server left running on the restored mass default:
`sglang-1789496308.scope`, `SGLANG_MOE_GATHER_DMA=1`, S184.
