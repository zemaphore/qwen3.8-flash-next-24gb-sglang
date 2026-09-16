# PP14: cross-layer cold-row prefetch (RTX 3090)

Date: 2026-09-16 UTC

Base commit: `b11051a` (PP13 checkpoint)

Decision: **DONE**. Cross-layer cold-row prefetch is accepted and enabled by
default for prefill chunks of at least 2,048 tokens. The launcher retains
`SGLANG_MOE_COLD_PREFETCH=0` as the immediate fallback and makes the threshold
overrideable.

## End-to-end result

Every canonical arm used presence placement, S184, chunk 4608, PP7 DMA
staging, PP11 batched submission, gather block 2048, and the accepted PLE/KV
stack. One first request per fresh server was excluded. The following five
requests were measured:

| Fresh-server arm | Canonical PP tok/s | Mean +/- sample SD | Decode mean |
|---|---|---:|---:|
| Adjacent flag-off control | 1915 / 1922 / 1912 / 1922 / 1870 | **1908.2 +/- 21.8** | 40.92 |
| PP14 prototype | 2511 / 2642 / 2633 / 2613 / 2607 | **2601.2 +/- 52.4** | 41.04 |
| Final guarded launcher default | 2470 / 2636 / 2595 / 2616 / 2600 | **2583.4 +/- 65.4** | 41.20 |

The prototype is **+36.32%** versus its adjacent control. The final guarded
default independently retains **+35.38%** versus that control. Decode is
unchanged (+0.29% prototype/control). The lower first measured PP14 sample is
retained; it is the first request that exercises all prefetch tables after the
excluded request discovers the layer streamers.

Both captured PP14 servers and the control match the established exactness
oracles: `m4_3090_untuned` is 0/0 and `lp2` is
0.168757/0.011632 (max/mean).

## Mechanism and trace

The implementation keeps one reusable full-expert device cache per tensor
kind. After layer L's fused MoE is queued, the fixed cold rows for L+1 are
copied from stable pinned slots on a dedicated stream. At L+1, the existing
table-gather kernel compacts resident and cached cold rows into the ordinary
staging layout. A release event protects cache reuse. Alternating caches were
unnecessary because the cache is no longer read after that compaction.

The profiled canonical request confirms that this is the source of the gain:

| Extend-span metric | PP13 accepted stack | PP14 | PP14 share |
|---|---:|---:|---:|
| CPU extend span | 2390.128 ms | **1743.301 ms** | -27.06% wall |
| Compute union | 1426.363 ms | 1595.895 ms | 91.54% |
| DMA union | 742.721 ms | 942.783 ms | 54.08% |
| Pinned HtoD union | 738.687 ms | 938.925 ms | 53.86% |
| Compute/DMA overlap | 10.879 ms | **901.272 ms** | **51.70%** |
| No GPU engine event | 231.923 ms | 105.895 ms | 6.07% |

PP14 intentionally transfers every cold row, so its HtoD union is larger. The
prefetch stream contains exactly 188 transfers (47 next layers x four tensor
kinds), while layer 0 retains the four ordinary selected-row transfers. The
extra bytes are profitable because 901 ms of DMA now overlaps compute.

## Memory and threshold

The four full cache tensors use 668 MB decimal (about 637 MiB). A 68,905-token
request completed at 2,473 PP tok/s and 33.5 decode tok/s with the expected
answer, no OOM/retraction, and a telemetry minimum of 1,221 MiB free VRAM.

Copying all cold rows is inappropriate for small interactive prompts. The
accepted launcher therefore defaults
`SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS=2048`; smaller chunks use the prior
selected-row DMA path and do not allocate the extra cache. A final fresh-server
smoke confirmed that the first 1,044-token request emitted zero cache-allocation
records and retained 3,639 MiB free. The subsequent first large request created
exactly four cache tensors; its following 4,565-token request reached 2,621
tok/s. Three 2,210-token probes on the final default produced
1427 / 1909 / 2078 tok/s (the first includes shape/path warmup), versus 1131 /
1499 / 1459 on the flag-off control. Three 1,044-token probes exercised the
sub-threshold path at 818 / 1016 / 1010 tok/s.

## Requested current-best-stack benchmarks

No A/B is attached to these secondary results.

`llama-benchy 0.4.0` command:

```bash
uvx llama-benchy \
  --base-url http://localhost:30001/v1 \
  --model /mnt/ai_models/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang \
  --extra-body return_token_ids=false \
  --pp 2048 --tg 256 --depth 2048 --runs 3
```

Coherence passed. `pp2048 @ d2048` measured **1218.09 +/- 42.17 tok/s** with
3368.54 +/- 113.85 ms end-to-end TTFT; `tg256 @ d2048` measured
**31.21 +/- 0.67 tok/s** (peak 31.67 +/- 0.94).

The authors' default command was run unchanged except for the 3090 endpoint:

```bash
SGLANG_URL=http://127.0.0.1:30001/generate \
  /root/quant/venv-sglang/bin/python tools/bench_speed.py
```

| Prompt tokens | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 101 | 0.35 | 292 | 42.1 |
| 421 | 0.57 | 745 | 38.2 |
| 1,701 | 1.02 | 1,669 | 35.0 |
| 6,821 | 2.72 | 2,504 | 37.6 |
| 10,001 | 4.05 | **2,468** | 38.8 |

For context, the authors' published reference results retained in
`docs/logs/tiers-validate.log` were:

| Prompt tokens | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 101 | 0.20 | 509 | 56.2 |
| 421 | 0.37 | 1,134 | 54.3 |
| 1,701 | 1.11 | 1,531 | 56.8 |
| 6,821 | 3.04 | 2,243 | 55.7 |
| 10,001 | 4.40 | **2,271** | 54.5 |

At 10,001 prompt tokens, the current PP14 result is 8.7% above the authors'
published prefill figure (2,468 versus 2,271 tok/s), while decode is lower
(38.8 versus 54.5 tok/s). This is context rather than a controlled A/B: the
published reference and this run used different GPUs and host state.

## Safety and retained evidence

All three arm manifests verify. Server logs contain no OOM, CUDA fault,
traceback, retraction, or request failure beyond the known harmless optional
model import warning. The final launcher and live capture both contain the
accepted flag and 2,048-token floor. The server scopes were stopped after the
measurements; the GPU was idle with 24,124 MiB free.

Evidence: [raw directory](raw/pp14_cross_layer_prefetch_3090_2026-09-16/).
