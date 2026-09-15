# Prefill chunk/residency sweep — RTX 3090

Date: 2026-09-15

## Setup

The retained PP2a control used a 1,024-token chunk, 224 resident experts per
layer, the 128 MiB exact PLE recent-row cache, and the 2,048-byte expert-gather
tile. Its three warmed code samples averaged **505.3 tok/s**.

The only throughput workload in this sweep was:

```bash
SGLANG_URL=http://127.0.0.1:30001/generate \
  python3 tools/bench_agentic.py --tokens 4096 --decode-tokens 256
```

It contains 4,565 prompt tokens. Each candidate kept the resident set at its
184-expert floor by setting `SGLANG_MOE_ELASTIC_FILL_MB=99999`; the only other
changed variable was `SGLANG_3090_CHUNKED_PREFILL_SIZE`. The launcher's normal
64-token warmup ran first, one full code request populated the PLE/JIT caches
and was excluded, and then three identical warmed requests were measured.

## Results

| Chunk | S | Excluded PP | Warm PP samples | Mean PP | Sample SD | Mean prefill |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 control | 224 | 418 | 505 / 496 / 515 | **505.3** | 9.5 | 9.04 s |
| 1,536 | 184 | 438 | 608 / 614 / 593 | **605.0** | 10.8 | 7.55 s |
| 2,048 | 184 | 455 | 648 / 637 / 624 | **636.3** | 12.0 | 7.17 s |

The 1,536 candidate improved warmed PP by **19.7%** over PP2a. The 2,048
candidate added another **5.2%** and improved PP by **25.9%** over PP2a.

Decode throughput was 37.6 / 37.6 / 40.1 tok/s at 1,536 and 38.9 / 42.3 /
39.8 tok/s at 2,048, versus 46.0 / 46.5 / 43.7 tok/s for the S=224 PP2a
control. Keeping S=184 through decode therefore trades some decode speed for
the PP gain. Even with the 256-token decode tail included, the mean estimated
prefill-plus-decode duration improved from about 14.68 s to 13.52 s at 2,048.
A phase-aware resident-set regrow remains a separate optimization.

## Safety and exactness

| Candidate | Idle free VRAM | Post-workload free VRAM | Post-workload host memory |
|---:|---:|---:|---:|
| 1,536 / S184 | 4,621 MiB | 3,421 MiB | 52.4 GiB |
| 2,048 / S184 | 4,621 MiB | 3,241 MiB | 52.1 GiB |

Both candidates stayed well above the 1.2 GiB operational VRAM floor. No OOM,
retraction, CUDA, or workload exception appeared. The scope briefly reached
its 56 GiB `MemoryMax` during checkpoint loading because of page cache, then
reclaimed to roughly 52 GiB under workload.

The winning 2,048 variant produced:

```text
LOGPROB_MAX=0.168757 LOGPROB_MEAN=0.011632
```

This is identical to PP1b and PP2a. The mean remains marginally above the old
0.010 reference gate, an existing reference-drift audit item rather than a PP3
change.

## Verdict

Accept the 2,048-token chunk with S=184 as the PP-oriented RTX 3090 default.
It is exact relative to the current accepted stack, exceeds the requested 10%
gain, and preserves more than 3 GiB free VRAM after this workload. Retain
environment overrides so mixed or decode-heavy deployments can select a
different chunk/residency balance.
