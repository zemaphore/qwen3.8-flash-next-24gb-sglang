# PP13: current-stack profile and kernel screens (RTX 3090)

Date: 2026-09-16 UTC

Base commit: `7717f31` (accepted PP12)

Decision: **REJECTED** for both tested kernel paths; launcher and production
assets unchanged.

## Result

A fresh accepted-stack server (presence placement, S184, chunk 4608, PP7 DMA,
PP11 batching, gather block 2048 and the accepted PLE settings) measured:

| Arm | Measured canonical PP tok/s | Mean +/- sample SD | Decode mean |
|---|---|---:|---:|
| current A | 1977 / 1968 / 1974 / 1958 / 1964 | **1968.2 +/- 7.6** | 44.46 |
| tuned high-M MoE | 1840 / 1815 / 1841 / 1842 / 1816 | **1830.8 +/- 14.0** | 41.68 |
| current C | 1920 / 1924 / 1933 / 1930 / 1923 | **1926.0 +/- 5.3** | 41.96 |
| pooled current | ten samples | **1947.1 +/- 23.1** | 43.21 |

The tuned arm is **-4.94%** versus its adjacent current-C control and **-5.97%**
versus pooled current. The earlier current-A decode rate was also higher, but
the adjacent tuned/current-C decode comparison is only -0.67%; thermal/clock
drift therefore does not explain the PP regression. Both captured arms match
the established exactness oracles (`m4_3090_untuned` 0/0 and `lp2`
0.168757/0.011632), and their server logs contain no OOM, retraction, CUDA
fault, request failure or traceback.

## Fresh profile

The profiled request is excluded from the table because collection lowered it
to 1848 tok/s. Inside its 2,390.128 ms CPU extend span, clipped GPU-event unions
were:

| Component | Union / total | Share of span |
|---|---:|---:|
| Any GPU engine | 2158.204 ms | 90.30% |
| Compute | 1426.363 ms | 59.68% |
| DMA | 742.721 ms | 31.07% |
| Pinned HtoD | 738.687 ms | 30.91% |
| Compute/DMA overlap | 10.879 ms | 0.46% |
| GPU idle | 231.923 ms | 9.70% |

Named kernels contribute 481.609 ms of routed INT2 MoE and 421.536 ms of
Marlin. The important remaining shape is therefore not an idle GPU or another
small CPU submission loop: substantial HtoD and compute work are both present
but almost completely serialized.

## GDN format screen

`tools/pp13_gdn_format_bench.py` reconstructed the real layer-0 W8 GDN
projections. At M=4565, dequantized BF16 improves `in_proj_qkvz` from 7.102 to
5.894 ms, but costs another 39.4 MiB per layer (about **1.42 GiB** across 36
GDN layers). Its ideal prompt saving is only about **43.5 ms / 1.8%**; dense
`out_proj` is slightly slower than Marlin. FP16 with boundary casts does not
win, while W8A8 is slower at M=2048, unsupported by cuBLAS at M=4565, and has
materially larger probe error. This path is rejected without a server A/B.

## Routed-MoE config screen

The synthetic uniform-route tuner found larger M=4565 tiles that looked better
than the production nearest M1024 config:

| Compact experts | Production-reference ms | Synthetic best ms | Kernel reduction |
|---:|---:|---:|---:|
| 432 | 11.984 | 9.284 | 22.53% |
| 480 | 12.647 | 10.076 | 20.33% |
| 512 | 12.027 | 10.648 | 11.46% |

The tuned files added exact E432/E480/E512 buckets with M4565 entries only in
an isolated asset tree. Server logs confirm those buckets were selected for the
actual 405-502-expert prefill layers. Despite the synthetic gains, the real
server regressed as shown above. The most likely cause is the tuner's uniform
random token-to-expert distribution: large 128-row tiles reduce launch count in
that synthetic distribution but waste padding on the real, skewed routing.
Do not promote these configs or continue synthetic-only config tuning.

## Next practical target

The profile supports one material next experiment: an opt-in double-buffered
cross-layer cold-row prefetch. While layer L computes, copy all cold rows for
layer L+1 into an alternate device cache; after L+1 routing is known, compact
only selected cold rows device-to-device and gather selected resident rows as
today. This replaces the route-dependent serialized HtoD wait with an early
fixed-order HtoD and a cheap post-route DtoD compact. It costs roughly one extra
full staging buffer (~0.6-0.7 GiB), which fits the measured margin but must pass
the existing long-prompt memory check. The 739 ms HtoD union and only 11 ms of
current overlap make its ceiling materially larger than either rejected PP13
screen.

Evidence: [raw directory](raw/pp13_current_profile_3090_2026-09-16/).
