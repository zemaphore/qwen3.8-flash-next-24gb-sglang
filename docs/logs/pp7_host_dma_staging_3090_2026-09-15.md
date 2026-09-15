# PP7 — hybrid cold-expert execution: low-token GEMV rejected, host-row DMA staging accepted

Date: 2026-09-15. Host: RTX 3090, PP3 baseline (2,048-token chunks, S184,
bulk pread + 128 MiB recent-row cache, 2,048-byte gather tile).

## What was asked

Execute routed experts that carry very few staged tokens (PP5 measured 22% of
cold rows at <= 2 tokens for 1% of token mass) directly from pinned host
memory instead of staging their ~1.3 MB rows, merging the two result paths.

## Transport anatomy on this box

Before building the merge, the three host-memory read paths were measured
offline (no model):

| Path | Bytes moved | Rate |
|---|---|---:|
| `cudaMemcpyAsync`, 400 MB contiguous pinned -> device | 400 MB | **22.6-23.0 GB/s** |
| production `_gather_rows_tab_kernel`, uint8, tile 2,048 | scattered rows | 7.2 GB/s |
| same kernel, tiles 4,096 / 8,192 | " | 7.5 GB/s |
| Triton row-per-CTA int32 vector copies (512-2,048 words) | " | 6.5-7.5 GB/s |
| Triton tile int32 copies | " | 6.9-7.2 GB/s |
| pointer-table GEMV host reads (`gemv/test_gemv_tab.py`) | decode row reads | 10 GB/s |
| per-row `Tensor.copy_` loop over the same scattered pinned slots | 210 x 0.82 MB rows | **21.8-22.2 GB/s** |

The SMs cannot exceed ~7.5-10 GB/s reading pinned host memory on this
machine in any geometry (the 51 GB/s figure in `expert_gemv.py` was measured
on the Gen5 RTX PRO 4000 host, not here). The DMA engine reaches 22+ GB/s
even for row-sized scattered transfers; per-210-row submit cost is only
0.9-1.2 ms CPU, hidden under the transfer.

## Low-token direct GEMV: REJECTED

`tools/pp7_hybrid_bench.py` replays real 2,048-token code-chunk routing
(`SGLANG_PREFILL_ROUTE_DUMP` artifacts) against synthetic S184 placement with
the production kernels. The staged path stays exact by construction (stage
the non-low experts, zero their top-k weights on the excluded pairs, run the
low pairs through the existing batch-GEMV with the pointer table, merge via
an fp32 `index_add`).

Layer 0, chunk 0 (478 distinct rows, atypically wide): base 56.4 ms of which
gather is 55.5 ms; hybrid T=2 is 58.3 ms (**+3.4%, slower**) with 462 staged
rows and 24 direct pairs; max output difference 0.0039 (bf16 reassociation).
Because both paths are SM-bound at the same ~7.5-10 GB/s, direct execution
sends T x the bytes (one row read per pair versus one staged row) and only
saves the staging writeback and padded-tile compute, which measurement shows
is not enough. The premise "22% of rows waste full-row staging" is only true
if host reads are cheaper; they are not.

## Host-row DMA staging: ACCEPTED (PP7 in its productive form)

The transfer anatomy turns the same split inside out: keep staging for every
expert, but move the cold rows with `cudaMemcpyAsync` instead of SM loads.

`patches/moe_host_dma_gather.py` (SGLang `expert_stream.py` +
`expert_elastic.py`) + `patches/enable_moe_host_dma_gather.py` (launcher,
`SGLANG_MOE_GATHER_DMA`, source default off, 3090 default 1):

* `expert_elastic.add_layer` exposes the per-expert host-slot home map in
  `layer._placed["home"]`; `resize_layer` mutates the same lists in place, so
  residency stays authoritative through elastic growth and shrink.
* `ExpertStreamer.gather` dispatches to `_gather_dma` when enabled and every
  gathered kind has a home map (plain placement and qzeros kinds fall back to
  the unchanged kernel path).
* `_gather_dma`: resident rows go through one existing tab-kernel launch into
  the staging prefix; each host row is copied into the staging tail via its
  pinned-slot view with `copy_(..., non_blocking=True)`; the top-k ids are
  renumbered onto the permuted staging order through a position lookup.
  Staged bytes per expert id are identical; only row order differs. The
  fused MoE kernel is untouched. **Bit-exact by construction** (the single
  per-call `uniq.tolist()` replaces the existing `numel()` sync, so no extra
  device round trip).

Offline audit (`/tmp` check with scattered elastic-like slots): 0 row-byte
mismatches versus the production kernel staging.

## End-to-end A/B (canonical protocol)

Only `SGLANG_MOE_GATHER_DMA` differs between sides. One excluded
population/JIT request, then warmed samples.

* Variant (DMA=1, restart #2 of the session): 1096 / 1122 / 1111 / 1110 /
  1120 tok/s; mean **1111.8**, sample SD 10.6. Prefill fell 9.0 -> ~4.1 s.
* Same-session control (DMA=0, restart #3, PLE re-warming, 4+ samples per
  the PP5 caveat): 597 / 617 / 643 / 629 / 645 / 611; mean **623.7**, SD
  18.4 - inside the documented 600-636 re-warm band.
* Documented steady baselines: PP3 636.3, PP4 control 636.0, PP5 steady 629.7.

**Accepted gain: +74.8% over the PP3 baseline (+78% versus the same-session
control).** Gather was ~46 ms per layer-chunk in PP4's profile (210 host rows
at ~7.5 GB/s); the DMA path cuts the host share to ~12.5 ms. No OOM,
retraction, CUDA fault, or exception; free VRAM after the workload 3,239 MiB
(PP3: 3,241 MiB). Decode measured 38.5-39.6 tok/s in both variant and
control - unchanged at S184, consistent with the PP3 phase note.

## Oracle

* `tools/logprob_diff.py check m4_3090_untuned`:
  `LOGPROB_MAX=0.000000 LOGPROB_MEAN=0.000000` - identical to the machine
  handover pass.
* `tools/logprob_diff.py check lp2`: `LOGPROB_MAX=0.168757
  LOGPROB_MEAN=0.011632` - bit-identical to PP1b/PP2a/PP3/PP4, proving zero
  numerical change.
* `tools/greedy_diff.py check oa` diverges on 2 of 3 prompts; that reference
  predates the 3090 campaign and is not the machine-local gate - the same
  result stands for the unchanged stack (transport is byte-identical).

## Consequences for the queue

* PP2 (direct/hybrid expert execution): the pointer-table fused kernel is no
  longer needed for PP; DMA staging subsumes the transfer saving.
* PP10 (host/PCIe audit): narrowed - the link negotiates ~23 GB/s (Gen4 x16)
  and is not the binding constraint for copies; SM-read inefficiency was.
  One actionable follow-up: the decode GEMV reads pinned rows at 10 GB/s via
  the SMs; a DMA prefetch or staging of the decode expert set is a
  DECODE_PERF_PLAN candidate, not PP work.
* Resident-row staging (~120 VRAM rows per layer-chunk, ~1 ms) is now the
  visible share of the remaining gather; skipping resident copies entirely is
  a small, later optimization.
* Housekeeping: `patches/enable_moe_gather_block.py` `--check` anchors were
  widened (the overlap patch's line had been inserted between the two anchor
  lines, causing a false MISMATCH while the environment was in fact set).

Server left running: `sglang-1789491753.scope`, `SGLANG_MOE_GATHER_DMA=1`.

Raw transport experiments: `/tmp/opencode/pp7_transport.py`,
`/tmp/opencode/pp7_dma2.py`, `/tmp/opencode/pp7_dma_exact.py` (session
scratch); the reproducible hybrid bench is `tools/pp7_hybrid_bench.py`.

Post-review hardening: `gemv/test_moe_host_dma_gather.py` now permanently
covers mixed/all-host/all-resident/duplicate-id staging and a simulated elastic
resize, and passed on the 3090. The source and launcher patch helpers now use
unambiguous state detection, reject mixed states, and round-trip cleanly; their
six CPU-side regression tests pass. The PP5 placement recheck did not alter or
supersede PP7: both placement arms retained DMA staging and its exact oracles.
