# PP15 real-routing MoE raw evidence

- `accept-m64-qsa/` is the same capture with the opt-in QSA host-lens patch
  active (`SGLANG_QSA_PREFILL_HOST_LENS=1`); its 5-sample canonical mean is
  2657.0, inside the noise of `accept-m64`'s 2687.4, and the pooled 22-sample
  comparison shows no effect. Kept for the audit.
- `accept-m64-l0/` is the same capture with the opt-in first-layer prefetch
  active (`SGLANG_MOE_PREFETCH_FIRST_LAYER=1`); its 5-sample canonical mean is
  2681.8, which does not separate from `accept-m64`'s 2687.4. Kept for the audit.
- `accept-m64/` is the accepted-stack canonical capture
  (`tools/capture_pp5b_arm.py --skip-heldout --chunk-size 4608`): one excluded
  request, five measured requests, both exactness oracles, telemetry,
  environment, launcher and status snapshots, and frozen server logs. The arm
  manifest verifies (`sha256sum -c SHA256SUMS` in that directory).
- `benchy-accept-m64.txt` is the required `llama-benchy 0.4.0`
  `pp 2048 --tg 256 --depth 2048 --runs 3` run on that server.
- `author-default-bench-accept-m64.txt` is the required unmodified
  `tools/bench_speed.py` run on that server.

The MoE config entries added by PP15 live in
`assets/moe_configs/configs/triton_3_7_1/E={384,432},...,dtype=int2_w2a16{,_down}.json`
at key `"4565"`. The real-routing tuning reports are
`/root/quant/logs/moe_int2_real_tuning{,_file2,_deep}.json`; the tuner is
`tools/tune_moe_int2_real.py` and the routing captures are in
`../pp15_prefill_retune_3090_2026-09-16/route_dump/`.
