# PP15 chunk/threshold + full-table raw evidence

All arms use presence placement, S184, chunk 4608, PP7 DMA staging, PP11
batching, gather block 2048, and the accepted PLE/KV stack unless the arm name
says otherwise. Each `*-canonical*.txt` / `c*.txt` / `ft1.txt` file holds the
raw `tools/bench_agentic.py` rows (4,565-token canonical first, then 12,000 and
30,000 target prompts).

| Path | Arm |
|---|---|
| `baseline-canonical-*.txt` | first session baseline, chunk 4608, threshold 2048 |
| `control2-canonical10.txt` | fresh 10-sample control for the PP15 A/B |
| `m64b-canonical10.txt` | 10-sample variant with the accepted M=4565 MoE config |
| `combined-canonical10.txt`, `combined-medium-long.txt`, `combined-long60000.txt` | M=4565 config + full-table + chunk 6144 (rejected combination) |
| `c8192-t2048.txt` | chunk 8192, threshold 2048 (unsafe memory) |
| `c6144-t2048.txt` | chunk 6144, threshold 2048 (medium gain, memory risk) |
| `c4608-t512*.txt` | chunk 4608, threshold 512 (regresses) |
| `ft1*.txt` | full-table gather, `SGLANG_MOE_PREFETCH_FULL_TABLE=1` |
| `control-bench_speed.txt`, `control-medium-long.txt`, `control-long-mem.csv` | control characterization and long-prompt telemetry |
| `route_dump/prouting_00001.pt`, `route_dump/prouting_00003.pt` | two canonical 48-layer real top-k captures (`sha256 01f39387...` / `4c6a4aff...`) used by the tuner |

See `docs/logs/pp15_real_routing_moe_3090_2026-09-16.md` for the analysis.
