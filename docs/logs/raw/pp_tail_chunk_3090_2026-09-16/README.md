# PP12 raw evidence — tail-chunk staging amortization

Date: 2026-09-16. Acceptance: chunk size 4,608 promoted over 2,048 (+50.66%
warmed PP on the canonical prompt, exact oracles). Full analysis:
[`docs/logs/pp12_tail_chunk_3090_2026-09-16.md`](../../pp12_tail_chunk_3090_2026-09-16.md).

| Directory | Contents |
|---|---|
| `ctrl-a-2048/` | Fresh-server 2,048 control captured with `tools/capture_pp5b_arm.py --skip-heldout` |
| `var-4608/` | Fresh-server 4,608 variant, same capture |
| `ctrl-b-2048/` | Fresh-server 2,048 control B, same capture |
| `probe-4608/` | Pre-bracket feasibility requests on one 4,608 server |
| `safety-4608/` | Long-prompt (4.6k/9.5k/20.3k/29.3k tokens) safety pass on a 4,608 server |

Each arm directory holds `metadata.json`, `results.json`, raw `bench-*.txt`
client output, `nvidia-smi.csv`, elastic status before/after, the launcher
snapshot, before/after server logs, both exactness oracles, and `SHA256SUMS`.
No raw artifact was replaced by a summary.
