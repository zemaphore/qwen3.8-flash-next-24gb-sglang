# PP5c raw-evidence checkpoint

Status: **complete — three captured arms, presence promoted**.

An earlier checkpoint here recorded that the execution environment had no
`/dev/nvidia*` nodes. A host-capable context then ran the full campaign on the
RTX 3090 with `/dev/nvidia*` and driver 580.178.04.

Three sequential fresh-server arms were captured under this directory, each with
its own server log and output directory:

1. [`mass-a/`](mass-a/) with `assets/expert_freq.pt`;
2. [`presence/`](presence/) with `assets/expert_presence_code.pt`;
3. [`mass-c/`](mass-c/) with `assets/expert_freq.pt`.

Each arm holds `metadata.json`, `results.json`, the unparsed `bench-*.txt`
client output for one excluded plus five measured canonical requests and one
excluded plus three measured requests for each held-out corpus, `nvidia-smi.csv`,
`elastic-status.before/after.txt`, `server-launcher.snapshot.sh`, before/after
server logs, both exactness oracles, and `SHA256SUMS`.

`sha256sum -c SHA256SUMS` verifies every file except the live `server.log`, which
kept receiving launcher output after the capture-time checksum was written; the
frozen `server.log.after.txt` verifies and is a confirmed append-only prefix of
the live log. No raw artifact was replaced by a summary.

Acceptance: canonical presence **+5.17%** over pooled mass (t = 4.68), held-out
code **-0.91%/+1.47%** (non-material), exact oracles, no errors; presence was
promoted to the general launcher default. Full analysis:
[`docs/logs/pp5c_presence_placement_3090_2026-09-15.md`](../../pp5c_presence_placement_3090_2026-09-15.md).
