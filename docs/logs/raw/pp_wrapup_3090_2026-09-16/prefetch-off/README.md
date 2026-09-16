# prefetch-off arm (non-promotion validation)

Accepted stack with `SGLANG_MOE_COLD_PREFETCH=0`: the PP14 cross-layer cold-row
prefetch is disabled, so every 4,565- and 4,608-token chunk uses the prior
selected-row path. The server log contains zero "cross-layer cold-row prefetch
active" lines. `oracle-check.txt` is run 1; the JSON repeats run1/run2 are in
`../long-oracle/`. `server.json` is the live server snapshot.
