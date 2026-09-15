# PLE recent-row cache: code-only end-to-end validation — RTX 3090

Date: 2026-09-15

The only throughput workload was:

```bash
SGLANG_URL=http://127.0.0.1:30001/generate \
  python3 tools/bench_agentic.py --tokens 4096 --decode-tokens 256
```

It tokenized to 4,565 prompt tokens and ran in five prefill chunks.  Benchy was
not run.

## Change

`patches/ple_recent_row_cache.py` adds an environment-controlled cache of the
deduplicated FP8 rows returned by the existing parallel-pread path.  Entries are
exact chunk working sets, retained up to
`SGLANG_QWEN4_PLE_RECENT_CACHE_MB`; decode remains on its prior path.  The
accepted launcher default is 128 MiB and profiling defaults off.

The cache stores both the sorted row IDs and their original 160-byte FP8 rows.
Each use applies the current inverse index, inside/outside-shard mask, FP8 to
BF16 conversion, scale, and device copy exactly as the miss path does.

## Same-day warmed control

Before applying PP1b, the already-warmed bulk-pread server produced:

| Sample | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 1 | 10.54 | 433 | 43.6 |
| 2 | 10.48 | 435 | 45.7 |
| 3 | 10.32 | 442 | 43.1 |

Mean PP was **436.7 tok/s**, sample standard deviation 4.7.

## Diagnostic run

After an unmapped `POSIX_FADV_DONTNEED`, PP1b was launched with a 128 MiB cap
and `SGLANG_QWEN4_PLE_PROFILE=1`.

- Cold code prompt: 11.40 s, **400 PP tok/s**, 43.8 decode tok/s.
- Its five chunks used pread, with 10,320 / 9,080 / 7,896 / 7,264 / 3,912
  unique rows.  CPU gather times were 304.5 / 198.0 / 178.3 / 146.7 / 84.3
  ms, and retained payload plus IDs totaled only 6.2 MiB.
- The repeated prompt used the recent cache for all five chunks.  CPU gather
  times fell to 4.7 / 2.0 / 1.5 / 1.4 / 0.7 ms; e2e PP was 469 tok/s with
  profiling still enabled.

## Clean acceptance run

The server was stopped again, the unmapped PLE file received
`POSIX_FADV_DONTNEED`, and the same build was started with profiling disabled.

Cold/population result:

| Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|
| 11.54 | **396** | 43.9 |

The two PP1b cold observations, 396 and 400 tok/s, are within 1.7% of PP1's
matched 403 tok/s cold result.  No meaningful cold regression was observed.

Three subsequent warmed samples, with no state or configuration changes:

| Sample | Prefill seconds | PP tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 1 | 9.50 | 481 | 44.1 |
| 2 | 9.37 | 487 | 43.1 |
| 3 | 9.37 | 487 | 42.1 |

Mean PP was **485.0 tok/s**, sample standard deviation 3.5.  This is **+11.1%**
over the same-day bulk-pread control and **+3.0%** over the earlier warmed mmap
mean of 471 tok/s.  Mean prefill latency fell from 10.45 to 9.41 seconds.

## Exactness and safety

The established `lp2` oracle reported `LOGPROB_MAX=0.168757` and
`LOGPROB_MEAN=0.011632`.  MAX passes its 0.5 gate; MEAN is marginally above the
old 0.010 gate.  Importantly, the first cache-miss oracle pass and the immediate
all-cache-hit repeat returned exactly the same per-prompt deltas and aggregate
values.  The byte-exact cache therefore introduced no observable numerical
change; the old-reference mean drift applies to the current baseline as well
and should be audited separately rather than attributed to PP1b.

There were no PLE errors, tracebacks, OOMs, or memory-leak reports.  Minimum
observed free GPU memory after the workload was 1,325 MiB, above the plan's
1.2 GiB floor.  The log contains the already-known CUDA VMM FABRIC permission
fallback, which successfully used its existing fallback path.

## Verdict

Keep PP1b enabled at a bounded 128 MiB.  It preserves the parallel-pread cold
behavior and removes the warmed repeated-chunk regression.  Exact chunk hits
are deliberately conservative; partial-overlap caching or asynchronous miss
prefetch remains possible future work, but PP2 is now the next controlled PP
experiment.

Final server state: PP1b active with profiling disabled on port 30001 in
`sglang-1789476015.scope`; `--sleep-on-idle` is also active.

