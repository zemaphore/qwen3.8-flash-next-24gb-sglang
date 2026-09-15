# PLE bulk pread: code-only end-to-end validation — RTX 3090

Date: 2026-09-15

Benchmark command (the only measured workload):

```bash
SGLANG_URL=http://127.0.0.1:30001/generate \
  python3 tools/bench_agentic.py --tokens 4096 --decode-tokens 256
```

The deterministic prompt tokenized to 4,565 tokens.  `llama-benchy` and the
repeated-sentence benchmark were not run.

## Matched cold-cache A/B

Each side used this protocol:

1. Stop the previous SGLang systemd scope.
2. With no process mapping the PLE file, issue
   `POSIX_FADV_DONTNEED` for `ple.f8_e4m3.bin`.
3. Start the otherwise identical 3090 launcher and wait for its normal warmup.
4. Run one code benchmark request.

| PLE bulk pread | Prefill seconds | PP tok/s | Decode tok/s |
|---|---:|---:|---:|
| disabled (mmap control) | 13.01 | 351 | 43.3 |
| enabled (16 workers, min unique 2048) | 11.34 | 403 | 43.1 |

Cold result: **+14.8% PP**, **-1.67 s / -12.8% prefill latency**.  Decode is
unchanged within noise.

## Warm behavior

Same-day warmed mmap control samples were 479 and 462 tok/s (mean 471).
Bulk-pread warmed samples were 448, 438, 463, and 441 tok/s (mean 448).
This is a roughly **4.9% warmed regression** because resident mmap pages are
much cheaper than issuing thousands of `pread` syscalls.

Early “first after restart” samples without an explicit unmapped cache drop
were 375 tok/s (pread) and 386 tok/s (mmap).  They are not used for the cold A/B:
the PLE page cache survives process restarts, and `posix_fadvise` does not evict
pages that remain mapped in the serving process.

## Verdict

The experiment validates the storage diagnosis but is not yet a universal
default.  It materially improves genuinely cold/high-entropy code prefill and
slightly regresses a page-cache-warm repeat.  The next implementation should
select mmap versus pread using cache residency (or asynchronously prefetch
misses) rather than unique-row count alone.

No request/runtime errors occurred.  The final state is the optimized server
running with bulk pread enabled in `sglang-1789470253.scope`.
