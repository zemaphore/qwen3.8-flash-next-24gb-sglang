#!/usr/bin/env python3
"""R1: evict retained prefix pages when lazy KV backing refuses a commit.

Mechanism (RC1-d, docs/logs/rc1_gpu_hybrid_3090_2026-09-16.md section 3.1):
``alloc_paged_token_slots_extend`` evicts only for the *logical* shortfall,
then hands out the lowest free pages. With a retained radix tree the free
list can be dominated by high-index pages that were never physically backed;
the lazy hook (``allocator/paged.py::_lazy_hook`` -> ``ensure_prefix``) then
fails against the driver and the allocator returns ``None``, which
``allocation.py`` turns into ``RuntimeError("Prefill out of memory")`` and
kills the scheduler, even with >100K evictable, already-backed pages in the
tree.

This patch retries once: on ``None`` it evicts ``num_tokens`` from the tree
(LRU, which returns backed low-index pages), merges and sorts the free list so
those pages come first, and calls ``alloc_extend`` again. Only if that also
fails does it raise as before.

Env: ``SGLANG_KV_EVICT_ON_PRESSURE`` (default ``0`` = original behaviour).

Target: ``$SGLANG/python/sglang/srt/mem_cache/allocation.py``.
Usage: ``apply`` / ``revert`` / ``--check``.
"""

from __future__ import annotations

import os
import sys

SG = os.environ.get("SGLANG", "/root/sglang")
TARGET = os.path.join(SG, "python/sglang/srt/mem_cache/allocation.py")

OLD = '''    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\\n"
'''

NEW = '''    if is_dsv4:
        bundle = out
        out_cache_loc = None if bundle is None else bundle.out_full_loc
        if batch is not None:
            batch.out_cache_loc_dsv4 = bundle
    else:
        out_cache_loc = out

    # R1 (kv_evict_on_physical_pressure): a None here means the lazy backing
    # refused to commit never-backed high-index pages. First re-sort the free
    # list so already-backed low-index pages are handed out (they need no
    # commit); if that is not enough, evict LRU prefixes (their pages are
    # backed too), re-sort, and retry once more.
    if out_cache_loc is None and not is_dsv4 and _EVICT_ON_PRESSURE and tree_cache is not None:
        allocator.merge_and_sort_free()
        out_cache_loc = allocator.alloc_extend(
            prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens, **extra_alloc_kwargs,
        )
        logger.warning("KV physical pressure: re-sorted free pages and retried allocation of %d tokens -> %s",
                       extend_num_tokens, "ok" if out_cache_loc is not None else "still refused")
        if out_cache_loc is None:
            evictable = 0 if tree_cache.is_chunk_cache() else tree_cache.evictable_size()
            if evictable > 0:
                evict_n = min(evictable, max(num_tokens, 4 * num_tokens))
                tree_cache.evict(EvictParams(num_tokens=evict_n))
                allocator.merge_and_sort_free()
                out_cache_loc = allocator.alloc_extend(
                    prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens, **extra_alloc_kwargs,
                )
                logger.warning("KV physical pressure: evicted up to %d retained tokens and retried allocation of %d tokens -> %s",
                               evict_n, extend_num_tokens, "ok" if out_cache_loc is not None else "still refused")
            else:
                # Sole owner of the pool: nothing to evict, so the strict floor (R3) would only
                # fail this request. Bypass it for one retry = the unpatched profile's behaviour.
                _kv = getattr(allocator, "_kvcache", None)
                _kv = getattr(_kv, "full_kv_pool", _kv)
                if _kv is not None and hasattr(_kv, "lazy_ensure"):
                    _kv._strict_bypass = True
                    try:
                        out_cache_loc = allocator.alloc_extend(
                            prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens, **extra_alloc_kwargs,
                        )
                    finally:
                        _kv._strict_bypass = False
                    logger.warning("KV physical pressure: nothing evictable, strict floor bypassed for %d tokens -> %s",
                                   extend_num_tokens, "ok" if out_cache_loc is not None else "still refused")

    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\\n"
'''

OLD_IMPORT = "logger = logging.getLogger(__name__)\n"
NEW_IMPORT = (
    "logger = logging.getLogger(__name__)\n"
    '_EVICT_ON_PRESSURE = os.environ.get("SGLANG_KV_EVICT_ON_PRESSURE", "0") == "1"\n'
)
OLD_OS = "import logging\n"
NEW_OS = "import logging\nimport os\n"


def read() -> str:
    with open(TARGET, encoding="utf-8") as f:
        return f.read()


def state() -> str:
    t = read()
    if NEW in t and NEW_IMPORT in t:
        return "applied"
    if OLD in t and NEW_IMPORT not in t:
        return "clean"
    return "mismatch"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one anchor: {old[:60]!r}")
    return text.replace(old, new)


def apply() -> None:
    if state() == "applied":
        print("already applied")
        return
    t = read()
    t = replace_once(t, OLD, NEW)
    t = replace_once(t, OLD_IMPORT, NEW_IMPORT)
    if "\nimport os\n" not in t:
        t = replace_once(t, OLD_OS, NEW_OS)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(t)
    print("applied")


def revert() -> None:
    if state() == "clean":
        print("already clean")
        return
    t = read()
    t = replace_once(t, NEW, OLD)
    t = replace_once(t, NEW_IMPORT, OLD_IMPORT)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(t)
    print("reverted")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if cmd == "--check":
        s = state()
        print(f"  {s.upper():<8} {TARGET}: R1 evict on physical pressure")
        if s == "mismatch":
            raise SystemExit(1)
    elif cmd == "apply":
        apply()
    elif cmd == "revert":
        revert()
    else:
        raise SystemExit(f"usage: {sys.argv[0]} apply|revert|--check")


if __name__ == "__main__":
    main()
