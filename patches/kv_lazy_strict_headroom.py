#!/usr/bin/env python3
"""R3: make the lazy-commit headroom watermark unconditional.

``MambaPool/KV lazy_ensure`` (memory_pool.py) keeps ``SGLANG_KV_LAZY_HEADROOM_MB``
(default 1536) driver-free after each commit by shrinking the elastic expert
cache, and raises when less than half of it would remain. Both the shrink and
the raise sit inside ``if can_shrink or now - last > 30.0:``, a rate limit
meant to avoid ``empty_cache()`` storms. Once the expert cache is at its floor
(``can_shrink`` False) and a shrink happened less than 30 s ago, the whole
block is skipped and the commit proceeds with no headroom check at all. That
is how RC1-d's 3072 arm committed at ``driver free 0.03 GB`` and then hit a
36 MiB activation OOM (docs/logs/rc1_gpu_hybrid_3090_2026-09-16.md section 3.1).

This patch evaluates the half-headroom refusal outside the rate limit, so a
commit that would leave < headroom/2 free is always refused. The refusal
surfaces as the allocator's ``None``; with R1 applied that becomes tree
eviction and a retry instead of a crash.

Env: ``SGLANG_KV_LAZY_STRICT_HEADROOM`` (default ``0`` = original behaviour).

Target: ``$SGLANG/python/sglang/srt/mem_cache/memory_pool.py``.
Usage: ``apply`` / ``revert`` / ``--check``.
"""

from __future__ import annotations

import os
import sys

SG = os.environ.get("SGLANG", "/root/sglang")
TARGET = os.path.join(SG, "python/sglang/srt/mem_cache/memory_pool.py")

OLD = '''                if torch.cuda.mem_get_info()[0] - delta < headroom // 2:
                    raise RuntimeError(f"KV lazy backing: no headroom for {want} tokens "
                                       f"(free {torch.cuda.mem_get_info()[0] >> 20} MB, need {(delta + headroom) >> 20} MB)")
        try:
'''

NEW = '''                if torch.cuda.mem_get_info()[0] - delta < headroom // 2:
                    raise RuntimeError(f"KV lazy backing: no headroom for {want} tokens "
                                       f"(free {torch.cuda.mem_get_info()[0] >> 20} MB, need {(delta + headroom) >> 20} MB)")
            # R3 (kv_lazy_strict_headroom): the refusal above is skipped by the rate limit once the
            # expert cache is at its floor; re-check unconditionally so a commit never leaves < headroom/2.
            if self._STRICT_HEADROOM and torch.cuda.mem_get_info()[0] - delta < headroom // 2:
                raise RuntimeError(f"KV lazy backing: strict headroom refused {want} tokens "
                                   f"(free {torch.cuda.mem_get_info()[0] >> 20} MB, need {(delta + headroom // 2) >> 20} MB)")
        try:
'''

OLD_FLAG = "    def lazy_ensure(self, num_tokens: int) -> None:\n"
NEW_FLAG = (
    '    _STRICT_HEADROOM = os.environ.get("SGLANG_KV_LAZY_STRICT_HEADROOM", "0") == "1"\n\n'
    "    def lazy_ensure(self, num_tokens: int) -> None:\n"
)


def read() -> str:
    with open(TARGET, encoding="utf-8") as f:
        return f.read()


def state() -> str:
    t = read()
    if NEW in t and NEW_FLAG in t:
        return "applied"
    if OLD in t and NEW_FLAG not in t:
        return "clean"
    return "mismatch"


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one anchor: {old[:60]!r}")
    return text.replace(old, new)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--check"
    s = state()
    if cmd == "--check":
        print(f"  {s.upper():<8} {TARGET}: R3 strict lazy-commit headroom")
        if s == "mismatch":
            raise SystemExit(1)
        return
    t = read()
    if cmd == "apply":
        if s == "applied":
            print("already applied"); return
        t = replace_once(t, OLD, NEW)
        t = replace_once(t, OLD_FLAG, NEW_FLAG)
        if "\nimport os\n" not in t:
            raise RuntimeError("memory_pool.py has no 'import os'; add it before applying")
    elif cmd == "revert":
        if s == "clean":
            print("already clean"); return
        t = replace_once(t, NEW, OLD)
        t = replace_once(t, NEW_FLAG, OLD_FLAG)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} apply|revert|--check")
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(t)
    print(cmd + "ed" if cmd == "revert" else "applied")


if __name__ == "__main__":
    main()
