#!/usr/bin/env python3
"""R2: abort the request, not the server, when prefill allocation fails.

``ScheduleBatch.prepare_for_extend`` -> ``alloc_for_extend`` raises
``RuntimeError("Prefill out of memory ...")`` when the KV allocator returns
``None`` (RC1-d: lazy backing refused under physical pressure). The exception
escapes ``Scheduler._get_new_batch_prefill_raw`` and the scheduler process
exits (SIGQUIT, whole tree killed).

This patch wraps ``new_batch.prepare_for_extend()``. On a prefill-OOM
``RuntimeError`` it rolls back the *fresh* requests of the batch (those with
no committed KV: ``req.kv is None``) — frees their mamba slot and ping-pong
buffer, returns the request slot, drops the prefix lock taken by the
``PrefillAdder`` — sends each an ``AbortReq`` with HTTP 503, clears
``self.chunked_req`` if it was one of them, and returns "no prefill batch"
so the scheduler keeps serving. A batch containing a request with already
committed KV (a continuing chunked prefill) is not rolled back in this
version: the original exception is re-raised.

Env: ``SGLANG_PREFILL_ALLOC_ABORT`` (default ``0`` = original behaviour).

Target: ``$SGLANG/python/sglang/srt/managers/scheduler.py``.
Usage: ``apply`` / ``revert`` / ``--check``.
"""

from __future__ import annotations

import os
import sys

SG = os.environ.get("SGLANG", "/root/sglang")
TARGET = os.path.join(SG, "python/sglang/srt/managers/scheduler.py")

OLD = '''        new_batch.prepare_for_extend()

        if self.tp_worker.model_runner.prefill_aware_swa:
'''

NEW = '''        try:
            new_batch.prepare_for_extend()
        except RuntimeError as _alloc_ex:
            # R2 (prefill_alloc_abort): fail the request(s), not the server.
            if not _PREFILL_ALLOC_ABORT or "out of memory" not in str(_alloc_ex):
                raise
            if any(r.kv is not None for r in can_run_list):
                logger.error("prefill allocation failed with committed KV in the batch; not recoverable here: %s", _alloc_ex)
                raise
            logger.error("prefill allocation failed; aborting %d request(s) instead of exiting: %s",
                         len(can_run_list), str(_alloc_ex).splitlines()[0])
            for _req in can_run_list:
                try:
                    if _req.mamba_pool_idx is not None and _req.req_pool_idx is not None:
                        self.req_to_token_pool.free_mamba_cache(_req)
                    if _req.req_pool_idx is not None:
                        self.req_to_token_pool.free(_req)
                    if getattr(_req, "last_node", None) is not None:
                        self.tree_cache.dec_lock_ref(_req.last_node)
                except Exception as _rb:
                    logger.error("rollback failed for %s: %s", _req.rid, _rb)
                _reason = FINISH_ABORT(
                    "Prefill out of memory under KV pressure; request aborted.",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
                _req.to_finish = _reason
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(finished_reason=_reason.to_json(), rid=_req.rid), _req
                )
            if self.chunked_req is not None and self.chunked_req in can_run_list:
                self.chunked_req.inflight_middle_chunks = max(0, self.chunked_req.inflight_middle_chunks - 1)
                self.chunked_req = None
            return None, running_batch

        if self.tp_worker.model_runner.prefill_aware_swa:
'''

OLD_FLAG = "from http import HTTPStatus\n"
NEW_FLAG = (
    "from http import HTTPStatus\n"
    '_PREFILL_ALLOC_ABORT = __import__("os").environ.get("SGLANG_PREFILL_ALLOC_ABORT", "0") == "1"\n'
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
        print(f"  {s.upper():<8} {TARGET}: R2 abort request on prefill allocation failure")
        if s == "mismatch":
            raise SystemExit(1)
        return
    t = read()
    if cmd == "apply":
        if s == "applied":
            print("already applied"); return
        t = replace_once(t, OLD, NEW)
        t = replace_once(t, OLD_FLAG, NEW_FLAG)
    elif cmd == "revert":
        if s == "clean":
            print("already clean"); return
        t = replace_once(t, NEW, OLD)
        t = replace_once(t, NEW_FLAG, OLD_FLAG)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} apply|revert|--check")
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(t)
    print("reverted" if cmd == "revert" else "applied")


if __name__ == "__main__":
    main()
