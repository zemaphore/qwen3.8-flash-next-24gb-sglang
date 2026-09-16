#!/usr/bin/env python3
"""Avoid the per-layer device sync in the QSA prefill metadata gather.

``QSAIndexerMetadata.get_prefill_mqa_inputs`` runs once per QSA layer (12 in
this model) and calls ``sequence_lengths.tolist()`` on a CUDA tensor purely to
get each sequence's block count on the host for a Python slicing loop.  The
result is data-independent per forward and the host copy ``seq_lens_cpu`` is
already carried by ``ForwardBatch``, so the round trip only forces a
``cudaStreamSynchronize`` (the canonical PP14 trace attributes 186 ms / 15.5 ms
per layer to ``tolist`` inside ``_compute_qsa_topk_indices``).

This opt-in patch threads ``forward_batch.seq_lens_cpu`` into the metadata and
uses it for the loop, falling back to the device ``tolist`` whenever the host
copy is missing or has a different row count.  The gathered compressed keys,
row ranges and returned ``sequence_lengths`` are unchanged.

Status: made opt-in default-off after an e2e A/B.  A 10-sample arm measured
2704.9 +/- 15.7 versus the M64-only 2679.8 +/- 23.5 (t~2.5), but pooling all
22 QSA samples (2685.1) against all 21 M64-only samples (2687.0) shows no
effect: the sync was already hidden behind queued GPU work.  The exactness
oracles are unchanged.  See docs/logs/pp15_real_routing_moe_3090_2026-09-16.md.

Usage:
  SGLANG=/root/sglang python3 patches/qsa_prefill_host_lens.py --check | apply | revert
"""

from __future__ import annotations

import os
import sys


SG = os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang"))
METADATA = os.path.join(
    SG, "python/sglang/srt/layers/attention/qsa/metadata.py"
)
BACKEND = os.path.join(
    SG, "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
)

BEFORE_FIELD = '''    graph_ring_group_locs: Optional[torch.Tensor] = None

    def get_seqlens_int32(self) -> torch.Tensor:
'''

AFTER_FIELD = '''    graph_ring_group_locs: Optional[torch.Tensor] = None
    # Host copy of the per-sequence lengths when available; avoids a
    # cudaStreamSynchronize in get_prefill_mqa_inputs.
    seq_lens_host: Optional[torch.Tensor] = None

    def get_seqlens_int32(self) -> torch.Tensor:
'''

BEFORE_TOLIST = '''        sequence_lengths = self.sequence_lengths.to(torch.int32)
        sequence_lengths_list = sequence_lengths.tolist()
'''

AFTER_TOLIST = '''        sequence_lengths = self.sequence_lengths.to(torch.int32)
        host_lengths = self.seq_lens_host if _HOST_LENS else None
        if host_lengths is not None and host_lengths.numel() == sequence_lengths.numel():
            sequence_lengths_list = host_lengths.tolist()
        else:
            sequence_lengths_list = sequence_lengths.tolist()
'''

BEFORE_GATE = '''from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
'''

AFTER_GATE = '''from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk

import os as _os

_HOST_LENS = _os.environ.get("SGLANG_QSA_PREFILL_HOST_LENS", "0") == "1"
'''

BEFORE_BUILD = '''            block_topk=self.token_to_kv_pool.qsa_block_topk,
            req_pool_indices=row_req_pool_indices,
'''

AFTER_BUILD = '''            block_topk=self.token_to_kv_pool.qsa_block_topk,
            seq_lens_host=forward_batch.seq_lens_cpu,
            req_pool_indices=row_req_pool_indices,
'''

EDITS = [
    (METADATA, BEFORE_GATE, AFTER_GATE),
    (METADATA, BEFORE_FIELD, AFTER_FIELD),
    (METADATA, BEFORE_TOLIST, AFTER_TOLIST),
    (BACKEND, BEFORE_BUILD, AFTER_BUILD),
]


def read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def states():
    values = []
    for path, before, after in EDITS:
        source = read(path)
        masked = source.replace(after, "\x00")
        values.append((masked.count(before) == 1, source.count(after) == 1))
    return values


def check():
    values = states()
    for (path, before, _), (clean, applied) in zip(EDITS, values):
        status = "APPLIED" if applied else ("clean" if clean else "MISMATCH")
        print(f"  {status:<8} {os.path.relpath(path, SG)}: {before.splitlines()[0][:60]}")
    if not (all(applied and not clean for clean, applied in values)
            or all(clean and not applied for clean, applied in values)):
        raise SystemExit(1)


def transform(reverse=False):
    values = states()
    wanted = all(a and not c for c, a in values) if reverse else all(c and not a for c, a in values)
    already = all(c and not a for c, a in values) if reverse else all(a and not c for c, a in values)
    if already:
        print("  already clean" if reverse else "  already applied")
        return
    if not wanted:
        check()
        raise RuntimeError("patch state mismatch")
    changed = {}
    edits = reversed(EDITS) if reverse else EDITS
    for path, before, after in edits:
        if path not in changed:
            changed[path] = read(path)
        old, new = (after, before) if reverse else (before, after)
        if changed[path].count(old) != 1:
            raise RuntimeError(f"expected one anchor in {path}")
        changed[path] = changed[path].replace(old, new, 1)
    for path, source in changed.items():
        with open(path, "w", encoding="utf-8") as output:
            output.write(source)
    print("  reverted" if reverse else "  applied (QSA prefill uses the host sequence lengths)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "apply": transform, "revert": lambda: transform(True)}[command]()
