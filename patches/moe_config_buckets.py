#!/usr/bin/env python3
"""Nearest-expert-count lookup for explicitly enabled INT2 MoE config buckets.

Elastic expert streaming compacts each prefill layer to the experts selected by
that layer.  The compact tensor's E therefore varies by layer and request, while
SGLang's config filename lookup normally requires an exact E.  With
``SGLANG_MOE_CONFIG_NEAREST_E=1``, this patch keeps exact lookup first and then
uses the nearest available E bucket for the same device, dtype, projection and
Triton version.

    SGLANG=/root/sglang python3 patches/moe_config_buckets.py --check
    SGLANG=/root/sglang python3 patches/moe_config_buckets.py apply
    SGLANG=/root/sglang python3 patches/moe_config_buckets.py revert
"""

import os
import sys


SG = os.path.join(
    os.environ.get("SGLANG", os.path.expanduser("~/quant/sglang")), "python/sglang"
)
TARGET = os.path.join(
    SG, "srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py"
)

BEFORE_HELPER = '''logger = logging.getLogger(__name__)
_is_hip = is_hip()
_LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES = 128 * 1024
'''

AFTER_HELPER = '''logger = logging.getLogger(__name__)
_is_hip = is_hip()
_LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES = 128 * 1024


def _find_nearest_expert_config(
    config_dir: str, version_dir: str, json_file_name: str
) -> Optional[Tuple[str, int]]:
    \"\"\"Find the nearest opt-in INT2 E bucket with every other key exact.\"\"\"
    if os.environ.get("SGLANG_MOE_CONFIG_NEAREST_E") != "1":
        return None
    if not json_file_name.startswith("E=") or ",dtype=int2_w2a16" not in json_file_name:
        return None
    requested_text, separator, suffix = json_file_name.partition(",")
    if not separator:
        return None
    try:
        requested = int(requested_text.removeprefix("E="))
    except ValueError:
        return None
    directory = os.path.join(config_dir, "configs", version_dir)
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    candidates = []
    for name in names:
        expert_text, candidate_separator, candidate_suffix = name.partition(",")
        if candidate_separator and candidate_suffix == suffix and expert_text.startswith("E="):
            try:
                experts = int(expert_text.removeprefix("E="))
            except ValueError:
                continue
            candidates.append((abs(experts - requested), experts, os.path.join(directory, name)))
    if not candidates:
        return None
    _, experts, path = min(candidates)
    return path, experts
'''

BEFORE_LOOKUP = '''    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance might still be suboptimal.
            # This is because the tuning environment might differ from your current environment.
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # Discover available triton config dirs on disk and search newest-first.
'''

AFTER_LOOKUP = '''    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance might still be suboptimal.
            # This is because the tuning environment might differ from your current environment.
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    nearest = _find_nearest_expert_config(config_dir, version_dir, json_file_name)
    if nearest is not None:
        nearest_path, nearest_experts = nearest
        with open(nearest_path) as f:
            requested_experts = int(json_file_name.partition(",")[0].removeprefix("E="))
            logger.info(
                "Using nearest MoE expert-count config E=%d for requested E=%d from %s.",
                nearest_experts,
                requested_experts,
                nearest_path,
            )
            return {int(key): val for key, val in json.load(f).items()}

    # Discover available triton config dirs on disk and search newest-first.
'''

EDITS = [(BEFORE_HELPER, AFTER_HELPER), (BEFORE_LOOKUP, AFTER_LOOKUP)]


def states():
    text = open(TARGET, encoding="utf-8").read()
    return [(before in text, after in text) for before, after in EDITS]


def check():
    for index, (before, after) in enumerate(states()):
        state = "APPLIED" if after else ("clean" if before else "MISMATCH")
        print(f"  {index} {state:<8} {os.path.relpath(TARGET, SG)}")


def apply():
    state = states()
    if not all(before or after for before, after in state):
        print("  [!] source mismatch")
        check()
        raise SystemExit(1)
    text = open(TARGET, encoding="utf-8").read()
    for (before, after), (_, applied) in zip(EDITS, state):
        if not applied:
            text = text.replace(before, after, 1)
    open(TARGET, "w", encoding="utf-8").write(text)
    print("  applied (SGLANG_MOE_CONFIG_NEAREST_E=1 enables nearest INT2 E buckets)")


def revert():
    text = open(TARGET, encoding="utf-8").read()
    for before, after in reversed(EDITS):
        if after in text:
            text = text.replace(after, before, 1)
    open(TARGET, "w", encoding="utf-8").write(text)
    print("  reverted")


if __name__ == "__main__":
    {"--check": check, "check": check, "apply": apply, "revert": revert}[
        sys.argv[1] if len(sys.argv) > 1 else "--check"
    ]()
