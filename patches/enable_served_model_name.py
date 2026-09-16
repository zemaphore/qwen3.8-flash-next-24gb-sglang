#!/usr/bin/env python3
"""Set the 3090 API model alias; supports --check, apply, and revert.

SERVED_MODEL_NAME overrides the alias at launch. Checkpoint loading is unchanged.
SGLANG_3090_LAUNCHER selects an alternative launcher for patching.
"""

import os
import sys
from pathlib import Path

LAUNCHER = Path(os.environ.get("SGLANG_3090_LAUNCHER", "/root/quant/serve-3090.sh"))
ANCHOR = '    --model-path "$MODEL" --max-total-tokens 262144 --context-length 262144'
ALIAS = '    --served-model-name "${SERVED_MODEL_NAME:-qwen38-flash-256K}" \\\n'


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if command not in ("--check", "apply", "revert"):
        raise SystemExit("usage: enable_served_model_name.py [--check|apply|revert]")
    source = LAUNCHER.read_text()
    applied = source.count(ALIAS + ANCHOR) == 1
    if source.count(ANCHOR) != 1 or source.count("--served-model-name") != int(applied):
        raise SystemExit("MISMATCH: launcher model-name anchor")
    if command == "--check":
        print("APPLIED" if applied else "clean")
        return
    if command == "apply" and not applied:
        LAUNCHER.write_text(source.replace(ANCHOR, ALIAS + ANCHOR, 1))
    elif command == "revert" and applied:
        LAUNCHER.write_text(source.replace(ALIAS + ANCHOR, ANCHOR, 1))
    print("applied" if command == "apply" else "reverted")


if __name__ == "__main__":
    main()
