#!/usr/bin/env python3
"""Snapshot the live port-30001 SGLang server argv + SGLANG_* environment.

Used to make the non-promotion validation arms (pre-PP15 MoE configs, prefetch
off) auditable without the full PP5c capture, which asserts the accepted
defaults and would refuse to run on an intentional one-variable deviation.

  python3 tools/snapshot_server.py OUT.json
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = Path("/root/quant/serve-3090.sh")


def live_server() -> tuple[int, list[str], dict[str, str]]:
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [
                item.decode(errors="replace")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            joined = " ".join(argv)
            if "sglang.launch_server" not in joined or "--port 30001" not in joined:
                continue
            env = {}
            for item in (proc / "environ").read_bytes().split(b"\0"):
                if b"=" in item:
                    key, value = item.split(b"=", 1)
                    env[key.decode(errors="replace")] = value.decode(errors="replace")
            matches.append((int(proc.name), argv, env))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"expected one port-30001 launch_server, found {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    pid, argv, env = live_server()
    selected = {k: v for k, v in sorted(env.items()) if k.startswith("SGLANG_")}
    config_dir = Path(selected.get("SGLANG_MOE_CONFIG_DIR", ""))
    configs = {}
    if config_dir.is_dir():
        for path in sorted(config_dir.rglob("*.json")):
            configs[str(path.relative_to(config_dir))] = sha256(path)
    payload = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "server_pid": pid,
        "server_argv": argv,
        "server_environment": selected,
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stdout=subprocess.PIPE
        ).stdout.strip(),
        "repo_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=REPO, text=True, stdout=subprocess.PIPE
        ).stdout,
        "sglang_source_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd="/root/sglang", text=True, stdout=subprocess.PIPE
        ).stdout.strip(),
        "launcher": str(LAUNCHER),
        "launcher_sha256": sha256(LAUNCHER),
        "config_dir": str(config_dir),
        "config_sha256": configs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
