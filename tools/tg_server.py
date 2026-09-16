#!/usr/bin/env python3
"""Small port-30001 lifecycle helper for TG0 boots (stop/start/wait).

  python3 tools/tg_server.py stop [--port 30001]
  python3 tools/tg_server.py start --log LOG --ctl CTL [--env K=V ...]
  python3 tools/tg_server.py wait [--port 30001]
  python3 tools/tg_server.py status
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LAUNCHER = Path("/root/quant/serve-3090.sh")


def server_pids(port: int) -> list[int]:
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [
                item.decode(errors="replace")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        joined = " ".join(argv)
        if "sglang.launch_server" in joined and f"--port {port}" in joined:
            pids.append(int(proc.name))
    return pids


def stop(port: int) -> None:
    for pid in server_pids(port):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 120
    while time.time() < deadline and server_pids(port):
        time.sleep(2)
    remaining = server_pids(port)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(3)
    print(f"  stopped; remaining pids: {server_pids(port)}")


def wait(port: int, timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    time.sleep(5)
                    print("  healthy")
                    return
        except Exception:  # noqa: BLE001
            time.sleep(10)
    raise SystemExit("server did not become healthy in time")


def start(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env.update({"SRVLOG": str(args.log), "CTL": str(args.ctl)})
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    Path(args.ctl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ctl).write_text("S 184\n")
    with open(args.launch_out, "w") as stream:
        subprocess.Popen(
            ["bash", str(LAUNCHER)],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wait(args.port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stop", "start", "wait", "status"))
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--log")
    parser.add_argument("--ctl")
    parser.add_argument("--launch-out", default="/tmp/tg_server_launch.out")
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    if args.command == "stop":
        stop(args.port)
    elif args.command == "start":
        if not args.log or not args.ctl:
            raise SystemExit("start requires --log and --ctl")
        start(args)
    elif args.command == "wait":
        wait(args.port)
    else:
        print(json.dumps({"port": args.port, "pids": server_pids(args.port)}))


if __name__ == "__main__":
    main()
