#!/usr/bin/env python3
"""Bounded PP14 prefetch on/off diagnostic for TG0.

The PP campaign left one unresolved long-prompt comparison: a prefetch-off run
reached max |dlogprob| 2.797 where the accepted run-to-run envelope reached
1.781.  TG0's bounded diagnostic takes three accepted and three prefetch-off
observations of every existing long oracle prompt in a balanced boot order
(accepted, off, off, accepted), retains per-token data for all six runs, and
compares per-position distributions rather than only a global maximum.

The driver owns the port-30001 server: it stops any running instance, boots the
launcher with the requested ``SGLANG_MOE_COLD_PREFETCH`` value, waits for
health, snapshots the live argv/environment, and collects the observations.
It never edits the serving tree.  The final boot restores prefetch-on.

  python3 tools/tg_pp14_diag.py --out docs/logs/raw/tg0_.../pp14-diag
  python3 tools/tg_pp14_diag.py --analyze-only --out docs/logs/raw/tg0_.../pp14-diag
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
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = Path("/root/quant/serve-3090.sh")
sys.path.insert(0, str(REPO / "tools"))
import long_logprob_oracle as oracle  # noqa: E402

# (arm, observations) in a balanced boot order; accepted and prefetch-off each
# receive exactly three observations.
SCHEDULE = (
    ("accepted", 2),
    ("prefetch-off", 2),
    ("prefetch-off", 1),
    ("accepted", 1),
)
PROMPTS = ("canonical", "long-2chunk", "long-4chunk")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def server_pids() -> list[int]:
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
        if "sglang.launch_server" in joined and "--port 30001" in joined:
            pids.append(int(proc.name))
    return pids


def wait_health(timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:30001/health", timeout=5):
                time.sleep(5)
                return
        except Exception:  # noqa: BLE001
            time.sleep(10)
    raise SystemExit("server did not become healthy in time")


def stop_server() -> None:
    pids = server_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 90
    while time.time() < deadline and server_pids():
        time.sleep(2)
    for pid in server_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(3)


def boot(arm: str, out: Path, boot_index: int) -> dict:
    stop_server()
    server_log = out / f"server-{arm}-boot{boot_index}.log"
    ctl = out / f"elastic-{arm}-boot{boot_index}.ctl"
    ctl.write_text("S 184\n")
    env = os.environ.copy()
    env.update(
        {
            "SRVLOG": str(server_log),
            "CTL": str(ctl),
            "SGLANG_MOE_COLD_PREFETCH": "1" if arm == "accepted" else "0",
        }
    )
    with (out / f"launch-{arm}-boot{boot_index}.out").open("w") as stream:
        subprocess.Popen(
            ["bash", str(LAUNCHER)],
            env=env,
            cwd=str(REPO),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wait_health()
    snapshot = out / f"server-{arm}-boot{boot_index}.json"
    subprocess.run(
        [
            "/root/quant/venv-sglang/bin/python",
            str(REPO / "tools" / "snapshot_server.py"),
            str(snapshot),
        ],
        check=True,
    )
    actual = json.loads(snapshot.read_text())["server_environment"].get(
        "SGLANG_MOE_COLD_PREFETCH"
    )
    expected = "1" if arm == "accepted" else "0"
    if actual != expected:
        raise SystemExit(f"boot {arm} has prefetch={actual}, expected {expected}")
    return {"arm": arm, "boot_index": boot_index, "server_log": str(server_log),
            "snapshot": str(snapshot), "prefetch": actual}


def collect(out: Path, arm: str, boot_index: int, run_index: int) -> Path:
    path = out / f"{arm}-boot{boot_index}-run{run_index}.json"
    oracle.URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30001/generate")
    payload = oracle.collect(None)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"  collected {path.name} ({sorted(payload)})", flush=True)
    return path


def analyze(out: Path) -> dict:
    files = sorted(out.glob("*-boot*-run*.json"))
    runs: dict[str, list[dict]] = {"accepted": [], "prefetch-off": []}
    meta = []
    for path in files:
        name = path.name
        arm = "prefetch-off" if name.startswith("prefetch-off") else "accepted"
        runs[arm].append(json.loads(path.read_text()))
        meta.append(name)
    for arm, items in runs.items():
        if len(items) != 3:
            raise SystemExit(f"expected 3 {arm} runs, found {len(items)}: {meta}")

    def delta_report(a: dict, b: dict) -> dict:
        report = oracle.stats(a, b)
        return {
            name: {
                "max": row["max"],
                "mean": row["mean"],
                "max_index": row["max_index"],
                "per_token_deltas": row["per_token_deltas"],
            }
            for name, row in report.items()
            if name != "_overall"
        } | {"_overall": report["_overall"]}

    within = {"accepted": [], "prefetch-off": []}
    for arm in runs:
        for i in range(len(runs[arm])):
            for j in range(i + 1, len(runs[arm])):
                within[arm].append(delta_report(runs[arm][i], runs[arm][j]))
    cross = []
    for a in runs["accepted"]:
        for b in runs["prefetch-off"]:
            cross.append(delta_report(a, b))

    # Per-position values at the index the closure review flagged (index 6).
    flagged_index = 6
    position_profile = {
        name: {
            "accepted_index6": [r[name]["logprobs"][flagged_index] for r in runs["accepted"]],
            "prefetch_off_index6": [
                r[name]["logprobs"][flagged_index] for r in runs["prefetch-off"]
            ],
        }
        for name in PROMPTS
    }

    def collect_values(pairs: list[dict], name: str) -> list[float]:
        return [p[name]["max"] for p in pairs if name in p]

    summary = {
        "schema": 1,
        "analyzed_utc": utc_now(),
        "runs": {arm: [p.name for p in sorted(out.glob(f"{arm}-boot*-run*.json"))]
                 for arm in runs},
        "within_arm": {
            arm: {
                name: {
                    "max_values": collect_values(within[arm], name),
                    "mean_values": [p[name]["mean"] for p in within[arm] if name in p],
                }
                for name in PROMPTS
            }
            for arm in within
        },
        "cross_arm": {
            name: {
                "max_values": collect_values(cross, name),
                "mean_values": [p[name]["mean"] for p in cross if name in p],
                "max_indices": [p[name]["max_index"] for p in cross if name in p],
            }
            for name in PROMPTS
        },
        "cross_arm_per_token": {
            name: [p[name]["per_token_deltas"] for p in cross if name in p]
            for name in PROMPTS
        },
        "within_arm_per_token": {
            arm: {name: [p[name]["per_token_deltas"] for p in within[arm] if name in p]
                  for name in PROMPTS}
            for arm in within
        },
        "flagged_index": flagged_index,
        "position_profile_index6": position_profile,
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out / 'analysis.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        boot_index = 0
        for arm, count in SCHEDULE:
            boot_index += 1
            info = boot(arm, out, boot_index)
            print(f"  boot {boot_index}: {arm} prefetch={info['prefetch']}", flush=True)
            for run_index in range(1, count + 1):
                collect(out, arm, boot_index, run_index)
        print("  prefetch-on restored by the final accepted boot", flush=True)
    analyze(out)


if __name__ == "__main__":
    main()
