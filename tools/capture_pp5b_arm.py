#!/usr/bin/env python3
"""Capture one auditable PP5c placement arm against an already-running server.

This tool deliberately does not start or stop SGLang. Give each fresh server a
separate log/output directory, then run this tool once:

  RAW=docs/logs/raw/pp5c_3090_2026-09-15/mass-a
  mkdir -p "$RAW"
  SRVLOG="$PWD/$RAW/server.log" \
    SGLANG_MOE_PLACEMENT="$PWD/assets/expert_freq.pt" \
    /root/quant/serve-3090.sh
  # Wait for /health, then:
  /root/quant/venv-sglang/bin/python tools/capture_pp5b_arm.py \
    --arm mass-a --placement assets/expert_freq.pt --output-dir "$RAW" \
    --server-log "$RAW/server.log"

Run mass-a, presence, mass-c on separate restarts and retain all three output
directories. The workload is code-only: the canonical corpus plus two held-out
repository code files. No benchy or repeated-sentence benchmark is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "tools" / "bench_agentic.py"
LAUNCHER = Path("/root/quant/serve-3090.sh")
DEFAULT_PYTHON = Path("/root/quant/venv-sglang/bin/python")
EXPECTED_ENV = {
    "SGLANG_KV_LAZY": "1",
    "SGLANG_KV_LAZY_SAFETY": "0.77",
    "SGLANG_KV_LAZY_TOKENS": "262144",
    "SGLANG_KV_TIERS_W": "8192",
    "SGLANG_MOE_CONFIG_NEAREST_E": "1",
    "SGLANG_MOE_EAGER_SHARED_OVERLAP_MAX_TOKENS": "0",
    "SGLANG_MOE_ELASTIC_FILL_MB": "99999",
    "SGLANG_MOE_ELASTIC_PIN_MB": "512",
    "SGLANG_MOE_ELASTIC_RESERVE_ROWS": "0",
    "SGLANG_MOE_GATHER_BLOCK": "2048",
    "SGLANG_MOE_GATHER_DMA": "1",
    "SGLANG_MOE_GATHER_DMA_BATCH": "1",
    "SGLANG_MOE_COLD_PREFETCH": "1",
    "SGLANG_MOE_COLD_PREFETCH_MIN_TOKENS": "2048",
    "SGLANG_MOE_PLACEMENT_S": "184",
    "SGLANG_MOE_EXPERT_STREAM": "1",
    "SGLANG_MOE_ELASTIC": "1",
    "SGLANG_QWEN4_PLE_BULK_PREAD": "1",
    "SGLANG_QWEN4_PLE_BULK_PREAD_MIN_UNIQUE": "2048",
    "SGLANG_QWEN4_PLE_PROFILE": "0",
    "SGLANG_QWEN4_PLE_RECENT_CACHE_MB": "128",
    "SGLANG_QWEN4_PLE_WORKERS": "16",
    "SGLANG_VLM_CACHE_SIZE_MB": "0",
}
CORPORA = (
    ("canonical", REPO / "sglang" / "qwen4exp-serving-73a255206f.patch", 5),
    ("heldout-quantize", REPO / "scripts" / "05_quantize.py", 3),
    ("heldout-phase1", REPO / "scripts" / "phase1.py", 3),
)
ROW_RE = re.compile(
    r"^\s*4096\s+(?P<actual>\d+)\s+(?P<prefill_s>[0-9.]+)\s+"
    r"(?P<prefill_tps>[0-9.]+)\s+(?P<decode_tps>[0-9.]+)\s*$",
    re.MULTILINE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise RuntimeError(f"expected exactly one port-30001 launch_server, found {len(matches)}")
    return matches[0]


def status_values(path: Path) -> tuple[str, dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    values = {}
    for line in raw.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            values[fields[0]] = fields[1]
    return raw, values


def status_path_for_server(
    server_env: dict[str, str], requested: Path | None = None
) -> Path:
    control = server_env.get("SGLANG_MOE_ELASTIC_CTL")
    if not control:
        raise ValueError("live server has no SGLANG_MOE_ELASTIC_CTL")
    expected = Path(control + ".status").resolve()
    if requested is not None and requested.resolve() != expected:
        raise ValueError(
            f"status file does not belong to the live server: expected {expected}, "
            f"got {requested.resolve()}"
        )
    return expected


def validate_status(path: Path, expected_signature: float) -> tuple[str, float]:
    raw, values = status_values(path)
    if values.get("S_min") != "184 S_max 184 floor 184":
        raise ValueError(f"elastic status is not fixed at S184: {values.get('S_min')!r}")
    try:
        actual_signature = float(values["mass_covered"])
    except (KeyError, ValueError) as exc:
        raise ValueError("elastic status has no valid mass_covered value") from exc
    if f"{actual_signature:.4f}" != f"{expected_signature:.4f}":
        raise ValueError(
            f"placement signature mismatch: expected {expected_signature:.4f}, "
            f"status has {actual_signature:.4f}"
        )
    return raw, actual_signature


def prepare_output_dir(output: Path, server_log: Path) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    allowed_live_log = server_log if server_log.parent == output else None
    unexpected = [
        item
        for item in output.iterdir()
        if allowed_live_log is None or item.resolve() != allowed_live_log
    ]
    if unexpected:
        names = ", ".join(sorted(item.name for item in unexpected))
        raise ValueError(
            f"refusing to use non-empty evidence directory {output}: {names}"
        )


def parse_benchmark_row(raw: str) -> dict[str, int | float]:
    matches = list(ROW_RE.finditer(raw))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one benchmark result row, found {len(matches)}")
    return {
        key: float(value) if key != "actual" else int(value)
        for key, value in matches[0].groupdict().items()
    }


def checksum_manifest(output: Path, excluded_names: set[str] | None = None) -> str:
    excluded = excluded_names or set()
    lines = []
    for path in sorted(item for item in output.iterdir() if item.is_file()):
        if path.name != "SHA256SUMS" and path.name not in excluded:
            lines.append(f"{sha256(path)}  {path.name}")
    return "\n".join(lines) + "\n"


def run_raw(argv: list[str], path: Path, env: dict[str, str], timeout: int = 1200) -> dict:
    started = utc_now()
    begin = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        path.write_text(partial, encoding="utf-8")
        raise RuntimeError(f"command timed out after {timeout}s; see {path}") from exc
    path.write_text(result.stdout, encoding="utf-8")
    record = {
        "argv": argv,
        "started_utc": started,
        "ended_utc": utc_now(),
        "wall_seconds": time.monotonic() - begin,
        "returncode": result.returncode,
        "raw_file": path.name,
        "raw_sha256": sha256(path),
    }
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {path}")
    return record


def health(url: str) -> dict:
    base = url.rsplit("/", 1)[0]
    with urllib.request.urlopen(base + "/health", timeout=30) as response:
        health_body = response.read().decode(errors="replace")
    with urllib.request.urlopen(base + "/model_info", timeout=30) as response:
        model_info = json.load(response)
    return {"health_body": health_body, "model_info": model_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument(
        "--status-file",
        type=Path,
        help=(
            "elastic status path; defaults to <live SGLANG_MOE_ELASTIC_CTL>.status "
            "and must match that path when supplied"
        ),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--url",
        default=os.environ.get("SGLANG_URL", "http://127.0.0.1:30001/generate"),
    )
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="expected live --chunked-prefill-size; must match the running server",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "capture requires the SGLang Python environment with torch; "
            "use /root/quant/venv-sglang/bin/python"
        ) from exc

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.arm):
        raise SystemExit("--arm must contain only lowercase letters, digits and hyphens")
    placement = args.placement.resolve(strict=True)
    server_log = args.server_log.resolve(strict=True)
    output = args.output_dir.resolve()
    try:
        prepare_output_dir(output, server_log)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    pid, server_argv, server_env = live_server()
    live_placement = Path(server_env.get("SGLANG_MOE_PLACEMENT", "")).resolve()
    if live_placement != placement:
        raise SystemExit(
            f"live placement mismatch: expected {placement}, server has {live_placement}"
        )
    mismatches = {
        key: {"expected": expected, "actual": server_env.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if server_env.get(key) != expected
    }
    if mismatches:
        raise SystemExit(f"live PP stack mismatch: {json.dumps(mismatches, sort_keys=True)}")
    try:
        status_file = status_path_for_server(server_env, args.status_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if "--chunked-prefill-size" not in server_argv:
        raise SystemExit("live server has no --chunked-prefill-size argument")
    chunk_index = server_argv.index("--chunked-prefill-size") + 1
    if chunk_index >= len(server_argv) or server_argv[chunk_index] != str(args.chunk_size):
        raise SystemExit(
            f"live server is not using chunked-prefill-size {args.chunk_size}"
        )

    placement_record = torch.load(placement, map_location="cpu", weights_only=True)
    score = placement_record["mass"].float()
    expected_signature = float(torch.topk(score, 184, dim=1).values.sum() / score.sum())
    try:
        status_raw, actual_signature = validate_status(status_file, expected_signature)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    repo_status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    repo_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    server_snapshot = output / "server-launcher.snapshot.sh"
    shutil.copyfile(LAUNCHER, server_snapshot)
    shutil.copyfile(server_log, output / "server.log.before.txt")
    (output / "elastic-status.before.txt").write_text(status_raw, encoding="utf-8")

    selected_env = {
        key: value
        for key, value in sorted(server_env.items())
        if key.startswith("SGLANG_")
    }
    live_log_name = server_log.name if server_log.parent == output else None
    metadata = {
        "schema": 2,
        "arm": args.arm,
        "started_utc": utc_now(),
        "repo_commit": repo_commit,
        "repo_status_porcelain": repo_status,
        "python": str(args.python.resolve()),
        "server_url": args.url,
        "server_pid": pid,
        "server_argv": server_argv,
        "server_environment": selected_env,
        "chunk_size": args.chunk_size,
        "launcher_snapshot": server_snapshot.name,
        "launcher_sha256": sha256(server_snapshot),
        "server_log_source": str(server_log),
        "server_log_before": "server.log.before.txt",
        "placement": str(placement),
        "placement_sha256": sha256(placement),
        "placement_signature_expected": round(expected_signature, 6),
        "placement_signature_status": actual_signature,
        "status_file": str(status_file),
        "checksum_excluded": [live_log_name] if live_log_name else [],
        "endpoint": health(args.url),
        "corpora": [
            {
                "name": name,
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "excluded": 1,
                "measured": measured,
            }
            for name, path, measured in CORPORA
            if not args.skip_heldout or name == "canonical"
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    monitor_path = output / "nvidia-smi.csv"
    monitor_stream = monitor_path.open("w", encoding="utf-8")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,driver_version,pstate,temperature.gpu,"
            "power.draw,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            "--loop-ms=250",
        ],
        stdout=monitor_stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    command_env = {**os.environ, "SGLANG_URL": args.url}
    records = []
    try:
        for corpus_name, corpus_path, measured in CORPORA:
            if args.skip_heldout and corpus_name != "canonical":
                continue
            for number in range(measured + 1):
                kind = "excluded" if number == 0 else "measured"
                raw_path = output / f"bench-{corpus_name}-{kind}-{number:02d}.txt"
                record = run_raw(
                    [
                        str(args.python),
                        str(BENCH),
                        "--corpus",
                        str(corpus_path),
                        "--tokens",
                        "4096",
                        "--decode-tokens",
                        "256",
                    ],
                    raw_path,
                    command_env,
                )
                try:
                    result_row = parse_benchmark_row(
                        raw_path.read_text(encoding="utf-8")
                    )
                except ValueError as exc:
                    raise RuntimeError(f"could not parse benchmark row in {raw_path}: {exc}") from exc
                record.update(
                    {
                        "corpus": corpus_name,
                        "sample_kind": kind,
                        "sample_index": number,
                        **result_row,
                    }
                )
                records.append(record)

        for oracle in ("m4_3090_untuned", "lp2"):
            records.append(
                {
                    "oracle": oracle,
                    **run_raw(
                        [str(args.python), str(REPO / "tools" / "logprob_diff.py"), "check", oracle],
                        output / f"oracle-{oracle}.txt",
                        command_env,
                    ),
                }
            )
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=10)
        monitor_stream.close()

    try:
        final_status_raw, final_signature = validate_status(
            status_file, expected_signature
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"final elastic status validation failed: {exc}") from exc
    if final_signature != actual_signature:
        raise RuntimeError(
            "elastic placement signature changed during capture: "
            f"{actual_signature} -> {final_signature}"
        )
    (output / "elastic-status.after.txt").write_text(final_status_raw, encoding="utf-8")
    shutil.copyfile(server_log, output / "server.log.after.txt")
    captured_server_log = output / "server.log"
    if server_log != captured_server_log:
        shutil.copyfile(server_log, captured_server_log)
    (output / "results.json").write_text(
        json.dumps(
            {"arm": args.arm, "completed_utc": utc_now(), "records": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    excluded_names = {live_log_name} if live_log_name else set()
    (output / "SHA256SUMS").write_text(
        checksum_manifest(output, excluded_names), encoding="utf-8"
    )
    print(f"captured {args.arm}: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        raise
