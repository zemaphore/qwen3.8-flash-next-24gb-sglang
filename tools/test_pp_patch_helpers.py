#!/usr/bin/env python3
"""Round-trip and mixed-state tests for PP launcher/source patch helpers."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.capture_pp5b_arm import (
    checksum_manifest,
    parse_benchmark_row,
    prepare_output_dir,
    status_path_for_server,
    status_values,
    validate_status,
)


REPO = Path(__file__).resolve().parents[1]


def run(script, command, env):
    return subprocess.run(
        ["python3", str(REPO / script), command],
        env={**os.environ, **env},
        check=False,
        capture_output=True,
        text=True,
    )


class LauncherHelperTests(unittest.TestCase):
    def round_trip(self, script, anchor, line):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "serve.sh"
            original = "#!/bin/sh\n" + anchor + "exec true\n"
            launcher.write_text(original, encoding="utf-8")
            env = {"SGLANG_3090_LAUNCHER": str(launcher)}

            self.assertEqual(run(script, "apply", env).returncode, 0)
            self.assertIn(line, launcher.read_text(encoding="utf-8"))
            self.assertEqual(run(script, "--check", env).returncode, 0)
            self.assertEqual(run(script, "revert", env).returncode, 0)
            self.assertEqual(launcher.read_text(encoding="utf-8"), original)
            self.assertEqual(run(script, "--check", env).returncode, 0)

    def test_dma_launcher_round_trip(self):
        self.round_trip(
            "patches/enable_moe_host_dma_gather.py",
            '      SGLANG_MOE_GATHER_BLOCK="${SGLANG_MOE_GATHER_BLOCK:-2048}" \\\n',
            '      SGLANG_MOE_GATHER_DMA="${SGLANG_MOE_GATHER_DMA:-1}" \\\n',
        )

    def test_gather_block_launcher_round_trip(self):
        self.round_trip(
            "patches/enable_moe_gather_block.py",
            "      SGLANG_MOE_EXPERT_STREAM=1 \\\n",
            '      SGLANG_MOE_GATHER_BLOCK="${SGLANG_MOE_GATHER_BLOCK:-2048}" \\\n',
        )

    def test_dma_batch_launcher_round_trip(self):
        self.round_trip(
            "patches/enable_moe_host_dma_batch.py",
            '      SGLANG_MOE_GATHER_DMA="${SGLANG_MOE_GATHER_DMA:-1}" \\\n',
            '      SGLANG_MOE_GATHER_DMA_BATCH="${SGLANG_MOE_GATHER_DMA_BATCH:-1}" \\\n',
        )

    def test_presence_placement_launcher_round_trip(self):
        self.round_trip(
            "patches/enable_prefill_presence_placement.py",
            '      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-$ASSETS/expert_freq.pt}" \\\n',
            '      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-$ASSETS/expert_presence_code.pt}" \\\n',
        )

    def test_presence_placement_launcher_migrates_old_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "serve.sh"
            old = '      SGLANG_MOE_PLACEMENT="$ASSETS/expert_freq.pt" \\\n'
            new = (
                '      SGLANG_MOE_PLACEMENT="${SGLANG_MOE_PLACEMENT:-'
                '$ASSETS/expert_presence_code.pt}" \\\n'
            )
            launcher.write_text("#!/bin/sh\n" + old + "exec true\n", encoding="utf-8")
            env = {"SGLANG_3090_LAUNCHER": str(launcher)}

            self.assertEqual(
                run("patches/enable_prefill_presence_placement.py", "apply", env).returncode,
                0,
            )
            self.assertIn(new, launcher.read_text(encoding="utf-8"))
            self.assertNotIn(old, launcher.read_text(encoding="utf-8"))
            self.assertEqual(
                run("patches/enable_prefill_presence_placement.py", "--check", env).returncode,
                0,
            )

    def test_dma_batch_launcher_migrates_old_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "serve.sh"
            old = '      SGLANG_MOE_GATHER_DMA_BATCH="${SGLANG_MOE_GATHER_DMA_BATCH:-0}" \\\n'
            new = '      SGLANG_MOE_GATHER_DMA_BATCH="${SGLANG_MOE_GATHER_DMA_BATCH:-1}" \\\n'
            launcher.write_text("#!/bin/sh\n" + old + "exec true\n", encoding="utf-8")
            env = {"SGLANG_3090_LAUNCHER": str(launcher)}

            self.assertNotEqual(
                run("patches/enable_moe_host_dma_batch.py", "--check", env).returncode,
                0,
            )
            self.assertEqual(
                run("patches/enable_moe_host_dma_batch.py", "apply", env).returncode,
                0,
            )
            self.assertIn(new, launcher.read_text(encoding="utf-8"))
            self.assertNotIn(old, launcher.read_text(encoding="utf-8"))


class ChunkResidencyHelperTests(unittest.TestCase):
    FILL_OLD = 'SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-2048}"'
    FILL_NEW = 'SGLANG_MOE_ELASTIC_FILL_MB="${SGLANG_MOE_ELASTIC_FILL_MB:-99999}"'
    CHUNK_OLD = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-1024}"'
    CHUNK_MID = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-2048}"'
    CHUNK_NEW = '--chunked-prefill-size "${SGLANG_3090_CHUNKED_PREFILL_SIZE:-4608}"'

    def test_ladder_up_then_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "serve.sh"
            launcher.write_text(
                "#!/bin/sh\n" + self.FILL_OLD + " \\\n" + self.CHUNK_OLD + " \\\n",
                encoding="utf-8",
            )
            env = {"SGLANG_3090_LAUNCHER": str(launcher)}
            script = "patches/enable_prefill_chunk_residency.py"

            # clean -> applied
            self.assertEqual(run(script, "apply", env).returncode, 0)
            text = launcher.read_text(encoding="utf-8")
            self.assertIn(self.FILL_NEW, text)
            self.assertIn(self.CHUNK_NEW, text)
            self.assertEqual(run(script, "--check", env).returncode, 0)

            # applied -> mid (undo PP12 only)
            self.assertEqual(run(script, "revert", env).returncode, 0)
            text = launcher.read_text(encoding="utf-8")
            self.assertIn(self.CHUNK_MID, text)
            self.assertNotIn(self.CHUNK_NEW, text)

            # mid -> clean (undo PP3 too)
            self.assertEqual(run(script, "revert", env).returncode, 0)
            text = launcher.read_text(encoding="utf-8")
            self.assertIn(self.FILL_OLD, text)
            self.assertIn(self.CHUNK_OLD, text)


class CaptureHelperTests(unittest.TestCase):
    def test_benchmark_row_parser(self):
        output = """  target   actual  prefill s  prefill t/s  decode t/s
  ---------------------------------------------------------
      4096     4565       3.67          1244        41.5
"""
        self.assertEqual(
            parse_benchmark_row(output),
            {
                "actual": 4565,
                "prefill_s": 3.67,
                "prefill_tps": 1244.0,
                "decode_tps": 41.5,
            },
        )

    def test_benchmark_row_parser_rejects_missing_and_duplicate_rows(self):
        row = "4096 4565 3.67 1244 41.5\n"
        with self.assertRaisesRegex(ValueError, "found 0"):
            parse_benchmark_row("no result\n")
        with self.assertRaisesRegex(ValueError, "found 2"):
            parse_benchmark_row(row + row)

    def test_elastic_status_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "elastic.status"
            raw = (
                "time 19:30:46\n"
                "S_min 184 S_max 184 floor 184\n"
                "mass_covered 0.4665\n"
            )
            status.write_text(raw, encoding="utf-8")
            actual_raw, values = status_values(status)
            self.assertEqual(actual_raw, raw)
            self.assertEqual(values["S_min"], "184 S_max 184 floor 184")
            self.assertEqual(values["mass_covered"], "0.4665")
            self.assertEqual(validate_status(status, 0.46654), (raw, 0.4665))

            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                validate_status(status, 0.5000)
            status.write_text(
                "S_min 183 S_max 184 floor 183\nmass_covered 0.4665\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not fixed at S184"):
                validate_status(status, 0.4665)

    def test_elastic_status_path_is_bound_to_live_control_file(self):
        env = {"SGLANG_MOE_ELASTIC_CTL": "/tmp/capture-test.ctl"}
        expected = Path("/tmp/capture-test.ctl.status")
        self.assertEqual(status_path_for_server(env), expected)
        self.assertEqual(status_path_for_server(env, expected), expected)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            status_path_for_server(env, Path("/tmp/stale.status"))
        with self.assertRaisesRegex(ValueError, "has no SGLANG_MOE_ELASTIC_CTL"):
            status_path_for_server({})

    def test_output_directory_allows_only_live_server_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "arm"
            output.mkdir()
            server_log = output / "server.log"
            server_log.write_text("live\n", encoding="utf-8")
            prepare_output_dir(output, server_log)
            (output / "partial.txt").write_text("partial\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partial.txt"):
                prepare_output_dir(output, server_log)

    def test_checksum_manifest_excludes_mutable_live_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "server.log").write_text("live\n", encoding="utf-8")
            (output / "server.log.after.txt").write_text("frozen\n", encoding="utf-8")
            manifest = checksum_manifest(output, {"server.log"})
            self.assertNotIn("  server.log\n", manifest)
            self.assertIn("  server.log.after.txt\n", manifest)


class SourcePatchTests(unittest.TestCase):
    def make_tree(self, root):
        import sys

        sys.path.insert(0, str(REPO / "patches"))
        import moe_host_dma_gather as patch

        stream = root / "python/sglang/srt/layers/moe/expert_stream.py"
        elastic = root / "python/sglang/srt/layers/moe/expert_elastic.py"
        stream.parent.mkdir(parents=True)
        stream.write_text(
            patch.BEFORE_DMA_CONST + patch.BEFORE_DISPATCH + patch.BEFORE_TAIL,
            encoding="utf-8",
        )
        elastic.write_text(patch.BEFORE_PLACED, encoding="utf-8")
        return patch, stream, elastic

    def test_source_patch_round_trip_and_mixed_state_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch, stream, elastic = self.make_tree(root)
            original_stream = stream.read_text(encoding="utf-8")
            original_elastic = elastic.read_text(encoding="utf-8")
            env = {"SGLANG": str(root)}
            script = "patches/moe_host_dma_gather.py"

            self.assertEqual(run(script, "apply", env).returncode, 0)
            self.assertEqual(run(script, "--check", env).returncode, 0)
            self.assertEqual(run(script, "revert", env).returncode, 0)
            self.assertEqual(stream.read_text(encoding="utf-8"), original_stream)
            self.assertEqual(elastic.read_text(encoding="utf-8"), original_elastic)

            self.assertEqual(run(script, "apply", env).returncode, 0)
            text = stream.read_text(encoding="utf-8")
            stream.write_text(
                text.replace(patch.AFTER_DISPATCH, patch.BEFORE_DISPATCH, 1),
                encoding="utf-8",
            )
            self.assertNotEqual(run(script, "--check", env).returncode, 0)
            self.assertNotEqual(run(script, "revert", env).returncode, 0)

    def test_dma_batch_source_patch_round_trip(self):
        import sys

        sys.path.insert(0, str(REPO / "patches"))
        import moe_host_dma_batch as patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stream = root / "python/sglang/srt/layers/moe/expert_stream.py"
            elastic = root / "python/sglang/srt/layers/moe/expert_elastic.py"
            stream.parent.mkdir(parents=True)
            original_stream = (
                patch.BEFORE_IMPORT
                + patch.BEFORE_CONST
                + patch.HELPER_ANCHOR
                + patch.BEFORE_GATHER
            )
            original_elastic = patch.BEFORE_INVALIDATE
            stream.write_text(original_stream, encoding="utf-8")
            elastic.write_text(original_elastic, encoding="utf-8")
            env = {"SGLANG": str(root)}
            script = "patches/moe_host_dma_batch.py"

            self.assertEqual(run(script, "apply", env).returncode, 0)
            self.assertEqual(run(script, "--check", env).returncode, 0)
            self.assertEqual(run(script, "revert", env).returncode, 0)
            self.assertEqual(stream.read_text(encoding="utf-8"), original_stream)
            self.assertEqual(elastic.read_text(encoding="utf-8"), original_elastic)


if __name__ == "__main__":
    unittest.main()
