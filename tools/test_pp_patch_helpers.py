#!/usr/bin/env python3
"""Round-trip and mixed-state tests for PP launcher/source patch helpers."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.capture_pp5b_arm import ROW_RE, status_values


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


class CaptureHelperTests(unittest.TestCase):
    def test_benchmark_row_parser(self):
        output = """  target   actual  prefill s  prefill t/s  decode t/s
  ---------------------------------------------------------
      4096     4565       3.67          1244        41.5
"""
        match = ROW_RE.search(output)
        self.assertIsNotNone(match)
        self.assertEqual(
            match.groupdict(),
            {
                "actual": "4565",
                "prefill_s": "3.67",
                "prefill_tps": "1244",
                "decode_tps": "41.5",
            },
        )

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
