#!/usr/bin/env python3
"""Focused CPU tests for tools/long_logprob_oracle.py (no server, no GPU)."""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

import long_logprob_oracle as oracle


def row(**overrides):
    base = {
        "prompt_tokens": 4565,
        "server_prompt_tokens": 4565,
        "forced_prompt_tokens": 4565 + 150,
        "prompt_chars": 14336,
        "continuation_tokens": 150,
        "continuation_ids": list(oracle.CONT),
        "forced_tokens": 150,
        "prompt_ids": [1, 2, 3],
        "logprobs": [-0.1 - i * 1e-6 for i in range(150)],
    }
    base.update(overrides)
    return base


class ValidateRowTests(unittest.TestCase):
    def test_valid_row_passes(self):
        oracle.validate_row("canonical", row())

    def test_truncated_logprobs_rejected(self):
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", row(logprobs=[-0.1] * 10))
        self.assertIn("truncated", str(ctx.exception))

    def test_extra_logprobs_rejected(self):
        with self.assertRaises(oracle.OracleValidationError):
            oracle.validate_row("canonical", row(logprobs=[-0.1] * 151))

    def test_non_finite_nan_rejected(self):
        bad = row()
        bad["logprobs"][7] = float("nan")
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", bad)
        self.assertIn("non-finite", str(ctx.exception))

    def test_non_finite_inf_rejected(self):
        bad = row()
        bad["logprobs"][3] = float("-inf")
        with self.assertRaises(oracle.OracleValidationError):
            oracle.validate_row("canonical", bad)

    def test_mismatched_prompt_ids_rejected(self):
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", row(), expect_ids=[9, 9, 9])
        self.assertIn("prompt_ids differ", str(ctx.exception))

    def test_mismatched_continuation_ids_rejected(self):
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row(
                "canonical", row(), expect_cont_ids=list(oracle.CONT[:-1]) + [999]
            )
        self.assertIn("continuation_ids differ", str(ctx.exception))

    def test_continuation_length_mismatch_rejected(self):
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", row(continuation_tokens=149))
        self.assertIn("continuation_tokens", str(ctx.exception))

    def test_inconsistent_forced_prompt_tokens_rejected(self):
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", row(forced_prompt_tokens=1))
        self.assertIn("forced_prompt_tokens", str(ctx.exception))

    def test_server_prompt_mismatch_rejected(self):
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", row(server_prompt_tokens=4566))
        self.assertIn("server_prompt_tokens", str(ctx.exception))

    def test_missing_field_rejected(self):
        bad = row()
        del bad["logprobs"]
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.validate_row("canonical", bad)
        self.assertIn("missing logprobs", str(ctx.exception))


class StatsTests(unittest.TestCase):
    def test_per_token_deltas_and_overall(self):
        ref = {"canonical": row()}
        cur = {"canonical": row(logprobs=[v + (1.0 if i == 4 else 0.0) for i, v in enumerate(row()["logprobs"])])}
        report = oracle.stats(ref, cur)
        self.assertEqual(report["canonical"]["max_index"], 4)
        self.assertAlmostEqual(report["canonical"]["max"], 1.0)
        self.assertEqual(len(report["canonical"]["per_token_deltas"]), 150)
        self.assertAlmostEqual(report["_overall"]["max"], 1.0)

    def test_missing_prompt_raises(self):
        with self.assertRaises(oracle.OracleValidationError):
            oracle.stats({"canonical": row()}, {})

    def test_mismatched_input_raises(self):
        ref = {"canonical": row()}
        cur = {"canonical": row(prompt_ids=[3, 2, 1])}
        with self.assertRaises(oracle.OracleValidationError) as ctx:
            oracle.stats(ref, cur)
        self.assertIn("prompt_ids differ", str(ctx.exception))

    def test_truncated_current_raises(self):
        ref = {"canonical": row()}
        cur = {"canonical": row(logprobs=[-0.1] * 10)}
        with self.assertRaises(oracle.OracleValidationError):
            oracle.stats(ref, cur)


class ForcedLogprobsTests(unittest.TestCase):
    def _meta(self, values, prompt_tokens=4715):
        return {"meta_info": {"input_token_logprobs": [[v, None, None] for v in values], "prompt_tokens": prompt_tokens}}

    def test_exact_shape_returns_continuation(self):
        values = [None] + [-0.2 - i * 1e-6 for i in range(150)]
        with mock.patch.object(oracle, "post", return_value=self._meta(values)):
            logprobs, prompt_tokens = oracle.forced_logprobs([1, 2, 3], oracle.CONT)
        self.assertEqual(len(logprobs), 150)
        self.assertAlmostEqual(logprobs[0], -0.2)
        self.assertEqual(prompt_tokens, 4715)

    def test_truncated_response_rejected(self):
        with mock.patch.object(oracle, "post", return_value=self._meta([None] + [-0.1] * 10)):
            with self.assertRaises(oracle.OracleValidationError) as ctx:
                oracle.forced_logprobs([1, 2, 3], oracle.CONT)
            self.assertIn("truncated", str(ctx.exception).lower())

    def test_non_finite_response_rejected(self):
        values = [None] + [-0.2] * 150
        values[5] = float("nan")
        with mock.patch.object(oracle, "post", return_value=self._meta(values)):
            with self.assertRaises(oracle.OracleValidationError) as ctx:
                oracle.forced_logprobs([1, 2, 3], oracle.CONT)
            self.assertIn("non-finite", str(ctx.exception))


class PayloadTests(unittest.TestCase):
    def test_validate_payload_empty_rejected(self):
        with self.assertRaises(oracle.OracleValidationError):
            oracle.validate_payload({})

    def test_validate_payload_rejects_bad_row(self):
        with self.assertRaises(oracle.OracleValidationError):
            oracle.validate_payload({"canonical": row(logprobs=[])})


if __name__ == "__main__":
    unittest.main(verbosity=2)
