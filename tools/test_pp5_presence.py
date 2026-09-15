#!/usr/bin/env python3
"""Regression tests for the PP5 placement analysis."""

import tempfile
import unittest
from pathlib import Path

import torch

from tools.pp5_presence import cold_rows, load_chunks, placement_score, rank_indices


class PP5PresenceTests(unittest.TestCase):
    def test_cold_rows_matches_each_layer_to_its_resident_set(self):
        routed = [[{0}, {1}], [{0}, {1}]]
        resident = [{0}, {1}]
        self.assertEqual(cold_rows(routed, resident, [0, 1]), 0.0)

    def test_mass_breaks_presence_ties(self):
        presence = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        mass = torch.tensor([[0.1, 0.9, 0.5, 2.0]])
        selected = rank_indices(presence, 2, mass)[0].tolist()
        self.assertEqual(selected, [1, 2])

        encoded = placement_score(presence, mass)
        self.assertEqual(torch.argsort(encoded[0], descending=True)[:2].tolist(), selected)

    def test_min_tokens_excludes_launcher_warmups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seq, m in enumerate((3, 6, 64, 469)):
                record = [(l, torch.zeros((m, 2), dtype=torch.int16)) for l in range(2)]
                torch.save(record, root / f"prouting_{seq:05d}.pt")
            chunks = load_chunks(str(root), min_tokens=64)
            self.assertEqual([int(ch[0].shape[0]) for ch in chunks], [64, 469])


if __name__ == "__main__":
    unittest.main()
