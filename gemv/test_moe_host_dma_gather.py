#!/usr/bin/env python3
"""GPU regression test for PP7 mixed resident/host DMA staging."""

import unittest

import torch

from sglang.srt.layers.moe.expert_stream import ExpertStreamer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class HostDMAGatherTests(unittest.TestCase):
    E = 8
    SPECS = {
        "pp7_test_i32": ((17,), torch.int32),
        "pp7_test_bf16": ((11,), torch.bfloat16),
    }

    def make_placement(self, residents):
        residents = list(residents)
        placed = {"addr": {}, "proto": {}, "home": {}, "E": self.E}
        expected = {}
        keep = []
        for name, (tail, dtype) in self.SPECS.items():
            values = torch.arange(self.E * tail[0], dtype=torch.float32).reshape(
                self.E, *tail
            )
            values = values.to(dtype)
            hot = values[residents].cuda().contiguous()
            addr = torch.empty(self.E, dtype=torch.int64)
            home = [None] * self.E
            row_bytes = values[0].numel() * values.element_size()
            for pos, expert in enumerate(residents):
                addr[expert] = hot.data_ptr() + pos * row_bytes
            for expert in set(range(self.E)) - set(residents):
                slab = torch.empty((2,) + tail, dtype=dtype, pin_memory=True)
                slot = expert % 2
                slab[slot].copy_(values[expert])
                home[expert] = (slab, slot)
                addr[expert] = slab.data_ptr() + slot * row_bytes
                keep.append(slab)
            placed["addr"][name] = addr.cuda()
            placed["proto"][name] = hot
            placed["home"][name] = home
            expected[name] = values
            keep.append(hot)
        placed["keep"] = keep
        return placed, expected

    def assert_case(self, placed, expected, ids):
        streamer = ExpertStreamer.__new__(ExpertStreamer)
        streamer.names = list(self.SPECS)
        streamer.device = torch.device("cuda")
        ids2d = torch.tensor(ids, dtype=torch.int32, device="cuda")
        flat = ids2d.reshape(-1)
        inverse = torch.arange(flat.numel(), dtype=torch.int32, device="cuda")
        new_ids, staged = streamer._gather_dma(placed, flat, inverse, ids2d)
        for name in self.SPECS:
            actual = staged[name].index_select(
                0, new_ids.reshape(-1).to(torch.int64)
            ).cpu()
            wanted = expected[name].index_select(0, flat.cpu().to(torch.int64))
            self.assertTrue(torch.equal(actual, wanted), name)

    def test_mixed_all_host_all_resident_and_duplicates(self):
        placed, expected = self.make_placement([1, 4, 6])
        self.assert_case(placed, expected, [[1, 2, 4, 7], [2, 2, 6, 3]])
        self.assert_case(placed, expected, [[0, 2, 3, 5]])
        self.assert_case(placed, expected, [[1, 4, 6, 1]])

        # Simulate elastic growth by mutating the home lists already exposed
        # through layer._placed, as ExpertElastic.resize_layer does.
        grown, expected = self.make_placement([1, 2, 3, 4, 6])
        for name in self.SPECS:
            placed["addr"][name] = grown["addr"][name]
            placed["proto"][name] = grown["proto"][name]
            placed["home"][name][:] = grown["home"][name]
        placed["keep"].extend(grown["keep"])
        self.assert_case(placed, expected, [[2, 3, 5, 7]])


if __name__ == "__main__":
    unittest.main()
