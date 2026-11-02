"""P3-14 wall predicate logic probe (CPU torch, no mjlab rollout needed)."""
from __future__ import annotations

import unittest

import torch

from alldog_mjlab.tasks.velocity.profiles.part2.predicates import _part2_wall_crossings


class TestWallCrossings(unittest.TestCase):
    def test_all_inside_span(self):
        prev_x = torch.tensor([[2.4, 2.4, 2.4, 2.4]])
        prev_y = torch.tensor([[-0.4, 0.4, -0.4, 0.4]])
        x = torch.tensor([[2.6, 2.6, 2.6, 2.6]])
        y = torch.tensor([[-0.4, 0.4, -0.4, 0.4]])
        valid, side = _part2_wall_crossings(prev_x, prev_y, x, y, 2.5, 0.7)
        self.assertTrue(bool(valid.all()))
        self.assertFalse(bool(side.any()))

    def test_side_bypass(self):
        prev_x = torch.tensor([[2.4, 2.4, 2.4, 2.4]])
        prev_y = torch.tensor([[-0.4, 1.2, -0.4, 0.4]])
        x = torch.tensor([[2.6, 2.6, 2.6, 2.6]])
        y = torch.tensor([[-0.4, 1.2, -0.4, 0.4]])
        valid, side = _part2_wall_crossings(prev_x, prev_y, x, y, 2.5, 0.7)
        self.assertTrue(bool(side[0, 1]))
        self.assertFalse(bool(valid[0, 1]))

    def test_no_crossing(self):
        prev_x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        prev_y = torch.zeros(1, 4)
        x = torch.tensor([[2.1, 2.1, 2.1, 2.1]])
        valid, side = _part2_wall_crossings(prev_x, prev_y, x, torch.zeros(1, 4), 2.5, 0.7)
        self.assertFalse(bool(valid.any()))
        self.assertFalse(bool(side.any()))


if __name__ == "__main__":
    unittest.main()
