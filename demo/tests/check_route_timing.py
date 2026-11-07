#!/usr/bin/env python3
"""Check Part 2 timing with synthetic state timestamps, without ROS nodes."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl_sar/scripts"))
import part2_motion as route  # noqa: E402


class StateEvent:
    def __init__(self, owner, sample_period, *, progress=True, advance_on_clear=False):
        self.owner, self.sample_period = owner, sample_period
        self.progress, self.advance_on_clear = progress, advance_on_clear
        self.waits = 0

    def clear(self):
        if self.advance_on_clear:
            self.owner._sim_t += self.sample_period
            self.advance_on_clear = False

    def wait(self, timeout):
        self.waits += 1
        self.owner.wall += 0.1
        if self.progress:
            self.owner._sim_t += self.sample_period
        return self.progress


class TimedRoute(route.Part2Motion):
    def __init__(self, sample_period, **event_options):
        # Exercise real _tick/_hold/_now without creating a Node or thread.
        self._steplock = True
        self._sim_t = self.wall = 0.0
        self.sent_at = []
        self._frame_evt = StateEvent(self, sample_period, **event_options)

    def _send_joy(self):
        self.sent_at.append(self._sim_t)


class RouteTimingTests(unittest.TestCase):
    def test_dense_and_sparse_feedback_keep_the_same_control_period_and_hold(self):
        histories = []
        for period in (0.002, 0.020):
            with self.subTest(feedback_period=period):
                node = TimedRoute(period)
                self.assertEqual(node._now(), 0.0, "zero simulation time used wall clock")
                node._tick()
                self.assertAlmostEqual(node._sim_t, 0.020)
                self.assertEqual(len(node.sent_at), 1)
                node._hold(0.12)
                self.assertAlmostEqual(node._sim_t, 0.14)
                self.assertEqual(len(node.sent_at), 7)
                histories.append(node.sent_at)
        for dense, sparse in zip(*histories):
            self.assertAlmostEqual(dense, sparse)

    def test_final_frame_between_event_clear_and_read_finishes_without_wait(self):
        node = TimedRoute(0.020, advance_on_clear=True)
        node._tick()
        self.assertAlmostEqual(node._sim_t, 0.020)
        self.assertEqual(node.sent_at, [0.0])
        self.assertEqual(node._frame_evt.waits, 0)

    def test_fractional_holds_round_to_the_same_whole_control_periods(self):
        for duration, expected in ((0.0, 0.0), (0.15, 0.16), (0.45, 0.46)):
            for period in (0.002, 0.020):
                with self.subTest(duration=duration, feedback_period=period):
                    node = TimedRoute(period)
                    node._hold(duration)
                    self.assertAlmostEqual(node._sim_t, expected)
                    self.assertEqual(len(node.sent_at), round(expected / 0.020))
                    for previous, following in zip(node.sent_at, node.sent_at[1:]):
                        self.assertAlmostEqual(following - previous, 0.020)

    def test_missing_state_progress_has_a_wall_clock_watchdog(self):
        node = TimedRoute(0.020, progress=False)
        with patch.object(route.time, "monotonic", side_effect=lambda: node.wall):
            with self.assertRaisesRegex(RuntimeError, "no /eval_state progress"):
                node._tick()
        self.assertLessEqual(node._frame_evt.waits, 21)
        self.assertEqual(node._sim_t, 0.0)
        self.assertEqual(node.sent_at, [0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
