"""Part 1 whole-course protocol: attempts, pause/recovery, skip, timer, bonus.

The course bookkeeping is pure Python (no MuJoCo): the fixture event streams are
replayed against ``Part1Course``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from robocon_eval import rules
from robocon_eval.part1.course import Part1Course, StableStopTracker

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "course"


def replay(events):
    course = Part1Course()
    course.begin_attempt(0.0)
    t = 0.0
    by_t = sorted(events, key=lambda event: event["t"])
    i = 0
    while t <= 3.0:
        while i < len(by_t) and by_t[i]["t"] <= t:
            event = by_t[i]
            if event["event"] == "pause":
                course.pause(t)
            elif event["event"] == "resume":
                course.resume(t)
            elif event["event"] == "abort":
                return course.abort_attempt(t), course.summary()
            i += 1
        course.update(t, (3.7, -9.0, 0.45))
        t = round(t + 0.02, 6)
    return course.end_attempt(t), course.summary()


class TestCourseEvents(unittest.TestCase):
    def test_pause_resume_does_not_consume_attempt(self):
        events = json.loads((FIX / "part1-course-events-example.json").read_text())
        _, summary = replay(events)
        self.assertEqual(summary["attempts_used"], 1)
        self.assertEqual(summary["recoveries"], 1)
        kinds = {event["event"] for event in summary["events"]}
        self.assertIn("resume", kinds)

    def test_abort_consumes_attempt(self):
        events = json.loads((FIX / "part1-course-events-abort.json").read_text())
        result, summary = replay(events)
        self.assertTrue(result.skipped)
        self.assertEqual(summary["attempts_used"], 1)
        self.assertEqual(result.reason, "operator_abort")

    def test_recovery_pose_violation_zeroes_attempt(self):
        course = Part1Course()
        self.assertTrue(course.begin_attempt(0.0))
        for obstacle in course.judge.obstacles.values():
            obstacle.result.status = "success"
            obstacle.result.score = 100.0
        course.update(0.1, (3.7, -9.0, 0.45))
        course.pause(0.2, (3.7, -9.0, 0.45), 0.0)
        # teleport far beyond the documented measurement slack -> attempt invalid
        course.resume(1.0, (2.0, -9.0, 0.45), 0.0)
        course.update(1.1, (3.7, -9.0, 0.45))
        result = course.end_attempt(1.2)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.reason, "recovery_pose_violation")

    def test_recovery_within_tolerance_is_allowed(self):
        course = Part1Course()
        self.assertTrue(course.begin_attempt(0.0))
        course.update(0.1, (3.7, -9.0, 0.45), 0.0)
        course.pause(0.2, (3.7, -9.0, 0.45), 0.0)
        moved = rules.PART1_RECOVERY_POS_TOL_M - 0.05
        course.resume(1.0, (3.7 + moved, -9.0, 0.45), 0.0)
        self.assertFalse(course.recovery_violation)

    def test_return_bonus_requires_full_course_then_start_area(self):
        course = Part1Course()
        self.assertTrue(course.begin_attempt(0.0))
        for obstacle in course.judge.obstacles.values():
            obstacle.result.status = "success"
            obstacle.result.score = 100.0
        course.update(0.0, (3.7, -9.0, 0.45))    # first start-area entry
        course.update(0.1, (2.8, -9.0, 0.45))    # leave
        course.update(0.2, (3.7, -9.0, 0.45))    # re-enter after all full -> bonus
        result = course.end_attempt(0.3)
        self.assertEqual(result.score, 900.0)
        self.assertEqual(result.return_bonus, 100.0)

    def test_partial_t_stairs_blocks_bonus(self):
        course = Part1Course()
        self.assertTrue(course.begin_attempt(0.0))
        for obstacle in course.judge.obstacles.values():
            obstacle.result.status = "success"
            obstacle.result.score = 50.0 if obstacle.name == "t_stairs" else 100.0
        course.update(0.1, (3.7, -9.0, 0.45))
        course.update(0.2, (2.8, -9.0, 0.45))
        course.update(0.3, (3.7, -9.0, 0.45))
        result = course.end_attempt(0.4)
        self.assertEqual(result.return_bonus, 0.0)
        self.assertEqual(result.score, 750.0)

    def test_obstacle_skip_scores_zero_and_keeps_attempt(self):
        course = Part1Course()
        self.assertTrue(course.begin_attempt(0.0))
        course.update(0.1, (3.7, -9.0, 0.45))
        course.skip_obstacle(0.2, "slalom")
        report = course.update(0.3, (3.7, -9.0, 0.45))
        self.assertEqual(report["states"]["slalom"]["status"], "skipped")
        result = course.end_attempt(0.4)
        self.assertFalse(result.skipped)
        self.assertEqual(result.score, 0.0)


class TestStableStopTracker(unittest.TestCase):
    def test_requires_hold_time(self):
        tracker = StableStopTracker()
        self.assertFalse(tracker.update(0.05, 3.0, 0.002))
        held = 0.0
        stopped = False
        while held < 1.0:
            held += 0.002
            stopped = tracker.update(0.05, 3.0, 0.002)
            if stopped:
                break
        self.assertTrue(stopped)
        self.assertAlmostEqual(tracker.held, rules.PART1_STOP_HOLD_S, places=9)

    def test_speed_spike_resets_hold(self):
        tracker = StableStopTracker()
        tracker.update(0.05, 3.0, 0.40)
        self.assertEqual(tracker.held, 0.40)
        tracker.update(0.30, 3.0, 0.002)
        self.assertEqual(tracker.held, 0.0)


if __name__ == "__main__":
    unittest.main()
