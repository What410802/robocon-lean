"""T-stairs traversal regressions (the 2026-11-02 judge fix).

Rules: climb one arm (levels 1->2->3) and descend **another** arm (3->2->1) for
100; up+down the same arm = 50; one direction only = 50; a skipped level = fail.

Two regressions are guarded here:

* the realistic robot is longer than a step, so front and rear supports are on
  different levels at the same sample and first-touch order alone is not
  monotone -- the judge must also accept a last-touch monotone order;
* the old judge froze the traversal evidence on the *first ground contact after
  the top* (front wheels touch the floor while the rear wheels are still on the
  lowest step), which discarded the descent evidence and commonly capped a legal
  full traversal at 50.  ``test_south_up_north_down_is_full`` exercises exactly
  that moment in the geometric feed.

Tags are produced from the real tagged scene geometry
(``demo/robots/blackW/scene_part1.xml``) so the feed is representative rather
than a hand-written tag sequence.
"""

from __future__ import annotations

import unittest

from robocon_eval.part1.judge import Part1JudgeV2

LEGS = ("FL", "FR", "RL", "RR")
# body-frame (dx, dy) of the four wheel supports: axles +-0.2 m, track +-0.3 m
OFFSETS = ((0.2, -0.3), (0.2, 0.3), (-0.2, -0.3), (-0.2, 0.3))

# Tagged scene geometry (t_stairs zones): x span of the south/north arms is
# [1.30, 2.30]; the east arm lies inside y [-4.00, -3.00].
ARM_X = (1.30, 2.30)
TOP_Y = (-4.00, -3.00)
SOUTH = (((-4.90, -4.60), "step:s:1"), ((-4.60, -4.30), "step:s:2"),
         ((-4.30, -4.00), "step:s:3"))
NORTH = (((-3.00, -2.70), "step:n:3"), ((-2.70, -2.40), "step:n:2"),
         ((-2.40, -2.10), "step:n:1"))
EAST = (((2.30, 2.60), "step:e:3"), ((2.60, 2.90), "step:e:2"),
        ((2.90, 3.20), "step:e:1"))


def _tag(x: float, y: float) -> str | None:
    if ARM_X[0] <= x <= ARM_X[1]:
        if TOP_Y[0] <= y <= TOP_Y[1]:
            return "step:top"
        for (lo, hi), tag in SOUTH + NORTH:
            if lo <= y <= hi:
                return tag
    if TOP_Y[0] <= y <= TOP_Y[1]:
        for (lo, hi), tag in EAST:
            if lo <= x <= hi:
                return tag
    return None


def _frame(t: float, x: float, y: float) -> dict:
    """One judged sample: supports from body-frame offsets, mapped to scene tags."""
    tags: dict[str, set[str]] = {leg: set() for leg in LEGS}
    ground: set[str] = set()
    for leg, (dx, dy) in zip(LEGS, OFFSETS):
        tag = _tag(x + dx, y + dy)
        if tag is None:
            ground.add(leg)
        else:
            tags[leg].add(tag)
    return {"t": t, "base": [x, y, 0.45], "foot_tags": tags, "ground_feet": ground}


def _feed_path(points: list[tuple[float, float]], *, dt: float = 0.02,
               speed: float = 0.5) -> Part1JudgeV2:
    judge = Part1JudgeV2()
    t = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(1, int(round(max(abs(x1 - x0), abs(y1 - y0)) / (speed * dt))))
        for i in range(steps + 1):
            x = x0 + (x1 - x0) * i / steps
            y = y0 + (y1 - y0) * i / steps
            frame = _frame(t, x, y)
            judge.update(t, tuple(frame["base"]), 0.0,
                         foot_tags=frame["foot_tags"],
                         ground_feet=frame["ground_feet"], bar_z=None)
            t += dt
    return judge


def _feed_samples(samples: list[dict], *, dt: float = 0.02) -> Part1JudgeV2:
    judge = Part1JudgeV2()
    for i, sample in enumerate(samples):
        tags = {leg: set(sample.get("tags", ())) for leg in LEGS}
        judge.update(i * dt, (sample["x"], sample["y"], sample.get("z", 0.45)), 0.0,
                     foot_tags=tags, ground_feet=set(sample.get("ground", ())),
                     bar_z=None)
    return judge


def _stairs(report: dict) -> dict:
    return report["states"]["t_stairs"]


class TestTStairsTraversal(unittest.TestCase):
    APPROACH_Y = -5.6
    TOP_Y = -3.5
    EXIT_NORTH_Y = -1.8
    EXIT_EAST_X = 3.5

    def test_south_up_north_down_is_full(self):
        """The pre-fix freeze-on-first-ground-touch bug: this used to score 50."""
        judge = _feed_path([(1.8, self.APPROACH_Y), (1.8, self.EXIT_NORTH_Y)])
        report = judge.finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "success", state)
        self.assertEqual(state["score"], 100.0, state)
        self.assertEqual(state["detail"]["up_arms"], ["s"])
        self.assertEqual(state["detail"]["down_arms"], ["n"])
        self.assertEqual(report["score"], 100.0)

    def test_south_up_east_down_is_full(self):
        judge = _feed_path([(1.8, self.APPROACH_Y), (1.8, self.TOP_Y),
                            (self.EXIT_EAST_X, self.TOP_Y)])
        report = judge.finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "success", state)
        self.assertEqual(state["score"], 100.0, state)
        self.assertEqual(state["detail"]["up_arms"], ["s"])
        self.assertEqual(state["detail"]["down_arms"], ["e"])

    def test_same_arm_up_down_is_partial(self):
        judge = _feed_path([(1.8, self.APPROACH_Y), (1.8, self.TOP_Y),
                            (1.8, self.APPROACH_Y)])
        report = judge.finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "success", state)
        self.assertEqual(state["score"], 50.0, state)

    def test_up_only_is_partial(self):
        judge = _feed_path([(1.8, self.APPROACH_Y), (1.8, self.TOP_Y)])
        # fall off the stairs to the entry ground: leaves the zone without descent
        exit_frame = _frame(999.0, 1.8, self.APPROACH_Y)
        judge.update(exit_frame["t"], tuple(exit_frame["base"]), 0.0,
                     foot_tags=exit_frame["foot_tags"],
                     ground_feet=exit_frame["ground_feet"], bar_z=None)
        report = judge.finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "success", state)
        self.assertEqual(state["score"], 50.0, state)

    def test_down_only_is_partial(self):
        judge = _feed_path([(1.8, self.TOP_Y), (self.EXIT_EAST_X, self.TOP_Y)],
                           speed=0.5)
        # start directly on the top platform (e.g. attempt starts there)
        report = judge.finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "success", state)
        self.assertEqual(state["score"], 50.0, state)

    def test_skipped_level_fails(self):
        skipped = [
            {"x": 1.8, "y": -5.4},
            {"x": 1.8, "y": -4.9, "tags": ["step:s:1"]},
            {"x": 1.8, "y": -4.5, "tags": ["step:s:3"]},   # s:2 skipped
            {"x": 1.8, "y": -3.8, "tags": ["step:top"]},
            {"x": 1.8, "y": -2.9, "tags": ["step:n:3"]},
            {"x": 1.8, "y": -2.55, "tags": ["step:n:2"]},
            {"x": 1.8, "y": -2.25, "tags": ["step:n:1"]},
            {"x": 1.8, "y": -1.9},
        ]
        report = _feed_samples(skipped).finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "fail", state)
        self.assertEqual(state["score"], 0.0, state)
        self.assertIn("incomplete_step_history", state["failures"])

    def test_incomplete_descent_does_not_upgrade_partial(self):
        # climb s to the top, dip two levels down the north arm and come back to
        # the top, then leave: no complete down direction, so one direction only
        # (50) -- the partial descent must not be read as a full crossing.
        samples = [
            {"x": 1.8, "y": -5.4},
            {"x": 1.8, "y": -4.9, "tags": ["step:s:1"]},
            {"x": 1.8, "y": -4.6, "tags": ["step:s:2"]},
            {"x": 1.8, "y": -4.3, "tags": ["step:s:3"]},
            {"x": 1.8, "y": -3.8, "tags": ["step:top"]},
            {"x": 1.8, "y": -3.0, "tags": ["step:n:3"]},
            {"x": 1.8, "y": -2.55, "tags": ["step:n:2"]},
            {"x": 1.8, "y": -3.8, "tags": ["step:top"]},
            {"x": 1.8, "y": -5.4},
        ]
        report = _feed_samples(samples).finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "success", state)
        self.assertEqual(state["score"], 50.0, state)

    def test_trace_replay_full_traversal(self):
        """Same geometric traversal fed through the JSONL trace replay path."""
        import json
        import tempfile
        from pathlib import Path

        from robocon_eval.trace import replay_part1

        raw = [_frame(0.02 * i, 1.8, -5.6 + 0.01 * i) for i in range(381)]
        frames = [{"t": frame["t"], "base": frame["base"],
                   "foot_tags": {leg: sorted(tags) for leg, tags in frame["foot_tags"].items()},
                   "ground_feet": sorted(frame["ground_feet"])}
                  for frame in raw]
        meta = {"kind": "meta", "schema": 1, "part": 1, "robot": "blackW"}
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t_stairs.jsonl"
            with trace.open("w") as fh:
                fh.write(json.dumps(meta) + "\n")
                for frame in frames:
                    fh.write(json.dumps({"kind": "frame", **frame}) + "\n")
            report = replay_part1(trace)
        self.assertEqual(report["states"]["t_stairs"]["status"], "success", report)
        self.assertEqual(report["states"]["t_stairs"]["score"], 100.0)
        self.assertEqual(report["score"], 100.0)

    def test_early_ground_contact_before_arm_complete_fails(self):
        samples = [
            {"x": 1.8, "y": -5.4},
            {"x": 1.8, "y": -4.9, "tags": ["step:s:1"]},
            {"x": 1.8, "y": -4.7, "ground": ["FL", "FR"]},
            {"x": 1.8, "y": -4.3},
        ]
        report = _feed_samples(samples).finalize("trace_end")
        state = _stairs(report)
        self.assertEqual(state["status"], "fail", state)
        self.assertTrue(any("two_feet_ground" in f for f in state["failures"]), state)


if __name__ == "__main__":
    unittest.main()
