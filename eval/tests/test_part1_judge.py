"""Part 1 judge conformance subset (V2.1 event judge, no MuJoCo needed).

Covers the rule-critical transitions that the migration must not break:
approach ground is not a failure, bridge traversal + landing-after-platform,
failure retry / success finality, height-bar gate + bar-drop, high-wall
support-point projection (and its legacy base-z fallback), slope projection,
and the attempt ledger.
"""

from __future__ import annotations

import unittest

from robocon_eval.part1.judge import AttemptManager, Part1JudgeV2

LEGS = ("FL", "FR", "RL", "RR")


def feed(judge: Part1JudgeV2, samples, *, dt: float = 0.02) -> dict:
    report = None
    for i, sample in enumerate(samples):
        tags = {leg: set(sample.get("tags", ())) for leg in LEGS}
        report = judge.update(
            i * dt, (sample["x"], sample["y"], sample.get("z", 0.45)),
            sample.get("yaw", 0.0), foot_tags=tags,
            ground_feet=set(sample.get("ground", ())), bar_z=sample.get("bar_z"),
            support_xy=sample.get("support"))
    return report


def state(report: dict, name: str) -> dict:
    return report["states"][name]


class TestApproachAndRetry(unittest.TestCase):
    def test_approach_ground_is_not_failure(self):
        cases = [
            ("bridge_blocks", [(0.8, 0.0), (1.3, 0.0)]),
            ("gravel_pit", [(5.0, -13.6), (5.0, -13.2)]),
            ("t_stairs", [(1.8, -5.4), (1.8, -4.9)]),
            ("slope", [(3.6, -5.0), (4.5, -5.0)]),
        ]
        for name, path in cases:
            report = feed(Part1JudgeV2(),
                          [{"x": x, "y": y, "ground": list(LEGS)} for x, y in path])
            self.assertEqual(state(report, name)["failures"], [], name)
            self.assertNotEqual(state(report, name)["status"], "success", name)

    def test_failure_can_be_retried_and_success_is_final(self):
        judge = Part1JudgeV2()
        report = feed(judge, [
            {"x": 0.5, "y": 0.0},
            {"x": 1.35, "y": 0.0, "tags": ["platform_left"]},
            {"x": 2.0, "y": 0.0, "tags": ["bridge"]},
            {"x": 2.3, "y": 0.0, "tags": ["bridge"], "ground": ["FL", "FR"]},
        ])
        self.assertTrue(state(report, "bridge_blocks")["failures"])
        legal = [
            {"x": 0.5, "y": 0.0},
            {"x": 1.35, "y": 0.0, "tags": ["platform_left"]},
            {"x": 2.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 3.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 4.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.6, "y": 0.0, "tags": ["platform_right"]},
            {"x": 6.4, "y": 0.0, "tags": ["platform_right"]},
        ]
        report = feed(judge, legal)
        self.assertEqual(state(report, "bridge_blocks")["status"], "success")
        self.assertEqual(state(report, "bridge_blocks")["score"], 100)


class TestBridge(unittest.TestCase):
    def test_legal_traversal_landing_and_mid_gap_fall(self):
        legal = [
            {"x": 0.5, "y": 0.0},
            {"x": 1.35, "y": 0.0, "tags": ["platform_left"]},
            {"x": 2.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 3.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 4.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.6, "y": 0.0, "tags": ["platform_right"]},
            {"x": 6.4, "y": 0.0, "tags": ["platform_right"]},
        ]
        report = feed(Part1JudgeV2(), legal)
        self.assertEqual(state(report, "bridge_blocks")["status"], "success")

        landing = [
            {"x": 0.5, "y": 0.0},
            {"x": 1.35, "y": 0.0, "tags": ["platform_left"]},
            {"x": 2.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 3.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 4.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.6, "y": 0.0, "tags": ["platform_right"]},
            {"x": 6.25, "y": 0.0, "tags": ["platform_right"], "ground": ["FL", "FR"]},
            {"x": 6.45, "y": 0.0},
        ]
        report = feed(Part1JudgeV2(), landing)
        self.assertEqual(state(report, "bridge_blocks")["status"], "success")

        mid_gap = [
            {"x": 0.5, "y": 0.0},
            {"x": 1.35, "y": 0.0, "tags": ["platform_left"]},
            {"x": 3.2, "y": 0.0, "tags": ["bridge"], "ground": ["FL", "FR"]},
            {"x": 4.0, "y": 0.0},
        ]
        report = feed(Part1JudgeV2(), mid_gap)
        self.assertTrue(state(report, "bridge_blocks")["failures"])

    def test_wrong_exit_side_fails(self):
        # enter and leave through the same x side: not a crossing
        same_side = [
            {"x": 1.35, "y": 0.0, "tags": ["platform_left"]},
            {"x": 2.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 3.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 4.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.2, "y": 0.0, "tags": ["bridge"]},
            {"x": 5.6, "y": 0.0, "tags": ["platform_right"]},
            {"x": 5.9, "y": 0.0, "tags": ["platform_right"], "ground": ["FL", "FR"]},
            {"x": 6.1, "y": 0.0},
            {"x": 5.5, "y": 0.0},
            {"x": 1.3, "y": 0.0, "ground": list(LEGS)},
        ]
        report = feed(Part1JudgeV2(), same_side)
        self.assertNotEqual(state(report, "bridge_blocks")["status"], "success")


class TestHeightBar(unittest.TestCase):
    def test_under_bar_success_drop_and_high_pass(self):
        base = [
            {"x": 5.7, "y": -10.4, "z": 0.25, "bar_z": 0.325},
            {"x": 5.7, "y": -9.5, "z": 0.25, "bar_z": 0.325},
            {"x": 5.7, "y": -9.0, "z": 0.25, "bar_z": 0.325},
            {"x": 5.7, "y": -8.5, "z": 0.25, "bar_z": 0.325},
            {"x": 5.7, "y": -8.2, "z": 0.25, "bar_z": 0.325},
        ]
        report = feed(Part1JudgeV2(), base)
        self.assertEqual(state(report, "height_bar")["status"], "success")

        high = [dict(sample, z=0.45) for sample in base]
        report = feed(Part1JudgeV2(), high)
        self.assertNotEqual(state(report, "height_bar")["status"], "success")
        self.assertTrue(any("not_under_bar" in f
                            for f in state(report, "height_bar")["failures"]))

        dropped = [dict(sample, bar_z=(0.20 if i >= 2 else 0.325))
                   for i, sample in enumerate(base)]
        report = feed(Part1JudgeV2(), dropped)
        self.assertTrue(any("bar_dropped" in f
                            for f in state(report, "height_bar")["failures"]))


class TestHighWall(unittest.TestCase):
    @staticmethod
    def _support(y):
        return {"FL": (1.55, y, 0.20), "FR": (2.05, y, 0.20),
                "RL": (1.55, y, 0.20), "RR": (2.05, y, 0.20)}

    def test_support_projection_success_and_side_bypass(self):
        path = [{"x": 1.8, "y": y, "support": self._support(y)}
                for y in (-5.9, -6.4, -7.2, -7.8, -8.1)]
        report = feed(Part1JudgeV2(), path)
        self.assertEqual(state(report, "high_wall")["status"], "success")
        self.assertEqual(state(report, "high_wall")["score"], 100)

        bad = []
        for sample in path:
            sample = dict(sample)
            support = dict(sample["support"])
            support["FR"] = (2.6, support["FR"][1], 0.20)   # outside wall x span
            sample["support"] = support
            bad.append(sample)
        report = feed(Part1JudgeV2(), bad)
        self.assertNotEqual(state(report, "high_wall")["status"], "success")

        stuck = []
        for sample in path:
            sample = dict(sample)
            support = dict(sample["support"])
            support["RR"] = (2.05, -6.9, 0.20)              # never crosses
            sample["support"] = support
            stuck.append(sample)
        report = feed(Part1JudgeV2(), stuck)
        self.assertNotEqual(state(report, "high_wall")["status"], "success")

    def test_legacy_base_z_climb_proxy(self):
        # old traces without support positions fall back to the base-z proxy
        climb = [
            {"x": 1.8, "y": -7.9},
            {"x": 1.8, "y": -7.4, "z": 0.75, "tags": ["wall"]},
            {"x": 1.8, "y": -7.0, "z": 0.85, "tags": ["wall"]},
            {"x": 1.8, "y": -6.6, "z": 0.75, "tags": ["wall"]},
            {"x": 1.8, "y": -6.0},
        ]
        report = feed(Part1JudgeV2(), climb)
        self.assertEqual(state(report, "high_wall")["status"], "success")

        walk_around = [{"x": 1.8, "y": -7.9}, {"x": 1.8, "y": -7.4},
                       {"x": 1.8, "y": -7.0}, {"x": 1.8, "y": -6.6},
                       {"x": 1.8, "y": -6.0}]
        report = feed(Part1JudgeV2(), walk_around)
        self.assertNotEqual(state(report, "high_wall")["status"], "success")


class TestSlope(unittest.TestCase):
    def test_long_axis_projection(self):
        legal = [{"x": 4.6, "y": -6.6}]
        legal += [{"x": 4.6, "y": -6.6 + 0.1 * i, "tags": ["slope"]} for i in range(1, 16)]
        legal += [{"x": 4.6, "y": -5.0, "tags": ["slope"]},
                  {"x": 4.6, "y": -3.8}]
        report = feed(Part1JudgeV2(), legal)
        self.assertEqual(state(report, "slope")["status"], "success")
        self.assertEqual(state(report, "slope")["score"], 100)

        spoof = [{"x": 3.6, "y": -5.0}, {"x": 4.5, "y": -5.0},
                 {"x": 5.5, "y": -5.0}, {"x": 6.5, "y": -5.0}]
        report = feed(Part1JudgeV2(), spoof)
        self.assertNotEqual(state(report, "slope")["status"], "success")


class TestAttemptLedger(unittest.TestCase):
    def test_attempts_and_summary(self):
        manager = AttemptManager(max_attempts=3)
        self.assertTrue(manager.begin(0.0))
        manager.note_recovery(1.0)          # recovery does not consume the attempt
        manager.end(2.0, {"score": 500.0})
        self.assertTrue(manager.begin(3.0))
        manager.skip(4.0)                   # operator skip consumes the attempt
        self.assertTrue(manager.begin(5.0))
        manager.end(9.0, {"score": 900.0})
        self.assertFalse(manager.begin(10.0))
        summary = manager.summary()
        self.assertEqual(summary["attempts_used"], 3)
        self.assertEqual(summary["final_score"], 900.0)
        self.assertEqual(summary["final_time_s"], 4.0)
        self.assertEqual(summary["skips"], 1)
        self.assertEqual(summary["recoveries"], 1)

    def test_finalize_marks_unfinished_obstacles_failed(self):
        judge = Part1JudgeV2()
        judge.update(0.0, (1.8, -7.9, 0.45))
        report = judge.finalize("timeout")
        self.assertEqual(report["states"]["high_wall"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
