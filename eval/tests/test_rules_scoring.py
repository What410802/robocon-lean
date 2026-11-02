"""The machine-readable standard: rules constants + scoring semantics.

Every score number lives in ``robocon_eval.rules``; these tests pin the official
curves (including the floor interpolation) and Part 1 attempt aggregation so a
migration change cannot silently drift a score.
"""

from __future__ import annotations

import unittest

from robocon_eval import rules, scoring


class TestRulesInvariants(unittest.TestCase):
    def test_part1_max_score(self):
        self.assertEqual(len(rules.PART1_OBSTACLES), 8)
        self.assertEqual(rules.PART1_OBSTACLE_SCORE, 100)
        self.assertEqual(rules.PART1_RETURN_BONUS, 100)
        self.assertEqual(rules.PART1_MAX_SCORE, 900)
        self.assertEqual(rules.part1()["max_score"], 900)

    def test_t_stairs_partials(self):
        self.assertEqual(rules.PART1_T_STAIRS_SCORE, 100)
        self.assertEqual(rules.PART1_T_STAIRS_PARTIAL_UP, 50)
        self.assertEqual(rules.PART1_T_STAIRS_PARTIAL_DOWN, 50)
        self.assertEqual(rules.PART1_T_STAIRS_SAME_ARM, 50)
        stairs = [o for o in rules.part1()["obstacles"] if o["id"] == "t_stairs"][0]
        self.assertEqual(stairs["partial_scores"]["one_side_up"], 50)
        self.assertEqual(stairs["partial_scores"]["one_side_down"], 50)

    def test_part2_attempts_rounding_and_obstacles(self):
        self.assertEqual(len(rules.PART2_OBSTACLES), 6)
        self.assertEqual(rules.PART2_ATTEMPTS_PER_OBSTACLE, 4)
        self.assertEqual(rules.PART2_ROUNDING, "floor")
        self.assertEqual(rules.PART2_MAX_SCORE, 1300)
        self.assertEqual(sum(rules.PART2_OBSTACLE_MAX_SCORE.values()), 1300)
        # Safety cap only: Part 2 is untimed by rule (do not use 15-20 s caps).
        self.assertGreaterEqual(rules.PART2_SAFETY_TIMEOUT_S, 120.0)
        for obstacle in rules.part2()["obstacles"]:
            for point in obstacle["score_curve"]:
                self.assertIn("value", point)
                self.assertIn("points", point)

    def test_robot_compliance_constants(self):
        self.assertEqual(rules.TORQUE_LIMITS_NM, {"hip": 25.0, "thigh": 25.0, "calf": 60.0})
        self.assertIn("imu", rules.SENSORS_ALLOWED)
        self.assertIn("camera", rules.SENSORS_ALLOWED)
        for forbidden in ("body_lin_vel", "heightmap", "foot_force", "foot_height",
                          "body_height"):
            self.assertIn(forbidden, rules.SENSORS_FORBIDDEN)
        self.assertTrue(rules.PART2_RL_ONLY)
        self.assertTrue(rules.PART1_FIXED_ACTIONS_ALLOWED)


class TestPart2Curves(unittest.TestCase):
    def setUp(self):
        self.p2 = rules.part2()

    def test_endpoints_and_clamps(self):
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", 10), 10)
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", 40), 200)
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", 5), 0)
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", 50), 200)
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", None), 0)

    def test_floor_interpolation(self):
        # 15 cm platform: 10 + 40*(5/10) = 30; 25 cm: 50 + 50*(5/10) = 75.
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", 15), 30)
        self.assertEqual(scoring.score_for_level(self.p2, "high_platform", 25), 75)
        # 17 cm stairs: 50 + 50*(2/5) = 70.
        self.assertEqual(scoring.score_for_level(self.p2, "stairs", 17), 70)
        # 12.3 cm platform: 10 + 40*2.3/10 = 19.2 -> floor 19.
        obstacle = next(o for o in self.p2["obstacles"] if o["id"] == "high_platform")
        self.assertEqual(scoring.curve_points_floor(obstacle, 12.3), 19)

    def test_trench_linear_and_wall_endpoints(self):
        self.assertEqual(scoring.score_for_level(self.p2, "trench", 10), 0)
        self.assertEqual(scoring.score_for_level(self.p2, "trench", 20), 150)
        self.assertEqual(scoring.score_for_level(self.p2, "trench", 30), 300)
        self.assertEqual(scoring.score_for_level(self.p2, "high_wall", 10), 0)
        self.assertEqual(scoring.score_for_level(self.p2, "high_wall", 20), 50)
        self.assertEqual(scoring.score_for_level(self.p2, "high_wall", 40), 200)

    def test_plum_bool(self):
        self.assertEqual(scoring.score_for_level(self.p2, "plum", 1), 200)
        self.assertEqual(scoring.score_for_level(self.p2, "plum", 0), 0)

    def test_total_is_1300(self):
        best = {"high_platform": 40, "stairs": 30, "slope": 45,
                "high_wall": 40, "trench": 30, "plum": 1}
        result = scoring.part2_score(self.p2, best)
        self.assertEqual(result["total"], 1300)
        self.assertEqual(set(result["by_obstacle"]), set(rules.PART2_OBSTACLES))
        self.assertEqual(scoring.part2_score(self.p2, {})["total"], 0)


class TestPart1Scoring(unittest.TestCase):
    def setUp(self):
        self.p1 = rules.part1()

    def _full_tasks(self):
        return {o["id"]: {"success": True, "score": 100, "skipped": False}
                for o in self.p1["obstacles"]}

    def test_full_course_with_return_bonus(self):
        result = scoring.part1_attempt_score(self.p1, self._full_tasks(),
                                             returned_to_start=True)
        self.assertEqual(result["obstacle_score"], 800)
        self.assertEqual(result["return_bonus"], 100)
        self.assertEqual(result["total"], 900)
        self.assertTrue(result["all_full"])

    def test_partial_t_stairs_blocks_return_bonus(self):
        tasks = self._full_tasks()
        tasks["t_stairs"] = {"success": True, "score": 50, "skipped": False}
        result = scoring.part1_attempt_score(self.p1, tasks, returned_to_start=True)
        self.assertEqual(result["obstacle_score"], 750)
        self.assertEqual(result["return_bonus"], 0)
        self.assertEqual(result["total"], 750)

    def test_skip_scores_zero_and_blocks_bonus(self):
        tasks = self._full_tasks()
        tasks["slalom"] = {"success": False, "score": 0, "skipped": True}
        result = scoring.part1_attempt_score(self.p1, tasks, returned_to_start=True)
        self.assertEqual(result["by_obstacle"]["slalom"], 0)
        self.assertEqual(result["return_bonus"], 0)
        self.assertFalse(result["all_full"])

    def test_session_best_score_then_fastest(self):
        attempts = [
            {"score": 700, "duration_s": 50.0},
            {"score": 900, "duration_s": 80.0},
            {"score": 900, "duration_s": 70.0},
            {"score": 0, "duration_s": 10.0, "skipped": True},
        ]
        summary = scoring.part1_session_summary(attempts)
        self.assertEqual(summary["final_score"], 900)
        self.assertEqual(summary["final_time_s"], 70.0)
        self.assertEqual(summary["attempts_used"], 4)


if __name__ == "__main__":
    unittest.main()
