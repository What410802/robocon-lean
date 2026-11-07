#!/usr/bin/env python3
"""Offline Part 2 replay regression checks (MuJoCo/NumPy, no ROS required).

    python demo/tests/check_part2_replay.py

Synthetic wheel-support samples test coordinate mapping and judge plumbing;
they do not represent a physical-policy success claim.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "field_replay_under_test", ROOT / "tools/route/replay_field_part2.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def wall_support_frames(centre_x, lane_y=21.0):
    frames = []
    offsets = (-0.7, -0.3, -0.1, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6)
    for i, offset in enumerate(offsets):
        x = centre_x + offset
        feet = {}
        for name, dx, dy in (("FL", 0.25, 0.20), ("FR", 0.25, -0.20),
                             ("RL", -0.25, 0.20), ("RR", -0.25, -0.20)):
            feet[f"{name}_foot"] = [x + dx, lane_y + dy, replay.WHEEL_RADIUS_M]
        frames.append({"t": i / 10, "base": [x, lane_y, 0.55],
                       "quat": [1.0, 0.0, 0.0, 0.0], "feet": feet})
    return frames


def crossing_window(name, start=0.0, end=1.0):
    return [{"t": start, "leg": name, "cmd": f"{name}(policy=fixture)"},
            {"t": end, "leg": name, "cmd": "ensure_transfer"}]


def platform_support_frames(lo, hi, *, wrong_axis=False):
    """Top supports on a west-east platform, or outside its west face."""
    frames = []
    for i, along in enumerate((-.6, -.2, .35, .65, *([.75] * 10))):
        x = float(lo[0]) - .6 if wrong_axis else float(lo[0]) + along
        # This legal near-edge lane exposes the old S->N projection: it maps
        # y=-.59 just short of the official profile's platform x-range.
        y = float(lo[1]) + along if wrong_axis else -.59
        bottom_z = float(hi[2]) if i >= 2 else 0.0
        feet = {name + "_foot": [x + dx, y + dy, bottom_z + replay.WHEEL_RADIUS_M]
                for name, dx, dy in (("FL", .25, .20), ("FR", .25, -.20),
                                     ("RL", -.25, .20), ("RR", -.25, -.20))}
        frames.append({"t": i / 10, "base": [x, y, .55 + bottom_z],
                       "quat": [1.0, 0.0, 0.0, 0.0], "feet": feet})
    return frames


def continuous_wall_fixture():
    """All wheels clear wall16, then hold on floor before touching wall20."""
    first, second = "high_wall_16cm", "high_wall_20cm"
    plan = [{"t": 0.0, "leg": first, "cmd": f"{first}(policy=fixture)"},
            {"t": 0.18, "leg": first, "cmd": "cross_complete"},
            {"t": 0.20, "leg": second, "cmd": f"{second}(policy=fixture)"},
            {"t": 0.80, "leg": second, "cmd": "cross_complete"}]
    positions = [5.9, 6.3, 6.5, 6.9] + [7.05] * 47
    frames = []
    for i, x in enumerate(positions):
        feet = {name + "_foot": [x + dx, 21.0 + dy, replay.WHEEL_RADIUS_M]
                for name, dx, dy in (("FL", 0.25, 0.20), ("FR", 0.25, -0.20),
                                     ("RL", -0.25, 0.20), ("RR", -0.25, -0.20))}
        frames.append({"t": round(i * 0.02, 6), "base": [x, 21.0, 0.55],
                       "quat": [1.0, 0.0, 0.0, 0.0], "feet": feet, "contacts": []})
    return frames, plan


class Part2ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(replay.FIELD_MAP))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="part2-replay-check-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_replay(self, frames, plan, *, require_all=True, expected_legs=None):
        states_path, plan_path, result_path = (
            self.root / "states.jsonl", self.root / "plan.jsonl", self.root / "result.json")
        states_path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
        plan_path.write_text("".join(json.dumps(row) + "\n" for row in plan))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = replay.run(states_path, plan_path, replay.FIELD_MAP, result_path,
                                require_all, expected_legs=expected_legs)
        return result, json.loads(result_path.read_text())

    def test_exact_target_bounds_cover_all_canonical_tiers(self):
        self.assertEqual(len(replay.PLAN_LEGS), 33)
        self.assertEqual(len(replay.PLAN_BY_NAME), 33)
        for leg in replay.PLAN_LEGS:
            with self.subTest(leg=leg["leg"]):
                lo, hi = replay.field_bbox(self.model, leg)
                np.testing.assert_allclose((lo, hi), leg["target_bounds"],
                                           atol=2e-5, rtol=0)
        plum = replay.PLAN_BY_NAME["high_plum"]
        _, hi = replay.field_bbox(self.model, plum)
        self.assertLess(hi[1], 13, "practice plum rows leaked into the competition bbox")
        broken = {**plum, "target_geoms": (*plum["target_geoms"], "missing_support_geom")}
        with self.assertRaises((ValueError, RuntimeError, KeyError)):
            replay.field_bbox(self.model, broken)

    def test_zero_padded_route_names_normalise_to_manifest_keys(self):
        cases = {"high_platform_05cm": "high_platform_5cm",
                 "stairs_04cm": "stairs_4cm", "slope_05deg": "slope_5deg",
                 "high_wall_04cm": "high_wall_4cm", "high_plum": "plum_1"}
        for name, expected in cases.items():
            with self.subTest(leg=name):
                self.assertEqual(replay.official_level_key(replay.PLAN_BY_NAME[name]), expected)
        known = {entry["key"] for entry in replay.official.load_levels()}
        for name in ("high_platform_10cm", "stairs_10cm", "slope_20deg",
                     "trench_20cm", "high_wall_20cm", "high_plum"):
            self.assertIn(replay.official_level_key(replay.PLAN_BY_NAME[name]), known)

    def test_west_to_east_wall_support_crossing_reaches_official_goal(self):
        leg = replay.PLAN_BY_NAME["high_wall_20cm"]
        result, document = self.run_replay(wall_support_frames(7.8),
                                          crossing_window(leg["leg"]),
                                          expected_legs=[leg["leg"]])
        self.assertEqual(result, 0, document)
        row = document["legs"][0]
        self.assertEqual(row["status"], "success", row)
        self.assertEqual(row["judge_scope"], "official_profile_projection")
        self.assertEqual(row["score"], replay.scoring.score_for_level(
            replay.rules.part2(), "high_wall", 20))
        official = replay.official.get_level("high_wall_20cm")
        flo, fhi = replay.field_bbox(self.model, leg)
        mapping = replay.make_map("high_wall", flo, fhi,
                                  official["obstacle_bbox"]["min"],
                                  official["obstacle_bbox"]["max"])
        self.assertLess(mapping(7.7, 21)[0], mapping(7.9, 21)[0],
                        "wall projection still mirrors the old east-to-west route")

    def test_official_projection_uses_the_explicit_heading_on_each_cardinal_axis(self):
        field_lo, field_hi = (3.0, -.8, 0.0), (4.5, .8, .1)
        level_lo, level_hi = (1.8, -.8, 0.0), (3.3, .8, .1)
        cases = ((0.0, (3.0, 0.0), (4.5, 0.0)),
                 (math.pi / 2, (3.75, -.8), (3.75, .8)),
                 (math.pi, (4.5, 0.0), (3.0, 0.0)),
                 (-math.pi / 2, (3.75, .8), (3.75, -.8)))
        for yaw, near, far in cases:
            with self.subTest(yaw=yaw):
                mapping = replay.make_map("high_platform", field_lo, field_hi,
                                          level_lo, level_hi, yaw=yaw)
                np.testing.assert_allclose(mapping(*near), (1.8, 0.0), atol=1e-12)
                np.testing.assert_allclose(mapping(*far), (3.3, 0.0), atol=1e-12)

    def test_west_to_east_platform_supports_score_with_canonical_heading(self):
        name = "high_platform_10cm"
        leg = {**replay.PLAN_BY_NAME[name], "lane_yaw": 0.0}
        lo, hi = replay.field_bbox(self.model, leg)
        frames = platform_support_frames(lo, hi)
        for frame in frames[2:]:
            for point in frame["feet"].values():
                self.assertTrue(lo[0] <= point[0] <= hi[0])
                self.assertTrue(lo[1] <= point[1] <= hi[1])
        with mock.patch.dict(replay.PLAN_BY_NAME, {name: leg}), \
                mock.patch.object(replay, "make_map", wraps=replay.make_map) as projection:
            result, document = self.run_replay(frames, crossing_window(name, end=1.3),
                                              expected_legs=[name])
        self.assertEqual(result, 0, document)
        row = document["legs"][0]
        self.assertEqual(row["status"], "success", row)
        self.assertEqual(row["judge_scope"], "official_profile_projection")
        self.assertEqual(row["target_geoms"], list(leg["target_geoms"]))
        self.assertEqual(row["score"], replay.scoring.score_for_level(
            replay.rules.part2(), "high_platform", 10))
        self.assertEqual(projection.call_args.kwargs, {"yaw": leg["lane_yaw"]})
        # A legacy omitted-yaw call remains compatible, while demonstrating
        # that this legal W->E fixture would fail with the former hardcoded axis.
        level = replay.official.get_level("high_platform_10cm")
        old_mapping = replay.make_map("high_platform", lo, hi,
                                      level["obstacle_bbox"]["min"], level["obstacle_bbox"]["max"])
        legacy = replay.judge_leg(frames, {"leg": name, "start": 0.0, "end": 1.3},
                                  level, old_mapping)
        self.assertEqual(legacy["status"], "fail", legacy)

    def test_platform_wrong_axis_support_heights_do_not_score_the_target(self):
        name = "high_platform_10cm"
        leg = {**replay.PLAN_BY_NAME[name], "lane_yaw": 0.0}
        lo, hi = replay.field_bbox(self.model, leg)
        frames = platform_support_frames(lo, hi, wrong_axis=True)
        with mock.patch.dict(replay.PLAN_BY_NAME, {name: leg}):
            result, document = self.run_replay(frames, crossing_window(name, end=1.3),
                                              expected_legs=[name])
        row = document["legs"][0]
        self.assertEqual(result, 1, document)
        self.assertEqual(row["status"], "fail", row)
        self.assertEqual(row["score"], 0)
        # This is rejection of off-target geometry through the corrected axis,
        # not a new direction rule: formal platform scoring only requires a top hold.
        level = replay.official.get_level("high_platform_10cm")
        old_mapping = replay.make_map("high_platform", lo, hi,
                                      level["obstacle_bbox"]["min"], level["obstacle_bbox"]["max"])
        legacy = replay.judge_leg(frames, {"leg": name, "start": 0.0, "end": 1.3},
                                  level, old_mapping)
        self.assertEqual(legacy["status"], "success", legacy)

    def test_west_to_east_platform_still_requires_all_four_top_supports(self):
        name = "high_platform_10cm"
        leg = {**replay.PLAN_BY_NAME[name], "lane_yaw": 0.0}
        lo, hi = replay.field_bbox(self.model, leg)
        for bad_support in ("missing", "high", "ground"):
            with self.subTest(support=bad_support):
                frames = platform_support_frames(lo, hi)
                for frame in frames[2:]:
                    if bad_support == "missing":
                        del frame["feet"]["RL_foot"]
                    else:
                        frame["feet"]["RL_foot"][2] = (
                            hi[2] + replay.WHEEL_RADIUS_M + .10
                            if bad_support == "high" else replay.WHEEL_RADIUS_M)
                with mock.patch.dict(replay.PLAN_BY_NAME, {name: leg}):
                    result, document = self.run_replay(frames, crossing_window(name, end=1.3),
                                                      expected_legs=[name])
                self.assertEqual(result, 1, document)
                self.assertEqual(document["legs"][0]["score"], 0)

    def test_intermediate_wall_uses_real_geometry_without_official_points(self):
        leg = replay.PLAN_BY_NAME["high_wall_04cm"]
        result, document = self.run_replay(wall_support_frames(3.0),
                                          crossing_window(leg["leg"]),
                                          expected_legs=[leg["leg"]])
        self.assertEqual(result, 0, document)
        row = document["legs"][0]
        self.assertEqual(row["status"], "success", row)
        self.assertNotEqual(row["judge_scope"], "official_profile_projection")
        self.assertEqual(row["score"], 0)
        self.assertEqual(document["total"], 0)

    def test_require_all_rejects_empty_and_partial_recordings(self):
        result, _ = self.run_replay([], [])
        self.assertNotEqual(result, 0, "empty recording passed --require-all")
        result, document = self.run_replay(wall_support_frames(7.8),
                                          crossing_window("high_wall_20cm"))
        self.assertEqual(document["legs"][0]["status"], "success")
        self.assertNotEqual(result, 0, "one successful crossing counted as the full route")
        incomplete = crossing_window("high_wall_20cm")[:1]
        result, _ = self.run_replay(wall_support_frames(7.8), incomplete,
                                    expected_legs=["high_wall_20cm"])
        self.assertNotEqual(result, 0, "an unclosed crossing window passed --require-all")

    def test_points_use_best_successful_level_per_category(self):
        names = ["high_platform_10cm", "high_platform_20cm"]
        plan = crossing_window(names[0], 0, 1) + crossing_window(names[1], 2, 3)

        def successful_judge(frames, window, level, mapped, *, scored=True):
            return {"leg": window["leg"], "status": "success", "reason": "fixture",
                    "level": level["key"], "category": level["category"],
                    "frames": 1, "detail": {}, "score": replay.scoring.score_for_level(
                        replay.rules.part2(), "high_platform", level["level"])}

        with mock.patch.object(replay, "judge_leg", side_effect=successful_judge):
            result, document = self.run_replay([], plan, expected_legs=names)
        self.assertEqual(result, 0, document)
        best = replay.scoring.score_for_level(replay.rules.part2(), "high_platform", 20)
        self.assertEqual(document["total"], best,
                         "lower-tier points were added repeatedly within one category")
        self.assertGreater(sum(row["score"] for row in document["legs"]), best)

    def test_continuous_expert_windows_need_explicit_crossing_completion(self):
        first, second = "stairs_04cm", "stairs_06cm"
        complete = [
            {"t": 0.0, "leg": first, "cmd": f"{first}(policy=fixture)"},
            {"t": 1.0, "leg": first, "cmd": "cross_complete"},
            {"t": 1.2, "leg": second, "cmd": f"{second}(policy=fixture)"},
            {"t": 2.0, "leg": second, "cmd": "cross_complete"},
        ]
        windows = replay.leg_windows(complete)
        self.assertEqual([row["leg"] for row in windows], [first, second])
        self.assertEqual([(row["start"], row["end"]) for row in windows],
                         [(0.0, 1.0), (1.2, 2.0)])
        self.assertTrue(all(row["closed"] for row in windows))
        interrupted = replay.leg_windows([complete[0], complete[2], complete[3]])
        self.assertEqual(len(interrupted), 2)
        self.assertFalse(interrupted[0]["closed"],
                         "starting the next tier silently completed the previous crossing")
        self.assertEqual(interrupted[0]["end"], 1.2)
        self.assertTrue(interrupted[1]["closed"])

    def test_policy_switch_status_update_does_not_open_a_duplicate_attempt(self):
        name = "high_plum"
        cmd = f"{name}(policy=plum_depth_final_v1, handover->vt_band_v1@x=+6.80)"
        plan = [
            {"t": 103.3, "leg": name, "cmd": cmd,
             "cli": "switch depth expert; cross the pile field"},
            {"t": 103.3, "leg": name, "cmd": cmd,
             "cli": "switch plum_depth_final_v1"},
            {"t": 106.22, "leg": name, "cmd": f"{name}(handover=vt_band_v1)"},
            {"t": 112.58, "leg": name, "cmd": "cross_complete"},
        ]
        self.assertEqual(replay.leg_windows(plan), [
            {"leg": name, "start": 103.3, "end": 112.58,
             "bypass": False, "closed": True, "policy": "plum_depth_final_v1",
             "end_cmd": "cross_complete", "continuation_from_previous": False}])

    def test_continuous_wall_tail_only_finishes_the_existing_terminal_hold(self):
        frames, plan = continuous_wall_fixture()
        result, document = self.run_replay(frames, plan, expected_legs=["high_wall_16cm"])
        self.assertEqual(result, 0, document)
        row = document["legs"][0]
        self.assertEqual(row["status"], "success", row)
        self.assertEqual(row["window"], [0.0, 0.18])
        self.assertGreater(row["observation_end"], row["window"][1])
        self.assertLessEqual(row["tail_s"], replay.WALL_OBSERVATION_TAIL_MAX_S)
        self.assertEqual(row["tail_stop"], "terminal_hold_complete")
        self.assertEqual(row["score"], 0, "field-only wall16 must remain unscored")

    def test_continuous_wall_tail_requires_a_hold_already_active_at_the_gate(self):
        frames, plan = continuous_wall_fixture()
        for frame in frames:
            if frame["t"] <= 0.18:
                frame["feet"]["RL_foot"][2] = 0.303
        result, document = self.run_replay(frames, plan, expected_legs=["high_wall_16cm"])
        self.assertEqual(result, 1)
        row = document["legs"][0]
        self.assertEqual(row["tail_s"], 0.0)
        self.assertEqual(row["tail_stop"], "no_pending_terminal_hold")

    def test_continuous_wall_tail_rejects_missing_or_malformed_contacts(self):
        for contacts in (None, [None], [["FL_wheel_collision"]], [["FL_wheel_collision", 5]]):
            with self.subTest(contacts=contacts):
                frames, plan = continuous_wall_fixture()
                frame = next(frame for frame in frames if frame["t"] == 0.20)
                if contacts is None:
                    del frame["contacts"]
                else:
                    frame["contacts"] = contacts
                result, document = self.run_replay(frames, plan, expected_legs=["high_wall_16cm"])
                self.assertEqual(result, 1)
                row = document["legs"][0]
                self.assertEqual(row["observation_end"], 0.18)
                self.assertEqual(row["tail_stop"], "invalid_contacts")

    def test_continuous_wall_tail_stops_before_next_obstacle_contact(self):
        frames, plan = continuous_wall_fixture()
        frame = next(frame for frame in frames if frame["t"] == 0.24)
        frame["contacts"] = [["FL_wheel_collision", "cross_bar_5"]]
        result, document = self.run_replay(frames, plan, expected_legs=["high_wall_16cm"])
        self.assertEqual(result, 1)
        row = document["legs"][0]
        self.assertLess(row["observation_end"], 0.24)
        self.assertEqual(row["tail_stop"], "next_target_contact")

    def test_continuous_wall_tail_cannot_restart_a_broken_terminal_hold(self):
        frames, plan = continuous_wall_fixture()
        frame = next(frame for frame in frames if frame["t"] == 0.22)
        frame["feet"]["RL_foot"][2] = 0.303
        result, document = self.run_replay(frames, plan, expected_legs=["high_wall_16cm"])
        self.assertEqual(result, 1)
        row = document["legs"][0]
        self.assertEqual(row["observation_end"], 0.22)
        self.assertEqual(row["tail_stop"], "terminal_hold_broken")

    def test_continuous_wall_tail_requires_adjacent_closed_same_policy_windows(self):
        _, plan = continuous_wall_fixture()
        windows = replay.leg_windows(plan)
        self.assertIsNotNone(replay.wall_observation_tail(windows, 0))
        mutations = ((0, "closed", False), (1, "closed", False),
                     (0, "end_cmd", "ensure_transfer"), (1, "policy", "different"),
                     (1, "leg", "high_wall_12cm"), (1, "start", 0.30))
        for index, key, value in mutations:
            with self.subTest(window=index, key=key, value=value):
                altered = [dict(window) for window in windows]
                altered[index][key] = value
                self.assertIsNone(replay.wall_observation_tail(altered, 0))
        plan.insert(2, {"t": 0.18, "leg": "high_wall_16cm", "cmd": "ensure_transfer"})
        self.assertIsNone(replay.wall_observation_tail(replay.leg_windows(plan), 0),
                          "an intervening connector switch counted as continuous expert tracking")

    def test_continuous_wall_tail_has_a_hard_half_second_limit(self):
        frames, plan = continuous_wall_fixture()
        windows = replay.leg_windows(plan)
        window = {**windows[0], "observation_tail": replay.wall_observation_tail(windows, 0)}
        leg = replay.PLAN_BY_NAME[window["leg"]]
        level, mapped = replay.field_profile(self.model, leg)
        level["judge"]["hold_s"] = 0.8
        row = replay.judge_leg(frames, window, level, mapped, scored=False)
        self.assertEqual(row["status"], "fail", row)
        self.assertEqual(row["tail_s"], 0.5)
        self.assertEqual(row["tail_stop"], "tail_limit")

    def test_repeated_policy_command_at_a_later_time_remains_a_new_attempt(self):
        name = "high_plum"
        cmd = f"{name}(policy=plum_depth_final_v1)"
        windows = replay.leg_windows([
            {"t": 1.0, "leg": name, "cmd": cmd},
            {"t": 3.0, "leg": name, "cmd": cmd},
            {"t": 5.0, "leg": name, "cmd": "cross_complete"},
        ])
        self.assertEqual([(row["start"], row["end"], row["closed"])
                          for row in windows], [(1.0, 3.0, False), (3.0, 5.0, True)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
