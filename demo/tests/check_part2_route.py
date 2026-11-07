#!/usr/bin/env python3
"""Geometry-derived Part 2 route regression checks; no ROS or policies needed.

Run with any repository Python environment containing MuJoCo and NumPy:
    python demo/tests/check_part2_route.py
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import math
from pathlib import Path
import re
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo/rl_sar/scripts"))
from part2_route_plan import AUDIT_SEGMENTS, LEGS, ROUTE_START, SKIPPED_TIERS  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "part2_geometry_audit", ROOT / "tools/route/audit_part2_route.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)

# The rule maxima are independent of the route's selected/previously successful
# tiers. A hard-coded deployment cap in both route and test hid missing upper
# terrain. Only the user's explicit 35-degree slope scope overrides the rules.
rule_spec = importlib.util.spec_from_file_location(
    "part2_rules", ROOT / "eval/robocon_eval/rules.py")
rules = importlib.util.module_from_spec(rule_spec)
rule_spec.loader.exec_module(rules)
RULE_CATEGORIES = {"platform": "high_platform", "stairs": "stairs",
                   "slope": "slope", "trench": "trench", "wall": "high_wall"}
CAPS = {category: max(rules.PART2_CURVES[rule_name]["values"])
        for category, rule_name in RULE_CATEGORIES.items()}
CAPS.update(slope=35, plum=20)


def field_groups(obstacles):
    groups = {}
    for obstacle in obstacles:
        name = obstacle.name
        category = prefix = None
        for candidate, pattern in (
                ("platform", r"(platform_\d+cm)$"),
                ("stairs", r"(stairs_\d+cm)_(?:up_\d+|landing|down_\d+)$"),
                ("slope", r"(slope_\d+deg)_(?:up|landing|down)$"),
                ("trench", r"(trench_\d+cm)_(?:entry|near|far|exit)$"),
                ("wall", r"(cross_bar_\d+)$"),
                ("plum", r"(plum)_(?:entry|exit|r[12]_\d+)$")):
            match = re.fullmatch(pattern, name)
            if match:
                category, prefix = candidate, match.group(1)
                break
        if category is not None:
            groups.setdefault((category, prefix), []).append(obstacle)
    return groups


def measured_difficulty(category, obstacles):
    """Read height, riser, angle or gap from the actual compiled collision map."""
    if category in ("platform", "wall", "plum"):
        return max(ob.zmax for ob in obstacles) * 100
    if category == "stairs":
        first = min((ob for ob in obstacles if "_up_" in ob.name),
                    key=lambda ob: ob.xmin)
        return first.zmax * 100
    if category == "slope":
        ramp = next(ob for ob in obstacles if ob.name.endswith("_up"))
        return math.degrees(math.atan2(ramp.zmax, ramp.xmax - ramp.xmin))
    if category == "trench":
        near = next(ob for ob in obstacles if ob.name.endswith("_near"))
        far = next(ob for ob in obstacles if ob.name.endswith("_far"))
        return (far.xmin - near.xmax) * 100
    raise AssertionError(f"unknown category {category}")


class Part2RouteGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obstacles = geometry.load_obstacles()
        cls.groups = field_groups(cls.obstacles)
        cls.groups_by_names = {
            frozenset(ob.name for ob in members): (category, prefix, members)
            for (category, prefix), members in cls.groups.items()
        }

    def test_clearance_checks_both_segment_endpoints_and_is_symmetric(self):
        obstacle = geometry.Obstacle("unit_box", 0, (0, 0, 0), (1, 1, 1))
        # The closest pair is endpoint B and the interior of the right edge.
        # Checking only endpoint A overstates this approach's clearance.
        for a, b in (((2.0, 2.0), (1.1, 0.5)),
                     ((1.1, 0.5), (2.0, 2.0))):
            with self.subTest(start=a, end=b):
                self.assertAlmostEqual(
                    geometry.aabb_seg_distance(obstacle, *a, *b), 0.1)

    def test_all_actual_lower_tiers_are_crossed_once(self):
        expected = {
            frozenset(ob.name for ob in members)
            for (category, _), members in self.groups.items()
            if measured_difficulty(category, members) <= CAPS[category] + 1e-4
            and not any(math.isclose(measured_difficulty(category, members), tier,
                                     abs_tol=1e-4)
                        for tier in SKIPPED_TIERS.get(category, {}))
        }
        actual = [frozenset(leg["target_geoms"]) for leg in LEGS]
        self.assertEqual(len(LEGS), 33, "the canonical demo must contain 33 crossings")
        self.assertEqual(len({leg["leg"] for leg in LEGS}), len(LEGS))
        self.assertEqual(len(set(actual)), len(actual), "a terrain tier is repeated")
        self.assertEqual(set(actual), expected,
                         "missing a lower physical tier, or added an unsupported tier")
        used_geoms = [name for leg in LEGS for name in leg["target_geoms"]]
        self.assertEqual(len(set(used_geoms)), len(used_geoms))

    def test_only_user_approved_platform40_is_explicitly_skipped(self):
        self.assertEqual(set(SKIPPED_TIERS), {"platform"})
        self.assertEqual(set(SKIPPED_TIERS["platform"]), {40})
        self.assertIn("user-approved", SKIPPED_TIERS["platform"][40])
        self.assertFalse(any(leg["category"] == "platform" and leg["difficulty"] == 40
                             for leg in LEGS))
        omitted = self.groups[("platform", "platform_40cm")]
        self.assertAlmostEqual(measured_difficulty("platform", omitted), 40)

    def test_rule_caps_and_explicit_omission_use_measured_size(self):
        # For example, the fourth platform is only 20 cm; the actual 40 cm
        # platform is seventh because the field also includes intermediate tiers.
        expected_counts = {"platform": 6, "stairs": 9, "slope": 7,
                           "trench": 3, "wall": 7, "plum": 1}
        for category, cap in CAPS.items():
            selected = [leg for leg in LEGS if leg["category"] == category]
            with self.subTest(category=category):
                self.assertEqual(len(selected), expected_counts[category])
                eligible = [measured_difficulty(category, members)
                            for (group_category, _), members in self.groups.items()
                            if group_category == category
                            and measured_difficulty(category, members) <= cap + 1e-4
                            and not any(math.isclose(measured_difficulty(category, members), tier,
                                                     abs_tol=1e-4)
                                        for tier in SKIPPED_TIERS.get(category, {}))]
                self.assertAlmostEqual(max(leg["difficulty"] for leg in selected),
                                       max(eligible), delta=1e-4)
        self.assertEqual(CAPS["platform"], 40)
        self.assertEqual(CAPS["stairs"], 30)
        self.assertEqual(CAPS["trench"], 30)
        self.assertEqual(CAPS["wall"], 40)

    def test_declared_difficulties_and_support_dimensions_match_field(self):
        for leg in LEGS:
            with self.subTest(leg=leg["leg"]):
                category, _, members = self.groups_by_names[
                    frozenset(leg["target_geoms"])]
                self.assertEqual(leg["category"], category)
                self.assertAlmostEqual(leg["difficulty"],
                                       measured_difficulty(category, members), delta=1e-4)
                self.assertAlmostEqual(leg["top_m"], max(ob.zmax for ob in members),
                                       delta=2e-5)
                actual_bounds = (
                    (min(ob.xmin for ob in members), min(ob.ymin for ob in members),
                     min(ob.zmin for ob in members)),
                    (max(ob.xmax for ob in members), max(ob.ymax for ob in members),
                     max(ob.zmax for ob in members)))
                np.testing.assert_allclose(leg["target_bounds"], actual_bounds,
                                           atol=2e-5, rtol=0)
                if category == "stairs":
                    up = sorted((ob for ob in members if "_up_" in ob.name),
                                key=lambda ob: ob.xmin)
                    down = sorted((ob for ob in members if "_down_" in ob.name),
                                  key=lambda ob: ob.xmin)
                    self.assertEqual((len(up), len(down)), (6, 6))
                    height = leg["difficulty"] / 100
                    np.testing.assert_allclose([ob.zmax for ob in up],
                                               np.arange(1, 7) * height, atol=1e-6)
                    np.testing.assert_allclose([ob.zmax for ob in down],
                                               np.arange(6, 0, -1) * height, atol=1e-6)
                    np.testing.assert_allclose([ob.xmax - ob.xmin for ob in up + down],
                                               0.3, atol=1e-6)
                    landing = next(ob for ob in members if ob.name.endswith("landing"))
                    self.assertAlmostEqual(landing.xmax - landing.xmin, 0.8)
                elif category == "trench":
                    self.assertAlmostEqual(leg["top_m"], 0.3)
                elif category == "plum":
                    cubes = [ob for ob in members if re.fullmatch(r"plum_r[12]_\d+", ob.name)]
                    self.assertEqual(len(cubes), 6)
                    np.testing.assert_allclose(
                        leg["cube_x_bounds"],
                        (min(cube.xmin for cube in cubes), max(cube.xmax for cube in cubes)),
                        atol=1e-6, rtol=0)
                    for cube in cubes:
                        np.testing.assert_allclose(
                            (cube.xmax - cube.xmin, cube.ymax - cube.ymin, cube.zmax),
                            (0.2, 0.2, 0.2), atol=1e-6)
                    self.assertAlmostEqual(actual_bounds[1][0] - actual_bounds[0][0], 3.5)

    def test_wall_approach_uses_one_west_then_north_corridor(self):
        first_wall = next(leg for leg in LEGS if leg["category"] == "wall")
        transit = first_wall["transit"]
        self.assertEqual(tuple(first_wall["stand"]), (1.2, 21.0))
        self.assertEqual(tuple(transit[-1]), (0.7, 21.0))
        np.testing.assert_allclose(transit[-2], (0.7, 13.2), atol=1e-9, rtol=0)
        # Horizontal subdivisions are fine; an extra east/west dogleg in the
        # northbound corridor asks the robot for two unnecessary right angles.
        self.assertTrue(all(abs(point[1] - 13.2) < 1e-9
                            for point in transit[:-1]))
        self.assertTrue(all(b[0] < a[0]
                            for a, b in zip(transit[:-2], transit[1:-1])))

    def test_crossings_face_forward_and_continuous_rows_do_not_turn_back(self):
        previous = None
        categories = []
        settle_segments = {segment["name"]: segment for segment in AUDIT_SEGMENTS
                           if segment["name"].endswith("_settle")}
        for leg in LEGS:
            with self.subTest(leg=leg["leg"]):
                stand, end = np.asarray(leg["stand"]), np.asarray(leg["cross_end"])
                direction = np.asarray((math.cos(leg["lane_yaw"]), math.sin(leg["lane_yaw"])))
                normal = np.asarray((-direction[1], direction[0]))
                self.assertGreater(float((end - stand) @ direction), 0.0,
                                   "crossing gate must be ahead of the expert")
                self.assertAlmostEqual(float((end - stand) @ normal), 0.0)
                np.testing.assert_allclose(stand, (leg["lane_x"], leg["lane_y"]))
                approach = np.asarray(leg.get("align_approach", leg["stand"]))
                approach_along = float((approach - stand) @ direction)
                approach_lateral = float((approach - stand) @ normal)
                if "align_approach" in leg:
                    self.assertEqual(leg["category"], "plum")
                    self.assertAlmostEqual(approach_lateral, -0.20)
                else:
                    self.assertAlmostEqual(approach_lateral, 0.0)
                self.assertLessEqual(approach_along, leg["along_max"])
                self.assertGreaterEqual(leg["settle_forward"], 0)
                available_settle_travel = leg["along_max"] - approach_along
                settle_segment = settle_segments.get(leg["leg"] + "_settle")
                settle_end = (np.asarray(settle_segment["pts"][-1])
                              if settle_segment else approach)
                settle_travel = float((settle_end - approach) @ direction)
                self.assertGreaterEqual(settle_travel, -1e-9)
                self.assertLessEqual(settle_travel, available_settle_travel + 1e-9)
                self.assertLessEqual(settle_travel, leg["settle_forward"] + 1e-9)
                members = self.groups_by_names[frozenset(leg["target_geoms"])][2]
                nearest_face = min(
                    min(np.asarray((x, y)) @ direction
                        for x in (ob.xmin, ob.xmax) for y in (ob.ymin, ob.ymax))
                    for ob in members)
                available_runway = nearest_face - float(stand @ direction)
                self.assertGreater(available_runway - leg["along_max"], 0.40,
                                   "alignment may consume the obstacle approach runway")
                if previous and previous["category"] == leg["category"]:
                    self.assertFalse(previous["exit"] or leg["transit"],
                                     "adjacent tiers must continue straight in their row")
                    displacement = stand - np.asarray(previous["cross_end"])
                    self.assertGreaterEqual(float(displacement @ direction), -0.02,
                                            "next alignment point is behind the robot")
                    self.assertAlmostEqual(float(displacement @ normal), 0.0)
                if not categories or categories[-1] != leg["category"]:
                    categories.append(leg["category"])
                previous = leg
        self.assertEqual(len(categories), len(set(categories)),
                         "the route returns to an earlier obstacle category")
        self.assertEqual(tuple(AUDIT_SEGMENTS[0]["pts"][0]), ROUTE_START)

    def test_platforms_follow_their_physical_row_without_corridor_loops(self):
        platforms = [leg for leg in LEGS if leg["category"] == "platform"]
        for index, leg in enumerate(platforms):
            with self.subTest(leg=leg["leg"]):
                self.assertAlmostEqual(leg["lane_yaw"], 0.0)
                self.assertAlmostEqual(leg["lane_y"], 0.0)
                self.assertTrue(leg["continuous_lane"])
                self.assertFalse(leg["transit"])
                if index < len(platforms) - 1:
                    self.assertFalse(leg["exit"])
                    following = platforms[index + 1]
                    np.testing.assert_allclose(leg["cross_end"], following["stand"])

    def test_final_platform_brakes_and_leaves_clear_of_the_omitted_tier(self):
        leg = next(leg for leg in LEGS if leg["leg"] == "high_platform_30cm")
        self.assertAlmostEqual(leg["cross_end"][0], 19.85)
        maximum_x = leg["cross_end"][0] + leg["exit_overshoot_tol"]
        self.assertAlmostEqual(maximum_x, 20.40)
        omitted = self.groups[("platform", "platform_40cm")][0]
        self.assertGreaterEqual(omitted.xmin - maximum_x, 0.60 - 1e-6)
        self.assertEqual(leg["exit"], [(20.15, 1.5)])
        departure = next(segment for segment in AUDIT_SEGMENTS
                         if segment["name"] == "high_platform_30cm_exit")
        self.assertEqual(departure["kind"], "exit")
        self.assertEqual(departure["target_geoms"], ("platform_30cm",))
        np.testing.assert_allclose(departure["pts"], [leg["cross_end"], (20.15, 1.5)])
        # Departing the completed platform permits its residual contact only.
        # Naming the omitted 40 cm terrain as a target must still be rejected.
        broken = copy.deepcopy(departure)
        broken["target_geoms"] = ("platform_40cm",)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertNotEqual(geometry.audit([broken], self.obstacles, 0.45), 0)

    def test_final_stair_alignment_keeps_its_ground_runway_and_entry_precision(self):
        leg = next(leg for leg in LEGS if leg["leg"] == "stairs_30cm")
        self.assertEqual(leg["stand"], (60.45, 3.0))
        self.assertEqual((leg["settle_forward"], leg["along_max"]), (1.10, 1.60))
        self.assertEqual((leg["align_xy_tol"], leg["align_yaw_tol_deg"]), (0.10, 2.5))
        latest_centre = leg["stand"][0] + leg["along_max"]
        front_face = leg["target_bounds"][0][0]
        self.assertAlmostEqual(front_face - latest_centre, 0.95)
        yaw = math.radians(leg["align_yaw_tol_deg"])
        front_extent = 0.35 * math.cos(yaw) + 0.30 * math.sin(yaw)
        self.assertGreater(front_face - latest_centre - front_extent, 0.58)

    def test_plum_alignment_envelope_covers_offset_corner_and_switch_runway(self):
        plum = next(leg for leg in LEGS if leg["category"] == "plum")
        envelope = next(segment for segment in AUDIT_SEGMENTS
                        if segment["name"] == "high_plum_alignment")
        np.testing.assert_allclose(plum["align_approach"], (1.2, 11.8))
        np.testing.assert_allclose(plum["stand"], (2.2, 12.0))
        self.assertEqual(envelope["kind"], "align_envelope")
        self.assertNotIn("target_geoms", envelope)
        np.testing.assert_allclose(envelope["bounds"], ((1.02, 11.62), (2.35, 12.18)))
        (xmin, ymin), (xmax, ymax) = envelope["bounds"]
        for point in (plum["align_approach"], plum["stand"],
                      (plum["stand"][0] + plum["along_max"], plum["stand"][1])):
            self.assertTrue(xmin <= point[0] <= xmax and ymin <= point[1] <= ymax)

    def test_alignment_envelope_rejects_contained_obstacle_without_exemptions(self):
        # All four boundary segments clear this obstacle by >0.45 m. Only a
        # filled-area check catches it, even if it is named as an expert target.
        obstacle = geometry.Obstacle("contained_target", 0, (1.4, 1.4, 0), (1.6, 1.6, 1))
        boundary = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
        envelope = {"name": "contained", "kind": "align_envelope",
                    "bounds": ((0.0, 0.0), (3.0, 3.0)), "pts": boundary,
                    "target_geoms": (obstacle.name,)}
        self.assertGreater(min(geometry.aabb_seg_distance(obstacle, *a, *b)
                               for a, b in zip(boundary[:-1], boundary[1:])), 0.45)
        self.assertEqual(geometry.aabb_rect_distance(obstacle, envelope["bounds"]), 0.0)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = geometry.audit([envelope], [obstacle], min_clear=0.45)
        self.assertNotEqual(result, 0, output.getvalue())
        self.assertIn("min_clear= 0.00", output.getvalue())

    def test_exact_targets_and_corridor_clearance(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = geometry.audit(AUDIT_SEGMENTS, self.obstacles, min_clear=0.45)
        self.assertEqual(result, 0, output.getvalue())

    def test_allowed_exit_overshoot_preserves_next_obstacle_centre_clearance(self):
        # A 0.45 m centre-path clearance is the audit's footprint allowance.
        # It does not assert that a physical expert can stop within this bound.
        for leg in LEGS:
            if leg["category"] not in ("wall", "slope"):
                continue
            with self.subTest(leg=leg["leg"]):
                targets = set(leg["target_geoms"])
                row_obstacles = [ob for ob in self.obstacles
                                 if ob.name not in targets
                                 and ob.ymin <= leg["lane_y"] <= ob.ymax
                                 and ob.xmin > leg["target_bounds"][1][0]]
                self.assertTrue(row_obstacles, "next physical tier was not found")
                next_face = min(ob.xmin for ob in row_obstacles)
                maximum_exit_x = leg["cross_end"][0] + leg["exit_overshoot_tol"]
                self.assertGreaterEqual(next_face - maximum_exit_x, 0.45 - 1e-6,
                                        "settled exit gate permits entering the next obstacle")
                # This is the concrete regression: the former generic 0.45 m
                # allowance consumed the footprint clearance in both rows.
                self.assertLess(next_face - (leg["cross_end"][0] + 0.45), 0.45)

    def test_wrong_obstacle_and_missing_target_fail_the_audit(self):
        crossing = next(segment for segment in AUDIT_SEGMENTS
                        if segment["name"] == "high_wall_04cm_cross")
        for names in (("cross_bar_2",), ("does_not_exist",), ()):
            with self.subTest(target_geoms=names):
                broken = copy.deepcopy(crossing)
                broken["target_geoms"] = names
                with contextlib.redirect_stdout(io.StringIO()):
                    result = geometry.audit([broken], self.obstacles, min_clear=0.45)
                self.assertNotEqual(result, 0,
                                    "the wrong obstacle was accepted as the crossing target")


if __name__ == "__main__":
    unittest.main(verbosity=2)
