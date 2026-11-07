#!/usr/bin/env python3
"""Part 2 controller checks using a synchronous ideal velocity plant.

    source demo/install/setup.bash
    .pixi/envs/demo/bin/python demo/tests/check_part2_controller.py

ROS Python imports must be available, but this script never initialises ROS,
creates nodes/publishers, or starts a simulator.  It executes the real route,
alignment and crossing loops against body-frame velocity integration at 50 Hz.
These checks validate controller logic, not policy dynamics or obstacle support.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo/rl_sar/scripts"))
os.environ.setdefault("PART1_TRANSFER_POLICY", "vt_locked12")
os.environ.setdefault("PART1_ROUTE_JUDGE", "0")
import part2_auto_route as controller  # noqa: E402
from part1_auto_route import _Pose, wrap  # noqa: E402


class MemoryLogger:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(str(message))

    warning = error = info


class IdealVelocityRoute(controller.Part2Route):
    """Replace only the external I/O boundary; keep actual control primitives."""
    def __init__(self, pose=None, *, immobile=False, self_driving_depth=False,
                 yaw_deadzone=0.0, lateral_deadzone=0.0):
        # Deliberately do not call either ROS Node constructor.
        self._plant_pose = pose or _Pose(0.0, 0.0, 0.55, 0.0, 0.0, 0.0)
        self._sim_t = self._plant_pose.t
        self._current_policy = "flat"
        self._exit_code = 0
        self._cmd, self._cli, self._leg = "start", "", ""
        self._corr = {}
        self._target = None
        self.output = (0.0, 0.0, 0.0)
        self.immobile = immobile
        self.self_driving_depth = self_driving_depth
        self.yaw_deadzone = yaw_deadzone
        self.lateral_deadzone = lateral_deadzone
        self.logger = MemoryLogger()
        self.reports, self.switches, self.rotations = [], [], []
        self.events = []
        self.alignments, self.motion = [], []
        self.commands = []
        self.ticks = 0

    def get_logger(self):
        return self.logger

    def pose(self):
        p = self._plant_pose
        return _Pose(p.x, p.y, p.z, p.yaw, p.tilt, p.t)

    def feet(self):
        # Synthetic wheel centres for the measured 0.53 x 0.42 m footprint.
        # These are telemetry fixtures, not a terrain/contact simulation.
        p = self.pose()
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        return {name + "_foot": (p.x + x * c - y * s,
                                p.y + x * s + y * c,
                                max(0.103, p.z - 0.447))
                for name, x, y in (("FL", 0.265, 0.21), ("FR", 0.265, -0.21),
                                   ("RL", -0.265, 0.21), ("RR", -0.265, -0.21))}

    def _now(self):
        return self._sim_t

    def command(self, vx, vy=0.0, wz=0.0):
        values = (float(vx), float(vy), float(wz))
        if not all(math.isfinite(value) for value in values):
            raise AssertionError(f"invalid command {values}")
        self.output = values
        self.commands.append(values)

    def _tick(self, fallback_s=0.02):
        self.ticks += 1
        if self.ticks > 400_000:
            raise AssertionError("controller exceeded 8,000 seconds of ideal-plant time")
        dt = 0.02
        p = self._plant_pose
        vx, vy, wz = self.output
        if self.self_driving_depth and self._current_policy == "plum_depth_final_v1":
            vx, vy, wz = 0.4, 0.0, 0.0
        if abs(wz) < self.yaw_deadzone:
            wz = 0.0
        if abs(vy) < self.lateral_deadzone:
            vy = 0.0
        if not self.immobile:
            p.x += (vx * math.cos(p.yaw) - vy * math.sin(p.yaw)) * dt
            p.y += (vx * math.sin(p.yaw) + vy * math.cos(p.yaw)) * dt
            p.yaw = wrap(p.yaw + wz * dt)
        self._sim_t += dt
        p.t = self._sim_t
        self.motion.append((self._sim_t, self._leg, p.x, p.y, p.yaw, self.output))

    def _hold(self, seconds):
        for _ in range(max(0, math.ceil(max(0.0, seconds) / 0.02))):
            self._tick()

    def _emit_plan(self):
        pass

    def switch(self, name):
        if self._current_policy != name:
            self.switches.append((self._now(), name))
            self.events.append(("switch", name))
            self._current_policy = name

    def switch_topic(self, name, hold_s=0.15):
        self.switch(name)
        self._hold(hold_s)

    def _press(self, buttons=(), hat=(0, 0), hold_s=0.12):
        self._hold(hold_s + 0.10)

    def wait_pose(self, timeout_s=20.0):
        return True

    def check_and_recover(self, *args, **kwargs):
        return True

    def report(self, name, ok, detail=""):
        self.reports.append((name, ok, detail))
        self.events.append(("report", name))
        if not ok:
            raise RuntimeError(f"{name}: FAIL {detail}")

    def _rotate_servo(self, target_yaw, timeout_s=10.0):
        self.rotations.append((self._leg, target_yaw))
        return super()._rotate_servo(target_yaw, timeout_s)

    def align_to(self, leg, policy, deadline_s=None):
        result = super().align_to(leg, policy, deadline_s)
        self.alignments.append((leg["leg"], result, self._corr.copy()))
        return result


def point_in_lane(leg, *, along=0.0, lateral=0.0, yaw_error=0.0):
    yaw = leg["lane_yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    return _Pose(leg["stand"][0] + along * c - lateral * s,
                 leg["stand"][1] + along * s + lateral * c,
                 0.55, wrap(yaw + yaw_error), 0.0, 0.0)


class LaggingYawRoute(IdealVelocityRoute):
    """First-order yaw response retains momentum after a zero command."""
    def __init__(self, pose, *, yaw_rate=0.65, response_s=0.40):
        super().__init__(pose)
        self.yaw_rate = yaw_rate
        self.response_s = response_s
        self.stopped_headings = []

    def _tick(self, fallback_s=0.02):
        requested = self.output
        self.yaw_rate += (requested[2] - self.yaw_rate) * 0.02 / self.response_s
        self.output = (requested[0], requested[1], self.yaw_rate)
        super()._tick(fallback_s)
        self.output = requested

    def _hold(self, seconds):
        super()._hold(seconds)
        if seconds > 0.0 and self.output == (0.0, 0.0, 0.0):
            self.stopped_headings.append(self.pose().yaw)


class LandingPhaseFaultRoute(IdealVelocityRoute):
    """Inject a fault only after three ticks of the requested landing phase."""
    def __init__(self, leg, phase, fault):
        x = 6.50 if phase == "cross" else 6.61
        super().__init__(_Pose(x, 12.0, 0.65, 0.0, 0.0, 0.0),
                         self_driving_depth=True)
        self.plum_leg, self.fault_phase, self.fault_kind = leg, phase, fault
        self.phase_ticks = {}
        self.injected = False

    def _tick(self, fallback_s=0.02):
        phase = None
        if self._current_policy == self.plum_leg["policy"]:
            phase = "cross"
        elif self._current_policy == self.plum_leg["handover_policy"]:
            vx = self.output[0]
            phase = "entry" if vx > 0.0 else "brake" if vx < 0.0 else "settle"
        super()._tick(fallback_s)
        if phase:
            self.phase_ticks[phase] = self.phase_ticks.get(phase, 0) + 1
        if phase == self.fault_phase and self.phase_ticks[phase] == 3:
            self.injected = True
            if self.fault_kind == "fallen":
                self._plant_pose.tilt = 65.0
            elif self.fault_kind == "heading":
                self._plant_pose.yaw = math.pi / 2

    def feet(self):
        return {name: (xyz[0], xyz[1],
                       0.137664601 if (self.injected and self.fault_kind == "support"
                                      and name == "RL_foot") else 0.303)
                for name, xyz in super().feet().items()}


class CrabbingPlumApproachRoute(IdealVelocityRoute):
    """Locked connector fixture with lateral slip during every forward tick."""
    def __init__(self, leg, crab_speed, *, initial_yaw=None,
                 negative_turn_crab=0.0, stop_coast=0.0):
        super().__init__(_Pose(*leg["align_approach"], 0.55,
                               leg["lane_yaw"] if initial_yaw is None else initial_yaw,
                               0.0, 0.0), yaw_deadzone=0.13)
        self.approach_leg = leg
        self.crab_speed = crab_speed
        self.negative_turn_crab = negative_turn_crab
        self.stop_coast = stop_coast
        self.coast_remaining = 0.0
        self.peak_along = controller.lane_position(leg, self.pose())[0]
        self.approach_stages = []

    def command(self, vx, vy=0.0, wz=0.0):
        if self.output[0] > 0.1 and vx == 0.0:
            self.coast_remaining = self.stop_coast
        super().command(vx, vy, wz)

    def _tick(self, fallback_s=0.02):
        vx, _, wz = self.output
        super()._tick(fallback_s)
        yaw = self.approach_leg["lane_yaw"]
        lateral_speed = self.crab_speed if vx > 0.1 else 0.0
        if abs(vx) < 0.1 and wz < -0.1:
            lateral_speed += self.negative_turn_crab
        self._plant_pose.x -= math.sin(yaw) * lateral_speed * 0.02
        self._plant_pose.y += math.cos(yaw) * lateral_speed * 0.02
        if vx == 0.0 and self.coast_remaining > 0.0:
            # Complete the requested residual travel over the 0.2 s stop hold.
            distance = min(self.coast_remaining, self.stop_coast * 0.1)
            self.coast_remaining -= distance
            self._plant_pose.x += math.cos(yaw) * distance
            self._plant_pose.y += math.sin(yaw) * distance
        self.peak_along = max(self.peak_along,
                              controller.lane_position(self.approach_leg, self.pose())[0])

    def _creep_along(self, *args, **kwargs):
        before = self.pose()
        super()._creep_along(*args, **kwargs)
        self.approach_stages.append(("creep", before, self.pose()))

    def _lane_settle(self, *args, **kwargs):
        before = self.pose()
        super()._lane_settle(*args, **kwargs)
        self.approach_stages.append(("settle", before, self.pose()))


class DriftingTrenchRoute(IdealVelocityRoute):
    """Synthetic northward-slip envelope, without terrain/contact dynamics.

    The initial 1.9 cm offset comes from v50. Weak small-command response uses
    the same deadzones as the other expert tracking checks. The 0.7 m/s slip
    over x=11.0..11.4 supplies about 0.4 m of drift, comparable to the recorded
    crossing envelope; it is not a fitted policy or support model.
    """
    def __init__(self, leg, *, slip_speed=0.7, canonical_feedback=False,
                 canonical_floor=False):
        super().__init__(point_in_lane(leg, lateral=0.019),
                         yaw_deadzone=0.4, lateral_deadzone=0.2)
        self.trench_leg = leg
        self.slip_speed = slip_speed
        self.canonical_feedback = canonical_feedback
        self.canonical_floor = canonical_floor or canonical_feedback
        self.pre_gap_lateral = None
        self.peak_lateral = 0.019

    def lane_cmd(self, p, prev, lane_x, lane_y, lane_yaw, vx):
        # Negative control: reproduce the former unshifted feedback input.
        if self.canonical_feedback:
            lane_y = self.trench_leg["lane_y"]
        return super().lane_cmd(p, prev, lane_x, lane_y, lane_yaw, vx)

    def _lane_offset(self, p, lane_x, lane_y, lane_yaw):
        # Negative control: shift adaptive input but leave the yaw-floor
        # residual on the geometric lane, suppressing the early correction.
        if self.canonical_floor:
            lane_y = self.trench_leg["lane_y"]
        return super()._lane_offset(p, lane_x, lane_y, lane_yaw)

    def _tick(self, fallback_s=0.02):
        before = self.pose()
        super()._tick(fallback_s)
        if 11.0 <= before.x < 11.4:
            self._plant_pose.y += self.slip_speed * 0.02
        lateral = controller.lane_position(self.trench_leg, self.pose())[1]
        if before.x < 11.0 <= self.pose().x:
            self.pre_gap_lateral = lateral
        self.peak_lateral = max(self.peak_lateral, lateral)


class Part2ControllerTests(unittest.TestCase):
    def test_expert_feedback_damps_body_lateral_velocity_on_each_lane_axis(self):
        # A robot travelling along the world lane with a 0.1 rad heading
        # offset has nonzero lateral velocity in its body frame. The official
        # controller damps that component even when world lateral speed is zero.
        for lane_yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            c, s = math.cos(lane_yaw), math.sin(lane_yaw)
            def sample(along, t):
                return _Pose(c * along - s * 0.05, s * along + c * 0.05,
                             0.55, wrap(lane_yaw + 0.1), 0.0, t)
            with self.subTest(lane_yaw=lane_yaw):
                vx, vy, wz = controller.Part2Route.lane_cmd(
                    sample(1.02, 1.02), sample(1.0, 1.0),
                    0.0, 0.0, lane_yaw, 0.5)
                self.assertEqual(vx, 0.5)
                self.assertAlmostEqual(vy, -0.025)
                self.assertAlmostEqual(wz, -0.24 + 0.6 * math.sin(0.1))

    def test_all_exit_gates_accept_only_the_correct_obstacle_lane(self):
        for leg in controller.LEGS:
            with self.subTest(leg=leg["leg"]):
                endpoint = _Pose(*leg["cross_end"], 0.55, leg["lane_yaw"], 0.0, 0.0)
                self.assertTrue(leg["stop"](endpoint))
                self.assertTrue(leg["post"](endpoint))
                self.assertEqual(controller.crossing_fault(leg, endpoint), "")
                self.assertFalse(leg["stop"](point_in_lane(leg)))
                along, _ = controller.lane_position(leg, endpoint)
                for lateral in (-2 * math.pi, 2 * math.pi):
                    wrong = point_in_lane(leg, along=along, lateral=lateral)
                    self.assertFalse(leg["stop"](wrong), "metres must not wrap modulo 2*pi")
                    self.assertEqual(controller.crossing_fault(leg, wrong), "off_lane")
                for neighbour in controller.LEGS:
                    if neighbour is leg or neighbour["category"] != leg["category"]:
                        continue
                    other_exit = _Pose(*neighbour["cross_end"], 0.55,
                                       leg["lane_yaw"], 0.0, 0.0)
                    self.assertFalse(leg["stop"](other_exit),
                                     f"{neighbour['leg']} must not satisfy {leg['leg']} gate")

    def test_heading_backwards_and_invalid_pose_fail_closed_on_both_axes(self):
        for leg in (controller.LEGS[0], next(leg for leg in controller.LEGS
                                            if leg["category"] == "stairs")):
            with self.subTest(leg=leg["leg"]):
                self.assertEqual(controller.crossing_fault(
                    leg, point_in_lane(leg, yaw_error=math.pi)), "wrong_heading")
                self.assertEqual(controller.crossing_fault(
                    leg, point_in_lane(leg, along=-0.5)), "wrong_direction")
                p = point_in_lane(leg)
                p.x = math.nan
                self.assertEqual(controller.crossing_fault(leg, p), "invalid_pose")
                ideal = IdealVelocityRoute(point_in_lane(leg, yaw_error=math.pi))
                crossed, reason, _ = ideal.expert_cross(leg)
                self.assertFalse(crossed)
                self.assertEqual(reason, "wrong_heading")
                self.assertEqual(ideal.output, (0.0, 0.0, 0.0))

    def test_unreachable_alignment_is_bounded_without_perpendicular_sidestep(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "slope")
        ideal = IdealVelocityRoute(point_in_lane(leg, lateral=0.20,
                                                yaw_error=math.pi), immobile=True)
        ideal.set_leg(leg["leg"])
        started = ideal._now()
        aligned = ideal.align_to(leg, leg["policy"])
        self.assertFalse(aligned[0])
        self.assertLessEqual(ideal._now() - started, leg["align_deadline_s"] + 0.8)
        self.assertTrue(ideal.rotations)
        for _, target in ideal.rotations:
            self.assertLessEqual(abs(wrap(target - leg["lane_yaw"])),
                                 math.radians(25.01),
                                 "alignment requested a perpendicular correction")
        self.assertFalse(hasattr(controller.Part2Route, "_sidestep_lane"))

    def test_plum_long_approach_corrects_crab_before_consuming_its_runway(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        for crab_speed in (-0.05, -0.04, -0.03, 0.03, 0.04, 0.05):
            with self.subTest(crab_speed=crab_speed):
                ideal = CrabbingPlumApproachRoute(leg, crab_speed)
                ideal.set_leg(leg["leg"])
                aligned = ideal.align_to(leg, leg["policy"])
                # Starting south of the lane with no preceding turn is a
                # separate adverse state: persistent southward 5 cm/s slip
                # cannot be removed within this runway and must fail closed.
                self.assertEqual(aligned[0], crab_speed != -0.05,
                                 (ideal._corr, ideal.logger.lines))
                if not aligned[0]:
                    self.assertGreater(abs(ideal._corr["e"]), leg["align_xy_tol"])
                    self.assertLessEqual(ideal.peak_along, leg["along_max"])
                    self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
                    continue
                stages = ideal.approach_stages
                creeps = [stage for stage in stages if stage[0] == "creep"]
                if creeps:
                    self.assertLess(controller.lane_position(leg, creeps[0][2])[0], -0.20,
                                    "long creep consumed the remaining correction runway")
                corrections = [stage for stage in stages if stage[0] == "settle"]
                self.assertTrue(corrections, "approach ignored its southern lane offset")
                self.assertLessEqual(len(corrections), 2,
                                     "approach deferred drift into repeated late corrections")
                for _, before, after in corrections:
                    self.assertLess(controller.lane_position(leg, before)[0], -0.20)
                    self.assertLessEqual(controller.lane_position(leg, after)[0],
                                         leg["along_max"])
                p = ideal.pose()
                along, lateral = controller.lane_position(leg, p)
                self.assertGreaterEqual(along, leg["along_min"])
                self.assertLessEqual(along, leg["along_max"])
                self.assertLessEqual(abs(lateral), leg["align_xy_tol"])
                self.assertLessEqual(abs(math.degrees(wrap(p.yaw - leg["lane_yaw"]))),
                                     leg["align_yaw_tol_deg"])
                self.assertLessEqual(ideal._now(), leg["align_deadline_s"])
                self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
                for _, target in ideal.rotations:
                    self.assertLessEqual(abs(wrap(target - leg["lane_yaw"])),
                                         math.radians(25.01))
                # Alignment finishes on the ground before the entry platform;
                # elevated wheel support is required only over the later cubes.
                feet = ideal.feet()
                self.assertTrue(all(xyz[0] < leg["target_bounds"][0][0]
                                    for xyz in feet.values()))
                self.assertEqual(controller.plum_support_fault(leg, p, feet), "")
                self.assertTrue(all(xyz[2] - controller.BLACKW_WHEEL_RADIUS_M
                                    < controller.PLUM_MIN_WHEEL_BOTTOM_M
                                    for xyz in feet.values()))

    def test_plum_full_entry_turn_reserves_room_for_rotation_drift_and_stopping(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        for negative_turn_crab in (0.013, 0.02):
            for crab_speed in (0.03, 0.04):
                for stop_coast in (0.0, 0.03, 0.06):
                    with self.subTest(negative_turn_crab=negative_turn_crab,
                                      crab_speed=crab_speed, stop_coast=stop_coast):
                        self.assert_plum_entry_turn(leg, crab_speed,
                                                    negative_turn_crab, stop_coast)

    def assert_plum_entry_turn(self, leg, crab_speed, negative_turn_crab, stop_coast):
        ideal = CrabbingPlumApproachRoute(
            leg, crab_speed, initial_yaw=math.pi / 2,
            negative_turn_crab=negative_turn_crab, stop_coast=stop_coast)
        ideal.set_leg(leg["leg"])
        aligned = ideal.align_to(leg, leg["policy"])
        self.assertTrue(aligned[0], (ideal._corr, ideal.logger.lines))
        # Include the initial -90 degree turn at the actual
        # southern approach, before any residual lane correction.
        first = ideal.approach_stages[0]
        self.assertEqual(first[0], "settle")
        self.assertGreater(abs(controller.lane_position(leg, first[1])[1]),
                           leg["align_xy_tol"])
        self.assertLessEqual(ideal.peak_along, leg["along_max"])
        along, lateral = controller.lane_position(leg, ideal.pose())
        self.assertGreaterEqual(along, leg["along_min"])
        self.assertLessEqual(abs(lateral), leg["align_xy_tol"])
        self.assertLessEqual(abs(ideal._corr["yaw_err_deg"]), leg["align_yaw_tol_deg"])
        self.assertLessEqual(ideal._corr["align_s"], leg["align_deadline_s"])
        self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)

    def test_plum_southern_approach_accepts_both_corner_errors_without_reacquiring(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        self.assertAlmostEqual(leg["align_approach"][1], leg["lane_y"] - 0.20)
        self.assertEqual(tuple(leg["transit"][-1]), leg["align_approach"])
        for corner_error in (-0.10, 0.10):
            for stop_coast in (0.0, 0.03, 0.06):
                with self.subTest(corner_error=corner_error, stop_coast=stop_coast):
                    ideal = CrabbingPlumApproachRoute(
                        leg, 0.04, initial_yaw=math.pi / 2,
                        negative_turn_crab=0.02, stop_coast=stop_coast)
                    ideal._plant_pose.x += 0.06
                    ideal._plant_pose.y += corner_error
                    initial_x = ideal.pose().x
                    ideal.set_leg(leg["leg"])
                    with mock.patch.object(ideal, "goto", wraps=ideal.goto) as goto:
                        aligned = ideal.align_to(leg, leg["policy"])
                    goto.assert_not_called()
                    self.assertTrue(aligned[0], (ideal._corr, ideal.logger.lines))
                    self.assertLessEqual(ideal.peak_along, leg["along_max"])
                    self.assertGreaterEqual(ideal._corr["along"], leg["along_min"])
                    self.assertLessEqual(abs(ideal._corr["e"]), leg["align_xy_tol"])
                    self.assertLessEqual(abs(ideal._corr["yaw_err_deg"]),
                                         leg["align_yaw_tol_deg"])
                    for _, target in ideal.rotations:
                        self.assertLessEqual(abs(wrap(target - leg["lane_yaw"])),
                                             math.radians(25.01))
                    self.assertTrue(all(vx >= -0.05 for vx, _, _ in ideal.commands),
                                    "alignment introduced a reverse driving segment")
                    self.assertGreaterEqual(min(row[2] for row in ideal.motion), initial_x - 0.03,
                                            "alignment backed toward the passed approach point")
                    self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)

    def test_plum_passed_approach_is_not_reacquired_outside_corner_tolerance(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        for passed_distance in (0.19, 0.20, 0.21):
            for corner_error in (-0.10, 0.10):
                with self.subTest(passed_distance=passed_distance, corner_error=corner_error):
                    ideal = CrabbingPlumApproachRoute(
                        leg, 0.04, initial_yaw=math.pi / 2,
                        negative_turn_crab=0.02, stop_coast=0.06)
                    ideal._plant_pose.x += passed_distance
                    ideal._plant_pose.y += corner_error
                    initial_x = ideal.pose().x
                    ideal.set_leg(leg["leg"])
                    with mock.patch.object(ideal, "goto", wraps=ideal.goto) as goto:
                        aligned = ideal.align_to(leg, leg["policy"])
                    goto.assert_not_called()
                    self.assertTrue(aligned[0], (ideal._corr, ideal.logger.lines))
                    self.assertLessEqual(ideal.peak_along, leg["along_max"])
                    self.assertGreaterEqual(ideal._corr["along"], leg["along_min"])
                    self.assertLessEqual(abs(ideal._corr["e"]), leg["align_xy_tol"])
                    self.assertLessEqual(abs(ideal._corr["yaw_err_deg"]),
                                         leg["align_yaw_tol_deg"])
                    self.assertTrue(all(vx >= -0.05 for vx, _, _ in ideal.commands))
                    self.assertGreaterEqual(min(row[2] for row in ideal.motion), initial_x - 0.03)
                    for _, target in ideal.rotations:
                        self.assertLessEqual(abs(wrap(target - leg["lane_yaw"])),
                                             math.radians(25.01))
                    self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)

    def test_plum_approach_refuses_depth_when_crab_exceeds_remaining_runway(self):
        original = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        leg = {**original, "transit": [], "exit": []}
        for crab_speed in (-0.12, 0.12):
            with self.subTest(crab_speed=crab_speed):
                ideal = CrabbingPlumApproachRoute(leg, crab_speed)
                with mock.patch.object(controller, "LEGS", [leg]), mock.patch.dict(
                        os.environ, {"PART2_ROUTE_START": "", "PART2_ROUTE_ONLY": ""}):
                    with self.assertRaisesRegex(RuntimeError, "alignment failed at leg high_plum"):
                        ideal._route()
                self.assertGreater(abs(ideal._corr["e"]), leg["align_xy_tol"])
                self.assertLessEqual(ideal._corr["along"], leg["along_max"])
                self.assertLessEqual(ideal._corr["align_s"], leg["align_deadline_s"])
                self.assertFalse(ideal.reports)
                self.assertNotIn(leg["policy"], [name for _, name in ideal.switches])
                self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
                self.assertEqual(ideal.output, (0.0, 0.0, 0.0))

    def test_full_canonical_route_completes_with_actual_control_loops(self):
        ideal = IdealVelocityRoute()
        # This ideal run checks route control, not terrain dynamics. Supply
        # explicitly safe synthetic supports over the plum cubes; the actual
        # measured dropped-wheel regressions are below.
        ordinary_feet = ideal.feet
        plum = next(leg for leg in controller.LEGS if leg["category"] == "plum")

        def synthetic_supports():
            feet = ordinary_feet()
            if (plum["cube_x_bounds"][0] - controller.PLUM_SUPPORT_WINDOW_MARGIN_M
                    <= ideal.pose().x
                    <= plum["cube_x_bounds"][1] + controller.PLUM_SUPPORT_WINDOW_MARGIN_M):
                return {name: (xyz[0], xyz[1], 0.303) for name, xyz in feet.items()}
            return feet

        ideal.feet = synthetic_supports
        with mock.patch.dict(os.environ, {"PART2_ROUTE_START": "", "PART2_ROUTE_ONLY": ""}):
            ideal._route()
        self.assertEqual([name for name, _, _ in ideal.reports],
                         [leg["leg"] for leg in controller.LEGS])
        self.assertTrue(all(ok for _, ok, _ in ideal.reports))
        self.assertEqual(len(ideal.alignments), 33)
        self.assertTrue(all(result[0] for _, result, _ in ideal.alignments))
        self.assertTrue(any(name == "plum_depth_final_v1" for _, name in ideal.switches))
        self.assertTrue(any(name == "vt_band_v1" for _, name in ideal.switches))
        for previous, following in zip(controller.LEGS, controller.LEGS[1:]):
            if (previous["category"] != following["category"] or previous["exit"]
                    or following["transit"] or previous["policy"] != following["policy"]):
                continue
            if following.get("reset_expert"):
                with self.subTest(reset_before=following["leg"]):
                    start = ideal.events.index(("report", previous["leg"]))
                    end = ideal.events.index(("report", following["leg"]))
                    self.assertIn(("switch", controller.TRANSFER_POLICY),
                                  ideal.events[start:end])
                    alignment = next(info for name, _, info in ideal.alignments
                                     if name == following["leg"])
                    self.assertFalse(alignment.get("continuation", False))
                    self.assertFalse(previous["keep_expert_moving"])
                continue
            with self.subTest(continuous_tiers=(previous["leg"], following["leg"])):
                start = ideal.events.index(("report", previous["leg"]))
                end = ideal.events.index(("report", following["leg"]))
                self.assertNotIn(("switch", controller.TRANSFER_POLICY),
                                 ideal.events[start:end],
                                 "straight consecutive tiers unloaded their expert")
                alignment = next(info for name, _, info in ideal.alignments
                                 if name == following["leg"])
                self.assertTrue(alignment.get("continuation"))
        self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
        self.assertEqual(ideal.output, (0.0, 0.0, 0.0))
        self.assertLess(ideal._now(), 8000.0)

    def test_final_platform_stops_before_the_omitted_tier(self):
        platforms = [leg for leg in controller.LEGS if leg["category"] == "platform"]
        leg = platforms[-1]
        self.assertEqual([p["difficulty"] for p in platforms], [5, 10, 15, 20, 25, 30])
        self.assertFalse(any(p.get("reset_expert") for p in platforms))
        self.assertFalse(leg["keep_expert_moving"])
        self.assertEqual(leg["cross_settle_s"], 0.50)
        self.assertAlmostEqual(leg["cross_end"][0], 19.85)
        maximum = 20.40
        self.assertAlmostEqual(leg["cross_end"][0] + leg["exit_overshoot_tol"], maximum)
        self.assertTrue(leg["post"](_Pose(maximum - 1e-6, 0.0, 0.55)))
        self.assertFalse(leg["post"](_Pose(maximum + 1e-6, 0.0, 0.55)))
        self.assertEqual(controller.crossing_fault(
            leg, point_in_lane(leg, lateral=0.450001)), "off_lane")
        self.assertEqual(controller.crossing_fault(
            leg, point_in_lane(leg, yaw_error=math.radians(65.001))), "wrong_heading")
        near_exit = IdealVelocityRoute(_Pose(19.84, 0.0, 0.55, 0.0, 0.0, 0.0))
        crossed, why, stopped = near_exit.expert_cross(leg)
        self.assertTrue(crossed)
        self.assertEqual(why, "gate")
        self.assertTrue(leg["post"](stopped))
        self.assertGreaterEqual(near_exit._now(), 0.50)
        self.assertEqual(near_exit.output, (0.0, 0.0, 0.0))

    def test_stair_continuation_preserves_recorded_drift_without_stopping(self):
        leg = next(leg for leg in controller.LEGS if leg["leg"] == "stairs_08cm")
        # v36's previous crossing passed here; stopping to realign then rolled
        # backwards onto that staircase. The continuing gait must stay loaded.
        ideal = IdealVelocityRoute(_Pose(15.65, 2.71, 0.55,
                                        math.radians(-4.4), 0.0, 0.0))
        ideal._current_policy = leg["policy"]
        ideal._continuing_expert = True
        ideal.set_leg(leg["leg"])
        ideal.command(leg["vx"], 0.0, 0.0)
        before = ideal.pose()
        result = ideal.align_to(leg, leg["policy"])
        self.assertTrue(result[0], (result, ideal.logger.lines))
        self.assertEqual(ideal.ticks, 0, "stair handover tried to reposition in the gap")
        self.assertEqual(ideal.pose().x, before.x)
        self.assertEqual(ideal.pose().y, before.y)
        self.assertEqual(ideal.pose().yaw, before.yaw)
        self.assertEqual(ideal.commands, [(leg["vx"], 0.0, 0.0)])
        self.assertTrue(ideal._corr["continuation"])
        self.assertEqual(ideal._current_policy, leg["policy"])
        self.assertFalse(ideal.switches)
        self.assertFalse(ideal.rotations)

    def test_high_stairs_allow_slow_climbs_but_keep_bounded_stall_and_timeout(self):
        class DelayedProgressRoute(IdealVelocityRoute):
            def _tick(self, fallback_s=0.02):
                # The historical tall-stair climb can remain at one riser for
                # longer than the ordinary 12 s stall budget before advancing.
                self.immobile = self._now() < 46.0
                super()._tick(fallback_s)

        stairs = [leg for leg in controller.LEGS if leg["category"] == "stairs"]
        self.assertEqual([leg["difficulty"] for leg in stairs
                          if leg["stall_window_s"] > controller.CROSS_STALL_WINDOW_S],
                         [25, 30])
        for leg in stairs[-2:]:
            with self.subTest(difficulty=leg["difficulty"]):
                self.assertEqual(leg["cross_timeout"], 120.0)
                self.assertEqual(leg["stall_window_s"], 60.0)
                slow = DelayedProgressRoute(point_in_lane(leg))
                crossed, why, _ = slow.expert_cross(leg)
                self.assertTrue(crossed, slow.logger.lines)
                self.assertEqual(why, "gate")
                self.assertGreater(slow._now(), 45.0)
                self.assertLess(slow._now(), leg["cross_timeout"])

        leg = stairs[-1]
        for override, expected in (({"stall_window_s": 12.0}, "stall"),
                                   ({"cross_timeout": 45.0}, "timeout")):
            with self.subTest(previous_limit=override):
                slow = DelayedProgressRoute(point_in_lane(leg))
                crossed, why, _ = slow.expert_cross({**leg, **override})
                self.assertFalse(crossed)
                self.assertEqual(why, expected)
        stuck = IdealVelocityRoute(point_in_lane(leg), immobile=True)
        crossed, why, _ = stuck.expert_cross(leg)
        self.assertFalse(crossed)
        self.assertEqual(why, "stall")
        self.assertLess(stuck._now(), 61.0)

    def test_final_stair_resets_before_its_tight_entry_gate(self):
        stairs = [leg for leg in controller.LEGS if leg["category"] == "stairs"]
        self.assertEqual([leg["difficulty"] for leg in stairs if leg.get("reset_expert")], [30])
        previous, leg = stairs[-2:]
        self.assertFalse(previous["keep_expert_moving"])
        self.assertEqual((leg["align_xy_tol"], leg["align_yaw_tol_deg"]), (0.10, 2.5))
        self.assertEqual((leg["settle_forward"], leg["along_max"]), (1.10, 1.60))
        # Full v82 stopped here after stairs25. A continuation must refuse the
        # drift; resetting the expert permits ground alignment without opening
        # the final staircase's 10 cm / 2.5 degree entry gates.
        for x, y, yaw in ((60.46, 3.122, 5.4), (60.896, 3.219, 6.0),
                          (60.896, 3.334, 0.0)):
            with self.subTest(recorded_pose=(x, y, yaw)):
                recorded = _Pose(x, y, 0.55, math.radians(yaw), 0.0, 0.0)
                continuation = IdealVelocityRoute(recorded)
                self.assertFalse(continuation.align_continuation(leg)[0])
                reset = IdealVelocityRoute(recorded)
                reset._current_policy = leg["policy"]
                aligned = reset.align_to(leg, leg["policy"])
                self.assertTrue(aligned[0], reset.logger.lines)
                self.assertFalse(reset._corr.get("continuation", False))
                self.assertTrue(any(name == controller.TRANSFER_POLICY for _, name in reset.switches))
                along, lateral = controller.lane_position(leg, reset.pose())
                self.assertLessEqual(abs(lateral), 0.10)
                self.assertLessEqual(along, 1.60)
                self.assertLessEqual(abs(math.degrees(controller.wrap(reset.pose().yaw - leg["lane_yaw"]))),
                                     2.5)

    def test_narrow_row_handover_keeps_forward_motion(self):
        pairs = [(previous, following)
                 for previous, following in zip(controller.LEGS, controller.LEGS[1:])
                 if previous["category"] in ("stairs", "slope", "wall", "trench")
                 and following["category"] == previous["category"]
                 and following["policy"] == previous["policy"]
                 and not following.get("reset_expert")]
        self.assertEqual(len(pairs), 19)
        for previous, following in pairs:
            with self.subTest(pair=(previous["leg"], following["leg"])):
                self.assertTrue(previous["continuous_lane"])
                self.assertTrue(following["continuous_lane"])
                self.assertFalse(previous["exit"])
                self.assertFalse(following["transit"])
                ideal = IdealVelocityRoute(point_in_lane(previous))
                ideal._current_policy = previous["policy"]
                ideal.set_leg(previous["leg"])
                crossed, reason, _ = ideal.expert_cross(previous)
                self.assertTrue(crossed, reason)
                self.assertEqual(reason, "gate")
                self.assertTrue(previous["post"](ideal.pose()))

                # Invoke the same continuation dispatch used by the route.
                # A stop at either side of this boundary consumes clearance.
                ideal._continuing_expert = True
                ideal.set_leg(following["leg"])
                ticks, commands = ideal.ticks, len(ideal.commands)
                aligned = ideal.align_to(following, following["policy"])
                self.assertTrue(aligned[0], ideal.logger.lines)
                self.assertEqual(ideal.ticks, ticks)
                self.assertEqual(len(ideal.commands), commands)
                self.assertTrue(ideal._corr["continuation"])
                along, lateral = controller.lane_position(following, ideal.pose())
                self.assertLessEqual(abs(lateral), following["align_xy_tol"])
                self.assertLessEqual(abs(aligned[5]), following["align_yaw_tol_deg"])
                self.assertGreaterEqual(along, following["along_min"])
                self.assertLessEqual(along, following["along_max"])

                crossed, reason, _ = ideal.expert_cross(following)
                self.assertTrue(crossed, reason)
                self.assertTrue(following["post"](ideal.pose()))
                self.assertFalse(ideal.switches)
                self.assertFalse(ideal.rotations)
                self.assertTrue(ideal.commands)
                commands = ideal.commands
                if following.get("cross_settle_s") or not following["keep_expert_moving"]:
                    self.assertEqual(commands[-1], (0.0, 0.0, 0.0),
                                     "row exit or policy change did not brake at its gate")
                    commands = commands[:-1]
                for vx, vy, wz in commands:
                    self.assertGreater(vx, 0.0, "continuous row commanded a brake/reverse")
                    if following["leg"] == "trench_20cm":
                        # This leg deliberately establishes southern margin
                        # while keeping the same forward gait and hard gates.
                        self.assertLessEqual(abs(vy), 0.2)
                        self.assertLessEqual(abs(wz), 0.7)
                    else:
                        self.assertAlmostEqual(vy, 0.0)
                        self.assertAlmostEqual(wz, 0.0, msg="straight handover commanded a turn")

    def test_trench_continuation_preserves_safe_drift_without_a_spin(self):
        leg = next(leg for leg in controller.LEGS if leg["leg"] == "trench_20cm")
        # Preserve v34's lateral/yaw disturbance at the new uninterrupted
        # boundary, before the old 0.5-second brake consumed the runway.
        ideal = IdealVelocityRoute(_Pose(leg["stand"][0] + 0.02, 8.768, 0.55,
                                        math.radians(-2.3), 0.0, 0.0))
        ideal._current_policy = leg["policy"]
        ideal._continuing_expert = True
        ideal.set_leg(leg["leg"])
        ideal.command(leg["vx"], 0.0, 0.0)
        before = ideal.pose()
        aligned = ideal.align_to(leg, leg["policy"])
        self.assertTrue(aligned[0], ideal.logger.lines)
        self.assertEqual(ideal.ticks, 0, "trench handover tried to reposition in the gap")
        self.assertFalse(ideal.rotations)
        self.assertFalse(ideal.switches)
        self.assertEqual(ideal.commands, [(leg["vx"], 0.0, 0.0)])
        self.assertEqual(ideal.pose().x, before.x)
        self.assertEqual(ideal.pose().y, before.y)
        self.assertEqual(ideal.pose().yaw, before.yaw)

    def test_final_handover_fall_latches_failure_after_a_successful_crossing(self):
        original = controller.LEGS[-1]
        leg = {**original, "transit": [], "exit": []}
        for only in ("", leg["leg"]):
            with self.subTest(debug_only=bool(only)):
                ideal = IdealVelocityRoute(point_in_lane(leg))
                ideal._finished = mock.Mock()
                ensure_transfer = ideal.ensure_transfer

                def fall_after_report(announce=False):
                    ensure_transfer(announce=announce)
                    if ideal.reports:
                        ideal._plant_pose.z = 0.10

                ideal.ensure_transfer = fall_after_report
                with mock.patch.object(controller, "LEGS", [leg]), mock.patch.dict(
                        os.environ, {"PART2_ROUTE_START": "", "PART2_ROUTE_ONLY": only}), \
                        mock.patch("part1_auto_route.threading.Timer"):
                    ideal._run()
                self.assertEqual(len(ideal.reports), 1)
                self.assertTrue(ideal.reports[0][1], "fall must occur after crossing PASS")
                self.assertEqual(ideal._exit_code, 1, ideal.logger.lines)
                self.assertTrue(ideal.fallen())
                self.assertEqual(ideal.output, (0.0, 0.0, 0.0))
                ideal._finished.set.assert_called_once()
                self.assertFalse(any("route finished:" in line for line in ideal.logger.lines))

    def test_narrow_alignment_preserves_each_gate_without_corrective_motion(self):
        for name in ("stairs_06cm", "slope_10deg", "trench_20cm", "high_wall_08cm"):
            leg = next(leg for leg in controller.LEGS if leg["leg"] == name)
            self.assertTrue(leg["continuous_lane"])
            cases = {
                "left": {"lateral": leg["align_xy_tol"] + 0.01},
                "right": {"lateral": -leg["align_xy_tol"] - 0.01},
                "yaw_left": {"yaw_error": math.radians(leg["align_yaw_tol_deg"] + 0.1)},
                "yaw_right": {"yaw_error": -math.radians(leg["align_yaw_tol_deg"] + 0.1)},
                "behind": {"along": leg["along_min"] - 0.01},
                "ahead": {"along": leg["along_max"] + 0.01},
                "inside_lower": {"along": leg["along_min"] + 0.001,
                                 "lateral": leg["align_xy_tol"] * 0.99,
                                 "yaw_error": math.radians(leg["align_yaw_tol_deg"] * 0.99)},
                "inside_upper": {"along": leg["along_max"] - 0.001,
                                 "lateral": -leg["align_xy_tol"] * 0.99,
                                 "yaw_error": -math.radians(leg["align_yaw_tol_deg"] * 0.99)},
            }
            for case, offsets in cases.items():
                with self.subTest(leg=name, gate=case):
                    ideal = IdealVelocityRoute(point_in_lane(leg, **offsets))
                    ideal._current_policy = leg["policy"]
                    ideal._continuing_expert = True
                    ideal.set_leg(name)
                    ideal.command(leg["vx"], 0.0, 0.0)
                    before = ideal.pose()
                    result = ideal.align_to(leg, leg["policy"])
                    self.assertEqual(result[0], case.startswith("inside_"))
                    self.assertEqual(ideal.ticks, 0, "narrow alignment moved before its verdict")
                    after = ideal.pose()
                    self.assertEqual((after.x, after.y, after.z, after.yaw, after.tilt, after.t),
                                     (before.x, before.y, before.z, before.yaw, before.tilt, before.t))
                    self.assertEqual(ideal.commands, [(leg["vx"], 0.0, 0.0)],
                                     "narrow alignment braked or steered in the gap")
                    self.assertEqual(ideal._current_policy, leg["policy"])
                    self.assertFalse(ideal.switches)
                    self.assertFalse(ideal.rotations)
                    self.assertTrue(ideal._corr["continuation"])
                    self.assertEqual(ideal._corr["align_s"], 0.0)

    def test_row_tracking_recentres_when_small_commands_are_weak(self):
        for name in ("high_platform_10cm", "stairs_08cm", "slope_10deg", "trench_10cm"):
            leg = next(leg for leg in controller.LEGS if leg["leg"] == name)
            for sign in (-1, 1):
                with self.subTest(leg=name, drift_direction=sign):
                    self.assert_moving_lane_correction(leg, sign)

    def test_trench20_feedback_establishes_margin_with_a_consistent_yaw_floor(self):
        leg = next(leg for leg in controller.LEGS if leg["leg"] == "trench_20cm")
        geometry = (leg["lane_y"], leg["stand"], leg["cross_end"],
                    leg.get("cross_lane_tol", 0.45))
        plants = {}
        for variant, kwargs in (("shifted", {}),
                                ("former_reference", {"canonical_feedback": True}),
                                ("stale_floor", {"canonical_floor": True})):
            with self.subTest(reference=variant):
                plant = DriftingTrenchRoute(leg, **kwargs)
                plant.set_leg(leg["leg"])
                crossed, reason, _ = plant.expert_cross(leg)
                self.assertTrue(crossed, reason)
                self.assertEqual(reason, "gate")
                self.assertIsNotNone(plant.pre_gap_lateral)
                for vx, vy, wz in plant.commands:
                    self.assertGreaterEqual(vx, 0.0)
                    self.assertLessEqual(abs(vy), 0.2)
                    self.assertLessEqual(abs(wz), 0.7)
                plants[variant] = plant

        shifted = plants["shifted"]
        self.assertLess(shifted.pre_gap_lateral, -0.03)
        self.assertLess(shifted.peak_lateral, 0.35)
        for variant in ("former_reference", "stale_floor"):
            with self.subTest(negative_control=variant):
                # Both still pass the original 0.45 m gate. They fail the
                # intended early-margin requirement, making this regression
                # sensitive to either omitted feedback-reference change.
                self.assertGreater(plants[variant].pre_gap_lateral, 0.0)
                self.assertGreater(plants[variant].peak_lateral, 0.35)
                self.assertGreater(plants[variant].peak_lateral - shifted.peak_lateral,
                                   0.06)
        self.assertEqual(geometry, (leg["lane_y"], leg["stand"],
                                    leg["cross_end"], leg.get("cross_lane_tol", 0.45)))
        self.assertEqual(leg["lane_y"], 9.0)

    def test_trench20_feedback_keeps_the_original_off_lane_failure_gate(self):
        leg = next(leg for leg in controller.LEGS if leg["leg"] == "trench_20cm")
        plant = DriftingTrenchRoute(leg, slip_speed=1.5)
        plant.set_leg(leg["leg"])
        crossed, reason, stopped = plant.expert_cross(leg)
        self.assertFalse(crossed)
        self.assertEqual(reason, "off_lane")
        self.assertGreater(controller.lane_position(leg, stopped)[1], 0.45)
        self.assertEqual(controller.crossing_fault(leg, stopped), "off_lane")
        self.assertFalse(leg["stop"](stopped))
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def assert_moving_lane_correction(self, leg, sign):
        ideal = IdealVelocityRoute(
            point_in_lane(leg, lateral=sign * 0.15,
                          yaw_error=sign * math.radians(2)),
            yaw_deadzone=0.4, lateral_deadzone=0.2)
        ideal.set_leg(leg["leg"])
        crossed, reason, stopped = ideal.expert_cross(leg)
        self.assertTrue(crossed, reason)
        self.assertEqual(reason, "gate")
        self.assertLess(abs(controller.lane_position(leg, stopped)[1]), 0.06)
        self.assertLess(abs(wrap(stopped.yaw - leg["lane_yaw"])), 0.035)
        self.assertTrue(any(abs(wz) >= 0.4 for _, _, wz in ideal.commands))
        for vx, _, wz in ideal.commands:
            self.assertGreater(vx, 0.0)
            self.assertLessEqual(abs(wz), 0.7)

    def test_connector_heading_servo_reacquires_after_stopped_pose_rebounds(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "stairs")
        lagging = LaggingYawRoute(point_in_lane(leg, yaw_error=math.radians(-6)))
        lagging._current_policy = controller.TRANSFER_POLICY
        lagging.set_leg(leg["leg"])
        before = lagging.pose()
        self.assertTrue(lagging._rotate_servo(leg["lane_yaw"], timeout_s=2.0),
                        lagging.stopped_headings)
        self.assertGreaterEqual(len(lagging.stopped_headings), 2,
                                "servo accepted its first transient zero crossing")
        self.assertGreater(abs(wrap(lagging.stopped_headings[0] - leg["lane_yaw"])),
                           math.radians(leg["align_yaw_tol_deg"]),
                           "yaw plant did not reproduce the measured rebound")
        self.assertLessEqual(abs(wrap(lagging.pose().yaw - leg["lane_yaw"])), 0.035)
        self.assertLessEqual(lagging._now() - before.t, 2.0 + 0.02)
        self.assertAlmostEqual(lagging.pose().x, before.x)
        self.assertAlmostEqual(lagging.pose().y, before.y)
        self.assertEqual(lagging._current_policy, controller.TRANSFER_POLICY)
        self.assertFalse(lagging.switches)
        self.assertTrue(any(wz > 0.0 for _, _, wz in lagging.commands))
        self.assertTrue(any(wz < 0.0 for _, _, wz in lagging.commands))
        self.assertEqual(lagging.output, (0.0, 0.0, 0.0))

    def test_plum_support_window_and_threshold_match_the_official_profile(self):
        sys.path.insert(0, str(ROOT / "eval"))
        sys.path.insert(0, str(ROOT / "terrain"))
        from robocon_eval.part2 import official
        from robocon_eval.part2.v5_judge import WHEEL_RADIUS_M
        from gen_part2_levels import _plum_judge

        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        level = official.get_level("plum_1")
        offset = leg["target_bounds"][0][0] - level["obstacle_bbox"]["min"][0]
        self.assertAlmostEqual(leg["target_bounds"][1][0] -
                               level["obstacle_bbox"]["max"][0], offset)
        self.assertEqual(controller.BLACKW_WHEEL_RADIUS_M, WHEEL_RADIUS_M)
        self.assertEqual(controller.PLUM_MIN_WHEEL_BOTTOM_M, level["judge"]["z_cross_min"])
        self.assertEqual(controller.PLUM_MIN_WHEEL_BOTTOM_M, _plum_judge()["z_cross_min"])
        for field_x, official_x in zip(
                (leg["cube_x_bounds"][0] - controller.PLUM_SUPPORT_WINDOW_MARGIN_M,
                 leg["cube_x_bounds"][1] + controller.PLUM_SUPPORT_WINDOW_MARGIN_M),
                level["judge"]["cube_x_range"]):
            self.assertAlmostEqual(field_x, official_x + offset)

    def test_plum_support_guard_rejects_recorded_v25_drop_and_missing_telemetry(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        # Measured first failed support frame, part2_plum_v25_states.jsonl,
        # t=106.08. Rear-left bottom is 0.034664601 m before x=6.8 handover.
        p = _Pose(6.66025275, 12.1799569, 0.642405411, 0.0, 0.0, 106.08)
        feet = {"FL_foot": (6.96193887, 12.1831437, 0.315495430),
                "FR_foot": (6.87236732, 11.9425090, 0.300071526),
                "RL_foot": (6.30616497, 12.3714152, 0.137664601),
                "RR_foot": (6.20137541, 12.0411293, 0.161668048)}
        fault = controller.plum_support_fault(leg, p, feet)
        self.assertEqual(fault, "plum_support_dropped:RL_foot=0.035")
        ideal = IdealVelocityRoute(p, self_driving_depth=True)
        ideal.feet = lambda: feet
        crossed, reason, _ = ideal.depth_plum_cross(leg)
        self.assertFalse(crossed)
        self.assertEqual(reason, fault)
        self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
        self.assertEqual(ideal.output, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(ideal.pose().x, p.x,
                               msg="self-driving actor advanced after known support failure")

        safe = {name: (xyz[0], xyz[1], 0.303) for name, xyz in feet.items()}
        self.assertEqual(controller.plum_support_fault(leg, p, safe), "")
        for x in (2.2, 6.66, 8.2):
            p.x = x
            for invalid in ({}, {name: xyz for name, xyz in safe.items() if name != "RL_foot"},
                            {**safe, "RL_foot": (6.3, math.nan, 0.303)}):
                with self.subTest(x=x, telemetry=invalid):
                    self.assertTrue(controller.plum_support_fault(leg, p, invalid).startswith(
                        "plum_support_telemetry:"))
        low = {name: (xyz[0], xyz[1], 0.103) for name, xyz in feet.items()}
        for x in (4.74, 6.76):
            p.x = x
            self.assertEqual(controller.plum_support_fault(leg, p, low), "")
        for x in (4.75, 6.75):
            p.x = x
            self.assertTrue(controller.plum_support_fault(leg, p, low).startswith(
                "plum_support_dropped:"))

    def test_plum_gate_rejects_heading_lost_during_final_settle(self):
        original = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        plum = {**original, "handover_brake_s": 0.0, "handover_settle_s": 0.0}
        for yaw in (-math.pi / 2, math.pi / 2):
            with self.subTest(stopped_yaw=yaw):
                ideal = IdealVelocityRoute(_Pose(*plum["cross_end"], 0.55,
                                                plum["lane_yaw"], 0.0, 0.0))
                tick = ideal._tick

                def heading_disturbance(fallback_s=0.02):
                    tick(fallback_s)
                    if (ideal._current_policy == plum["handover_policy"]
                            and ideal.output == (0.0, 0.0, 0.0)):
                        ideal._plant_pose.yaw = yaw

                ideal._tick = heading_disturbance
                crossed, reason, stopped = ideal.depth_plum_cross(plum)
                self.assertTrue(plum["post"](stopped), "position must remain inside the exit gate")
                self.assertFalse(ideal.fallen(), "disturbance must isolate the heading check")
                self.assertEqual(controller.crossing_fault(plum, stopped), "wrong_heading")
                self.assertFalse(crossed, "position alone accepted a sideways final pose")
                self.assertEqual(reason, "settled_gate_missed")
                self.assertEqual(ideal.output, (0.0, 0.0, 0.0))

    def test_plum_missing_support_telemetry_unloads_actor_before_advancing(self):
        leg = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        ideal = IdealVelocityRoute(point_in_lane(leg), self_driving_depth=True)
        ideal.feet = lambda: {}
        before = ideal.pose()
        crossed, reason, stopped = ideal.depth_plum_cross(leg)
        self.assertFalse(crossed)
        self.assertTrue(reason.startswith("plum_support_telemetry:"), reason)
        self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
        self.assertEqual(ideal.output, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(stopped.x, before.x)
        self.assertAlmostEqual(stopped.y, before.y)

    def test_plum_support_guard_observes_crossing_entry_brake_and_settle_frames(self):
        original = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        plum = {**original, "handover_x": 6.60}
        for phase in ("cross", "entry", "brake", "settle"):
            with self.subTest(phase=phase):
                ideal = LandingPhaseFaultRoute(plum, phase, "support")
                crossed, reason, _ = ideal.depth_plum_cross(plum)
                self.assertFalse(crossed)
                self.assertEqual(reason, "plum_support_dropped:RL_foot=0.035")
                self.assertTrue(ideal.injected)
                self.assertEqual(ideal.phase_ticks[phase], 3,
                                 "guard missed the fault or failed in an earlier phase")
                self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
                self.assertEqual(ideal.output, (0.0, 0.0, 0.0))
                if phase in ("brake", "settle"):
                    self.assertGreaterEqual(ideal.phase_ticks["entry"] * 0.02,
                                            controller.PLUM_HANDOVER_ENTRY_S)
                if phase == "settle":
                    self.assertGreaterEqual(ideal.phase_ticks["brake"] * 0.02,
                                            plum["handover_brake_s"])

    def test_plum_landing_checks_fallen_and_heading_during_each_hold(self):
        original = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        plum = {**original, "handover_x": 6.60}
        for phase in ("entry", "brake", "settle"):
            for fault, expected in (("fallen", "fallen"), ("heading", "wrong_heading")):
                with self.subTest(phase=phase, fault=fault):
                    ideal = LandingPhaseFaultRoute(plum, phase, fault)
                    crossed, reason, _ = ideal.depth_plum_cross(plum)
                    self.assertFalse(crossed)
                    self.assertEqual(reason, expected)
                    self.assertTrue(ideal.injected)
                    self.assertEqual(ideal.phase_ticks[phase], 3)

    def test_plum_landing_keeps_recorded_forward_reverse_and_zero_inputs(self):
        plum = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        ideal = IdealVelocityRoute(_Pose(plum["handover_x"], 12.10, 0.65,
                                        math.radians(3), 0.0, 0.0))
        ideal.feet = lambda: {name: (xyz[0], xyz[1], 0.303)
                             for name, xyz in IdealVelocityRoute.feet(ideal).items()}
        # An actual lane error would request steering. It may be used only
        # after all three landing holds have completed.
        ideal.lane_cmd = mock.Mock(side_effect=RuntimeError("landing holds completed"))
        with self.assertRaisesRegex(RuntimeError, "landing holds completed"):
            ideal.depth_plum_cross(plum)
        ideal.lane_cmd.assert_called_once()
        phases = []
        for _, _, _, _, _, command in ideal.motion:
            self.assertEqual(command[1:], (0.0, 0.0),
                             "landing introduced lateral/yaw feedback")
            if not phases or phases[-1][0] != command[0]:
                phases.append([command[0], 0])
            phases[-1][1] += 1
        expected = ((plum["vx"], controller.PLUM_HANDOVER_ENTRY_S),
                    (plum["handover_brake_vx"], plum["handover_brake_s"]),
                    (0.0, plum["handover_settle_s"]))
        self.assertEqual(len(phases), len(expected))
        for (vx, ticks), (expected_vx, duration) in zip(phases, expected):
            self.assertEqual(vx, expected_vx)
            self.assertGreaterEqual(ticks * 0.02 + 1e-9, duration)
            self.assertLessEqual(ticks * 0.02, duration + 0.020001)

    def test_failed_self_driving_plum_is_unloaded_before_failure_report(self):
        original = next(leg for leg in controller.LEGS if leg["category"] == "plum")
        plum = {**original, "transit": [], "exit": [], "cross_timeout": 0.20}
        ideal = IdealVelocityRoute(point_in_lane(plum), self_driving_depth=True)
        with mock.patch.object(controller, "LEGS", [plum]), mock.patch.dict(
                os.environ, {"PART2_ROUTE_START": "", "PART2_ROUTE_ONLY": ""}):
            with self.assertRaisesRegex(RuntimeError, "high_plum: FAIL"):
                ideal._route()
        self.assertEqual(len(ideal.reports), 1)
        self.assertFalse(ideal.reports[0][1])
        self.assertEqual(ideal._current_policy, controller.TRANSFER_POLICY)
        self.assertEqual(ideal.output, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
