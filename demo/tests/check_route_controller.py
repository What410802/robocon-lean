#!/usr/bin/env python3
"""Closed-loop regressions for Part 2 transit (demo Python, no ROS node).

The plant below deliberately has delayed velocity/yaw response and lateral
drift.  These checks exercise the real Part2Motion.goto() rather than duplicating
its steering equations; they do not establish terrain/expert performance.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl_sar/scripts"))
import part1_auto_route as base  # noqa: E402
import part2_motion as route  # noqa: E402


class TransitPlant(route.Part2Motion):
    """Simple nonholonomic plant with optional disturbances."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, *, drift=0.0,
                 displacement=None, stuck=False, fall_at=None,
                 turn_drift=0.0, forward_deadzone=0.0, forward_gain=1.0,
                 flat_stationary_turn=True, flat_yaw_deadzone=0.0):
        # Skip Node construction: command/feedback are driven synchronously.
        self.p = base._Pose(x=x, y=y, z=0.55, yaw=yaw)
        self.output = (0.0, 0.0, 0.0)
        self.velocity = self.turn_rate = 0.0
        self.history = []
        self.policy_history = []
        self.drift = drift
        self.displacement = displacement
        self.displaced = False
        self.stuck = stuck
        self.fall_at = fall_at
        self.turn_drift = turn_drift
        self.forward_deadzone = forward_deadzone
        self.forward_gain = forward_gain
        self.flat_stationary_turn = flat_stationary_turn
        self.flat_yaw_deadzone = flat_yaw_deadzone
        self._plan_fh = None
        self._current_policy = route.TRANSFER_POLICY
        self._cmd = self._cli = ""
        self._corr = {}
        self.switches = []
        self.stops = []

    def ensure_transfer(self, announce=False):
        # Exercise the inherited real braking/settling sequence too.
        super().ensure_transfer(announce)

    def switch(self, name):
        if name != self._current_policy:
            self.switches.append((self.p.t, name, self.p.x, self.p.y, self.output))
            self._current_policy = name

    def _hold(self, seconds):
        for _ in range(math.ceil(max(0.0, seconds) / 0.02)):
            self._tick()

    def _now(self):
        return self.p.t

    def pose(self):
        return base._Pose(self.p.x, self.p.y, self.p.z,
                           self.p.yaw, self.p.tilt, self.p.t)

    def command(self, vx, vy=0.0, wz=0.0):
        self.output = (vx, vy, wz)

    def _tick(self, fallback_s=0.02):
        dt = 0.02
        vx, _, wz = self.output
        # Slower yaw response and a command dead zone reflect the connector
        # failure mode; these are stress-test parameters, not a fitted model.
        yaw_target = 0.5 * wz if abs(wz) >= 0.10 else 0.0
        if (self._current_policy == route.FLAT_TRANSIT_POLICY
                and abs(wz) < self.flat_yaw_deadzone):
            yaw_target = 0.0
        if (self._current_policy == route.FLAT_TRANSIT_POLICY
                and not self.flat_stationary_turn and abs(vx) < 0.01):
            yaw_target = 0.0
        forward_target = (self.forward_gain * vx
                          if abs(vx) >= self.forward_deadzone else 0.0)
        self.velocity += (forward_target - self.velocity) * dt / 0.12
        self.turn_rate += (yaw_target - self.turn_rate) * dt / 0.10
        if not self.stuck:
            self.p.x += self.velocity * math.cos(self.p.yaw) * dt
            self.p.y += (self.velocity * math.sin(self.p.yaw) + self.drift) * dt
            # A 90-degree turn at this setting shifts the centre by 15 cm.
            turn_shift = self.turn_drift * abs(self.turn_rate) * dt
            self.p.x -= turn_shift * math.cos(self.p.yaw)
            self.p.y -= turn_shift * math.sin(self.p.yaw)
            self.p.yaw = route.wrap(self.p.yaw + self.turn_rate * dt)
        self.p.t += dt
        if (self.displacement is not None and not self.displaced
                and self.p.x >= self.displacement[0]):
            self.p.x += self.displacement[1]
            self.p.y += self.displacement[2]
            self.displaced = True
        if self.fall_at is not None and self.p.t >= self.fall_at:
            self.p.z = 0.10
        self.history.append((self.p.x, self.p.y, self.p.yaw, self.output))
        self.policy_history.append((self._current_policy, self.output, self.p.yaw))

    def stop(self, settle_s=0.4):
        self.stops.append((self.p.t, self.p.x, self.p.y, self._current_policy))
        self.command(0.0)
        for _ in range(round(settle_s / 0.02)):
            self._tick()


class RouteTransitChecks(unittest.TestCase):
    def setUp(self):
        self.transfer = route.TRANSFER_POLICY
        self.base_transfer = base.TRANSFER_POLICY

    def tearDown(self):
        route.TRANSFER_POLICY = self.transfer
        base.TRANSFER_POLICY = self.base_transfer

    def set_policy(self, policy):
        route.TRANSFER_POLICY = base.TRANSFER_POLICY = policy

    def test_drift_on_straight_transit_does_not_turn_toward_endpoint(self):
        for policy in ("vt_pose_v1", "vt_locked12"):
            with self.subTest(policy=policy):
                self.set_policy(policy)
                plant = TransitPlant(drift=0.025)
                self.assertTrue(plant.goto((10.0, 0.0), speed=0.30,
                                           timeout_s=60.0))
                self.assertLess(max(abs(row[2]) for row in plant.history),
                                math.radians(30.0))
                self.assertLess(abs(plant.p.y), 0.12)
                self.assertLess(abs(plant.p.x - 10.0), 0.15)

    def test_endpoint_pass_inside_corridor_finishes_without_spinback(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(displacement=(1.75, 0.35, 0.06))
        self.assertTrue(plant.goto((2.0, 0.0), speed=0.30))
        self.assertTrue(plant.displaced)
        self.assertLess(plant.p.t, 10.0)
        self.assertLess(max(abs(row[2]) for row in plant.history),
                        math.radians(10.0))

    def test_endpoint_miss_stops_instead_of_chasing_point_behind(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(displacement=(1.75, 0.35, 0.30))
        self.assertFalse(plant.goto((2.0, 0.0), speed=0.30))
        self.assertTrue(plant.displaced)
        self.assertLess(plant.p.t, 10.0)
        self.assertLess(max(abs(row[2]) for row in plant.history),
                        math.radians(10.0))
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_successive_corners_and_wrapped_heading_still_arrive(self):
        for policy in ("vt_pose_v1", "vt_locked12"):
            with self.subTest(policy=policy):
                self.set_policy(policy)
                plant = TransitPlant(yaw=math.radians(179.0))
                for target in ((-2.0, -0.05), (-2.0, 2.0), (0.0, 2.0)):
                    self.assertTrue(plant.goto(target, speed=0.25,
                                               timeout_s=40.0), target)
                    self.assertLess(math.hypot(plant.p.x - target[0],
                                               plant.p.y - target[1]), 0.15)

    def test_stall_and_fall_are_failures_with_zero_output(self):
        for plant in (TransitPlant(stuck=True), TransitPlant(fall_at=0.5)):
            with self.subTest(stuck=plant.stuck):
                self.assertFalse(plant.goto((2.0, 0.0), timeout_s=2.0))
                self.assertLess(plant.p.t, 2.4)
                self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_short_corner_tolerates_turn_drift_and_forward_deadzone(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(yaw=math.pi / 2, drift=0.015,
                             turn_drift=0.15 / (math.pi / 2),
                             forward_deadzone=0.10, forward_gain=1.7)
        self.assertTrue(plant.goto((1.5, 0.0), speed=0.22,
                                   timeout_s=25.0))
        self.assertLess(math.hypot(plant.p.x - 1.5, plant.p.y), 0.15)
        self.assertLess(abs(plant.p.yaw), math.radians(30.0))

    def test_transit_subdivisions_merge_but_real_corners_and_reversals_remain(self):
        self.assertEqual(route.transit_corners(
            [(4.0, 1.5), (8.0, 1.5), (12.0, 1.5), (12.0, 3.0)],
            origin=(0.0, 1.52)), [(12.0, 1.5), (12.0, 3.0)])
        self.assertEqual(route.transit_corners(
            [(4.0, 0.0), (4.0, 1.5)], origin=(0.0, 0.0)),
            [(4.0, 0.0), (4.0, 1.5)])
        self.assertEqual(route.transit_corners(
            [(4.0, 0.0), (8.0, 0.0), (4.0, 0.0)], origin=(0.0, 0.0)),
            [(8.0, 0.0), (4.0, 0.0)])

    def test_long_flat_corridor_has_no_subdivision_stops_and_returns_to_locked(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(drift=0.01)
        plant.transit_path([(4.0, 0.0), (8.0, 0.0), (12.0, 0.0),
                            (12.0, 1.5)], speed=0.50)
        self.assertEqual([row[1] for row in plant.switches], ["flat", "vt_locked12"])
        handover = plant.switches[1]
        self.assertLess(handover[2], 11.0, "flat handover consumed the corner runway")
        self.assertEqual(handover[4], (0.0, 0.0, 0.0), "switch preceded braking")
        self.assertTrue(all(abs(x - 4.0) > 0.3 and abs(x - 8.0) > 0.3
                            for _, x, _, _ in plant.stops))
        self.assertLess(math.hypot(plant.p.x - 12.0, plant.p.y - 1.5), 0.17)
        self.assertEqual(plant._current_policy, "vt_locked12")
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_short_transit_keeps_locked_and_empty_path_does_nothing(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant()
        plant.transit_path([])
        self.assertEqual(plant.p.t, 0.0)
        self.assertEqual(plant.switches, [])
        plant.transit_path([(1.5, 0.0)], speed=0.50)
        self.assertEqual(plant.switches, [])
        self.assertLess(abs(plant.p.x - 1.5), 0.17)

    def test_westbound_flat_corridor_finishes_turn_in_locked(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(flat_stationary_turn=False)
        plant.transit_path([(-4.0, 0.0), (-8.0, 0.0), (-12.0, 0.0)])
        self.assertLess(math.hypot(plant.p.x + 12.0, plant.p.y), 0.17)
        flat_rows = [row for row in plant.policy_history if row[0] == "flat"]
        self.assertTrue(flat_rows)
        self.assertLess(abs(route.wrap(flat_rows[0][2] - math.pi)), 0.10)
        self.assertFalse(any(abs(output[0]) < 0.01 and abs(output[2]) > 0.0
                             for _, output, _ in flat_rows))

    def test_failed_locked_heading_never_loads_flat(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(stuck=True)
        with self.assertRaisesRegex(RuntimeError, "locked heading failed"):
            plant.transit_path([(-6.0, 0.0)])
        self.assertEqual(plant.switches, [])
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_flat_corridor_corrects_crab_without_lateral_policy_authority(self):
        class CrabbingFlat(TransitPlant):
            def _tick(self, fallback_s=0.02):
                moving_flat = (self._current_policy == "flat" and self.output[0] > 0.0)
                super()._tick(fallback_s)
                if moving_flat:
                    self.p.x -= math.sin(self.lane_yaw) * self.crab_speed * 0.02
                    self.p.y += math.cos(self.lane_yaw) * self.crab_speed * 0.02

        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            for drift in (-0.035, 0.035):
                with self.subTest(yaw=yaw, drift=drift):
                    plant = CrabbingFlat(yaw=yaw, flat_yaw_deadzone=0.30,
                                         forward_gain=1.5)
                    plant.lane_yaw, plant.crab_speed = yaw, drift
                    target = (20.0 * math.cos(yaw), 20.0 * math.sin(yaw))
                    self.assertTrue(plant.goto(target, tol=0.15, speed=0.50, timeout_s=45.0,
                                               policy="flat"))
                    cross = lambda row: -row[0] * math.sin(yaw) + row[1] * math.cos(yaw)
                    self.assertLess(max(abs(cross(row)) for row in plant.history), 0.15)
                    self.assertLess(math.dist((plant.p.x, plant.p.y), target), 0.17)
                    moving = [output for _, output, _ in plant.policy_history
                              if output[0] > 0.0]
                    self.assertTrue(any(abs(output[1]) > 0.01 for output in moving))
                    self.assertTrue(any(abs(output[2]) >= 0.40 for output in moving))
                    self.assertTrue(all(abs(output[1]) <= 0.20 and abs(output[2]) <= 0.45
                                        for output in moving))
                    self.assertEqual(plant.output, (0.0, 0.0, 0.0))

        # Reproduce the former yaw ceiling with the same nonholonomic plant:
        # vy is ignored and every remaining yaw command is in its dead zone.
        class FormerYawCeiling(CrabbingFlat):
            def command(self, vx, vy=0.0, wz=0.0):
                super().command(vx, 0.0, max(-0.20, min(0.20, wz)))

        old = FormerYawCeiling(flat_yaw_deadzone=0.30, forward_gain=1.5)
        old.lane_yaw, old.crab_speed = 0.0, 0.035
        self.assertFalse(old.goto((20.0, 0.0), tol=0.15, speed=0.50,
                                  timeout_s=45.0, policy="flat"))
        self.assertGreater(abs(old.p.y), 0.45)

    def test_flat_direct_goto_does_not_attempt_a_stationary_turn(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(flat_stationary_turn=False)
        self.assertFalse(plant.goto((-6.0, 0.0), policy="flat"))
        self.assertLess(plant.p.t, 0.5)
        self.assertTrue(all(output == (0.0, 0.0, 0.0)
                            for _, output, _ in plant.policy_history))

    def test_small_heading_error_starts_feedback_while_moving(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(yaw=math.radians(4.0))
        self.assertTrue(plant.goto((3.0, 0.0), speed=0.25))
        self.assertGreater(plant.history[0][3][0], 0.0)

    def test_failed_flat_segment_brakes_and_restores_locked_before_raising(self):
        self.set_policy("vt_locked12")
        plant = TransitPlant(stuck=True)
        with self.assertRaisesRegex(RuntimeError, "transit failed"):
            plant.transit_path([(6.0, 0.0)], speed=0.50)
        self.assertEqual([row[1] for row in plant.switches], ["flat", "vt_locked12"])
        self.assertEqual(plant._current_policy, "vt_locked12")
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
