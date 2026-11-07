#!/usr/bin/env python3
"""Part 1 gravel handover regressions against the actual route primitives.

Run with `pixi run -e demo python demo/tests/check_part1_handover.py`.
Only command/feedback I/O is replaced by ideal body-frame integration at 50 Hz;
no ROS node or physics is started. These tests establish steering behavior,
not policy dynamics, wheel locking, or terrain support.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl_sar/scripts"))
import part1_auto_route as route  # noqa: E402


class HandoverPlant(route.RouteNode):
    """Keep RouteNode motion/fall/stop logic; replace only its external I/O."""

    def __init__(self, x, y, yaw, *, fallen=False, fall_at=None,
                 lateral_response=1.0, yaw_deadzone=0.0, yaw_response=1.0,
                 reverse_response=1.0, east_crab=0.0, turn_east_drift=0.0):
        # Skip ROS Node construction and start from the supplied pose fixture.
        self.p = route._Pose(x, y, 0.706, yaw, 81.0 if fallen else 0.0)
        self.output = (0.0, 0.0, 0.0)
        self.commands = []
        self.motion = []
        self.fall_at = fall_at
        self.lateral_response = lateral_response
        self.yaw_deadzone = yaw_deadzone
        self.yaw_response = yaw_response
        self.reverse_response = reverse_response
        self.east_crab = east_crab
        self.turn_east_drift = turn_east_drift
        self._lock = threading.Lock()
        self._feet = {}
        self._contacts = ()

    def ensure_transfer(self, announce=False):
        pass

    def _emit_plan(self):
        pass

    def pose(self):
        p = self.p
        return route._Pose(p.x, p.y, p.z, p.yaw, p.tilt, p.t)

    def _now(self):
        return self.p.t

    def command(self, vx, vy=0.0, wz=0.0):
        self.output = (float(vx), float(vy), float(wz))
        if not all(math.isfinite(value) for value in self.output):
            raise AssertionError(f"nonfinite command: {self.output}")
        self.commands.append((self.p.t, self.output))

    def _tick(self, fallback_s=0.02):
        dt = 0.02
        vx, vy, wz = self.output
        vy *= self.lateral_response
        vx *= self.reverse_response if vx < 0.0 else 1.0
        wz = 0.0 if abs(wz) < self.yaw_deadzone else wz * self.yaw_response
        c, s = math.cos(self.p.yaw), math.sin(self.p.yaw)
        self.p.x += (vx * c - vy * s) * dt
        if vx:
            self.p.x += self.east_crab * dt
        elif wz:
            self.p.x += self.turn_east_drift * dt
        self.p.y += (vx * s + vy * c) * dt
        self.p.yaw = route.wrap(self.p.yaw + wz * dt)
        self.p.t += dt
        if self.fall_at is not None and self.p.t >= self.fall_at:
            self.p.tilt = 81.0
        self.motion.append(self.pose())

    def _hold(self, seconds):
        for _ in range(math.ceil(max(0.0, seconds) / 0.02)):
            self._tick()


class Part1HandoverChecks(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(route, "TRANSFER_POLICY", "vt_locked12")
        patcher.start()
        self.addCleanup(patcher.stop)

    def move_to_corner(self, plant):
        return plant._pose_move_to(plant.p.x, -12.55, plant.p.yaw,
                                   tol=0.08, timeout_s=10.0, speed=0.12)

    def test_v54_late_handover_reverses_without_turning_back(self):
        heading = math.radians(-81.4)
        plant = HandoverPlant(5.85, -12.727, heading)
        self.assertTrue(self.move_to_corner(plant))
        moving = [command for _, command in plant.commands if any(command)]
        self.assertTrue(moving)
        self.assertLess(moving[0][0], 0.0)
        self.assertTrue(all(-0.12 <= command[0] <= 0.0 for command in moving))
        self.assertLess(max(abs(command[2]) for command in moving), 0.05)
        self.assertLess(max(abs(route.wrap(p.yaw - heading)) for p in plant.motion),
                        math.radians(2.0))
        self.assertLess(math.hypot(plant.p.x - 5.85, plant.p.y + 12.55), 0.08)
        self.assertGreater(plant.p.y, -12.65)
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_earlier_supported_gate_requests_south_motion_without_turnback(self):
        # v78's first y<-11.80 sample: all four wheel collisions contacted
        # gravel_pit_b. Check its measured wheel BODY ORIGINS against the
        # actual scene, not an assumed pit centre or tire-edge clearance.
        scene = Path(__file__).resolve().parents[1] / "robots/blackW/scene_part1.xml"
        geom = ET.parse(scene).find(".//geom[@name='gravel_pit_b']")
        self.assertIsNotNone(geom)
        cx, cy, cz = map(float, geom.attrib["pos"].split())
        hx, hy, hz = map(float, geom.attrib["size"].split())
        wheel_origins = (
            (6.2376, -12.0983, 0.2527),  # FL
            (5.8815, -12.1674, 0.2537),  # FR
            (6.1535, -11.6144, 0.2561),  # RL
            (5.8342, -11.6546, 0.2552),  # RR
        )
        for x, y, z in wheel_origins:
            self.assertGreater(min(x - (cx - hx), cx + hx - x), 0.09)
            self.assertGreater(min(y - (cy - hy), cy + hy - y), 0.60)
            self.assertAlmostEqual(z - (cz + hz), 0.103, delta=0.01)
        # This is the pre-braking pose, not the later settled handover pose.
        # Observe 0.2 s of commands; reaching the corner takes longer.
        heading = math.radians(-83.376)
        plant = HandoverPlant(6.01758905, -11.8144722, heading)
        self.assertFalse(plant._pose_move_to(plant.p.x, -12.55, heading,
                                            tol=0.08, timeout_s=0.20, speed=0.12))
        moving = [command for _, command in plant.commands if any(command)]
        self.assertGreater(moving[0][0], 0.0)
        self.assertTrue(all(0.0 <= command[0] <= 0.12 for command in moving))
        self.assertLess(max(abs(route.wrap(p.yaw - heading)) for p in plant.motion),
                        math.radians(2.0))
        self.assertLess(plant.p.y, -11.83)
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_forward_correction_after_braking_arrives_without_turnback(self):
        # A 0.15 m forward correction after the expert's stopping interval.
        # It complements the measured late handover's 0.177 m reverse case.
        heading = math.radians(-83.2)
        plant = HandoverPlant(5.783, -12.40, heading)
        self.assertTrue(self.move_to_corner(plant))
        moving = [command for _, command in plant.commands if any(command)]
        self.assertTrue(moving)
        self.assertTrue(all(0.0 <= command[0] <= 0.12 for command in moving))
        self.assertLess(max(abs(route.wrap(p.yaw - heading)) for p in plant.motion),
                        math.radians(2.0))
        self.assertLess(math.hypot(plant.p.x - 5.783, plant.p.y + 12.55), 0.08)
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_fallen_handover_is_rejected_before_motion(self):
        plant = HandoverPlant(5.85, -12.727, math.radians(-81.4), fallen=True)
        self.assertFalse(self.move_to_corner(plant))
        self.assertTrue(all(command == (0.0, 0.0, 0.0) for _, command in plant.commands))
        self.assertEqual((plant.p.x, plant.p.y), (5.85, -12.727))
        self.assertLess(plant.p.t, 0.3)

    def test_fall_during_correction_stops_without_waiting_for_deadline(self):
        plant = HandoverPlant(5.783, -11.952, math.radians(-83.2), fall_at=0.20)
        self.assertFalse(self.move_to_corner(plant))
        self.assertTrue(any(command[0] > 0.0 for _, command in plant.commands))
        self.assertTrue(all(command == (0.0, 0.0, 0.0)
                            for t, command in plant.commands if t >= 0.20))
        self.assertLess(plant.p.t, 0.5)
        self.assertEqual(plant.output, (0.0, 0.0, 0.0))

    def test_old_goto_negative_control_requests_large_reverse_turn(self):
        # Execute the old primitive for one tick: its atan2 target points north,
        # so the same measured pose requests +0.65 rad/s instead of reversing.
        plant = HandoverPlant(5.85, -12.727, math.radians(-81.4))
        plant.goto((5.85, -12.55), tol=0.10, timeout_s=0.02, speed=0.15)
        first = plant.commands[0][1]
        self.assertEqual(first[:2], (0.0, 0.0))
        self.assertGreater(first[2], 0.6)



if __name__ == "__main__":
    unittest.main()
