#!/usr/bin/env python3
"""Part2 full-field route wiring (experts + flat connector).

Route order (one chosen difficulty per row):
  high_platform_20cm (east approach) -> stairs_20cm -> slope_20deg
  -> trench_20cm -> 3-pair plum -> bridge_20cm -> high_wall_20cm.
Unused rows (rough/cross-slope/low-friction/plum tiers) are skipped; slope45 is
skipped by design.  Segments use the deployment experts:
  row2p (platform/slope/bridge/wall), st30 (stairs), s1d_trench (trench),
  obstacle (plum).  Position/holding uses PART1_TRANSFER_POLICY=vt_locked12.

Run with the fake kinematic sim first:
  PART2_ROUTE_STEPLOCK=1 PART1_TRANSFER_POLICY=vt_locked12 \
  PART1_RECORD_PATH=... PART1_RECORD_PLAN_PATH=... \
  python demo/rl_sar/scripts/part2_auto_route.py
"""
from __future__ import annotations

import math
import os
import sys

from part1_auto_route import (  # noqa: E402
    BTN_A, BTN_LB, BTN_RB, BTN_START, RouteNode, TRANSFER_POLICY,
    wrap,
)


class Part2Route(RouteNode):
    def switch(self, name: str) -> None:
        # Part1's binding table lacks the Part2 trench specialist.
        if name == "s1d_trench_reward_v75_m100":
            if name == self._current_policy:
                return
            self._press(buttons=[BTN_RB], hat=(1, 0), hold_s=0.15)
            self._current_policy = name
            self._cmd = "policy"
            self._cli = f"switch {name}"
            self._emit_plan()
            self._hold(0.45)
            return
        super().switch(name)

    def cross_x(self, policy: str, y: float, vx: float, stop, timeout: float,
                label: str, lane_yaw: float = 0.0) -> bool:
        """Cross an obstacle along +x with a y lane hold and fake-friendly gates."""
        self.switch(policy)
        self._cmd = f"{label}(policy={policy}, y={y:+.2f}, vx={vx:+.2f})"
        self._cli = f"switch {policy}; forward {vx:+.2f}m/s + lane hold"
        self._corr = {}
        self._emit_plan()
        prev = None
        t0 = self._now()
        while self._now() - t0 < timeout:
            p = self.pose()
            if stop(p):
                self.stop(0.5)
                return True
            if self.fallen():
                return False
            e = p.y - y
            if abs(wrap(lane_yaw)) < 0.1:      # +x lane
                vy = max(-0.2, min(0.2, -0.5 * e))
            else:                              # -x lane (body +y = world -y)
                vy = max(-0.2, min(0.2, 0.5 * e))
            wz = max(-0.35, min(0.35, 1.2 * wrap(lane_yaw - p.yaw) + 0.8 * e))
            self.command(vx, vy, wz)
            prev = p
            self._tick()
        self.stop(0.5)
        return False

    def _route(self) -> None:
        log = self.get_logger()
        if not self.wait_pose(30.0):
            log.error("no /eval_state")
            return
        log.info("Part2 route: standing up / entering RL")
        self.switch("row2p_p40w40_v1")
        self._press(buttons=[BTN_A], hold_s=0.15)
        self.wait_stable_z(0.46, tilt_max=8.0, hold_s=0.5, timeout_s=15.0)
        self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)
        self._hold(1.5)
        self.wait_stable_z(0.40, tilt_max=15.0, hold_s=0.3, timeout_s=5.0)
        self.switch(TRANSFER_POLICY)
        log.info("Part2 route start at (%.2f,%.2f,yaw %.2f)" %
                 (self.pose().x, self.pose().y, self.pose().yaw))

        # 1) high_platform_20cm at x=12.75,y=0, east-side approach.
        self.set_leg("high_platform_20cm")
        self.require((0.0, -2.5), tol=0.25, timeout_s=12.0, speed=0.35)
        self.require((14.0, -2.5), tol=0.30, timeout_s=45.0, speed=0.45)
        self.require((14.0, -0.2), tol=0.15, timeout_s=20.0, speed=0.25)
        self.rotate_hold(math.pi, timeout_s=15.0)  # face -x
        ok = self.cross_x("row2p_p40w40_v1", 0.0, 0.40,
                          lambda p: p.x < 11.4, 20.0, "platform20",
                          lane_yaw=math.pi)
        self.report("high_platform_20cm", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 2) stairs_20cm at x=48.15..52.4,y=3, west approach.
        self.set_leg("stairs_20cm")
        self.require((14.0, -2.5), tol=0.30, timeout_s=12.0, speed=0.35)
        self.require((46.5, -2.5), tol=0.30, timeout_s=95.0, speed=0.55)
        self.require((46.5, 3.0), tol=0.15, timeout_s=30.0, speed=0.30)
        self.rotate_hold(0.0, timeout_s=15.0)  # face +x
        ok = self.cross_x("st30_ppo_st30_s100_600", 3.0, 0.45,
                          lambda p: p.x > 53.0, 25.0, "stairs20")
        self.report("stairs_20cm", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 3) slope_20deg at x=22.3..24.5,y=6; platforms/down slope are in the
        #    modified route scene.  West approach.
        self.set_leg("slope_20deg")
        self.require((53.0, 4.5), tol=0.25, timeout_s=15.0, speed=0.30)
        self.require((20.2, 4.5), tol=0.30, timeout_s=85.0, speed=0.55)
        self.require((20.2, 6.0), tol=0.15, timeout_s=20.0, speed=0.28)
        self.rotate_hold(0.0, timeout_s=15.0)
        ok = self.cross_x("row2p_p40w40_v1", 6.0, 0.45,
                          lambda p: p.x > 26.2, 20.0, "slope20")
        self.report("slope_20deg", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 4) trench_20cm at x=9.7..12.5,y=9, west approach.
        self.set_leg("trench_20cm")
        self.require((26.2, 7.5), tol=0.25, timeout_s=15.0, speed=0.30)
        self.require((8.5, 7.5), tol=0.30, timeout_s=55.0, speed=0.50)
        self.require((8.5, 9.0), tol=0.15, timeout_s=20.0, speed=0.28)
        self.rotate_hold(0.0, timeout_s=15.0)
        ok = self.cross_x("s1d_trench_reward_v75_m100", 9.0, 0.45,
                          lambda p: p.x > 13.0, 20.0, "trench20")
        self.report("trench_20cm", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 5) reduced long plum at y=12: entry 4.4, three pairs at 5.0/6.2,
        #    exit platform 7.0; original side ramps removed.
        self.set_leg("plum_3pair")
        self.require((13.0, 10.5), tol=0.25, timeout_s=12.0, speed=0.30)
        self.require((2.5, 10.5), tol=0.30, timeout_s=30.0, speed=0.45)
        self.require((2.5, 12.0), tol=0.15, timeout_s=20.0, speed=0.28)
        self.rotate_hold(0.0, timeout_s=15.0)
        ok = self.cross_x("obstacle", 12.0, 0.40,
                          lambda p: p.x > 7.6, 20.0, "plum3")
        self.report("plum_3pair", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 6) bridge_20cm at x=18.3..21.7,y=18, west approach.
        self.set_leg("bridge_20cm")
        self.require((7.6, 13.5), tol=0.25, timeout_s=12.0, speed=0.30)
        self.require((16.8, 13.5), tol=0.30, timeout_s=20.0, speed=0.45)
        self.require((16.8, 18.0), tol=0.15, timeout_s=25.0, speed=0.30)
        self.rotate_hold(0.0, timeout_s=15.0)
        ok = self.cross_x("row2p_p40w40_v1", 18.0, 0.40,
                          lambda p: p.x > 22.5, 20.0, "bridge20")
        self.report("bridge_20cm", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 7) high_wall_20cm = cross_bar_5 at x=7.8,y=21, west approach.
        self.set_leg("high_wall_20cm")
        self.require((22.5, 19.5), tol=0.25, timeout_s=12.0, speed=0.30)
        self.require((5.5, 19.5), tol=0.30, timeout_s=55.0, speed=0.50)
        self.require((5.5, 21.0), tol=0.15, timeout_s=20.0, speed=0.28)
        self.rotate_hold(0.0, timeout_s=15.0)
        ok = self.cross_x("row2p_p40w40_v1", 21.0, 0.40,
                          lambda p: p.x > 10.0, 15.0, "wall20")
        self.report("high_wall_20cm", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")

        log.info("Part2 route finished")
        self.stop(0.5)


def main() -> int:
    import rclpy
    rclpy.init()
    node = Part2Route()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
