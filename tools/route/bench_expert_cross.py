#!/usr/bin/env python3
"""Deployment-side expert-cross bench: drive one policy on an official Part2
single-level scene with the same state-feedback lane command the route uses.

Used to compare the deployed sim_node+rl_sim stack against the in-process
``robocon_eval.policy_runner`` bench (same command, same level, same spawn).

    source demo/install/setup.bash
    ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=87 PART1_ROUTE_STEPLOCK=1 \
    PART1_TRANSFER_POLICY=vt_locked12 \
    .pixi/envs/demo/bin/python tools/route/bench_expert_cross.py \
        --vx 0.5 --lane-y 0.0 --lane-yaw 0 --seconds 14 --out artifacts/part2_route3/bench_platform40.jsonl

Run it against a launch started as::

    ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=2 \
        level:=high_platform_40cm policy_name:=row2p_p40w40_v1 \
        viewer:=false realtime:=true eval_state_every:=10
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String

BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_BACK, BTN_START = range(8)
AX_LX, AX_LY, AX_LT, AX_RX, AX_RY, AX_RT, AX_HATX, AX_HATY = range(8)
SCALE = 1.5


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Bench(Node):
    def __init__(self, args) -> None:
        super().__init__("bench_expert_cross")
        self.args = args
        self._lock = threading.Lock()
        self._axes = [0.0] * 8
        self._buttons = [0] * 11
        self._pose = (0.0, 0.0, 0.0, 0.0, 0.0)  # x,y,z,yaw,tilt
        self._sim_t = 0.0
        self._have = False
        self._evt = threading.Event()
        self._joy = self.create_publisher(Joy, "joy", QoSProfile(depth=1))
        self.create_subscription(String, "eval_state", self._on_eval,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self._rec = open(args.out, "w", buffering=1) if args.out else None

    def _on_eval(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
            x, y, z = (float(v) for v in d["base"])
            qw, qx, qy, qz = (float(v) for v in d["quat"])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, r22))))
            self._sim_t = float(d.get("t", self._sim_t))
            self._pose = (x, y, z, yaw, tilt)
            self._have = True
            if self._rec is not None:
                self._rec.write(msg.data + "\n")
        except Exception:  # noqa: BLE001
            return
        self._evt.set()

    def _send(self) -> None:
        m = Joy()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "bench_expert_cross"
        with self._lock:
            m.axes = list(self._axes)
            m.buttons = list(self._buttons)
        self._joy.publish(m)

    def tick(self) -> None:
        if self.args.steplock:
            self._send()
            if self._evt.wait(timeout=0.5):
                self._evt.clear()
        else:
            self._send()
            time.sleep(0.02)

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / 0.02)))):
            self.tick()

    def press(self, buttons=(), hat=(0, 0), hold_s: float = 0.15) -> None:
        with self._lock:
            for b in buttons:
                self._buttons[b] = 1
            self._axes[AX_HATX] = float(hat[0])
            self._axes[AX_HATY] = float(hat[1])
        self.hold(hold_s)
        with self._lock:
            for b in buttons:
                self._buttons[b] = 0
            self._axes[AX_HATX] = 0.0
            self._axes[AX_HATY] = 0.0
        self.hold(0.10)

    def command(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        with self._lock:
            self._axes[AX_LX] = max(-1.0, min(1.0, vy / SCALE))
            self._axes[AX_LY] = max(-1.0, min(1.0, vx / SCALE))
            self._axes[AX_RX] = max(-1.0, min(1.0, wz / SCALE))

    def run(self) -> int:
        a = self.args
        log = self.get_logger()
        t0 = time.monotonic()
        while not self._have and time.monotonic() - t0 < 30.0:
            self.hold(0.05)
        if not self._have:
            log.error("no /eval_state")
            return 2
        log.info("bench: stand up / enter RL")

        def wait_z(zmin: float, timeout: float) -> bool:
            t = time.monotonic()
            stable = 0
            while time.monotonic() - t < timeout:
                if self._pose[2] >= zmin:
                    stable += 1
                    if stable >= 10:
                        return True
                else:
                    stable = 0
                self.hold(0.05)
            return False

        self.hold(0.3)
        self.press(buttons=[BTN_A])
        self.hold(2.0)
        self.press(buttons=[BTN_RB], hat=(0, 1))  # EnterRL
        self.hold(1.5)
        self.hold(0.3)
        px, py, pz, pyaw, _ = self._pose
        log.info("bench start pose=(%.2f,%.2f,%.2f,%.1fdeg)" % (px, py, pz, math.degrees(pyaw)))
        prev = None
        n = int(round(a.seconds / 0.02))
        max_z = pz
        crossed = False
        for i in range(n):
            p = self._pose
            max_z = max(max_z, p[2])
            if p[4] > 80.0 or p[2] < 0.12:
                log.error("bench: fall/tilt at (%.2f,%.2f,%.2f) tilt=%.1f"
                          % (p[0], p[1], p[2], p[4]))
                break
            # lane command identical to the route's lane_cmd (adaptive gains)
            e = (-(p[0] - a.lane_x)) * math.sin(a.lane_yaw) \
                + (p[1] - a.lane_y) * math.cos(a.lane_yaw)
            h = wrap(p[3] - a.lane_yaw)
            vx_w = vy_w = 0.0
            if prev is not None and self._sim_t > prev[0]:
                dt = self._sim_t - prev[0]
                vx_w = (p[0] - prev[1]) / dt
                vy_w = (p[1] - prev[2]) / dt
            vy_meas = -math.sin(a.lane_yaw) * vx_w + math.cos(a.lane_yaw) * vy_w
            vy = max(-0.2, min(0.2, -0.5 * e))
            wz = max(-0.7, min(0.7, -0.8 * e - 2.0 * h - 0.6 * vy_meas))
            self.command(a.vx, vy * math.cos(h), wz)
            prev = (self._sim_t, p[0], p[1])
            self.tick()
            if i % 25 == 0:
                log.info("bench t=%.1f pose=(%.2f,%.2f,%.2f) tilt=%.1f"
                         % (self._sim_t, p[0], p[1], p[2], p[4]))
            # +x course crossing check (official level platform at x 1.8..3.3)
            if p[0] > 3.6 and p[2] > 0.75:
                crossed = True
        self.command(0.0, 0.0, 0.0)
        self.hold(0.5)
        px, py, pz, _, _ = self._pose
        log.info("bench END pose=(%.2f,%.2f,%.2f) max_z=%.3f crossed_top=%s"
                 % (px, py, pz, max_z, crossed))
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--lane-x", type=float, default=0.0)
    ap.add_argument("--lane-y", type=float, default=0.0)
    ap.add_argument("--lane-yaw", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--out", default="")
    # Wall-clock pacing by default: steplock is only meaningful when the sim is
    # paced by the route (it is NOT valid for full-field validation with
    # realtime=false -- the sim then outruns the route's /eval_state processing).
    ap.add_argument("--steplock", type=int, default=0)
    args = ap.parse_args()
    rclpy.init()
    node = Bench(args)
    import threading
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    try:
        rc = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
