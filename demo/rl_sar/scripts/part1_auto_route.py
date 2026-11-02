#!/usr/bin/env python3
"""Part1 whole-course auto route for the MuJoCo window (Plan A demo).

Launched by::

    ros2 launch rl_sar rl_sim_all.launch.py robot:=blackW part:=1 \
        policy_name:=vt_pose_v1 route:=part1

The node behaves like a virtual gamepad on /joy:

* continuous axes: body-frame vx/vy/wz are mapped through the runtime
  `joy_command_scale` (1.5 in rl_sim_all) exactly like a stick;
* button edges: policy switches (LB+D-pad left = vt_pose_v1, Start+D-pad down =
  row2p, RB+D-pad down = st30, Start+D-pad right = obstacle) and fixed actions
  (Back+D-pad down = low_bar_drive, Back+D-pad right = bridge_drive);
* pose feedback comes from `sim_node` `/eval_state` (std_msgs/String JSON). The launch
  default is `eval_state_every:=1` (one sample per 2 ms physics frame, judged trace rate);
  the route only reads the latest frame, so 50 Hz (`eval_state_every:=10`) is enough for
  interactive driving and lighter on IO (the route raises 0 to 10).

Route order follows the legacy Part1 diagnostic route:
high_wall -> t_stairs -> slope -> bridge_blocks -> bridge_bands (fixed) ->
height_bar (fixed) -> gravel -> slalom (predefined S path around the poles) ->
return to start.  Fixed-action segments use the real FSM states.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String

JOY_COMMAND_SCALE = 1.5
PUBLISH_HZ = 50.0
TRANSFER_POLICY = os.environ.get("PART1_TRANSFER_POLICY", "vt_pose_v1").strip() or "vt_pose_v1"

BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_BACK, BTN_START = range(8)
AX_LX, AX_LY, AX_LT, AX_RX, AX_RY, AX_RT, AX_HATX, AX_HATY = range(8)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class _Pose:
    __slots__ = ("x", "y", "z", "yaw", "tilt", "t")

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, tilt=0.0, t=0.0):
        self.x, self.y, self.z, self.yaw, self.tilt, self.t = x, y, z, yaw, tilt, t


class RouteNode(Node):
    def __init__(self) -> None:
        super().__init__("part1_auto_route")
        self._lock = threading.Lock()
        self._axes = [0.0] * 8
        self._buttons = [0] * 11
        self._pose = _Pose()
        self._feet = {}
        self._have_pose = False
        self._recoveries = 0
        # Step-lock (PART1_ROUTE_STEPLOCK=1): control loops wait for the next
        # /eval_state frame (one per 10 sim steps) instead of wall-clock
        # sleeps, so the commanded sequence is a pure function of sim state —
        # repeated runs are reproducible (the route variance is control-loop
        # timing, not RNG; MuJoCo and the fixed spawn are deterministic).
        # Pair with realtime:=false so the sim is not wall-throttled.
        self._steplock = os.environ.get("PART1_ROUTE_STEPLOCK", "").strip() == "1"
        self._frame_evt = threading.Event()
        self._last_eval_wall = 0.0
        self._current_policy = ""
        self._record = os.environ.get("PART1_RECORD_PATH", "").strip()
        self._record_fh = open(self._record, "w", buffering=1) if self._record else None
        self._plan_path = os.environ.get("PART1_RECORD_PLAN_PATH", "").strip()
        self._plan_fh = open(self._plan_path, "w", buffering=1) if self._plan_path else None
        self._leg = "start"
        self._target = None
        self._cmd = "start"
        self._cli = ""
        self._corr = {}
        self._sim_t = 0.0
        self._last_pose_wall = 0.0
        self._joy_pub = self.create_publisher(Joy, "joy", QoSProfile(depth=1))
        self.create_subscription(
            String, "eval_state", self._on_eval,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)
        self.get_logger().info("part1_auto_route: virtual gamepad on /joy (waiting for /eval_state)")
        self._thread = threading.Thread(target=self._run, name="route", daemon=True)
        self._thread.start()

    def _tick(self, fallback_s: float = 0.02) -> None:
        """Advance one control step: wait for the next state frame (step-lock)
        or sleep the wall-clock fallback."""
        if self._steplock:
            if self._frame_evt.wait(timeout=0.5):
                self._frame_evt.clear()
            return
        self._hold(fallback_s)

    def adaptive_lane_cmd(self, p, prev, lane_yaw: float, center: float,
                          axis: str, vx: float):
        """Repo AdaptiveCommand generalised to a lane along `axis`.

        Gains are the training-side adaptive lane (adaptive_command.py:
        vy_gain 0.5/max 0.2, k_y 0.8, k_yaw 2.0, k_vy 0.6/max_wz 0.7) — the
        experts were trained with this feedback, so feeding it is
        in-distribution.  axis 'x': lane along +x, lateral = world y.
        axis 'y': lane along +y, lateral = world x (the T-stairs line).
        Returns (vx, vy, wz) in the body frame.
        """
        spsi, cpsi = math.sin(lane_yaw), math.cos(lane_yaw)
        if axis == "x":
            e = p.y - center
        else:
            e = -(p.x - center)
        vx_w = vy_w = 0.0
        if prev is not None and p.t > prev.t:
            dt = p.t - prev.t
            vx_w = (p.x - prev.x) / dt
            vy_w = (p.y - prev.y) / dt
        vy_b_meas = -spsi * vx_w + cpsi * vy_w
        h = wrap(p.yaw - lane_yaw)
        vy_lat = max(-0.2, min(0.2, -0.5 * e))
        wz = max(-0.7, min(0.7, -0.8 * e - 2.0 * h - 0.6 * vy_b_meas))
        vy_body = vy_lat * math.cos(h)
        return (vx, vy_body, wz)

    def _hold(self, seconds: float) -> None:
        """Hold the current output for `seconds`: frame-counted in step-lock
        (deterministic), wall-clock otherwise."""
        if not self._steplock:
            self._hold(seconds)
            return
        for _ in range(max(1, int(round(seconds / 0.02)))):
            self._tick()

    def _on_eval(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
            x, y, z = (float(v) for v in d["base"])
            qw, qx, qy, qz = (float(v) for v in d["quat"])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, r22))))
        except Exception:  # noqa: BLE001
            return
        try:
            raw = json.loads(msg.data)
            self._sim_t = float(raw.get("t", self._sim_t))
            feet = raw.get("feet") or {}
            with self._lock:
                self._feet = {
                    str(k): (float(v[0]), float(v[1]), float(v[2]))
                    for k, v in feet.items() if len(v) >= 3
                }
        except Exception:  # noqa: BLE001
            pass
        if self._record_fh is not None:
            self._record_fh.write(msg.data + "\n")
        with self._lock:
            self._pose = _Pose(x, y, z, yaw, tilt, time.monotonic())
            self._have_pose = True
            self._last_pose_wall = time.monotonic()
        if self._steplock:
            self._frame_evt.set()

    def _publish(self) -> None:
        m = Joy()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "part1_auto_route"
        with self._lock:
            m.axes = list(self._axes)
            m.buttons = list(self._buttons)
        self._joy_pub.publish(m)

    def pose(self) -> _Pose:
        with self._lock:
            return _Pose(self._pose.x, self._pose.y, self._pose.z,
                         self._pose.yaw, self._pose.tilt, self._pose.t)

    def feet(self):
        with self._lock:
            return dict(self._feet)

    def wheel_level_spread(self) -> float:
        feet = self.feet()
        zs = [feet[n][2] for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
              if n in feet]
        return (max(zs) - min(zs)) if len(zs) == 4 else 0.0

    def check_and_recover(self, spread_tol: float = 0.06,
                          timeout_s: float = 12.0) -> bool:
        """Closed-loop edge recovery (legacy part1_route parity, 2026-11-01).

        When the four wheels sit on two planes on flat ground (a wheel dropped
        off an edge during a turn or an expert handover), switch to the
        edge-recovery specialist (vt_edge_v1) and drive toward the
        majority-plane wheel centroid until all four share one level.  Slope
        gate: on an incline the wheels sit on two planes by geometry (~0.15 m
        spread on a 20 deg slope) while the body tilt is large — do not
        "recover" there, it drives the robot back up the slope.
        """
        if self.wheel_level_spread() < spread_tol:
            return True
        if self.pose().tilt > 10.0:
            return False
        self.get_logger().info(
            "wheel-off-edge detected (spread %.2f); edge recovery" %
            self.wheel_level_spread())
        self.switch("vt_edge_v1")
        self._recoveries += 1
        t_end = time.monotonic() + timeout_s
        try:
            while time.monotonic() < t_end:
                if self.wheel_level_spread() < spread_tol or self.fallen():
                    break
                feet = self.feet()
                names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
                zs = {n: feet[n][2] for n in names if n in feet}
                if len(zs) < 4:
                    break
                lo = min(zs.values())
                hi = [n for n, z in zs.items() if z > lo + spread_tol]
                if not hi or len(hi) == 4:
                    hi = sorted(zs, key=zs.get)[-1:]
                tx = sum(feet[n][0] for n in hi) / len(hi)
                ty = sum(feet[n][1] for n in hi) / len(hi)
                p = self.pose()
                err = wrap(math.atan2(ty - p.y, tx - p.x) - p.yaw)
                if abs(err) > 0.7:
                    self.command(0.0, 0.0, math.copysign(0.45, err))
                else:
                    self.command(0.20, 0.0, max(-0.20, min(0.20, err)))
                self._tick()
        finally:
            self.stop(0.2)
            self.ensure_transfer()
        ok = self.wheel_level_spread() < max(0.10, spread_tol)
        self.get_logger().info("edge recovery done: level=%s" % ok)
        return ok

    def wheels_inside(self, xr, yr) -> bool:
        feet = self.feet()
        if len(feet) < 4:
            return False
        for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            p = feet.get(name)
            if p is None:
                return False
            if not (xr[0] <= p[0] <= xr[1] and yr[0] <= p[1] <= yr[1]):
                return False
        return True

    def command(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        with self._lock:
            self._axes[AX_LX] = max(-1.0, min(1.0, vy / JOY_COMMAND_SCALE))
            self._axes[AX_LY] = max(-1.0, min(1.0, vx / JOY_COMMAND_SCALE))
            self._axes[AX_RX] = max(-1.0, min(1.0, wz / JOY_COMMAND_SCALE))

    def fixed_command(self, x: float, yaw: float = 0.0) -> None:
        with self._lock:
            self._axes[AX_LY] = max(-1.0, min(1.0, x / JOY_COMMAND_SCALE))
            self._axes[AX_RX] = max(-1.0, min(1.0, yaw / JOY_COMMAND_SCALE))

    def _press(self, buttons=(), hat=(0, 0), hold_s: float = 0.12) -> None:
        with self._lock:
            for b in buttons:
                self._buttons[b] = 1
            self._axes[AX_HATX] = float(hat[0])
            self._axes[AX_HATY] = float(hat[1])
        self._hold(hold_s)
        with self._lock:
            for b in buttons:
                self._buttons[b] = 0
            self._axes[AX_HATX] = 0.0
            self._axes[AX_HATY] = 0.0
        self._hold(0.10)

    def switch(self, name: str) -> None:
        combos = {
            "vt_pose_v1": ([BTN_LB], (-1, 0)),
            "vt_edge_v1": ([BTN_BACK], (0, 1)),
            "row2p_p40w40_v1": ([BTN_START], (0, -1)),
            "st30_ppo_st30_s100_600": ([BTN_RB], (0, -1)),
            "obstacle": ([BTN_START], (1, 0)),
            "flat": ([BTN_START], (-1, 0)),
            "stair": ([BTN_START], (0, 1)),
        }
        if name not in combos:
            raise ValueError(f"no gamepad binding for policy {name}")
        if name == self._current_policy:
            return
        buttons, hat = combos[name]
        self._press(buttons=buttons, hat=hat, hold_s=0.15)
        self._current_policy = name
        self.get_logger().info(f"switch -> {name}")
        self._hold(0.45)

    def ensure_transfer(self) -> None:
        self.switch(TRANSFER_POLICY)

    def enter_fixed(self, kind: str) -> None:
        self._cmd = f"enter_fixed({kind})"
        self._emit_plan()
        combos = {"low_bar": ([BTN_BACK], (0, -1)), "bridge": ([BTN_BACK], (1, 0))}
        buttons, hat = combos[kind]
        self._press(buttons=buttons, hat=hat, hold_s=0.15)
        self.get_logger().info(f"fixed action -> {kind}")
        self._hold(0.9)

    def exit_fixed(self) -> None:
        self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)
        self._hold(0.5)

    def stop(self, settle_s: float = 0.4) -> None:
        self.command(0.0, 0.0, 0.0)
        self._hold(settle_s)

    def wait_pose(self, timeout_s: float = 20.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self._have_pose and time.monotonic() - self._last_pose_wall < 1.0:
                return True
            self._hold(0.05)
        return False

    def wait_z(self, z_min: float, timeout_s: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.pose().z > z_min:
                return True
            self._hold(0.05)
        return False

    def wait_stable_z(self, z_min: float, tilt_max: float = 10.0,
                      hold_s: float = 0.4, timeout_s: float = 10.0) -> bool:
        """Wait for a settled upright standing pose (z high, tilt small)."""
        t0 = time.monotonic()
        above_since = None
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            now = time.monotonic()
            if p.z > z_min and p.tilt < tilt_max:
                if above_since is None:
                    above_since = now
                elif now - above_since >= hold_s:
                    return True
            else:
                above_since = None
            self._hold(0.05)
        return False

    def fallen(self) -> bool:
        p = self.pose()
        return p.z < 0.15 or p.tilt > 60.0 or time.monotonic() - p.t > 1.5

    def rotate_by(self, delta: float, tol: float = 0.10, timeout_s: float = 10.0) -> bool:
        self.ensure_transfer()
        target = wrap(self.pose().yaw + delta)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            err = wrap(target - self.pose().yaw)
            if abs(err) < tol:
                break
            wz = math.copysign(max(0.12, min(0.45, 1.2 * abs(err))), err)
            self.command(0.0, 0.0, wz)
            self._tick()
        self.stop(0.3)
        return abs(wrap(target - self.pose().yaw)) < 0.16

    def rotate_hold(self, target_yaw: float, timeout_s: float = 8.0) -> bool:
        """In-place yaw turn with position feedback to stay on a small platform."""
        self._cmd = f"rotate_hold({target_yaw:.2f})"
        self._emit_plan()
        self.ensure_transfer()
        p0 = self.pose()
        x0, y0 = p0.x, p0.y
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            err = wrap(target_yaw - p.yaw)
            ex, ey = x0 - p.x, y0 - p.y
            fwd = ex * math.cos(p.yaw) + ey * math.sin(p.yaw)
            vx = max(-0.16, min(0.16, 1.2 * fwd))
            wz = math.copysign(max(0.10, min(0.35, 1.0 * abs(err))), err) if abs(err) > 0.04 else 0.0
            self.command(vx, 0.0, wz)
            self._tick()
            if abs(err) < 0.08 and math.hypot(ex, ey) < 0.10:
                break
        self.stop(0.3)
        p = self.pose()
        return abs(wrap(target_yaw - p.yaw)) < 0.14 and math.hypot(x0 - p.x, y0 - p.y) < 0.22

    def _record_corr(self, dx: float, dy: float, dyaw: float) -> None:
        self._corr = {"dx": round(dx, 3), "dy": round(dy, 3),
                      "dyaw": round(dyaw, 3)}
        self._emit_plan()

    def _pose_move_to(self, tx: float, ty: float, target_yaw: float,
                      tol: float = 0.01, timeout_s: float = 10.0,
                      speed: float = 0.15) -> bool:
        """Closed-loop relative move using pose policy, reporting residual error."""
        self._cmd = f"pose_move_to({tx:.2f},{ty:.2f})"
        self._emit_plan()
        self.ensure_transfer()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            dx, dy = tx - p.x, ty - p.y
            dyaw = wrap(target_yaw - p.yaw)
            self._record_corr(dx, dy, dyaw)
            if math.hypot(dx, dy) < tol and abs(dyaw) < math.radians(2.0):
                self.stop(0.25)
                return True
            if abs(dyaw) > 0.7:
                self.command(0.0, 0.0, math.copysign(min(0.35, max(0.10, 1.2 * abs(dyaw))), dyaw))
            else:
                fwd = dx * math.cos(p.yaw) + dy * math.sin(p.yaw)
                side = -dx * math.sin(p.yaw) + dy * math.cos(p.yaw)
                vx = max(-speed, min(speed, 1.5 * fwd))
                wz = max(-0.20, min(0.20, 1.2 * dyaw - 0.8 * side))
                self.command(vx, 0.0, wz)
            self._tick()
        self.stop(0.25)
        return False

    def rel_forward(self, meters: float, tol: float = 0.01,
                    max_corrections: int = 3) -> bool:
        p = self.pose()
        tx = p.x + meters * math.cos(p.yaw)
        ty = p.y + meters * math.sin(p.yaw)
        self._cli = f"forward {meters*100:+.1f} cm"
        self._emit_plan()
        for _ in range(max_corrections):
            if self._pose_move_to(tx, ty, p.yaw, tol=tol, timeout_s=8.0):
                return True
        return False

    def rel_rotate(self, deg: float, tol_deg: float = 1.0,
                   max_corrections: int = 3) -> bool:
        p = self.pose()
        target = wrap(p.yaw + math.radians(deg))
        self._cli = f"rotate {deg:+.0f} deg"
        self._emit_plan()
        for _ in range(max_corrections):
            ok = self._pose_move_to(p.x, p.y, target,
                                    tol=0.02, timeout_s=8.0, speed=0.10)
            if ok and abs(wrap(target - self.pose().yaw)) < math.radians(tol_deg):
                return True
        return False

    def face(self, yaw: float) -> None:
        self._cmd = f"face({yaw:.2f})"
        self._emit_plan()
        if not self.rotate_by(wrap(yaw - self.pose().yaw), timeout_s=12.0):
            p = self.pose()
            raise RuntimeError(f"face {yaw:.2f} failed at yaw {p.yaw:.2f} ({p.x:.2f},{p.y:.2f})")

    def goto(self, target, tol: float = 0.12, timeout_s: float = 30.0,
             speed: float = 0.18) -> bool:
        self.ensure_transfer()
        tx, ty = float(target[0]), float(target[1])
        self._cmd = f"goto(({tx:.2f},{ty:.2f}))"
        self._emit_plan()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            ex, ey = tx - p.x, ty - p.y
            dist = math.hypot(ex, ey)
            if dist < tol:
                break
            err = wrap(math.atan2(ey, ex) - p.yaw)
            if abs(err) > 0.7:
                wz = math.copysign(max(0.12, min(0.45, 1.2 * abs(err))), err)
                self.command(0.0, 0.0, wz)
            else:
                v = min(speed, max(0.08, 0.7 * dist))
                wz = max(-0.20, min(0.20, 1.0 * err))
                self.command(v, 0.0, wz)
            self._tick()
        self.stop(0.3)
        p = self.pose()
        return math.hypot(tx - p.x, ty - p.y) < max(0.20, 1.5 * tol)

    def set_leg(self, leg: str) -> None:
        self._leg = leg
        self._emit_plan()

    def set_cmd(self, cmd: str) -> None:
        self._cmd = cmd
        self._emit_plan()

    def _emit_plan(self) -> None:
        if self._plan_fh is None:
            return
        self._plan_fh.write(json.dumps({
            "t": self._sim_t, "leg": self._leg, "cmd": self._cmd,
            "cli": self._cli, "corr": self._corr,
            "target": list(self._target) if self._target is not None else None,
        }) + "\n")

    def require(self, target, tol: float = 0.12, timeout_s: float = 30.0,
                recover: bool = True, speed: float = 0.18) -> None:
        if recover:
            self.check_and_recover()
        if not self.goto(target, tol=tol, timeout_s=timeout_s, speed=speed):
            p = self.pose()
            raise RuntimeError(
                f"arrive {target} failed at ({p.x:.2f},{p.y:.2f},{p.yaw:.2f})")

    def spin_fixed(self, target_yaw: float, tol: float = 0.08,
                   timeout_s: float = 15.0, rate: float = 0.45,
                   max_drift: float = 0.30) -> bool:
        """Skid-steer in-place spin through the fixed-action differential drive.

        User proposal (2026-11-01): hold the legs at the yaml pose (kp-damped,
        near-standing wide stance) and drive the wheels differentially — the
        contacts scrub instead of rolling the body sideways, so the centroid
        stays put (unlike the RL in-place turn whose swept footprint is
        ~0.65 m radius — probe_turn_footprint; that is what kept falling off
        the 1.0 m bridge A west platform).  The fixed state interpolates in
        over prepare_cycles (deterministic) and this loop closes on /eval_state
        yaw; drift is watched and capped.

        Returns True when |yaw - target| < tol.  Logs the centroid drift so
        route runs carry the footprint evidence.
        """
        p0 = self.pose()
        self.enter_fixed("bridge")
        ok = False
        t0 = time.monotonic()
        max_seen = 0.0
        # FSM channel is direct (edge11/12 evidence): commanded yaw sign ==
        # body rotation sign.  edge11: negative cmd -> -123 deg; edge12:
        # positive cmd -> +95 deg.  (An earlier auto-flip heuristic misfired
        # on the 1.5 s prepare interpolation and is gone.)
        # Burst+dwell instead of a continuous P-loop: the fixed differential
        # drive is momentum-dominated (wheels scrub, body inertia carries past
        # sign flips — spin_probe3/edge13 limit cycles), so command fixed
        # bursts toward the target, damp, re-measure.
        # Direction calibration: the differential channel is direct MOST runs
        # (edge18-30 five spins passed), but the camber coupling can overwhelm
        # it depending on the initial leg state (edge31: +110 deg while
        # commanded negative).  A short calibration burst measures the actual
        # sign before the main loop commits.
        cal0 = self.pose().yaw
        self.fixed_command(0.0, 0.45)
        self._hold(0.5)
        self.fixed_command(0.0, 0.0)
        self._hold(0.2)
        dyaw_cal = wrap(self.pose().yaw - cal0)
        sign = 1.0 if dyaw_cal >= 0 else -1.0
        self.get_logger().info(
            "spin cal: dyaw=%+.1fdeg -> cmd sign %+.0f" %
            (math.degrees(dyaw_cal), sign))
        try:
            while time.monotonic() - t0 < timeout_s:
                p = self.pose()
                err = wrap(target_yaw - p.yaw)
                drift = math.hypot(p.x - p0.x, p.y - p0.y)
                max_seen = max(max_seen, drift)
                if drift > max_drift:
                    self.get_logger().warning(
                        "spin drift %.2f m exceeds cap; stopping" % drift)
                    break
                if abs(err) < tol:
                    self.stop(0.8)
                    p = self.pose()
                    if abs(wrap(target_yaw - p.yaw)) < tol:
                        ok = True
                    break
                if self.fallen():
                    break
                burst = min(1.2, 0.8 + abs(err))
                cmd = sign * math.copysign(min(rate, max(0.30, 0.8 * abs(err))), err)
                t_b = time.monotonic()
                while time.monotonic() - t_b < burst:
                    self.fixed_command(0.0, cmd)
                    self._tick()
                    if self.fallen():
                        break
                self.stop(0.6)  # damp, then re-measure
        finally:
            self.stop(0.6)
            self.exit_fixed()
        p1 = self.pose()
        self.get_logger().info(
            "spin_fixed: %s dyaw=%.1fdeg drift=%.3f m (max %.3f) end=(%.2f,%.2f)" %
            ("PASS" if ok else "FAIL",
             math.degrees(wrap(p1.yaw - target_yaw)),
             math.hypot(p1.x - p0.x, p1.y - p0.y), max_seen, p1.x, p1.y))
        return ok

    def drive_expert(self, policy: str, vx: float, target_yaw: float,
                     until, timeout_s: float) -> bool:
        """Open-loop velocity profile after precise pose alignment.

        No yaw feedback during the crossing: the obstacle policies track yaw
        poorly, and a yaw correction command was observed to veer the robot off
        the T-stairs.  Alignment is the pose primitive's job.
        """
        self.switch(policy)
        self._cmd = f"drive_expert({policy}, vx={vx:+.2f})"
        self._emit_plan()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.command(vx, 0.0, 0.0)
            self._tick()
            if until(self.pose()):
                self.stop(0.6)
                return True
            if self.fallen():
                break
        self.stop(0.6)
        return False

    def turn_expert(self, policy: str, target_yaw: float,
                    timeout_s: float = 6.0, max_wz: float = 0.40) -> bool:
        """Rotate in place while staying in an obstacle policy (platform turns)."""
        self.switch(policy)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            err = wrap(target_yaw - p.yaw)
            if abs(err) < 0.08:
                self.stop(0.25)
                return True
            wz = math.copysign(min(max_wz, max(0.08, 1.2 * abs(err))), err)
            self.command(0.0, 0.0, wz)
            self._tick()
            if self.fallen():
                break
        self.stop(0.3)
        return False

    def drive_expert_heading(self, policy: str, vx: float, target_yaw: float,
                             until, timeout_s: float, x_hold=None,
                             kp: float = 1.0, max_wz: float = 0.25) -> bool:
        """Expert drive with optional yaw / cross-track hold.

        Used on the slope where row2p drifts east; small yaw corrections keep
        the robot on the long axis without changing the policy's forward speed.
        """
        self.switch(policy)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            desired = target_yaw
            if x_hold is not None:
                desired = wrap(desired - 0.8 * (p.x - x_hold))
            err = wrap(desired - p.yaw)
            wz = math.copysign(min(max_wz, kp * abs(err)), err) if abs(err) > 0.02 else 0.0
            self.command(vx, 0.0, wz)
            self._tick()
            if until(p):
                self.stop(0.6)
                return True
            if self.fallen():
                break
        self.stop(0.6)
        return False

    def drive_fixed(self, x_cmd: float, target_yaw: float, until, timeout_s: float) -> bool:
        self._cmd = f"fixed_drive(x={x_cmd:+.2f})"
        self._emit_plan()
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            p = self.pose()
            err = wrap(target_yaw - p.yaw)
            yaw_cmd = math.copysign(max(0.10, min(0.40, 1.5 * abs(err))), err) if abs(err) > 0.03 else 0.0
            self.fixed_command(x_cmd, yaw_cmd)
            self._tick()
            if until(p):
                self.fixed_command(0.0, 0.0)
                return True
            if self.fallen():
                break
        self.fixed_command(0.0, 0.0)
        return False

    def _run(self) -> None:
        try:
            self._route()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"part1_auto_route aborted: {exc}")
            self.stop(0.5)

    def _route(self) -> None:
        log = self.get_logger()
        if not self.wait_pose(30.0):
            log.error("no /eval_state; launch with route:=part1 (route raises eval_state_every:=0 to 10)")
            return
        log.info("sim detected; standing up")
        self._press(buttons=[BTN_A], hold_s=0.15)
        if not self.wait_stable_z(0.46, tilt_max=8.0, hold_s=0.5, timeout_s=15.0):
            log.warning("stand-up not stable (z/tilt); continuing")
        self._hold(0.3)
        self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)
        self._hold(1.5)
        if not self.wait_stable_z(0.40, tilt_max=12.0, hold_s=0.3, timeout_s=4.0):
            log.warning("RL entry not stable; pressing EnterRL again")
            self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)
            self._hold(1.2)
            self.wait_stable_z(0.40, tilt_max=15.0, hold_s=0.3, timeout_s=4.0)
        self.switch(TRANSFER_POLICY)
        self.stop(0.3)
        log.info("route start at (%.2f, %.2f, yaw %.2f)" % (self.pose().x, self.pose().y, self.pose().yaw))
        only = os.environ.get("PART1_ROUTE_ONLY", "").strip()
        if only == "transfer_bench":
            # Flat-area position-tracking benchmark for route-transfer models.
            # Each waypoint is tracked with the same closed-loop controller; we
            # record the final position error rather than aborting on a miss.
            self.ensure_transfer()
            pts = [(2.5, -9.0), (2.5, -8.0), (4.0, -8.0), (4.0, -9.0), (3.0, -9.5)]
            errs = []
            t0 = time.monotonic()
            for pt in pts:
                ok = self.goto(pt, tol=0.10, timeout_s=15.0, speed=0.22)
                p = self.pose()
                errs.append(round(math.hypot(pt[0] - p.x, pt[1] - p.y), 3))
                log.info(f"transfer_bench waypoint {pt} ok={ok} err={errs[-1]:.3f}")
                if self.fallen():
                    log.warning("transfer_bench: fallen")
                    break
            log.info(f"transfer_bench policy={TRANSFER_POLICY} errors={errs} "
                     f"mean={sum(errs)/max(1,len(errs)):.3f} max={max(errs) if errs else 0:.3f} "
                     f"time={time.monotonic()-t0:.1f}s")
            return
        if only == "t_stairs_hold":
            policy = os.environ.get("PART1_T_POLICY", "st30_ppo_st30_s100_600")
            speed = float(os.environ.get("PART1_T_SPEED", "0.45"))
            xhold = os.environ.get("PART1_T_XHOLD", "")
            self.face(-math.pi / 2.0)
            ok = self.drive_expert_heading(policy, speed, -math.pi / 2.0,
                                           lambda p: p.y < -4.9, 15.0,
                                           x_hold=float(xhold) if xhold else None,
                                           kp=0.30, max_wz=0.10)
            p = self.pose()
            log.info(f"t_stairs-hold xhold={xhold}: {'PASS' if ok else 'FAIL'} "
                     f"at ({p.x:.2f},{p.y:.2f},{p.tilt:.1f}deg)")
            self.stop(0.5)
            return
        if only == "t_stairs":
            policy = os.environ.get("PART1_T_POLICY", "st30_ppo_st30_s100_600")
            speed = float(os.environ.get("PART1_T_SPEED", "0.45"))
            self.face(-math.pi / 2.0)
            ok = self.drive_expert(policy, speed, -math.pi / 2.0,
                                   lambda p: p.y < -4.9, 15.0)
            self.stop(0.5)
            log.info(f"t_stairs-only policy={policy} speed={speed}: "
                     f"{'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f},{self.pose().tilt:.1f}deg)")
            return
        if only == "bridge_A_fixed_demo":
            self.set_leg("bridge_A_fixed_demo")
            stop_x = float(os.environ.get("PART1_A_STOP_X", "5.65"))
            ok = self.drive_expert(
                "row2p_p40w40_v1", 0.45, 0.0,
                lambda p: p.x > stop_x, 15.0)
            log.info(f"A demo coarse: {'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f}) feet={self.feet()}")
            self.ensure_transfer()
            self.stop(0.6)
            self._pose_move_to(5.7, -0.5, 0.0, tol=0.03, timeout_s=6.0, speed=0.08)
            ok2 = self.rel_rotate(-90.0)
            p = self.pose()
            log.info(f"A demo centre rotate: {'PASS' if ok2 else 'FAIL'} at ({p.x:.2f},{p.y:.2f},{p.yaw:.2f},{p.tilt:.1f}deg)")
            self.enter_fixed("bridge")
            ok3 = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -3.9, 20.0)
            self.exit_fixed()
            p = self.pose()
            log.info(f"A demo bridge B: {'PASS' if ok3 else 'FAIL'} at ({p.x:.2f},{p.y:.2f},{p.yaw:.2f},{p.tilt:.1f}deg)")
            self.stop(0.5)
            return
        if only == "bridge_A_rel":
            self.set_leg("bridge_A_rel")
            ok = self.drive_expert(
                "row2p_p40w40_v1", 0.45, 0.0,
                lambda p: self.wheels_inside((5.25, 6.15), (-0.95, -0.05)), 15.0)
            log.info(f"bridge_A_rel coarse: {'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f}) feet={self.feet()}")
            self.ensure_transfer()
            self.stop(0.6)
            self._pose_move_to(5.7, -0.5, 0.0, tol=0.03, timeout_s=6.0, speed=0.08)
            ok2 = self.rel_rotate(-90.0)
            p = self.pose()
            log.info(f"bridge_A_rel centre rotate: {'PASS' if ok2 else 'FAIL'} at ({p.x:.2f},{p.y:.2f},{p.yaw:.2f},{p.tilt:.1f}deg)")
            self.stop(0.5)
            return
        if only == "center_cli":
            self.set_leg("center_cli")
            self._pose_move_to(5.7, -0.5, 0.0, tol=0.02, timeout_s=5.0, speed=0.06)
            ok1 = self.rel_forward(0.02)
            ok2 = self.rel_rotate(-90.0)
            ok3 = self.rel_forward(0.05)
            ok4 = self.rel_rotate(90.0)
            p = self.pose()
            log.info(f"center_cli: fwd2cm={ok1} rot-90={ok2} fwd5cm={ok3} rot+90={ok4} "
                     f"end=({p.x:.3f},{p.y:.3f},{p.yaw:.3f},{p.tilt:.1f}deg)")
            self.stop(0.5)
            return
        if only == "turn_hold":
            ok = self.rotate_hold(0.0, 8.0)
            p = self.pose()
            log.info(f"turn_hold: {'PASS' if ok else 'FAIL'} at ({p.x:.2f},{p.y:.2f},{p.tilt:.1f}deg) yaw={p.yaw:.2f}")
            self.stop(0.5)
            return
        if only == "turn_fixed":
            self.enter_fixed("bridge")
            ok = self.drive_fixed(0.0, 0.0, lambda p: abs(wrap(0.0 - p.yaw)) < 0.10, 6.0)
            self.exit_fixed()
            p = self.pose()
            log.info(f"turn_fixed: {'PASS' if ok else 'FAIL'} at ({p.x:.2f},{p.y:.2f},{p.tilt:.1f}deg) yaw={p.yaw:.2f}")
            self.stop(0.5)
            return
        if only == "turn_platform":
            pol = os.environ.get("PART1_TURN_POLICY", "row2p_p40w40_v1")
            ok = self.turn_expert(pol, 0.0, 6.0)
            p = self.pose()
            log.info(f"turn_platform policy={pol}: {'PASS' if ok else 'FAIL'} "
                     f"at ({p.x:.2f},{p.y:.2f},{p.tilt:.1f}deg) yaw={p.yaw:.2f}")
            self.stop(0.5)
            return
        if only == "slope_hold":
            self.require((4.6, -7.7), tol=0.18, timeout_s=20.0, speed=0.25)
            self.face(math.pi / 2.0)
            ok = self.drive_expert_heading("row2p_p40w40_v1", 0.5, math.pi / 2.0,
                                           lambda p: p.y > -4.0, 15.0,
                                           x_hold=4.6, kp=1.0, max_wz=0.20)
            p = self.pose()
            log.info(f"slope-only hold: {'PASS' if ok else 'FAIL'} at ({p.x:.2f},{p.y:.2f},{p.tilt:.1f}deg)")
            self.stop(0.5)
            return
        if only == "bridge_bands":
            # North-to-south crossing (the bands sit north of the slope/platform).
            self.face(-math.pi / 2.0)
            self.enter_fixed("bridge")
            ok = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -3.9, 25.0)
            self.exit_fixed()
            log.info(f"bridge_bands-only: {'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f})")
            return
        if only == "bridge_bands_east":
            # Spawned east of the platform at (6.9, -3.5), facing west.  Use
            # row2p to climb the 0.2 m platform (the flat pose policy cannot),
            # then switch to the bridge fixed action on top of it.
            ok = self.drive_expert("row2p_p40w40_v1", 0.4, math.pi, lambda p: p.x < 5.80, 10.0)
            log.info(f"bridge-east platform climb: {'ok' if ok else 'fail'} at ({self.pose().x:.2f},{self.pose().y:.2f})")
            self.switch(TRANSFER_POLICY)
            self.face(math.pi / 2.0)
            self.enter_fixed("bridge")
            ok = self.drive_fixed(0.32, math.pi / 2.0, lambda p: p.y > -0.5, 20.0)
            self.exit_fixed()
            log.info(f"bridge_bands-east: {'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f})")
            return
        if only == "spin_probe":
            # Fixed-action skid-steer spin footprint (user directive 2026-11-01:
            # verify the required ground area before running whole-course tests).
            # Sweeps both spin directions and 90/180 deg targets on open floor,
            # recording the swept wheel footprint like the legacy route
            # probe_turn_footprint does for the RL turns.
            names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

            def feet_xy():
                f = self.feet()
                return [f[n][:2] for n in names if n in f]

            for tgt_deg in (90, -90, 180):
                tgt = math.radians(tgt_deg)
                self.ensure_transfer()
                self.stop(1.0)
                p0 = self.pose()
                yaw0 = p0.yaw
                f0 = feet_xy()
                if len(f0) < 4:
                    log.error("spin_probe: no feet feedback")
                    return
                c0 = (sum(p[0] for p in f0) / 4.0, sum(p[1] for p in f0) / 4.0)
                c_, s_ = math.cos(-yaw0), math.sin(-yaw0)

                def to_start_frame(pts):
                    return [(c_ * (x - c0[0]) - s_ * (y - c0[1]),
                             s_ * (x - c0[0]) + c_ * (y - c0[1])) for x, y in pts]

                rest = to_start_frame(f0)
                self.enter_fixed("bridge")
                t0 = time.monotonic()
                samples = list(rest)
                ok = False
                while time.monotonic() - t0 < 25.0:
                    p = self.pose()
                    err = wrap(tgt - p.yaw)
                    samples.extend(to_start_frame(feet_xy()))
                    if abs(err) < 0.08:
                        ok = True
                        break
                    cmd = math.copysign(0.45, err)
                    self.fixed_command(0.0, cmd)
                    self._tick()
                self.stop(0.6)
                self.exit_fixed()
                p1 = self.pose()
                xs = [q[0] for q in samples]
                ys = [q[1] for q in samples]
                bbox = [min(xs), max(xs), min(ys), max(ys)]
                radius = max(math.hypot(q[0], q[1]) for q in samples)
                drift = math.hypot(p1.x - p0.x, p1.y - p0.y)
                log.info(
                    "spin_probe %+ddeg: %s dyaw=%+.1fdeg t=%.1fs drift=%.3f m "
                    "bbox=[%.2f,%.2f]x[%.2f,%.2f] radius=%.2f" %
                    (tgt_deg, "PASS" if ok else "FAIL",
                     math.degrees(wrap(p1.yaw - tgt)),
                     time.monotonic() - t0, drift,
                     bbox[0], bbox[1], bbox[2], bbox[3], radius))
            self.stop(0.5)
            return
        if only == "height_bar":
            self.face(math.pi / 2.0)
            self.enter_fixed("low_bar")
            ok = self.drive_fixed(0.32, math.pi / 2.0, lambda p: p.y > -8.0, 15.0)
            self.exit_fixed()
            log.info(f"height_bar-only: {'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f})")
            return

        # ---------------------------------------------------------------- course
        # Correct Part1 topology (XML geometry + PDF map):
        # high_wall (+y) -> around T to north arm -> T-stairs (-y, north up /
        # south down) -> short slope (+y) -> bridge A / block bridge (+x, west
        # platform to east platform) -> bridge B / narrow bands (-y, high platform
        # to low platform) -> height bar (-y) -> gravel pit (west in, north out)
        # -> slalom S (start B, posts 3-2-1-0, corner, end A) -> start.

        # ------------------------------------------------------------ macro route
        # start -> high wall S/N -> T south up / north down -> short ramp to
        # bridge A west -> in-place turn +x -> bridge A (+x) -> east -> in-place
        # turn -> bridge B high->low (-y) -> straight S through the long slope
        # -> height bar fixed action -> gravel L (N in, turn W) -> slalom
        # keyframes -> start.

        # 1) high wall S -> N: cross y=-7 in +y, all supports x in [1.30,2.30]
        self.set_leg("high_wall")
        # 60 s: the spawn faces +x and the first waypoint is ~155 deg behind,
        # so the opening U-turn spends its rotation-drift budget; run-to-run
        # variance there consumed a 35 s timeout (edge4) while a twin run
        # passed in 20 s (edge2).
        self.require((1.8, -8.15), tol=0.14, timeout_s=60.0, speed=0.30)
        self.face(math.pi / 2.0)
        ok = self.drive_expert("row2p_p40w40_v1", 0.5, math.pi / 2.0,
                               lambda p: p.y > -6.0, 8.0)
        log.info(f"high_wall: {'PASS' if ok else 'FAIL'}")
        self.ensure_transfer()

        # 2) T stairs: south arm up, north arm down (different arms), exit north
        self.set_leg("t_stairs")
        self.require((1.8, -5.55), tol=0.16, timeout_s=18.0, speed=0.25)
        self.face(math.pi / 2.0)
        # Adaptive lane crossing (user feedback: the left-front wheel dropped
        # off the T-top's SE corner — the x_hold blend was too weak; the repo
        # adaptive formula is what st30 was trained with).  Lane x=1.8 along
        # +y; exit past the north arm foot.
        self.switch("st30_ppo_st30_s100_600")
        prev_t = None
        ok = False
        t0t = time.monotonic()
        while time.monotonic() - t0t < 20.0:
            p = self.pose()
            if p.y > -1.9 and abs(p.x - 1.8) < 0.45:
                ok = True
                break
            if abs(p.x - 1.8) > 0.85:
                break
            if self.fallen():
                break
            self.command(*self.adaptive_lane_cmd(
                p, prev_t, math.pi / 2.0, 1.8, "y", 0.45))
            prev_t = p
            self._tick()
        self.stop(0.5)
        log.info(f"t_stairs (S up / N down): {'PASS' if ok else 'FAIL'}")
        self.ensure_transfer()

        # 3+4) Bridge A: mount the west platform straight from the floor at
        # yaw 0 (row2p climbs the 0.2 m face) and cross east.  The old
        # choreography (south ramp -> on-platform 90 deg turn) required a
        # ~0.65 m turn footprint with zero margin on the 1.0 m platform
        # (probe_turn_footprint + spin_probe evidence; edge8-11 failures).
        # The floor mount is the legacy-harness choreography (5/6 legs).
        self.set_leg("bridge_A_west")
        # Two-step approach (legacy harness parity): straight to (0.8, 0.0)
        # from the t_stairs exit cuts across the ramp edge and slides off west
        # (edge14 ended at x=-0.26); go west in the south corridor first, then
        # north up the clear column.
        self.require((0.8, -2.0), tol=0.18, timeout_s=25.0, speed=0.30)
        self.require((0.8, 0.0), tol=0.18, timeout_s=20.0, speed=0.20)
        self.face(0.0)
        ok = False
        for vx in (0.45, 0.50, 0.40):
            self.goto((1.8, 0.0), tol=0.22, timeout_s=10.0, speed=0.15)
            self.rotate_hold(0.0)
            # Y-held crossing: a bare vx/heading drive let y drift to +1.3 m
            # (off the blocks, edge13) while x kept growing — exit on x alone
            # is not enough.  wz = heading error + lateral position term;
            # fail fast when the line is lost.
            self.switch("row2p_p40w40_v1")
            t0c = time.monotonic()
            # Adaptive lane command (user suggestion 2026-11-01): the repo's
            # AdaptiveCommand formula (eval policy_runner.py:243, gains =
            # the training-side adaptive lane the policy was trained with).
            # Feeding the same feedback structure is in-distribution, unlike
            # the hand-rolled wz blends (edge17/19 drift-outs).
            prev = None
            last_x = None
            stall_t = None
            while time.monotonic() - t0c < 15.0:
                p = self.pose()
                # Exit at the platform centre, not the far edge, then brake:
                # the earlier taper (0.55x before x=4.6) killed the momentum
                # needed to roll the 0.15 m block4->platform gap (edge28:
                # stalled at x=5.0 for 65 s, then walked off north).
                if p.x > 5.7 and abs(p.y) < 0.35:
                    ok = True
                    break
                if abs(p.y) > 0.85:
                    break
                if self.fallen():
                    break
                # Stall kick at the gap: momentum roll-through.
                if last_x is not None and abs(p.x - last_x) < 0.01:
                    stall_t = stall_t or time.monotonic()
                else:
                    stall_t = None
                last_x = p.x
                vx_eff = vx
                if stall_t is not None and time.monotonic() - stall_t > 1.5                         and 4.85 < p.x < 5.35:
                    vx_eff = 0.6
                self.command(*self.adaptive_lane_cmd(
                    p, prev, 0.0, 0.0, "x", vx_eff))
                prev = p
                self._tick()
            if ok:
                # Active brake: hold -0.2 for ~0.4 s so the landing momentum
                # does not carry the robot toward the 6.2 edge.
                self.command(-0.2, 0.0, 0.0)
                self._hold(0.4)
                self.stop(0.5)
            self.stop(0.5)
            if ok:
                break
            log.warning(f"bridge A retry vx={vx} at "
                        f"({self.pose().x:.2f},{self.pose().y:.2f})")
            # Re-approach from the west via the south corridor: after a lost
            # line the robot is often north of the blocks where the flat pose
            # policy cannot cross back over the 0.2 m tops (edge17 retries
            # 2/3 started from a stuck state and failed together).
            self.ensure_transfer()
            self.command(-0.25, 0.0, 0.0)
            self._hold(1.0)
            self.stop(0.3)
            # Detour back via the east clear column: a straight return to the
            # west crosses the blocks (edge19 aborted wedged at block 4).
            p = self.pose()
            if p.y > 0.6:
                self.goto((6.9, p.y), tol=0.3, timeout_s=12.0, speed=0.25)
            self.goto((6.9, -2.0), tol=0.3, timeout_s=18.0, speed=0.25)
            self.goto((0.8, -2.0), tol=0.25, timeout_s=25.0, speed=0.25)
            self.goto((0.8, 0.0), tol=0.22, timeout_s=15.0, speed=0.18)
            self.face(0.0)
        log.info(f"bridge_blocks: {'PASS' if ok else 'FAIL'}")
        self.ensure_transfer()

        # 5) A east platform: stop immediately when the centre region is
        # reached; do not keep the old row2p forward command.  Switch to the
        # relative CLI/pose layer and turn -y at the platform centre.
        self.set_leg("bridge_A_center")
        self.ensure_transfer()
        self.stop(0.8)
        self._cli = "center correction"
        self._emit_plan()
        # Spin FIRST (burst+dwell, drift <=0.04 m in 5 consecutive runs), then
        # drive SOUTH along the x=5.7 lane with the adaptive formula — the old
        # diagonal pose_move_to relied on lateral correction, which is the
        # zero-authority axis for every model (edge29/30 wedged at the west
        # lip, 0.7 m off-centre).
        ok_spin = self.spin_fixed(-math.pi / 2.0, tol=0.10, timeout_s=20.0,
                                  max_drift=0.40)
        if not ok_spin:
            log.warning("east centre spin retry")
            ok_spin = self.spin_fixed(-math.pi / 2.0, tol=0.12, timeout_s=20.0,
                                      max_drift=0.50)
        log.info(f"bridge A east centre spin: "
                 f"{'PASS' if ok_spin else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f})")
        # Lane-south to the bridge B entry (lateral = x about 5.7).
        prev_p = None
        t0s = time.monotonic()
        while time.monotonic() - t0s < 15.0:
            p = self.pose()
            if p.y < -0.55 and abs(p.x - 5.7) < 0.25:
                break
            if self.fallen():
                break
            self.command(*self.adaptive_lane_cmd(
                p, prev_p, -math.pi / 2.0, 5.7, "y", 0.15))
            prev_p = p
            self._tick()
        self.stop(0.4)

        self.set_leg("bridge_B")
        p = self.pose()
        if not (5.25 <= p.x <= 6.15 and -1.4 <= p.y <= 0.4):
            log.warning(f"bridge B entry off-platform ({p.x:.2f},{p.y:.2f}); re-placing")
            self._pose_move_to(5.7, -0.5, 0.0, tol=0.08, timeout_s=12.0, speed=0.12)
        self.enter_fixed("bridge")
        ok = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -3.9, 20.0)
        self.exit_fixed()
        log.info(f"bridge_bands: {'PASS' if ok else 'FAIL'}")
        self.ensure_transfer()

        # 6) straight south through the long slope, then height bar fixed action
        self.set_leg("long_slope")
        self.face(-math.pi / 2.0)
        # Exit only on flat ground south of the ramp: a y-only condition can
        # leave the robot on the slope lip where the flat pose policy and the
        # edge-recovery logic both operate on an incline (legacy final3:
        # slid off sideways, tilt 173 deg).
        ok = self.drive_expert("row2p_p40w40_v1", 0.50, -math.pi / 2.0,
                               lambda p: p.y < -7.2 and p.z < 0.54, 12.0)
        log.info(f"long slope straight S: {'PASS' if ok else 'FAIL'} at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer()
        self.set_leg("height_bar")
        self.require((5.7, -8.2), tol=0.16, timeout_s=18.0, speed=0.25)
        self.face(-math.pi / 2.0)
        self.enter_fixed("low_bar")
        ok = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -10.2, 15.0)
        self.exit_fixed()
        log.info(f"height_bar: {'PASS' if ok else 'FAIL'}")
        self.ensure_transfer()

        # 7) gravel L: enter from the north, turn in the middle, exit west
        self.set_leg("gravel")
        self.face(-math.pi / 2.0)
        ok = self.drive_expert("obstacle", 0.5, -math.pi / 2.0,
                               lambda p: p.y < -12.3, 10.0)
        # rotate_hold's 10 s budget is not enough for the obstacle policy's
        # weak in-place yaw (edge18/20 stalled at yaw -2.0 with 1.14 rad left);
        # close the heading first, untranslated, with a generous budget.
        t0r = time.monotonic()
        while time.monotonic() - t0r < 20.0:
            p = self.pose()
            yaw_err = wrap(math.pi - p.yaw)
            if abs(yaw_err) < 0.15 or self.fallen():
                break
            self.command(0.0, 0.0, math.copysign(0.45, yaw_err))
            self._tick()
        self.stop(0.4)
        p = self.pose()
        log.info(f"gravel west turn: yaw_err {math.degrees(wrap(math.pi - p.yaw)):+.1f}deg")
        # Y-held west leg through the pits (edge18: bare heading drive stalled
        # at x=7.59 — the pits drag the line south/north); same pattern as the
        # bridge A crossing.
        ok2 = False
        t0g = time.monotonic()
        # World-frame pure pursuit (same pattern as the bridge crossing): a
        # hand-derived lateral sign was INVERTED for the west heading
        # (edge21: positive wz from yaw=pi steers south, y ran to -20).
        while time.monotonic() - t0g < 18.0:
            p = self.pose()
            if p.x < 4.2 and abs(p.y + 12.5) < 0.8:
                ok2 = True
                break
            if self.fallen():
                break
            bearing = math.atan2(-12.5 - p.y, (p.x - 0.8) - p.x)
            wz = max(-0.40, min(0.40, 1.2 * wrap(bearing - p.yaw)))
            self.command(0.5, 0.0, wz)
            self._tick()
        self.stop(0.5)
        log.info(f"gravel_pit: {'PASS' if (ok and ok2) else 'FAIL'} "
                 f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer()

        # 8) slalom keyframes: B start -> north of post3 -> south of post2 ->
        # corner / west of post2 -> east of post1 -> west of post0 -> north of
        # post0 / A.  Keyframe headings are chosen to keep the wheel rectangle
        # clear of each pole.
        self.set_leg("slalom")
        self.require((3.2, -12.5), tol=0.18, timeout_s=15.0, speed=0.20)
        slalom = [
            ((2.8, -12.15), 150.0),
            ((1.8, -12.85), 225.0),
            ((1.55, -12.75), 180.0),
            ((1.45, -12.5), 90.0),
            ((2.15, -11.5), 120.0),
            ((1.45, -10.5), 55.0),
            ((1.8, -10.1), 90.0),
        ]
        reached = 0
        for pt, deg in slalom:
            if self.goto(pt, tol=0.12, timeout_s=9.0, speed=0.14):
                reached += 1
        log.info(f"slalom keyframes: {reached}/{len(slalom)}")

        # 9) return to start
        ok = self.goto((3.7, -9.0), tol=0.20, timeout_s=20.0, speed=0.25)
        self.stop(0.5)
        log.info(f"return_start: {'PASS' if ok else 'FAIL'}")
        log.info("part1_auto_route finished (wheel recoveries: %d)" % self._recoveries)


def main() -> int:
    rclpy.init()
    node = RouteNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.command(0.0, 0.0, 0.0)
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
