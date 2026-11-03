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
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String

# Route prechecks should use the same Part 1 geometry as the judge.  Import the
# authoritative constants from eval/ (pure Python, no ROS); the support-crossing
# high-wall check below is the judge's support-point rule, not a local heuristic.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_ROOT = _REPO_ROOT / "eval"
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))
try:
    from robocon_eval.part1 import judge as _p1_judge
    from robocon_eval.trace import classify_contacts as _classify_contacts
    P1_ZONES = dict(_p1_judge.ZONES)
    WALL_CROSS_Y = float(_p1_judge.WALL_CROSS_Y)
    WALL_X_MIN = float(_p1_judge.WALL_X_MIN)
    WALL_X_MAX = float(_p1_judge.WALL_X_MAX)
    JUDGE_GEOMETRY = True
except Exception:  # noqa: BLE001 - demo env without the eval package
    P1_ZONES = {"start_area": (3.2, 4.2, -9.5, -8.5)}
    WALL_CROSS_Y, WALL_X_MIN, WALL_X_MAX = -7.0, 1.30, 2.30
    JUDGE_GEOMETRY = False

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
        # Stop at the first FAIL (default on): a wedged/fallen robot should not
        # sit in a timeout for minutes.  PART1_ROUTE_STOP_ON_FAIL=0 resumes the
        # old "log FAIL and keep going" behaviour.
        self._stop_on_fail = os.environ.get("PART1_ROUTE_STOP_ON_FAIL", "1").strip() != "0"
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
        self._fixed_mode = None
        self._corr = {}
        self._sim_t = 0.0
        self._last_pose_wall = 0.0
        self._joy_pub = self.create_publisher(Joy, "joy", QoSProfile(depth=1))
        self.create_subscription(
            String, "eval_state", self._on_eval,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_timer(1.0 / PUBLISH_HZ, self._publish)
        # Live official judge: route PASS/FAIL for obstacle segments is the same
        # Part1JudgeV2 the eval replay uses (single source of truth).  Falls back
        # to local prechecks if the eval package is not importable.
        self._judge = None
        self._judge_key = {
            "high_wall": "high_wall",
            "t_stairs (S up / N down)": "t_stairs",
            "bridge_blocks": "bridge_blocks",
            "bridge_bands": "bridge_bands",
            "long slope straight S": "slope",
            "height_bar": "height_bar",
            "gravel_pit": "gravel_pit",
            "slalom keyframes": "slalom",
        }
        if JUDGE_GEOMETRY and os.environ.get("PART1_ROUTE_JUDGE", "1") != "0":
            try:
                self._judge = _p1_judge.Part1JudgeV2()
                self.get_logger().info("route judge: live Part1JudgeV2 enabled")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f"route judge unavailable: {exc}")
        self.get_logger().info("part1_auto_route: virtual gamepad on /joy (waiting for /eval_state)")
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name="route", daemon=True)
        self._thread.start()

    def _tick(self, fallback_s: float = 0.02) -> None:
        """Advance one control step: wait for the next state frame (step-lock)
        or sleep the wall-clock fallback."""
        if self._steplock:
            # Publish once per control frame, immediately before waiting for
            # the next /eval_state frame.  The 50 Hz timer is disabled in
            # step-lock so command/button edges are tied to policy frames rather
            # than ROS timer scheduling.
            self._send_joy()
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

    def _now(self) -> float:
        """Route clock: simulation time in step-lock, wall-clock otherwise.

        All control-loop deadlines use this clock so timeout decisions are tied
        to /eval_state frames (deterministic) rather than Python scheduling.
        """
        if self._steplock and self._sim_t > 0.0:
            return self._sim_t
        return time.monotonic()

    def _hold(self, seconds: float) -> None:
        """Hold the current output for `seconds`: frame-counted in step-lock
        (deterministic), wall-clock sleep otherwise (never recurse: the old
        implementation called itself and crashed any wall-clock route run)."""
        if not self._steplock:
            time.sleep(max(0.0, float(seconds)))
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
                if self._judge is not None:
                    try:
                        support = {
                            leg: self._feet[f"{leg}_foot"]
                            for leg in ("FL", "FR", "RL", "RR")
                            if f"{leg}_foot" in self._feet
                        }
                        tags = {leg: set() for leg in ("FL", "FR", "RL", "RR")}
                        ground: set[str] = set()
                        contacts = raw.get("contacts") or []
                        if contacts:
                            ctags, cground = _classify_contacts(contacts)
                            for leg in tags:
                                tags[leg].update(ctags.get(leg, set()))
                            ground.update(cground)
                        self._judge.update(
                            float(raw.get("t", self._sim_t)),
                            (x, y, z), yaw,
                            foot_tags=tags, ground_feet=ground,
                            bar_z=raw.get("bar_z"), support_xy=support or None,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._judge = None
                        self.get_logger().warning(f"route judge disabled: {exc}")
        except Exception:  # noqa: BLE001
            pass
        if self._record_fh is not None:
            self._record_fh.write(msg.data + "\n")
        with self._lock:
            self._pose = _Pose(x, y, z, yaw, tilt, self._now())
            self._have_pose = True
            self._last_pose_wall = self._now()
        if self._steplock:
            self._frame_evt.set()

    def _send_joy(self) -> None:
        m = Joy()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "part1_auto_route"
        with self._lock:
            m.axes = list(self._axes)
            m.buttons = list(self._buttons)
        self._joy_pub.publish(m)

    def _publish(self) -> None:
        # Timer path: inert in step-lock; _tick drives publishing instead.
        if self._steplock:
            return
        self._send_joy()

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
        t_end = self._now() + timeout_s
        try:
            while self._now() < t_end:
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
            "vt_track_v1": ([BTN_LB], (1, 0)),
            "vt_track_v2": ([BTN_LB], (0, 1)),
            "vt_locked12": ([BTN_LB], (0, -1)),
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
        # Do not overwrite a motion primitive's CLI here: ensure_transfer()
        # switches internally during goto/rotate_hold, and the overlay should
        # keep showing the operator command (e.g. "turn(+36deg)"), not the
        # transient model load.  Explicit drive_* methods overwrite the CLI
        # with "switch <policy>; forward ..." immediately afterwards.
        if not self._cmd or self._cmd == "start":
            self._cli = f"switch {name}"
            self._emit_plan()
        self.get_logger().info(f"switch -> {name}")
        self._hold(0.45)

    def ensure_transfer(self, announce: bool = False) -> None:
        # The locked-wheel connector is BC-initialised and much more sensitive
        # to the expert handover state than the rolling vt_track models: loading
        # it while the expert is still walking (first locked-route run, after
        # t_stairs) put the robot on its side within 1 s.  Stop first, let the
        # command limiter settle, then load the connector.  Its zero-command
        # creep is only ~0.02 m/s, so the old "zero command rolls back" failure
        # does not apply.
        if (TRANSFER_POLICY.startswith("vt_locked")
                and self._current_policy != TRANSFER_POLICY):
            self.command(0.0, 0.0, 0.0)
            self._hold(0.6)
            self.stop(0.3)
            self.switch(TRANSFER_POLICY)
            # Show the connector switch during the settle, not only for one
            # frame before the next goto overwrites it.
            if announce:
                self._cmd = "ensure_transfer"
                self._cli = f"switch {TRANSFER_POLICY}"
                self._corr = {}
                self._emit_plan()
            # Let the connector settle in its own stance before route commands
            # resume; an immediate goto after the expert handover rolled the
            # locked policy over in the first route run.
            self.command(0.0, 0.0, 0.0)
            self._hold(1.2)
            return
        self.switch(TRANSFER_POLICY)
        if announce:
            self._cmd = "ensure_transfer"
            self._cli = f"switch {TRANSFER_POLICY}"
            self._corr = {}
            self._emit_plan()

    def enter_fixed(self, kind: str) -> None:
        self._fixed_mode = kind
        self._cmd = f"enter_fixed({kind})"
        self._cli = f"switch mode {kind}"
        self._emit_plan()
        combos = {"low_bar": ([BTN_BACK], (0, -1)), "bridge": ([BTN_BACK], (1, 0))}
        buttons, hat = combos[kind]
        self._press(buttons=buttons, hat=hat, hold_s=0.15)
        self.get_logger().info(f"fixed action -> {kind}")
        self._hold(0.9)

    def exit_fixed(self) -> None:
        self._fixed_mode = None
        self._cli = "switch mode walk"
        self._emit_plan()
        self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)
        self._hold(0.5)

    def stop(self, settle_s: float = 0.4) -> None:
        self.command(0.0, 0.0, 0.0)
        self._hold(settle_s)

    def wait_pose(self, timeout_s: float = 20.0) -> bool:
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            if self._have_pose and self._now() - self._last_pose_wall < 1.0:
                return True
            self._hold(0.05)
        return False

    def wait_z(self, z_min: float, timeout_s: float) -> bool:
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            if self.pose().z > z_min:
                return True
            self._hold(0.05)
        return False

    def wait_stable_z(self, z_min: float, tilt_max: float = 10.0,
                      hold_s: float = 0.4, timeout_s: float = 10.0) -> bool:
        """Wait for a settled upright standing pose (z high, tilt small)."""
        t0 = self._now()
        above_since = None
        while self._now() - t0 < timeout_s:
            p = self.pose()
            now = self._now()
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
        return p.z < 0.15 or p.tilt > 60.0 or self._now() - p.t > 1.5

    def _transfer_yaw_gains(self) -> tuple[float, float, float]:
        """Return (min_wz, max_wz, gain) for the active transfer model.

        vt_track_v1 was trained for velocity accuracy; its low-wz response has a
        dead zone (see docs/VT-RECIPE.md §7).  The route therefore uses a higher
        floor and a bang-bang-ish gain for this model.  vt_pose_v1 keeps the
        historical gains that the route was tuned with.
        """
        if TRANSFER_POLICY.startswith(("vt_track", "vt_locked")):
            # vt_track and the locked-wheel connector both have a low-wz dead
            # zone; use the higher floor/gain that the route needs.
            return (0.25, 0.65, 2.5)
        return (0.12, 0.45, 1.2)

    def rotate_by(self, delta: float, tol: float = 0.10, timeout_s: float = 10.0) -> bool:
        self.ensure_transfer()
        target = wrap(self.pose().yaw + delta)
        self._set_turn_cli(target)
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            err = wrap(target - self.pose().yaw)
            if abs(err) < tol:
                break
            wz_min, wz_max, wz_gain = self._transfer_yaw_gains()
            wz = math.copysign(max(wz_min, min(wz_max, wz_gain * abs(err))), err)
            self.command(0.0, 0.0, wz)
            self._tick()
        self.stop(0.3)
        return abs(wrap(target - self.pose().yaw)) < 0.16

    def rotate_hold(self, target_yaw: float, timeout_s: float = 8.0) -> bool:
        """In-place yaw turn with position feedback to stay on a small platform."""
        self._cmd = f"rotate_hold({target_yaw:.2f})"
        self._corr = {}
        self._set_turn_hold_cli(target_yaw)
        self._emit_plan()
        self.ensure_transfer()
        p0 = self.pose()
        x0, y0 = p0.x, p0.y
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            p = self.pose()
            err = wrap(target_yaw - p.yaw)
            ex, ey = x0 - p.x, y0 - p.y
            fwd = ex * math.cos(p.yaw) + ey * math.sin(p.yaw)
            vx = max(-0.16, min(0.16, 1.2 * fwd))
            wz_min, wz_max, wz_gain = self._transfer_yaw_gains()
            wz = math.copysign(max(wz_min * 0.8, min(wz_max * 0.8, wz_gain * abs(err))), err) if abs(err) > 0.04 else 0.0
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
        self._set_move_cli(tx, ty, target_yaw)
        self._emit_plan()
        self.ensure_transfer()
        t0 = self._now()
        while self._now() - t0 < timeout_s:
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
        self._corr = {}
        self._set_turn_cli(yaw)
        self._emit_plan()
        if not self.rotate_by(wrap(yaw - self.pose().yaw), timeout_s=18.0):
            p = self.pose()
            raise RuntimeError(
                f"face {yaw:.2f} failed at yaw {p.yaw:.2f} ({p.x:.2f},{p.y:.2f})")

    def goto(self, target, tol: float = 0.12, timeout_s: float = 30.0,
             speed: float = 0.18) -> bool:
        self.ensure_transfer()
        tx, ty = float(target[0]), float(target[1])
        self._cmd = f"goto(({tx:.2f},{ty:.2f}))"
        self._corr = {}
        p_cli = self.pose()
        # goto also turns toward the target while translating; include the
        # target heading in the relative CLI so the operator command matches
        # the full motion.
        target_yaw = math.atan2(ty - p_cli.y, tx - p_cli.x)
        self._set_move_cli(tx, ty, target_yaw)
        self._emit_plan()
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            p = self.pose()
            ex, ey = tx - p.x, ty - p.y
            dist = math.hypot(ex, ey)
            if dist < tol:
                break
            err = wrap(math.atan2(ey, ex) - p.yaw)
            if abs(err) > 0.7:
                wz_min, wz_max, wz_gain = self._transfer_yaw_gains()
                wz = math.copysign(max(wz_min, min(wz_max, wz_gain * abs(err))), err)
                self.command(0.0, 0.0, wz)
            else:
                v = min(speed, max(0.08, 0.7 * dist))
                if TRANSFER_POLICY.startswith(("vt_track", "vt_locked")):
                    # The low-wz dead zone also hurts small heading corrections
                    # while driving (goto veered off the T-stairs approach).
                    wz = max(-0.40, min(0.40, 1.6 * err))
                    if abs(err) > 0.05 and abs(wz) < 0.14:
                        wz = math.copysign(0.14, err)
                else:
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

    def report(self, name: str, ok: bool, detail: str = "") -> None:
        """Log one segment verdict; raise (and shut the route down) on the first FAIL."""
        judge_note = ""
        if self._judge is not None:
            key = self._judge_key.get(name)
            try:
                report = self._judge.report()
                if key is not None:
                    status = report["statuses"].get(key, "pending")
                    judge_note = f" judge={status}"
                    if status != "success":
                        ok = False
                        detail = f"{detail} judge_reason={report['states'][key].get('reason', '')}".strip()
                elif name == "return_start":
                    returned = bool(report.get("returned_to_start"))
                    judge_note = f" judge_returned_to_start={returned}"
                    ok = ok and returned
            except Exception:  # noqa: BLE001
                pass
        line = f"{name}: {'PASS' if ok else 'FAIL'}" + (f" {detail}" if detail else "") + judge_note
        log = self.get_logger()
        if ok:
            log.info(line)
            return
        log.error(line)
        if self._stop_on_fail:
            raise RuntimeError(line)

    def _set_move_cli(self, tx: float, ty: float, yaw=None) -> None:
        """Pseudo-REPL relative move, matching what an operator would type."""
        p = self.pose()
        ex, ey = float(tx) - p.x, float(ty) - p.y
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        fwd = ex * c + ey * s
        left = -ex * s + ey * c
        parts = [f"fwd={fwd:+.2f}m", f"left={left:+.2f}m"]
        if yaw is not None:
            dyaw = wrap(float(yaw) - p.yaw)
            if abs(dyaw) > 0.02:
                parts.append(f"turn={math.degrees(dyaw):+.0f}deg")
        self._cli = "move(" + ",".join(parts) + ")"

    def _set_turn_cli(self, target_yaw: float) -> None:
        dyaw = wrap(float(target_yaw) - self.pose().yaw)
        self._cli = f"turn({math.degrees(dyaw):+.0f}deg)"

    def _set_turn_hold_cli(self, target_yaw: float) -> None:
        """rotate_hold also commands body-frame vx to hold (x0,y0)."""
        dyaw = wrap(float(target_yaw) - self.pose().yaw)
        self._cli = f"turn_hold({math.degrees(dyaw):+.0f}deg)"

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
        self._cli = f"mode bridge; spin {math.degrees(wrap(target_yaw - p0.yaw)):+.0f}deg"
        self._emit_plan()
        ok = False
        t0 = self._now()
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
            while self._now() - t0 < timeout_s:
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
                t_b = self._now()
                while self._now() - t_b < burst:
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
        self._corr = {}
        self._cli = f"switch {policy}; forward {vx:+.2f}m/s"
        self._emit_plan()
        t0 = self._now()
        while self._now() - t0 < timeout_s:
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
        t0 = self._now()
        while self._now() - t0 < timeout_s:
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
        self._cmd = f"drive_expert_heading({policy}, vx={vx:+.2f}, x_hold={x_hold})"
        self._corr = {}
        self._cli = f"switch {policy}; forward {vx:+.2f}m/s + lane hold"
        self._emit_plan()
        t0 = self._now()
        while self._now() - t0 < timeout_s:
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
        self._corr = {}
        mode = self._fixed_mode or "fixed"
        self._cli = f"mode {mode}; forward {x_cmd:+.2f}m/s"
        self._emit_plan()
        t0 = self._now()
        while self._now() - t0 < timeout_s:
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

    # ---- smooth slalom path helpers -------------------------------------
    def _slalom_arc_waypoints(self):
        """Dense forward+turn path around the four slalom posts.

        The path is an arc/line approximation: it leaves B eastward, goes
        north around post3, crosses the post3-post2 gap perpendicularly,
        goes south around post2, detours through the corner checkpoint,
        crosses the post2-post1 gap horizontally, arcs east of post1,
        crosses post1-post0 horizontally and ends north of post0.  The
        mid-gap crossings are perpendicular to the two-post line, which is
        what avoids clipping the poles.
        """
        # Control polyline, checked for clearance from post centres below.
        ctrl = [
            (3.55, -12.50),  # east side of B circle, clear of post3
            (3.55, -11.90),
            (3.50, -11.95),
            (2.80, -11.75),  # north of post3, ~0.75 m
            (2.30, -11.95),
            (2.30, -12.50),  # perpendicular crossing post3/post2
            (2.30, -12.95),
            (1.80, -13.20),  # south of post2, ~0.70 m
            (1.45, -12.85),  # corner checkpoint, 0.14 m from corner, 0.49 m from P2
            (1.20, -12.45),
            (1.15, -12.00),  # west of post2
            (1.80, -12.00),  # perpendicular crossing post2/post1
            (2.45, -12.00),
            (2.50, -11.50),  # east of post1, ~0.70 m
            (2.45, -11.00),
            (1.80, -11.00),  # perpendicular crossing post1/post0
            (1.10, -11.00),
            (1.10, -10.50),  # west of post0, ~0.70 m
            (1.10, -9.80),
            (1.80, -9.80),   # north of post0, end A
            (1.80, -9.90),
        ]
        dense = []
        step = 0.04
        for (x0, y0), (x1, y1) in zip(ctrl[:-1], ctrl[1:]):
            d = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(d / step))
            for i in range(n):
                t = i / n
                dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        dense.append(ctrl[-1])
        return dense

    def follow_path(self, pts, speed: float = 0.14, lookahead: float = 0.22,
                    tol: float = 0.12, timeout_s: float = 120.0) -> bool:
        """Pure-pursuit forward+turn follower for a dense point path."""
        self._cmd = "follow_path"
        self._corr = {}
        self._cli = "follow path"
        self._emit_plan()
        self.ensure_transfer()
        t0 = self._now()
        idx = 0
        n = len(pts)
        while self._now() - t0 < timeout_s:
            p = self.pose()
            if self.fallen():
                return False
            while idx < n - 1 and math.hypot(pts[idx][0] - p.x,
                                            pts[idx][1] - p.y) < lookahead:
                idx += 1
            # lookahead target: first path point at least `lookahead` ahead
            j = idx
            acc = 0.0
            last = idx
            while j < n - 1:
                seg = math.hypot(pts[j + 1][0] - pts[j][0],
                                 pts[j + 1][1] - pts[j][1])
                acc += seg
                last = j + 1
                if acc >= lookahead:
                    break
                j += 1
            tx, ty = pts[last]
            if last == n - 1 and math.hypot(tx - p.x, ty - p.y) < tol:
                self.stop(0.3)
                return True
            err = wrap(math.atan2(ty - p.y, tx - p.x) - p.yaw)
            if abs(err) > 1.2:
                wz = math.copysign(max(0.20, min(0.60, 1.8 * abs(err))), err)
                self.command(0.0, 0.0, wz)
            else:
                v = max(0.04, speed * math.cos(err)) if abs(err) < 1.2 else 0.0
                wz = max(-0.50, min(0.50, 1.8 * err))
                self.command(v, 0.0, wz)
            self._tick()
        self.stop(0.3)
        return False

    def _run(self) -> None:
        try:
            self._route()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"part1_auto_route aborted: {exc}")
            self.stop(0.5)
        finally:
            # Release the main spin loop and let launch shut the sim nodes down.
            # The watchdog also covers a hung rclpy teardown after an abort.
            self._finished.set()
            threading.Timer(2.0, lambda: os._exit(0)).start()

    def _route(self) -> None:
        log = self.get_logger()
        if not self.wait_pose(30.0):
            log.error("no /eval_state; launch with route:=part1 (route raises eval_state_every:=0 to 10)")
            return
        log.info("sim detected; standing up")
        # The sim loads policy_name:=vt_pose_v1 at process start.  If EnterRL is
        # pressed without preselecting, InitRL loads that previous model; select
        # the locked connector first so the very first RL entry is locked.
        locked_policy = os.environ.get("PART1_LOCKED_POLICY", "vt_locked12").strip() or "vt_locked12"
        self.switch(locked_policy)
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
            t0 = self._now()
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
                     f"time={self._now()-t0:.1f}s")
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
            ok3 = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -3.50, 20.0)
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
            ok = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -3.50, 25.0)
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
                t0 = self._now()
                samples = list(rest)
                ok = False
                while self._now() - t0 < 25.0:
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
                     self._now() - t0, drift,
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
        # Stop closer to the wall (0.85 m instead of 1.15 m): with the 10 s
        # open-loop expert run the old approach accumulated several degrees of
        # yaw before the support points reached y=-7.  Measured full-course
        # switch pose was x=1.75, y=-8.28, yaw=95.7 deg (5.7 deg error).
        self.require((1.8, -7.85), tol=0.10, timeout_s=60.0, speed=0.30)
        self.face(math.pi / 2.0)
        # face() accepts up to ~0.16 rad; tighten to 2-3 deg before handing to
        # the yaw-open-loop expert, which never corrects heading itself.
        if abs(wrap(math.pi / 2.0 - self.pose().yaw)) > 0.05:
            self.rotate_hold(math.pi / 2.0, timeout_s=10.0)
        p = self.pose()
        log.info("high_wall expert start: x=%.3f y=%.3f yaw=%.1fdeg err=%+.1fdeg" %
                 (p.x, p.y, math.degrees(p.yaw),
                  math.degrees(wrap(math.pi / 2.0 - p.yaw))))
        # Judge-consistent precheck (eval/.../judge.py HighWall support path): every
        # wheel support must cross y=-7 inside the wall x span; crossing outside the
        # span is a side bypass and fails.  Route previously stopped merely at y>-6.
        self.switch("row2p_p40w40_v1")
        self._cmd = "high_wall_cross(policy=row2p, vx=+0.50)"
        self._cli = "switch row2p_p40w40_v1; forward +0.50m/s"
        self._corr = {}
        self._emit_plan()
        support_prev: dict[str, tuple[float, float, float]] = {}
        crossed: set[str] = set()
        side_bypass = False
        ok = False
        t_hw = self._now()
        while self._now() - t_hw < 10.0:
            # Keep the high-wall crossing open-loop: the adaptive lane command
            # (tried 2026-11-03) steered the support points out of the wall span
            # in full10 (3/4 supports + side_bypass), worse than the open-loop
            # 0.5 m/s expert.  Adaptive lane is used where the expert was
            # trained with it (T stairs, bridge blocks, gravel/long-slope hold).
            self.command(0.5, 0.0, 0.0)
            self._tick()
            p = self.pose()
            for name, pos in self.feet().items():
                x, y = float(pos[0]), float(pos[1])
                prev = support_prev.get(name)
                if prev is not None:
                    px, py = prev[0], prev[1]
                    if (py - WALL_CROSS_Y) * (y - WALL_CROSS_Y) <= 0 and abs(y - py) < 1.0:
                        if abs(y - py) < 1e-9:
                            x_at = (px + x) / 2.0
                        else:
                            x_at = px + (x - px) * (WALL_CROSS_Y - py) / (y - py)
                        if WALL_X_MIN <= x_at <= WALL_X_MAX:
                            crossed.add(name)
                        else:
                            side_bypass = True
                support_prev[name] = (x, y, float(pos[2]))
            if p.y > -6.0:
                ok = not side_bypass and len(crossed) >= 4
                break
            if self.fallen():
                break
        self.stop(0.6)
        self.report("high_wall", ok,
                    f"supports_crossed={len(crossed)}/4 side_bypass={side_bypass}")
        self.ensure_transfer(announce=True)

        # 2) T stairs: south arm up, north arm down (different arms), exit north
        self.set_leg("t_stairs")
        self.require((1.8, -5.55), tol=0.16, timeout_s=18.0, speed=0.25)
        self.face(math.pi / 2.0)
        # Adaptive lane crossing (user feedback: the left-front wheel dropped
        # off the T-top's SE corner — the x_hold blend was too weak; the repo
        # adaptive formula is what st30 was trained with).  Lane x=1.8 along
        # +y; exit past the north arm foot.
        self.switch("st30_ppo_st30_s100_600")
        self._cmd = "t_stairs_adaptive(policy=st30, lane=+y)"
        self._cli = "switch st30_ppo_st30_s100_600; forward +0.45m/s + lane hold"
        self._corr = {}
        self._emit_plan()
        prev_t = None
        ok = False
        t0t = self._now()
        while self._now() - t0t < 20.0:
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
        self.report("t_stairs (S up / N down)", ok)
        self.ensure_transfer(announce=True)

        # 3+4) Bridge A: mount the west platform straight from the floor at
        # yaw 0 (row2p climbs the 0.2 m face) and cross east.  The old
        # choreography (south ramp -> on-platform 90 deg turn) required a
        # ~0.65 m turn footprint with zero margin on the 1.0 m platform
        # (probe_turn_footprint + spin_probe evidence; edge8-11 failures).
        # The floor mount is the legacy-harness choreography (5/6 legs).
        self.set_leg("bridge_A_west")
        # Direct NW ramp approach (T-stairs top / ramp_blocks_l /
        # bridge_blocks_platform_l are all aligned at x=1.8): after descending
        # the T stairs, face +y and climb the ramp straight onto the NW
        # platform, then turn clockwise (yaw +90 -> 0) to face +x.  The old
        # west detour via (0.8,-2.0)->(0.8,0) was for the weak-rotation pose
        # model; the 12-DoF/row2p model no longer needs it.
        # T-stairs can exit at y=-1.3..-1.9 and x up to 2.3.  Align only the
        # x lane at the current y: driving to a fixed y=-1.75 made full30
        # re-enter the T top and abort at (2.30,-1.28).  The ramp is directly
        # north at x=1.8, so keep the exit y and just centre the lane.
        p_stair = self.pose()
        p_stair_y = max(-1.60, min(-1.10, p_stair.y))
        self.goto((1.8, p_stair_y), tol=0.10, timeout_s=8.0, speed=0.12)
        self.face(math.pi / 2.0)
        ok_ramp = self.drive_expert_heading(
            "row2p_p40w40_v1", 0.40, math.pi / 2.0,
            lambda p: p.y > -0.10, 15.0,
            x_hold=1.8, kp=0.8, max_wz=0.20)
        self.stop(0.4)
        p = self.pose()
        log.info("NW ramp: %s at (%.2f,%.2f,z=%.2f,yaw=%.1fdeg)" %
                 ("PASS" if ok_ramp else "FAIL", p.x, p.y, p.z,
                  math.degrees(p.yaw)))
        # The ramp expert can drift to the platform edge (full23 x=2.06);
        # centre on platform_l at x=1.8 before the clockwise turn and expert
        # handover, aligning with the first bridge block.
        self.goto((1.8, 0.0), tol=0.10, timeout_s=8.0, speed=0.12)
        self.stop(0.3)
        # Clockwise turn on the platform: +y heading (pi/2) -> +x (0).
        self.rotate_hold(0.0, timeout_s=15.0)
        self.stop(0.3)
        ok = False
        for vx in (0.45, 0.50, 0.40):
            self.goto((1.8, 0.0), tol=0.22, timeout_s=10.0, speed=0.15)
            self.rotate_hold(0.0)
            # Y-held crossing: a bare vx/heading drive let y drift to +1.3 m
            # (off the blocks, edge13) while x kept growing — exit on x alone
            # is not enough.  wz = heading error + lateral position term;
            # fail fast when the line is lost.
            self.switch("row2p_p40w40_v1")
            self._cmd = f"bridge_blocks_adaptive(policy=row2p, vx={vx:+.2f}, lane=y=0)"
            self._cli = f"switch row2p_p40w40_v1; forward {vx:+.2f}m/s + lane hold"
            self._corr = {}
            self._emit_plan()
            t0c = self._now()
            # Adaptive lane command (user suggestion 2026-11-01): the repo's
            # AdaptiveCommand formula (eval policy_runner.py:243, gains =
            # the training-side adaptive lane the policy was trained with).
            # Feeding the same feedback structure is in-distribution, unlike
            # the hand-rolled wz blends (edge17/19 drift-outs).
            prev = None
            last_x = None
            stall_t = None
            while self._now() - t0c < 15.0:
                p = self.pose()
                # Exit at the platform centre, not the far edge, then brake:
                # the earlier taper (0.55x before x=4.6) killed the momentum
                # needed to roll the 0.15 m block4->platform gap (edge28:
                # stalled at x=5.0 for 65 s, then walked off north).
                # Hand over ~0.45 m before the target.  Switching at full
                # crossing speed overshot to x=7.1 (1.7 m handover drift, off the
                # platform); slow the old expert to a crawl before switching.
                if p.x > 5.25 and abs(p.y) < 0.35:
                    ok = True
                    break
                if abs(p.y) > 0.85:
                    break
                if self.fallen():
                    break
                # Stall kick at the gap: momentum roll-through.
                if last_x is not None and abs(p.x - last_x) < 0.01:
                    stall_t = stall_t or self._now()
                else:
                    stall_t = None
                last_x = p.x
                vx_eff = vx
                if stall_t is not None and self._now() - stall_t > 1.5                         and 4.85 < p.x < 5.35:
                    vx_eff = 0.6
                self.command(*self.adaptive_lane_cmd(
                    p, prev, 0.0, 0.0, "x", vx_eff))
                prev = p
                self._tick()
            if ok:
                # Handover to the transfer model while the forward command is
                # still active; it brakes and holds at the centre (no old-model
                # negative/zero command, no backward coast).
                p_pre = self.pose()
                # Slow the old expert first (still forward, never zero/negative:
                # row2p zero-command coasting is what rolled the robot back).
                self.command(0.12, 0.0, 0.0)
                self._hold(0.25)
                self.ensure_transfer()
                self.command(0.0, 0.0, 0.0)
                self._hold(0.4)
                if not self.goto((5.65, -0.45), tol=0.10, timeout_s=15.0, speed=0.20):
                    log.warning("bridge A east handover recentre failed")
                self.command(0.0, 0.0, 0.0)
                self._hold(0.5)
                p_post = self.pose()
                target_err = math.hypot(p_post.x - 5.65, p_post.y + 0.45)
                log.info("bridge A east handover: pre=(%.2f,%.2f) post=(%.2f,%.2f) "
                         "moved=%.3f m target_err=%.3f m" %
                         (p_pre.x, p_pre.y, p_post.x, p_post.y,
                          math.hypot(p_post.x - p_pre.x, p_post.y - p_pre.y),
                          target_err))
            else:
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
        self.report("bridge_blocks", ok)
        self.ensure_transfer(announce=True)

        # 5) A east platform: stop immediately when the centre region is
        # reached; do not keep the old row2p forward command.  Switch to the
        # relative CLI/pose layer and turn -y at the platform centre.
        self.set_leg("bridge_A_center")
        self.ensure_transfer()
        self.stop(0.8)
        # The bridge-blocks exit condition allows the robot to coast ~0.2 m back
        # from the east platform centre; a fixed skid spin started against the
        # block edge stalls (observed 30-40 deg in 40 s).  Re-centre with the
        # transfer primitive first; failure is non-fatal (spin/stop_on_fail will
        # report it).
        if not self.goto((5.65, -0.45), tol=0.10, timeout_s=15.0, speed=0.20):
            log.warning("bridge A east centre recentre failed; spin will report")
        self._cli = "lane hold (relative center)"
        self._emit_plan()
        # Turn with the 12-DoF locked policy, not the fixed-action differential
        # skid.  The fixed spin is momentum/friction sensitive: in the 12-DoF
        # full-course run it stalled twice (33.5 deg, 31.6 deg) on the 1 m
        # platform and then wedged in FixedBridgeDrive.  rotate_hold is the
        # position-loop turn that passed the position harness (drift 0.067 m).
        self.ensure_transfer()
        ok_spin = self.rotate_hold(-math.pi / 2.0, timeout_s=35.0)
        if not ok_spin:
            log.warning("east centre spin retry with locked policy")
            ok_spin = self.rotate_hold(-math.pi / 2.0, timeout_s=35.0)
        self.report("bridge A east centre spin", ok_spin,
                    f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        # Lane-south to the bridge B entry.  Exit only when both y and the
        # narrow-band lateral coordinate are tight; the old 0.20 m x window let
        # the ideal-model run stop at x=5.79, i.e. on the east edge of the three
        # 0.2 m bands.  Keep the robot's centre at x=5.65.
        # Dedicated lane command: adaptive_lane_cmd's axis='y' formula was
        # derived for the +y (T-stairs) lane.  With lane_yaw=-pi/2 its lateral
        # sign is inverted: the ideal-model run drove x from 5.72 to 8.71
        # before entering bridge B.  World x = +vy_body at yaw=-90, so the
        # correction is vy = -k*(x-5.65).
        # This is a real southward translation after the turn; give it its own
        # CLI instead of leaving turn_hold(...) on screen.
        self._cmd = "lane_south(x_hold=5.65, y_stop=-0.90)"
        self._corr = {}
        self._set_move_cli(5.65, -0.90, -math.pi / 2.0)
        self._emit_plan()
        t0s = self._now()
        while self._now() - t0s < 15.0:
            p = self.pose()
            if p.y < -0.90:
                break
            if self.fallen():
                break
            ex = p.x - 5.65
            vy = max(-0.15, min(0.15, -0.8 * ex))
            dyaw = wrap(-math.pi / 2.0 - p.yaw)
            wz = max(-0.25, min(0.25, 1.2 * dyaw))
            self.command(0.15, vy, wz)
            self._tick()
        self.stop(0.4)

        self.set_leg("bridge_B")
        # A pose_move_to with a fixed south heading cannot correct a ~0.14 m
        # lateral error (it only commands forward + yaw); the fake model exposed
        # exactly that: the robot faced south, x stayed at 5.79 and the goto
        # timed out.  Do a dedicated lateral lane hold instead.
        p_lat = self.pose()
        self._cmd = "lane_hold_x(5.65)"
        self._corr = {}
        self._set_move_cli(5.65, p_lat.y, -math.pi / 2.0)
        self._emit_plan()
        for _ in range(150):
            p = self.pose()
            if abs(p.x - 5.65) < 0.035:
                break
            ex = p.x - 5.65
            vy = max(-0.15, min(0.15, -0.8 * ex))
            dyaw = wrap(-math.pi / 2.0 - p.yaw)
            wz = max(-0.25, min(0.25, 1.2 * dyaw))
            self.command(0.0, vy, wz)
            self._tick()
        if abs(wrap(-math.pi / 2.0 - self.pose().yaw)) > 0.04:
            self.rotate_hold(-math.pi / 2.0, timeout_s=8.0)
        self.stop(0.3)
        p = self.pose()
        log.info("bridge B entry: x=%.2f y=%.2f yaw=%.1fdeg" %
                 (p.x, p.y, math.degrees(p.yaw)))
        if not (5.58 <= p.x <= 5.72 and -1.5 <= p.y <= 0.1):
            log.warning("bridge B alignment failed; fixed action will report")
        self.enter_fixed("bridge")
        ok = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -3.50, 20.0)
        self.exit_fixed()
        self.report("bridge_bands", ok)
        self.ensure_transfer(announce=True)

        # 6) straight south through the long slope, then height bar fixed action
        self.set_leg("long_slope")
        self.face(-math.pi / 2.0)
        # Exit only on flat ground south of the ramp: a y-only condition can
        # leave the robot on the slope lip where the flat pose policy and the
        # edge-recovery logic both operate on an incline (legacy final3:
        # slid off sideways, tilt 173 deg).
        ok = self.drive_expert_heading(
            "row2p_p40w40_v1", 0.50, -math.pi / 2.0,
            lambda p: p.y < -7.2 and p.z < 0.54, 12.0,
            x_hold=5.70, kp=0.8, max_wz=0.18)
        self.report("long slope straight S", ok, f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)
        self.set_leg("height_bar")
        # The old single require((5.70,-8.20)) picked a diagonal target
        # heading; the video showed >90 deg of yaw during the move, then
        # face(-pi/2) pulled back ~110 deg and left a lateral error that hit
        # the east post.  Face south first, then the goto to (5.70,-8.20) has
        # almost the same heading and only a small lateral correction.
        self.face(-math.pi / 2.0)
        self.require((5.70, -8.20), tol=0.08, timeout_s=12.0,
                     recover=False, speed=0.15)
        self.face(-math.pi / 2.0)
        self.enter_fixed("low_bar")
        ok = self.drive_fixed(0.32, -math.pi / 2.0, lambda p: p.y < -10.2, 15.0)
        self.exit_fixed()
        self.report("height_bar", ok)
        self.ensure_transfer(announce=True)

        # Position alignment at the gravel entrance face.  The east column is
        # commanded straight south; without a planar correction the expert
        # enters off-centre and the first climb drifts to an arm edge.
        self.require((5.70, -10.20), tol=0.10, timeout_s=10.0, speed=0.15)
        p = self.pose()
        log.info("gravel entrance align: x=%.2f y=%.2f yaw=%.1fdeg" %
                 (p.x, p.y, math.degrees(p.yaw)))

        # 7) gravel L: expert climbs into the EAST arm, switch to 12-DoF at
        # its middle; the 12-DoF connector then drives to the SE corner, turns
        # west through the bottom row, and tries the descent.  The old route
        # used the gravel expert for the whole L and put the left wheels
        # outside the pit at the south-east corner.
        self.set_leg("gravel")
        self.face(-math.pi / 2.0)
        # Best run so far (full12): 0.5 m/s expert to y<-12.15 with x_hold
        # 5.95; then the 12-DoF goto to the bottom-row centre.  Slower or
        # pre-biased entries over/under-shot (full14-17); revert to this preset.
        ok_entry = self.drive_expert_heading(
            "obstacle", 0.5, -math.pi / 2.0,
            lambda p: p.y < -12.15, 12.0,
            x_hold=5.95, kp=1.2, max_wz=0.35)
        p = self.pose()
        log.info("gravel east-arm entry: %s at (%.2f,%.2f,%.1fdeg)" %
                 ("PASS" if ok_entry else "FAIL", p.x, p.y, math.degrees(p.yaw)))
        self.ensure_transfer()

        # Go straight south from the expert's actual x: full22 entered at
        # x=5.54 and the lateral goto to 5.72 turned the 12-DoF policy west off
        # the arm.  Only the y waypoint needs to move.
        p_sw = self.pose()
        ok_fwd = self.goto((p_sw.x, -12.55), tol=0.10, timeout_s=10.0, speed=0.15)
        self.stop(0.4)
        p = self.pose()
        log.info("gravel east-arm corner: %s at (%.2f,%.2f,%.1fdeg)" %
                 ("PASS" if ok_fwd else "FAIL", p.x, p.y, math.degrees(p.yaw)))

        # Turn west on the 0.15 m platform with position feedback.
        ok_turn = self.rotate_hold(math.pi, timeout_s=20.0)
        if not ok_turn:
            ok_turn = self.rotate_hold(math.pi, timeout_s=20.0)
        self.stop(0.3)
        p = self.pose()
        log.info("gravel west turn: %s yaw=%s at (%.2f,%.2f)" %
                 ("PASS" if ok_turn else "FAIL", round(math.degrees(p.yaw), 1), p.x, p.y))

        # Bottom row is x in [4.336, 6.336]: keep the base at 4.70 so the
        # left wheel stays inside, then try the 12-DoF descent.
        ok_west = self.goto((4.70, -12.50), tol=0.12, timeout_s=15.0, speed=0.18)
        self.stop(0.4)
        # full11: the expert was handed yaw≈159 deg (21 deg off west); its
        # open-loop exit drifted north from y=-12.36 to y=-11.79 and fell off
        # the pit's north edge.  Align to west with the position-loop policy
        # before handing the descent to the expert.
        self.face(math.pi)
        self.stop(0.3)
        # Per the full-course plan after bridge A: use the obstacle expert for
        # the descent directly.  The 12-DoF descent was tried in full5 and
        # failed; keeping the fallback branch only added a loss-of-position
        # step, so the expert exit is now deterministic.
        self.switch("obstacle")
        # Continue past the pit edge: stopping at x=4.50 left the base
        # straddling the 0.15 m lip (full12: FL/FR at z=0.10, RL/RR at z=0.25)
        # and the 12-DoF recovery could not move.  Let the expert finish the
        # descent to x<3.85 (clear floor) before switching.
        ok_exit = self.drive_expert(
            "obstacle", 0.5, math.pi, lambda p: p.x < 3.85, 12.0)
        self.stop(0.4)
        p = self.pose()
        log.info("gravel exit: %s at (%.2f,%.2f,%.1fdeg)" %
                 ("PASS" if ok_exit else "FAIL", p.x, p.y, math.degrees(p.yaw)))
        ok = ok_entry and ok_fwd and ok_turn and ok_west and ok_exit
        self.report("gravel_pit", ok,
                    f"at ({self.pose().x:.2f},{self.pose().y:.2f})")
        self.ensure_transfer(announce=True)

        # 8) slalom keyframes: B start -> north of post3 -> south of post2 ->
        # corner / west of post2 -> east of post1 -> west of post0 -> north of
        # post0 / A.  Keyframe headings are chosen to keep the wheel rectangle
        # clear of each pole.
        self.set_leg("slalom")
        # Enter the B circle on its east side: the arc starts at x=3.55, so the
        # robot does not have to turn around next to post3 (x=2.8).
        self.require((3.55, -12.5), tol=0.18, timeout_s=15.0, speed=0.20)
        # Keyframe centres widened to >=0.60 m from the post centres (posts:
        # (2.8,-12.5), (1.8,-12.5), (1.8,-11.5), (1.8,-10.5), radius 0.35 m).
        # The old 0.35 m offsets put the wheel rectangle on the poles.
        # Smooth arc/line path; pure pursuit keeps the robot moving forward
        # while steering, and the mid-gap legs cross perpendicular to the line
        # joining the two posts (the old point-to-point chords cut the poles).
        slalom_path = self._slalom_arc_waypoints()
        ok_slalom = self.follow_path(slalom_path, speed=0.14,
                                     lookahead=0.22, tol=0.12,
                                     timeout_s=150.0)
        self.report("slalom keyframes", ok_slalom,
                    f"arc path {len(slalom_path)} pts")

        # 9) return to start
        sx0, sx1, sy0, sy1 = P1_ZONES["start_area"]
        ok = self.goto(((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0),
                       tol=0.20, timeout_s=20.0, speed=0.25)
        self.stop(0.5)
        self.report("return_start", ok)
        log.info("part1_auto_route finished (wheel recoveries: %d)" % self._recoveries)


def main() -> int:
    rclpy.init()
    node = RouteNode()
    try:
        while rclpy.ok() and not node._finished.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.command(0.0, 0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
