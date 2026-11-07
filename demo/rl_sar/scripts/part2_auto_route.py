#!/usr/bin/env python3
"""Execute the single Part 2 demo plan in part2_route_plan.py.

The plan crosses the lower tiers through platform 30 cm, stairs 30 cm per
step, slope 35 degrees, trench 30 cm, plum 1 and wall 40 cm. Platform 40 cm
is explicitly omitted with user approval pending a stable descent. Its geometry is
also consumed by the clearance audit; policy and speed tuning do not select
different routes. PART2_ROUTE_START / PART2_ROUTE_ONLY select a debug window
within this same plan.

The locked-wheel connector follows segments and uses bounded, shallow steering
for expert alignment. No perpendicular sidestep or point-chasing endgame is
used. Integration gates check metres in the lane frame, direction and progress;
official support-point verdicts remain the responsibility of the eval judge.
"""
from __future__ import annotations

import math
import os

from std_msgs.msg import String

# Set before importing the shared controller, including for direct script runs.
os.environ.setdefault("PART1_TRANSFER_POLICY", "vt_locked12")
os.environ.setdefault("PART1_ROUTE_JUDGE", "0")
from part1_auto_route import BTN_A, BTN_RB, TRANSFER_POLICY, wrap
from part2_motion import Part2Motion
from part2_options import approach_point
from part2_route_plan import AUDIT_SEGMENTS, LEGS as PLAN_LEGS, ROUTE_START, SKIPPED_TIERS


def _policy_env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _vx_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


EXPERT_SPECS = {
    "platform": {"policy": "row2p_p40w40_v1", "vx": 0.50, "mode": "adaptive"},
    "stairs": {"policy": "st30_ppo_st30_s100_600", "vx": 0.45, "mode": "adaptive"},
    "slope": {"policy": "row2p_p40w40_v1", "vx": 0.70, "mode": "adaptive"},
    "trench": {"policy": "row2p_p40w40_v1", "vx": 0.70, "mode": "adaptive"},
    "plum": {"policy": "plum_depth_final_v1", "vx": 0.40, "mode": "depth"},
    "wall": {"policy": "row2p_p40w40_v1", "vx": 0.40, "mode": "adaptive"},
}
for _category, _spec in EXPERT_SPECS.items():
    _spec["policy"] = _policy_env(f"PART2_EXPERT_{_category.upper()}", _spec["policy"])
    _spec["vx"] = _vx_env(f"PART2_VX_{_category.upper()}", _spec["vx"])
    if _spec["vx"] <= 0:
        raise ValueError(f"PART2_VX_{_category.upper()} must be positive")

ALIGN_DEADLINE_S = 45.0
TRANSIT_SPEED = 0.30
RECOVER_ENABLED = os.environ.get("PART2_RECOVER", "1").lower() not in ("0", "false", "off")
CROSS_STALL_WINDOW_S = 12.0
CROSS_STALL_PROGRESS_M = 0.03
CROSS_BACKOFF_M = 0.60
RECOVER_MAX_GROUND_Z = 0.75
CROSS_RETRY_ALIGN_S = 30.0
PLUM_GATE_SETTLE_S = _vx_env("PART2_PLUM_GATE_SETTLE_S", 1.5)
# Preserve the successful composition's short forward entry into its wheeled
# policy before braking; do not mix yaw/lateral corrections into this landing.
PLUM_HANDOVER_ENTRY_S = 0.16
# Match terrain/gen_part2_levels.py:_plum_judge and the official support judge.
# The monitored base window extends 15 cm beyond the physical six-cube span;
# the wheel-bottom threshold is 5 cm above the floor, using blackW's radius.
PLUM_SUPPORT_WINDOW_MARGIN_M = 0.15
PLUM_MIN_WHEEL_BOTTOM_M = 0.05
BLACKW_WHEEL_RADIUS_M = 0.103
PLUM_SUPPORT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def lane_position(leg: dict, p) -> tuple[float, float]:
    """Along/cross-track position relative to this obstacle's approach, in m."""
    dx, dy = p.x - leg["stand"][0], p.y - leg["stand"][1]
    c, s = math.cos(leg["lane_yaw"]), math.sin(leg["lane_yaw"])
    return dx * c + dy * s, -dx * s + dy * c


def crossing_fault(leg: dict, p) -> str:
    if not all(math.isfinite(v) for v in (p.x, p.y, p.yaw)):
        return "invalid_pose"
    along, lateral = lane_position(leg, p)
    if abs(lateral) > float(leg.get("cross_lane_tol", 0.45)):
        return "off_lane"
    if along < float(leg["along_min"]) - 0.30:
        return "wrong_direction"
    if abs(wrap(p.yaw - leg["lane_yaw"])) > math.radians(65):
        return "wrong_heading"
    return ""


def plum_support_fault(leg: dict, p, feet: dict) -> str:
    """Fail closed on support telemetry; apply the official drop rule on cubes.

    This is an integration guard, not a substitute for the recorded support
    judge. Floor-height wheels are expected before entry and after the cubes.
    """
    if not math.isfinite(p.x):
        return "plum_support_invalid_pose"
    bottoms = {}
    for name in PLUM_SUPPORT_NAMES:
        try:
            xyz = tuple(float(v) for v in feet[name])
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                return "plum_support_telemetry:" + name
        except (KeyError, TypeError, ValueError, OverflowError):
            return "plum_support_telemetry:" + name
        bottoms[name] = xyz[2] - BLACKW_WHEEL_RADIUS_M
    low, high = leg["cube_x_bounds"]
    if low - PLUM_SUPPORT_WINDOW_MARGIN_M <= p.x <= high + PLUM_SUPPORT_WINDOW_MARGIN_M:
        for name, bottom in bottoms.items():
            if bottom < PLUM_MIN_WHEEL_BOTTOM_M:
                return "plum_support_dropped:%s=%.3f" % (name, bottom)
    return ""


def at_exit(leg: dict, p, *, settled: bool = False) -> bool:
    along, lateral = lane_position(leg, p)
    dx = leg["cross_end"][0] - leg["stand"][0]
    dy = leg["cross_end"][1] - leg["stand"][1]
    end = dx * math.cos(leg["lane_yaw"]) + dy * math.sin(leg["lane_yaw"])
    return (end - (0.05 if settled else 0.0) <= along
            <= end + float(leg.get("exit_overshoot_tol", 0.45))
            and abs(lateral) <= float(leg.get("cross_lane_tol", 0.45)))


def _build_runtime_legs() -> list[dict]:
    legs = []
    for planned in PLAN_LEGS:
        leg = {
            "align_deadline_s": ALIGN_DEADLINE_S, "transit_speed": TRANSIT_SPEED,
            "align_xy_tol": 0.08, "align_yaw_tol_deg": 2.5,
            "along_min": -0.15, "along_max": 0.50, "settle_forward": 0.25,
            "start_dy": 0.0, "no_recover": False, "fixed": False,
            "handover_policy": "", "handover_x": 0.0,
            "handover_settle_s": 0.0, "handover_vx": 0.0,
            "handover_brake_s": 0.0, "handover_brake_vx": 0.0,
            **planned, **EXPERT_SPECS[planned["category"]],
        }
        if leg["category"] == "stairs" and leg["difficulty"] == 20:
            # The final 20 cm rear-wheel riser can pin the gait at 0.45.
            # 0.50 is the validated st30 evaluation command; keep the gentler
            # command on lower tiers and honor an explicit category override.
            leg["vx"] = _vx_env("PART2_VX_STAIRS", 0.50)
        if leg["category"] == "stairs" and leg["difficulty"] >= 25:
            leg.update(policy=_policy_env("PART2_EXPERT_STAIRS", "vt_band_v1"),
                       vx=_vx_env("PART2_VX_STAIRS", 0.60))
        if leg["category"] == "slope" and leg["difficulty"] >= 35:
            leg["vx"] = _vx_env("PART2_VX_SLOPE", 0.50)
        if leg["category"] == "trench" and leg["difficulty"] >= 30:
            leg.update(policy=_policy_env("PART2_EXPERT_TRENCH", "s1d_trench_reward_v75_m100"),
                       vx=_vx_env("PART2_VX_TRENCH", 0.80))
        if leg["category"] == "wall":
            # The archived 40 cm success uses this actor at adaptive 0.5.
            # Keep it through the entire row so the narrow gaps need no
            # stationary policy change or second run-up.
            leg.update(policy=_policy_env("PART2_EXPERT_WALL", "row2p_p40w40_v1"),
                       vx=_vx_env("PART2_VX_WALL", 0.50))
        if leg["category"] == "plum":
            leg.update(handover_policy=_policy_env("PART2_PLUM_HANDOVER", leg["handover_policy"]),
                       handover_x=_vx_env("PART2_PLUM_HANDOVER_X", leg["handover_x"]),
                       handover_settle_s=_vx_env("PART2_PLUM_SETTLE_S", leg["handover_settle_s"]),
                       handover_vx=_vx_env("PART2_PLUM_DRIVE_VX", leg["handover_vx"]),
                       handover_brake_s=_vx_env("PART2_PLUM_BRAKE_S", leg["handover_brake_s"]),
                       handover_brake_vx=_vx_env("PART2_PLUM_BRAKE_VX", leg["handover_brake_vx"]),
                       no_recover=True)
            if leg["handover_policy"].lower() in ("none", "off", "0"):
                raise ValueError("The demo plum crossing requires its exit handover policy")
            if not 6.70 <= leg["handover_x"] <= 7.25:
                raise ValueError("PART2_PLUM_HANDOVER_X must lie on the exit platform [6.70, 7.25]")
        leg["stop"] = lambda p, leg=leg: at_exit(leg, p)
        leg["post"] = lambda p, leg=leg: at_exit(leg, p, settled=True)
        legs.append(leg)
    for index, leg in enumerate(legs):
        following = legs[index + 1] if index + 1 < len(legs) else None
        leg["keep_expert_moving"] = bool(
            following and not leg["exit"] and not following["transit"]
            and following["category"] == leg["category"]
            and following["policy"] == leg["policy"]
            and not following.get("reset_expert"))
    return legs


LEGS = _build_runtime_legs()


def _resolve_start_leg(raw: str) -> int:
    for index, leg in enumerate(LEGS):
        if raw.strip().lower() == leg["leg"].lower():
            return index
    try:
        index = int(raw) - 1
    except ValueError:
        return -1
    return index if 0 <= index < len(LEGS) else -1


def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class Part2Route(Part2Motion):
    def __init__(self) -> None:
        obsolete = [key for key in (
            "PART2_PLATFORM_LEVEL", "PART2_STAIRS_LEVEL", "PART2_SLOPE_LEVEL",
            "PART2_TRENCH_LEVEL", "PART2_PLUM_MODE", "PART2_ROUTE_WALL",
            "PART2_ROUTE_WALL_VARIANT") if key in os.environ]
        if obsolete:
            raise ValueError("Part2 has one fixed route; unset obsolete switches: "
                             + ", ".join(obsolete))
        for key in ("PART2_ROUTE_START", "PART2_ROUTE_ONLY"):
            if os.environ.get(key, "").strip() and _resolve_start_leg(os.environ[key]) < 0:
                raise ValueError(f"{key} does not name a leg in part2_route_plan.py")
        raw_start, raw_only = os.environ.get("PART2_ROUTE_START", ""), os.environ.get("PART2_ROUTE_ONLY", "")
        if raw_start.strip() and raw_only.strip() and _resolve_start_leg(raw_only) < _resolve_start_leg(raw_start):
            raise ValueError("PART2_ROUTE_ONLY must follow PART2_ROUTE_START in the canonical plan")
        super().__init__()
        # /policy_switch is rl_sim's gamepad-less policy loader (it takes a
        # policy dir name).  Create the publisher upfront: a lazily created
        # endpoint would race discovery on its first sample.  This is never
        # published to while rclpy.spin() has not started (RouteNode starts a
        # thread, but that thread's first step waits for /eval_state).
        self._switch_pub = self.create_publisher(String, "/policy_switch", 1)

    # ---- policy switching -------------------------------------------------
    def switch(self, name: str) -> None:
        # Part1's binding table lacks the Part2 trench/band specialists.
        # RB+D-pad up is EnterRL, so the band expert uses RB+D-pad left / B
        # (wired in input_map.hpp + rl_sim.cpp; see docs/joystick.md §2.2).
        extra = {
            "s1d_trench_reward_v75_m100": (BTN_RB, (1, 0)),
            "vt_band_v1": (BTN_RB, (-1, 0)),
        }
        if name in extra:
            if name == self._current_policy:
                return
            button, hat = extra[name]
            self._press(buttons=[button], hat=hat, hold_s=0.15)
            self._current_policy = name
            self._cli = f"switch {name}"
            self._emit_plan()
            self._hold(0.45)
            return
        try:
            super().switch(name)
        except ValueError:
            # rl_sim can load any policy directory, but the gamepad map only
            # binds a fixed roster.  The depth plum expert has no combo: rl_sim
            # takes its dir name on /policy_switch (std_msgs/String).  A fixed
            # wait is enough there, and the fake kinematic sim ignores the
            # topic (it keeps following the route's lane command).
            self._switch_topic(name)

    def switch_topic(self, name: str, hold_s: float = 0.15) -> None:
        """Load a policy through /policy_switch (no gamepad combo, no press).

        This avoids the gamepad combo's ~0.7 s button sequence. Loading still
        costs a few ms of wall time while physics advances, so fast simulation
        can travel farther before the first output from the new policy.
        """
        if name == self._current_policy:
            return
        self._switch_pub.publish(String(data=name))
        self._current_policy = name
        self._cli = f"switch {name}"
        self._emit_plan()
        self.get_logger().info(f"switch -> {name} (/policy_switch)")
        self._hold(hold_s)

    def _switch_topic(self, name: str) -> None:
        self.switch_topic(name, hold_s=1.0)

    # ---- adaptive lane command (exact training-side gains) ----------------
    @staticmethod
    def lane_cmd(p, prev, lane_x: float, lane_y: float, lane_yaw: float,
                 vx: float):
        """Training-side AdaptiveCommand, generalised to any axis-aligned lane.

        Gains are the ones the experts were trained with: vy=clip(-0.5*e,±0.2),
        wz=clip(-0.8*e - 2.0*heading_err - 0.6*vy_body, ±0.7).  ``e`` is the
        lateral offset in the lane frame (positive = left of the lane
        direction) relative to the lane line through (lane_x, lane_y) and
        ``vy_body`` the measured body-frame lateral velocity; this form mirrors correctly
        for -x and +y lanes without duplicating the formula.
        Returns body-frame (vx, vy, wz).
        """
        spsi, cpsi = math.sin(lane_yaw), math.cos(lane_yaw)
        e = -(p.x - lane_x) * spsi + (p.y - lane_y) * cpsi
        vx_w = vy_w = 0.0
        if prev is not None and p.t > prev.t:
            dt = p.t - prev.t
            vx_w = (p.x - prev.x) / dt
            vy_w = (p.y - prev.y) / dt
        vy_lat_meas = -spsi * vx_w + cpsi * vy_w
        h = wrap(p.yaw - lane_yaw)
        vx_lane_meas = cpsi * vx_w + spsi * vy_w
        vy_body_meas = -math.sin(h) * vx_lane_meas + math.cos(h) * vy_lat_meas
        vy_lat = max(-0.2, min(0.2, -0.5 * e))
        wz = max(-0.7, min(0.7, -0.8 * e - 2.0 * h - 0.6 * vy_body_meas))
        return (vx, vy_lat, wz)

    # ---- pre-expert alignment --------------------------------------------
    def _rotate_servo(self, target_yaw: float, timeout_s: float = 10.0) -> bool:
        """Rotate on the flat approach using the loaded policy's yaw range.

        The locked connector responds to 0.14 rad/s, while the stair expert
        barely turns at that command. Its 0.45 floor stays inside the expert's
        trained +/-0.7 range. A bounded position hold limits rollback during
        this short correction; alignment still checks the measured pose.
        """
        timeout_s = self._timeout(timeout_s)
        p0 = self.pose()
        t0 = self._now()
        expert = self._current_policy != TRANSFER_POLICY
        if timeout_s <= 0.20:
            self.command(0.0, 0.0, 0.0)
            return abs(wrap(target_yaw - p0.yaw)) <= 0.035
        while self._now() - t0 < timeout_s - 0.20:
            p = self.pose()
            err = wrap(target_yaw - p.yaw)
            if abs(err) <= 0.02:
                # An expert can carry yaw momentum through the target. Check
                # the stopped pose before accepting alignment; a transient
                # zero crossing previously rebounded beyond the hard gate.
                self.stop(0.20)
                if abs(wrap(target_yaw - self.pose().yaw)) <= 0.035:
                    return not self.fallen()
                continue
            cap = 0.65 if expert else (0.28 if abs(err) > 0.35 else 0.20)
            floor = 0.45 if expert else 0.14
            wz = math.copysign(min(cap, max(floor, 0.8 * abs(err))), err)
            dx, dy = p0.x - p.x, p0.y - p.y
            drift = math.hypot(dx, dy)
            vx = 0.0
            if drift > 0.02:
                fwd = dx * math.cos(p.yaw) + dy * math.sin(p.yaw)
                vx = max(-0.05, min(0.05, 1.2 * fwd))
            self.command(vx, 0.0, wz)
            self._tick()
            if self.fallen():
                break
        self.stop(0.20)
        p = self.pose()
        return abs(wrap(target_yaw - p.yaw)) <= 0.035

    def _lane_offset(self, p, lane_x: float, lane_y: float,
                     lane_yaw: float) -> float:
        """Lateral offset in the lane frame (positive = left of lane dir)."""
        return (-(p.x - lane_x)) * math.sin(lane_yaw) \
            + (p.y - lane_y) * math.cos(lane_yaw)

    def _lane_settle(self, lane_x: float, lane_y: float, lane_yaw: float,
                     timeout_s: float, max_travel: float = 0.35) -> None:
        """Correct residual lateral offset with a short bang-bang S-turn.

        vt_locked12 has no vy, so a lateral offset e is corrected by: steer
        theta=clamp(12*|e|, 8deg, 25deg) toward the lane line, drive straight
        until |e|<=0.012 (or max_travel), steer back to lane_yaw.  The forward
        travel is capped so the expert runway is not eaten; the final in-place
        turn leaves only ~1-2 cm of rotation drift.
        """
        t_end = self._now() + self._timeout(timeout_s)
        origin = self.pose()
        unit_x, unit_y = math.cos(lane_yaw), math.sin(lane_yaw)
        along0 = (origin.x - lane_x) * unit_x + (origin.y - lane_y) * unit_y
        for _ in range(2):
            if self._now() >= t_end - 1.0:
                return
            p = self.pose()
            e = self._lane_offset(p, lane_x, lane_y, lane_yaw)
            if abs(e) <= 0.018:
                break
            theta = abs(e) * 12.0
            theta = min(math.radians(25.0), max(math.radians(8.0), theta))
            heading = lane_yaw - math.copysign(theta, e)
            self._rotate_servo(heading,
                               timeout_s=self._remaining(t_end - 0.40, 8.0))
            t_drive = self._remaining(t_end - 6.0, 12.0)
            t1 = self._now()
            while self._now() - t1 < t_drive:
                p = self.pose()
                e = self._lane_offset(p, lane_x, lane_y, lane_yaw)
                along = ((p.x - lane_x) * unit_x
                         + (p.y - lane_y) * unit_y)
                if abs(e) <= 0.018 or (along - along0) >= max_travel:
                    break
                err = wrap(heading - p.yaw)
                wz = max(-0.15, min(0.15, 1.2 * err))
                self.command(0.18, 0.0, wz)
                self._tick()
                if self.fallen():
                    return
            self.stop(0.20)
        # steer back to the lane heading
        if self._now() < t_end:
            self._rotate_servo(lane_yaw,
                               timeout_s=self._remaining(t_end, 8.0))

    def _creep_along(self, lane_x: float, lane_y: float, lane_yaw: float,
                     target_along: float, timeout_s: float,
                     vx: float = 0.18) -> None:
        """Follow the fixed lane up to its target progress, without turning back.

        The locked gait can crab even at zero yaw. Shallow cross-track steering
        compensates that drift during a long staged approach; final heading is
        polished separately after this bounded forward movement.
        """
        ux, uy = math.cos(lane_yaw), math.sin(lane_yaw)
        t0 = self._now()
        while self._now() - t0 < self._timeout(timeout_s):
            p = self.pose()
            along = (p.x - lane_x) * ux + (p.y - lane_y) * uy
            if along >= target_along:
                break
            lateral = self._lane_offset(p, lane_x, lane_y, lane_yaw)
            correction = max(-math.radians(25), min(math.radians(25),
                             math.atan2(-lateral, 0.35)))
            err = wrap(lane_yaw + correction - p.yaw)
            wz = max(-0.25, min(0.25, 1.2 * err))
            if abs(err) > 0.02 and abs(wz) < 0.14:
                wz = math.copysign(0.14, err)
            self.command(vx, 0.0, wz)
            self._tick()
            if self.fallen():
                break
        self.stop(0.20)

    def align_to(self, leg: dict, policy: str, deadline_s: float | None = None):
        """One forward approach, bounded shallow correction, then a hard gate.

        Never turn perpendicular to remove centimetres of residual error, or
        circle back to an already passed stand point. Tight gaps retain their
        runway; an unreachable alignment fails with its measured residual.
        """
        if (getattr(self, "_continuing_expert", False)
                and self._current_policy == policy and leg["mode"] == "adaptive"):
            return self.align_continuation(leg)
        started = self._now()
        deadline = started + self._timeout(float(leg["align_deadline_s"]) if deadline_s is None
                                            else deadline_s)
        self.ensure_transfer()
        log = self.get_logger()
        yaw = float(leg["lane_yaw"])
        staged = "align_approach" in leg
        tx, ty = leg["stand"]
        xy_tol, yaw_tol = leg["align_xy_tol"], leg["align_yaw_tol_deg"]
        along, lateral = lane_position(leg, self.pose())
        log.info("align start leg=%s policy=%s stand=(%+.2f,%+.2f) yaw=%+.1fdeg"
                 % (leg["leg"], policy, tx, ty, math.degrees(yaw)))
        if along < -0.12:
            approach = tuple(leg.get("align_approach", (tx, ty)))
            p_approach = self.pose()
            distance = _dist((p_approach.x, p_approach.y), approach)
            approach_progress = ((p_approach.x - approach[0]) * math.cos(yaw)
                                 + (p_approach.y - approach[1]) * math.sin(yaw))
            # The final transit already accepts this corner within 15 cm.
            # Reacquiring it with a tighter point tolerance can turn sideways
            # again before the staged runway even begins.
            if not staged or (distance > 0.18 and approach_progress < 0.0):
                if not self.goto(approach, tol=0.10, speed=0.18,
                                 timeout_s=self._remaining(deadline, max(12.0, 3.5 * distance / 0.18))):
                    log.warning("approach missed for %s" % leg["leg"])
        if self.fallen():
            raise RuntimeError("fallen during alignment at " + leg["leg"])
        self._rotate_servo(yaw, timeout_s=self._remaining(deadline, 20.0))
        along, lateral = lane_position(leg, self.pose())
        travel = min(float(leg["settle_forward"]),
                     max(0.0, float(leg["along_max"]) - along - 0.03))
        if abs(lateral) > xy_tol and travel > 0.03 and self._now() < deadline - 2:
            self._lane_settle(leg["lane_x"], leg["lane_y"], yaw,
                              timeout_s=self._remaining(deadline, 30.0),
                              max_travel=travel)
        along, _ = lane_position(leg, self.pose())
        if along < leg["along_min"] and not leg.get("no_creep"):
            # Stop the staged plum creep 25 cm before its nominal stand,
            # leaving room for one measured correction and the stopping drift.
            self._creep_along(leg["lane_x"], leg["lane_y"], yaw,
                              -0.25 if staged else 0.0,
                              timeout_s=self._remaining(deadline, 16.0))
            if staged:
                along, lateral = lane_position(leg, self.pose())
                # Travel is measured before the S-turn; its stopping and
                # return-to-heading drift need their own forward allowance.
                travel = min(0.40, max(0.0, leg["along_max"] - along - 0.10))
                if abs(lateral) > 0.5 * xy_tol and travel > 0.03:
                    self._lane_settle(leg["lane_x"], leg["lane_y"], yaw,
                                      timeout_s=self._remaining(deadline, 20.0),
                                      max_travel=travel)
                along, _ = lane_position(leg, self.pose())
                if along < leg["along_min"]:
                    self._creep_along(leg["lane_x"], leg["lane_y"], yaw,
                                      float(leg["along_min"]) + 0.02,
                                      timeout_s=self._remaining(deadline, 10.0))
        if abs(math.degrees(wrap(self.pose().yaw - yaw))) > yaw_tol:
            self._rotate_servo(yaw, timeout_s=self._remaining(deadline, 10.0))
        p = self.pose()
        along, lateral = lane_position(leg, p)
        yaw_err = math.degrees(wrap(p.yaw - yaw))
        elapsed = self._now() - started
        ok = (not self.fallen() and abs(lateral) <= xy_tol
              and abs(yaw_err) <= yaw_tol
              and leg["along_min"] <= along <= leg["along_max"]
              and self._now() <= deadline + 0.5)
        log.info("expert switch leg=%s policy=%s lane_center=(x=%+.2f,y=%+.2f,"
                 "yaw=%+.1fdeg) start_dy=+0.000 stand=(x=%+.2f,y=%+.2f) "
                 "actual=(x=%+.2f,y=%+.2f,yaw=%+.1fdeg) e=%+.3f along=%+.2f "
                 "yaw_err=%+.1fdeg gates=(%.2fm,%.1fdeg) align_s=%.1f %s"
                 % (leg["leg"], policy, leg["lane_x"], leg["lane_y"],
                    math.degrees(yaw), tx, ty, p.x, p.y, math.degrees(p.yaw),
                    lateral, along, yaw_err, xy_tol, yaw_tol, elapsed,
                    "OK" if ok else "ALIGN_FAIL"))
        self._corr = {"x": p.x, "y": p.y, "e": lateral, "along": along,
                      "yaw_err_deg": yaw_err, "align_s": elapsed}
        self._emit_plan()
        return ok, p.x, p.y, p.yaw, lateral, yaw_err

    def align_continuation(self, leg: dict):
        """Check a forward handover while the same expert keeps tracking its lane.

        Consecutive row boundaries coincide with the preceding measured gate.
        The actor remains at its validated crossing speed. Stationary turns or
        a low-speed alignment gait can destabilise these experts and consume
        the next obstacle's runway, so an out-of-envelope pose fails directly.
        """
        p = self.pose()
        along, lateral = lane_position(leg, p)
        yaw_error = math.degrees(wrap(p.yaw - leg["lane_yaw"]))
        ok = (bool(leg.get("continuous_lane")) and not self.fallen()
              and not crossing_fault(leg, p)
              and abs(lateral) <= leg["align_xy_tol"]
              and abs(yaw_error) <= leg["align_yaw_tol_deg"]
              and leg["along_min"] <= along <= leg["along_max"])
        self.get_logger().info(
            "expert switch leg=%s policy=%s continuation=1 "
            "actual=(x=%+.2f,y=%+.2f,yaw=%+.1fdeg) e=%+.3f along=%+.2f "
            "yaw_err=%+.1fdeg gates=(%.2fm,%.1fdeg) align_s=0.0 %s"
            % (leg["leg"], leg["policy"], p.x, p.y, math.degrees(p.yaw),
               lateral, along, yaw_error, leg["align_xy_tol"],
               leg["align_yaw_tol_deg"], "OK" if ok else "ALIGN_FAIL"))
        self._corr = {"e": lateral, "along": along, "yaw_err_deg": yaw_error,
                      "align_s": 0.0, "continuation": True}
        self._emit_plan()
        return ok, p.x, p.y, p.yaw, lateral, yaw_error

    # ---- expert crossing --------------------------------------------------
    def expert_cross(self, leg: dict, attempt: int = 1):
        """Run one expert leg with the adaptive lane command until its gate.

        Returns (crossed, why, pose).  ``why`` is one of gate / timeout / stall
        / fallen.  A stall means the lane-projected position made less than
        ``CROSS_STALL_PROGRESS_M`` progress for ``CROSS_STALL_WINDOW_S``; the
        caller may recover (back off + re-align + retry once).

        ``mode="depth"`` (plum): the policy has no command input and ignores
        /joy, so the lane command below is only for the fake kinematic sim, and
        the stop/reverse primitives are skipped -- the caller hands the body to
        the transfer policy right after the gate.
        """
        policy = leg["policy"]
        vx = float(leg["vx"])
        lane_x = float(leg["lane_x"])
        lane_y = float(leg["lane_y"])
        lane_yaw = float(leg["lane_yaw"])
        # Leave the geometric lane/gates unchanged; reserve southern margin
        # before the 20 cm trench's repeatable northward crossing drift.
        feedback_y = lane_y - 0.06 if leg["leg"] == "trench_20cm" else lane_y
        depth = leg.get("mode") == "depth"
        if depth:
            return self.depth_plum_cross(leg)
        self.switch(policy)
        retry_note = "" if attempt <= 1 else f" (recovery retry {attempt - 1})"
        self._cmd = ("%s(policy=%s, vx=%+.2f, lane=(%+.2f,%+.2f))%s"
                     % (leg["leg"], policy, vx, lane_x, lane_y, retry_note))
        self._cli = (f"switch {policy}; forward {vx:+.2f} m/s "
                     f"+ adaptive lane hold")
        self._cli += retry_note
        self._corr = {}
        self._emit_plan()
        log = self.get_logger()
        cos_l, sin_l = math.cos(lane_yaw), math.sin(lane_yaw)
        prev = None
        why = "timeout"
        t0 = self._now()
        p0 = self.pose()
        best_along = ((p0.x - lane_x) * cos_l + (p0.y - lane_y) * sin_l)
        last_progress = t0
        while self._now() - t0 < self._timeout(float(leg["cross_timeout"])):
            p = self.pose()
            if self.fallen():
                why = "fallen"
                break
            fault = crossing_fault(leg, p)
            if fault:
                why = fault
                break
            if leg["stop"](p):
                why = "gate"
                break
            # stall detector: lane-projected progress must keep coming
            along = (p.x - lane_x) * cos_l + (p.y - lane_y) * sin_l
            now = self._now()
            if along > best_along + CROSS_STALL_PROGRESS_M:
                best_along = along
                last_progress = now
            elif now - last_progress >= self._timeout(float(leg.get("stall_window_s", CROSS_STALL_WINDOW_S))):
                why = "stall"
                log.info("expert stall leg=%s after %.1fs along=%+.2f "
                         "(%.2fm behind best; no new %.2fm progress for %.1fs)"
                         % (leg["leg"], now - t0, along, best_along - along,
                            CROSS_STALL_PROGRESS_M, now - last_progress))
                break
            if leg.get("fixed"):
                # fixed-command expert (s1d trench): no adaptive vy/wz, the
                # pre-leg alignment gate is what keeps it on the lane
                self.command(vx, 0.0, 0.0)
            else:
                command = self.lane_cmd(p, prev, lane_x, feedback_y, lane_yaw, vx)
                weak_yaw_actor = (
                    ((leg["category"] == "trench"
                      or (leg["category"] == "slope" and leg["difficulty"] <= 30)
                      or (leg["category"] == "platform" and leg["difficulty"] <= 25))
                     and policy == "row2p_p40w40_v1")
                    or (leg["category"] == "stairs" and policy == "st30_ppo_st30_s100_600"))
                if weak_yaw_actor:
                    # Deployment traces show weak correction from small yaw
                    # commands on these actors; 0.4--0.7 turns them. Compensate only outside the small lane/heading band;
                    # retain the trained feedback sign and 0.7 rad/s limit.
                    lateral = self._lane_offset(p, lane_x, feedback_y, lane_yaw)
                    heading = wrap(p.yaw - lane_yaw)
                    if ((abs(lateral) > 0.06 or abs(heading) > 0.035)
                            and 0.0 < abs(command[2]) < 0.45):
                        command = (*command[:2], math.copysign(0.45, command[2]))
                self.command(*command)
            prev = p
            self._tick()
        settle_s = float(leg.get("cross_settle_s", 0.0))
        if not leg.get("keep_expert_moving", leg.get("continuous_lane")) or why != "gate" or settle_s:
            self.stop(settle_s or 0.50)
        p = self.pose()
        crossed = (why == "gate" and not self.fallen() and not crossing_fault(leg, p))
        return crossed, why, p

    def depth_plum_cross(self, leg: dict):
        """Composed plum attempt: depth expert crosses the piles, wheeled expert descends.

        The depth actor has no command term (it self-drives and ignores /joy).
        In the eval runner it also completes the exit-platform step-down, but in
        the deployment stack its own descent is knife-edge: the body pitches
        forward off the 0.2 m edge and falls.  So the route splits the attempt:
        the depth expert crosses the entry platform and pile field, then at
        ``handover_x`` a wheeled expert takes over as the front supports reach
        the exit platform. Rear supports can still be on the final piles:
        this mixed-support handover remains sensitive to speed. It brakes with a short
        reverse command, stands, and walks down under a slow lane command.  Both
        halves are learned policies. The plan records one attempt with an
        explicit handover; support-point replay verifies the crossing.

        """
        log = self.get_logger()
        policy = leg["policy"]
        handover = str(leg["handover_policy"])
        handover_x = float(leg["handover_x"])
        settle_s = float(leg["handover_settle_s"])
        drive_vx = float(leg["handover_vx"])
        lane_y = float(leg["lane_y"])
        lane_yaw = float(leg["lane_yaw"])

        def support_failure(p):
            fault = plum_support_fault(leg, p, self.feet())
            if fault:
                # A depth actor ignores zero velocity; unload it before any
                # hold/tick can move the robot further after a dropped support.
                self.command(0.0, 0.0, 0.0)
                self.switch_topic(TRANSFER_POLICY, hold_s=0.0)
                self.stop(0.20)
                return False, fault, self.pose()
            return None

        def hold_landing_command(vx, duration):
            started = self._now()
            while self._now() - started < duration:
                p = self.pose()
                if self.fallen():
                    return False, "fallen", p
                failed = support_failure(p)
                if failed:
                    return failed
                fault = crossing_fault(leg, p)
                if fault:
                    return False, fault, p
                self.command(vx, 0.0, 0.0)
                self._tick()
            return None

        self._cmd = ("%s(policy=%s, vx=%+.2f, handover->%s@x=%+.2f)"
                     % (leg["leg"], policy, float(leg["vx"]), handover,
                        handover_x))
        self._cli = (f"switch {policy}; depth expert self-drives the pile "
                     f"field; at field x {handover_x:+.2f} switch {handover} "
                     f"(/policy_switch), stand {settle_s:.1f} s, then forward "
                     f"{drive_vx:+.2f} m/s down to the gate")
        self._corr = {}
        self._emit_plan()
        # Open the recorded attempt and start monitoring before loading the
        # self-driving actor. The generic fallback switch waits a full second,
        # during which this actor can already reach the entry platform.
        self.switch_topic(policy, hold_s=0.0)
        log.info("plum cross: depth policy %s crosses the pile field; handover "
                 "to %s at field x=%+.2f (level x=%+.2f) -- the depth actor "
                 "cannot be steered and its own exit step-down falls in the "
                 "deployment stack"
                 % (policy, handover, handover_x, handover_x - 2.20))
        t0 = self._now()
        prev = None
        driven = False
        while self._now() - t0 < self._timeout(float(leg["cross_timeout"])):
            p = self.pose()
            if self.fallen():
                return False, "fallen", p
            failed = support_failure(p)
            if failed:
                return failed
            fault = crossing_fault(leg, p)
            if fault:
                self.switch_topic(TRANSFER_POLICY, hold_s=0.0)
                self.stop(0.20)
                return False, fault, self.pose()
            if not driven and p.x >= handover_x:
                log.info("plum handover: switch %s at (%+.2f,%+.2f) level x=%+.2f"
                         % (handover, p.x, p.y, p.x - 2.20))
                self.set_cmd("%s(handover=%s@field_x=%+.2f)"
                             % (leg["leg"], handover, handover_x))
                self._cli = (f"switch {handover} (/policy_switch); stand "
                             f"{settle_s:.1f} s; forward {drive_vx:+.2f} m/s "
                             f"+ lane hold")
                self._emit_plan()
                # The recorded successful composition gave the wheeled actor
                # one short forward entry before braking. Preserve that input
                # sequence, with every landing frame checked. Immediate reverse
                # plus lateral/yaw feedback changed the landing dynamics.
                self.command(float(leg["vx"]), 0.0, 0.0)
                self.switch_topic(handover, hold_s=0.0)
                for vx, duration in (
                        (float(leg["vx"]), PLUM_HANDOVER_ENTRY_S),
                        (float(leg.get("handover_brake_vx", 0.0)),
                         float(leg.get("handover_brake_s", 0.0))),
                        (0.0, settle_s)):
                    failed = hold_landing_command(vx, duration)
                    if failed:
                        return failed
                prev = self.pose()
                driven = True
                continue
            if driven:
                if leg["stop"](p):
                    # Stop the wheels before the caller switches to the locked
                    # connector: locking wheel hinges while the body is still
                    # rolling lurches the connector backwards (measured, real_v24
                    # needed +0.5 m forward and ended -0.5 m back).
                    self.command(0.0, 0.0, 0.0)
                    t_gate = self._now()
                    while self._now() - t_gate < PLUM_GATE_SETTLE_S:
                        failed = support_failure(self.pose())
                        if failed:
                            return failed
                        self._tick()
                    stopped = self.pose()
                    failed = support_failure(stopped)
                    if failed:
                        return failed
                    reached = (at_exit(leg, stopped, settled=True)
                               and not self.fallen() and not crossing_fault(leg, stopped))
                    return reached, "gate" if reached else "settled_gate_missed", stopped
                self.command(*self.lane_cmd(p, prev, 2.20, lane_y, lane_yaw,
                                            drive_vx))
            else:
                # Pre-handover the depth policy ignores commands; the lane
                # command is what keeps the fake kinematic sim on the lane.
                self.command(*self.lane_cmd(p, prev, 2.20, lane_y, lane_yaw,
                                            float(leg["vx"])))
            prev = p
            self._tick()
        return False, "timeout", self.pose()

    def recover_cross(self, leg: dict, why: str) -> bool:
        """Back off along -lane, re-align and report readiness for one retry.

        Only allowed on flat ground: reversing on elevated terrain (stairs top
        landing, slope, platform top) tips/rolls the robot -- real v10 stalled
        on the stairs top, the recovery backed down the stairs and fell.  When
        the base is high the caller gets False and fails the leg immediately.
        """
        log = self.get_logger()
        p0 = self.pose()
        if p0.z >= RECOVER_MAX_GROUND_Z:
            log.info("expert_recover leg=%s skipped_on_terrain z=%.2f "
                     "reason=%s (no reverse on elevated terrain)"
                     % (leg["leg"], p0.z, why))
            return False
        lane_yaw = float(leg["lane_yaw"])
        dirx, diry = math.cos(lane_yaw), math.sin(lane_yaw)
        self.ensure_transfer()
        bx = p0.x - dirx * CROSS_BACKOFF_M
        by = p0.y - diry * CROSS_BACKOFF_M
        log.info("expert_recover leg=%s reason=%s dist=%.2f "
                 "from=(%+.2f,%+.2f,%+.1fdeg) back=(%+.2f,%+.2f)"
                 % (leg["leg"], why, CROSS_BACKOFF_M, p0.x, p0.y,
                    math.degrees(p0.yaw), bx, by))
        self.set_leg(leg["leg"])
        # Reverse in the lane frame. A goto behind the body would rotate it
        # 180 degrees before driving, consuming the gap and crossing a neighbour.
        t_reverse = self._now()
        while self._now() - t_reverse < self._timeout(14.0):
            p = self.pose()
            remaining = (p.x - bx) * dirx + (p.y - by) * diry
            if remaining <= 0.10 or self.fallen():
                break
            heading_error = wrap(lane_yaw - p.yaw)
            self.command(-min(0.20, max(0.06, remaining)), 0.0,
                         max(-0.25, min(0.25, 1.2 * heading_error)))
            self._tick()
        self.stop(0.3)
        if self.fallen():
            raise RuntimeError("fallen during crossing recovery at leg %s"
                               % leg["leg"])
        aligned = self.align_to(leg, leg["policy"],
                                deadline_s=CROSS_RETRY_ALIGN_S)
        if not aligned[0]:
            log.warning("expert_recover leg=%s re-align missed tolerance "
                        "(e=%+.3f yaw_err=%+.1fdeg); crossing retry refused"
                        % (leg["leg"], aligned[4], aligned[5]))
        return bool(aligned[0])

    # ---- route ------------------------------------------------------------
    def _log_table(self) -> None:
        log = self.get_logger()
        log.info("Part2 canonical route: %d legs, connector=%s"
                 % (len(LEGS), TRANSFER_POLICY))
        for category, tiers in SKIPPED_TIERS.items():
            for difficulty, reason in tiers.items():
                log.warning("SKIPPED %s %s: %s" % (category, difficulty, reason))
        for index, leg in enumerate(LEGS):
            log.info("  leg%d %s difficulty=%s top=%.3fm policy=%s "
                     "stand=%s end=%s targets=%s"
                     % (index + 1, leg["leg"], leg["difficulty"], leg["top_m"],
                        leg["policy"], leg["stand"], leg["cross_end"],
                        ",".join(leg["target_geoms"])))

    def search_goto(self, target, speed: float = 0.50) -> None:
        """goto with a distance-scaled sim-time timeout (long transits)."""
        p = self.pose()
        dist = _dist((p.x, p.y), target)
        timeout = max(20.0, 2.5 * dist / max(0.20, speed) + 12.0)
        # Corridor corners retain 15 cm of position allowance; the audited
        # narrowest gap has 60 cm centre clearance. Expert alignment below
        # uses its separate, tighter lane/heading gates.
        self.require(tuple(target), tol=0.15, timeout_s=timeout, speed=speed)

    def _route(self) -> None:
        log = self.get_logger()
        if not self.wait_pose(30.0):
            raise RuntimeError("no /eval_state; launch with route:=part2")
        log.info("Part2 route: standing up / entering RL with %s" % TRANSFER_POLICY)
        self._log_table()
        # Preselect the connector so the first RL entry loads the locked model
        # (the sim boots policy_name:=flat; EnterRL would otherwise load it).
        self.switch(TRANSFER_POLICY)
        self._press(buttons=[BTN_A], hold_s=0.15)
        if not self.wait_stable_z(0.46, tilt_max=8.0, hold_s=0.5, timeout_s=15.0):
            raise RuntimeError("Part2 stand-up failed its height/tilt gate")
        self._hold(0.3)
        self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)  # EnterRL
        self._hold(1.5)
        if not self.wait_stable_z(0.40, tilt_max=12.0, hold_s=0.3, timeout_s=4.0):
            log.warning("RL entry not stable; pressing EnterRL again")
            self._press(buttons=[BTN_RB], hat=(0, 1), hold_s=0.15)
            self._hold(1.2)
            self.wait_stable_z(0.40, tilt_max=15.0, hold_s=0.3, timeout_s=4.0)
        self.ensure_transfer()
        self.stop(0.3)
        p = self.pose()
        log.info("route start at (%.2f, %.2f, yaw %.2f) policy=%s"
                 % (p.x, p.y, p.yaw, self._current_policy))
        start_index = 0
        only_index = _resolve_start_leg(os.environ.get("PART2_ROUTE_ONLY", ""))
        raw_start = os.environ.get("PART2_ROUTE_START", "").strip()
        if raw_start:
            start_index = _resolve_start_leg(raw_start)
            if start_index < 0:
                raise ValueError("PART2_ROUTE_START names no leg: " + raw_start)
            else:
                entry = approach_point(LEGS[start_index])
                log.info("PART2_ROUTE_START=%s: starting at leg %d/%d %s "
                         "(earlier legs and selected entry transit skipped; alignment approach (%+.2f,%+.2f))"
                         % (raw_start, start_index + 1, len(LEGS),
                            LEGS[start_index]["leg"], entry[0], entry[1]))

        for index, leg in enumerate(LEGS):
            if index < start_index:
                # Single-leg real runs: no transit/exit/expert switch of the
                # earlier legs; the selected leg's own goto chain drives in
                # from wherever the robot currently stands.
                log.info("skip leg %d/%d %s (PART2_ROUTE_START)"
                         % (index + 1, len(LEGS), leg["leg"]))
                continue
            self.set_leg(leg["leg"])
            log.info("=== leg %d/%d %s (policy=%s, vx=%+.2f) ==="
                     % (index + 1, len(LEGS), leg["leg"], leg["policy"], leg["vx"]))
            hop_speed = float(leg.get("transit_speed", TRANSIT_SPEED))
            if not (raw_start and index == start_index):
                self.transit_path(leg["transit"], speed=hop_speed)
            aligned = self.align_to(leg, leg["policy"])
            if not aligned[0]:
                raise RuntimeError(
                    "alignment failed at leg %s: y_err=%+.3f yaw_err=%+.1fdeg"
                    % (leg["leg"], aligned[4], aligned[5]))
            align_s = float(self._corr.get("align_s", 0.0))
            t_cross_start = self._now()
            crossed, why, pend = self.expert_cross(leg)
            recoveries = 0
            if (not crossed and RECOVER_ENABLED and why in ("stall", "timeout")
                    and not leg.get("no_recover") and not self.fallen()):
                if self.recover_cross(leg, why):
                    recoveries = 1
                    crossed, why2, pend = self.expert_cross(leg, attempt=2)
                    if not crossed:
                        why = "%s+retry_%s" % (why, why2)
                    else:
                        why = "gate_recovered"
                else:
                    # recovery refused (elevated terrain): fail with residual
                    why = "%s+no_reverse_on_terrain" % why
            t_cross = self._now() - t_cross_start
            post_ok = bool(leg["post"](pend)) and not self.fallen()
            ok = bool(crossed and post_ok)
            detail = ("policy=%s vx=%+.2f end=(%+.2f,%+.2f,yaw=%+.1fdeg) gate=%s"
                      " post=%s y_err_at_switch=%+.3f yaw_err=%+.1fdeg"
                      " align_s=%.1f cross_s=%.1f recover=%d mode=%s"
                      % (leg["policy"], leg["vx"], pend.x, pend.y,
                         math.degrees(pend.yaw), why,
                         "ok" if post_ok else "out", aligned[4], aligned[5],
                         align_s, t_cross,
                         recoveries, leg.get("mode", "")))
            # A failed self-driving depth actor must be unloaded before
            # report() can raise and terminate route execution.
            if leg["mode"] == "depth" and not ok:
                self.switch_topic(TRANSFER_POLICY, hold_s=0.0)
                self.stop(0.2)
            self.report(leg["leg"], ok, detail)
            if ok:
                self.set_cmd("cross_complete")
            # For the depth plum leg this is the immediate post-gate handover:
            # zero /joy commands would not stop it, so the transfer policy is
            # loaded right here (the move_to exit hops then run on the connector).
            next_leg = LEGS[index + 1] if index + 1 < len(LEGS) else None
            self._continuing_expert = bool(
                ok and next_leg and leg["keep_expert_moving"])
            if not self._continuing_expert:
                self.ensure_transfer(announce=True)
            self.transit_path(leg["exit"])
            if index == only_index:
                self.stop(0.4)
                if self.fallen():
                    raise RuntimeError("fallen after final handover at " + leg["leg"])
                log.info("PART2_ROUTE_ONLY: stopping after leg %d (debug)" % (index + 1))
                return

        completion = "Part2 debug subset finished" if raw_start else "Part2 route finished"
        if self._exit_code:
            log.error("%s with failed legs; see FAIL records above" % completion)
            self.stop(0.5)
            return
        self.stop(0.5)
        if not self.wait_stable_z(0.40, tilt_max=15.0, hold_s=0.3, timeout_s=4.0):
            raise RuntimeError("Part2 final handover did not settle upright")
        log.info("%s: %d/%d legs PASS "
                 "(integration gates; official support evidence uses the eval judge)"
                 % (completion, len(LEGS) - start_index, len(LEGS)))


def main() -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init()
    node = Part2Route()
    try:
        while rclpy.ok() and not node._finished.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.command(0.0, 0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return node._exit_code


if __name__ == "__main__":
    raise SystemExit(main())
