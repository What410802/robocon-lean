#!/usr/bin/env python3
"""Part 2 terminal-state judge (v5 lineage, ``v8.0`` team-migration revision).

The judge is a pure Python terminal-state predicate: the runner (MuJoCo /
deployment trace) feeds it the trunk pose, the tilt and the four support-point
(wheel-bottom) coordinates; it decides success, failure or running.

Success semantics (team operationalisation of the official "cross the
obstacle" rule, see ``eval/SPEC.md`` §4): the attempt only counts once the
robot has *traversed* the obstacle -- climbed / crossed, held, then left it and
returned to normal ground -- not when the trunk merely crosses a line or stands
on the obstacle.  Per obstacle:

* platform: four wheels on the platform top (tight top band), hold, then all
  wheels off the far edge and back near the ground;
* stairs / slope: reach the top landing, hold, then descend to the ground;
* wall: every wheel crosses the wall plane inside the wall y span, then back on
  the ground past the wall;
* trench (gap): cross the slot, reach the far plateau, descend the exit ramp to
  the ground;
* plum: cross the pile field without a wheel dropping below ``z_cross_min``,
  reach the exit platform, then step off its far edge onto normal ground.

Failure is a rule violation (fall, lane/forbidden footprint, wheel inside a
trench slot, plum wheel on the ground) or the attempt ending without success; it
is never a reward term.
"""
from __future__ import annotations

import numpy as np

FALL_TILT_DEG = 75.0
# A support point exactly on the y lane edge is not a side-ground contact; allow
# a small geometric/contact-point tolerance, and give a short grace window so a
# real fall can take reason priority over an earlier wall/edge grazing.
SIDE_MARGIN_M = 0.02
SIDE_GRACE_S = 0.6
FALL_Z_M = 0.15
# Judge version for score traceability (bumped whenever a predicate changes;
# old result JSONs keep the version string they were produced with).
#   v7.5 (2026-10-31): trench terminal state = descend the exit ramp to ground.
#   v7.6 (2026-10-31): the same traverse semantics for the official plum field.
#   v8.0 (2026-11-02, team migration): plum rebuilt to the owner-confirmed 3-pile
#        geometry (no ramps; exit platform far edge is the terminal); the top
#        "on surface" upper band for platform/plum tightened from +0.20 m to
#        +0.05 m (a wheel hovering well above the top is not standing on it).
JUDGE_VERSION = "v8.0-plum-3pile-2026-11-02"
# Wheel geometry / terminal-state tolerances (blackW wheel radius 0.103 m).
WHEEL_RADIUS_M = 0.103
# A wheel counts as "on a support surface" when its bottom is within this
# distance of that surface (the default pose has ~2 cm static penetration).
WHEEL_ON_SURFACE_TOL_M = 0.06
# Tightened upper half of the top-surface band (platform / plum exit platform):
# a wheel bottom more than 5 cm above the surface is not resting on it.
WHEEL_TOP_BAND_M = 0.05
# Wall crossing: all wheels must be past the wall and back on the ground.
WALL_BACK_CLEARANCE_M = 0.16
WALL_SPAN_HALF_M = 0.7

# P3-12 support geometry interface: wheel contact points are the terminal support.
SUPPORT_KIND = "wheel"
GROUND_WHEEL_TOL_M = 0.08
# Stairs: all four wheels must be past the end of the down flight and back on
# the ground (the official staircase is up + 0.8 m landing + down).
STEP_DESCEND_CLEARANCE_M = 0.10
# Extra hold beyond the rule's 0.2 s so the terminal state is genuinely stable.
TERMINAL_HOLD_EXTRA_S = 0.1



class Part2OfficialJudge:
    support_kind = SUPPORT_KIND
    """Rule-faithful Part 2 capability judge on official-geometry levels.

    Success is a **terminal-state predicate on the robot's support points**: the
    attempt only counts once all four wheels have reached the target support
    surface and stay there (``hold_s``), not when the trunk merely crosses a
    line.  Failure is a rule violation (fall, lane bypass, wheel inside a trench
    slot) or timeout; it is a termination, never a reward term.

    Per obstacle:
      * platform: all four wheels on the platform top inside the platform
        x-range and the *tightened* top band (bottom within
        ``[-WHEEL_ON_SURFACE_TOL_M, +WHEEL_TOP_BAND_M]`` of the surface), stable
        ``hold_s``, then off the far edge back near the ground;
      * stairs / slope: all four wheels on the top landing, past the goal x,
        then descended back to the ground;
      * wall: crossed the wall plane, all four wheels back on the ground past
        the wall, stable;
      * gap (trench): cross the slot, reach the far plateau, then traverse the
        1 m exit ramp and come back to normal ground past the ramp end (all
        four wheels on the ground, stable ``hold_s``); crossing the slot /
        standing on the far plateau is no longer success.  A wheel dipping into
        the slot deeper than ``min(0.5*depth, radius)`` invalidates the attempt
        (official trench depth = 0.30 m, wheel radius 0.103 m -> 0.103 m
        allowed drop);
      * plum: cross the owner-confirmed 3-pile field (entry platform, three
        0.2 m cubes in one row, exit platform; **no ramps**) without any wheel
        bottom dropping below ``z_cross_min`` while over the field, reach the
        exit platform (four wheels inside the tightened 0.2 m top band), then
        step off its far edge and come back to normal ground (``exit_ramp_end``
        is the exit-platform far edge in the new geometry, kept under its
        historical manifest name); standing on the exit platform is evidence
        only, not success.  Any wheel touching the ground while over the pile
        field invalidates the attempt.
    """

    def __init__(self, level: dict) -> None:
        self.level = level
        self.j = level["judge"]
        self.kind = self.j["kind"]
        # Fail closed (audit P1-5): judge knobs must be explicit.  The old silent
        # defaults (lane 0.7 / hold 0.2 s) could hide a missing rule, and
        # np.all([])==True let a missing wheel body pass the four-wheel checks.
        name = level.get("key", "?")
        for key in ("lane_half_width", "hold_s"):
            if key not in self.j:
                raise KeyError(f"judge level {name!r}: missing required key {key!r}")
        if self.kind == "stairs" and "descend_end_x" not in self.j:
            raise KeyError(
                f"judge level {name!r}: stairs needs 'descend_end_x' (a fallback to "
                f"goal_x_min would silently score a non-descended robot as success)")
        if self.kind == "plum":
            # Reaching the exit platform is evidence, not success: the level must
            # state where the exit platform ends so the terminal state can require
            # the step down to normal ground.
            for key in ("exit_ramp_end", "exit_ground_tol"):
                if key not in self.j:
                    raise KeyError(
                        f"judge level {name!r}: plum needs {key!r} (exit-platform "
                        f"step-down; falling back would silently score a robot still "
                        f"on the exit platform as success)")
        self.t = 0.0
        self.last_t = None
        self.held = 0.0
        self.status = "running"
        self.reason = ""
        self.detail: dict = {"max_z": 0.0, "max_x": 0.0, "min_z_gap": 9.9,
                             "min_wheel_z": 9.9, "min_wheel_z_slot": 9.9}
        self.crossed = False
        self.max_z_near = 0.0
        self.gap_near_crossed = False
        self.gap_far_crossed = False
        self.top_seen = False
        self.top_ok = False
        self.top_held = 0.0
        self.exit_platform_seen = False
        self.exited = False
        self.prev = None
        self.last_base = None
        self.last_wheel_z = None
        self._wall_side_bypass = False
        self._prev_wx = None
        self._prev_wy = None
        self._wall_crossed_mask = None
        self.side_since: float | None = None

    def _lateral_ok(self, y: float) -> bool:
        return abs(y) <= float(self.j["lane_half_width"])

    def update(self, t: float, base: np.ndarray, tilt_deg: float,
               wheel_x: np.ndarray | None = None,
               wheel_z: np.ndarray | None = None,
               wheel_y: np.ndarray | None = None) -> str:
        if self.status != "running":
            return self.status
        if (wheel_x is None or wheel_y is None or wheel_z is None
                or len(wheel_x) != 4 or len(wheel_y) != 4 or len(wheel_z) != 4
                or not (np.isfinite(wheel_x).all() and np.isfinite(wheel_y).all()
                        and np.isfinite(wheel_z).all())):
            self.status, self.reason = "fail", "missing_support_evidence"
            return self.status
        self.t = float(t)
        x, y, z = (float(base[0]), float(base[1]), float(base[2]))
        self.last_base = (x, y, z)
        self.last_wheel_z = None if wheel_z is None else np.asarray(wheel_z, dtype=float).copy()
        dt = 0.02 if self.last_t is None else max(0.0, self.t - self.last_t)
        self.detail["max_z"] = round(max(self.detail["max_z"], z), 3)
        self.detail["max_x"] = round(max(self.detail["max_x"], x), 3)
        lateral_ok = self._lateral_ok(y)
        if wheel_z is not None and len(wheel_z):
            self.detail["min_wheel_z"] = round(min(self.detail["min_wheel_z"],
                                                   float(np.min(wheel_z))), 3)

        if tilt_deg > FALL_TILT_DEG or z < FALL_Z_M:
            self.status, self.reason = "fail", (
                f"fell:tilt={tilt_deg:.1f}" if tilt_deg > FALL_TILT_DEG else f"fell:z={z:.2f}")
            return self.status

        # Side corridor / forbidden footprint: before success, no support point
        # or base may leave the y-bounded course.  This is the Part2 analogue of
        # the high-wall y-span rule: all obstacles are bounded by two parallel
        # y planes.  Wall is excluded because its terminal is crossing (and
        # post-crossing drift is allowed by the v7.2 fix).
        side_active = False
        if self.kind in {"platform", "stairs", "slope", "gap", "plum"} and wheel_y is not None:
            lane = float(self.j["lane_half_width"])
            limit = lane + SIDE_MARGIN_M
            side_active = (abs(y) > limit
                           or bool(np.any(np.abs(wheel_y) > limit)))
            if side_active:
                if self.side_since is None:
                    self.side_since = self.t
                if self.t - self.side_since >= SIDE_GRACE_S:
                    self.status, self.reason = "fail", "footprint_forbidden"
                    return self.status
            else:
                self.side_since = None

        cond = False
        kind = self.kind
        if kind == "platform":
            lo, hi = self.j["x_range"]
            top = float(self.j["top_z"])
            # "On the surface": bottom no more than WHEEL_ON_SURFACE_TOL_M below
            # the top and no more than WHEEL_TOP_BAND_M above it (a wheel
            # hovering high above the platform is not standing on it).
            top_cond = (lo <= x <= hi and lateral_ok
                        and wheel_z is not None
                        and np.all((wheel_z >= top - WHEEL_ON_SURFACE_TOL_M)
                                   & (wheel_z <= top + WHEEL_TOP_BAND_M)))
            if top_cond:
                self.top_seen = True
                self.top_held += dt
                if self.top_held >= float(self.j["hold_s"]) + TERMINAL_HOLD_EXTRA_S:
                    self.top_ok = True
            else:
                self.top_held = 0.0
            exit_x = float(self.j.get("exit_x_min", float(hi) + 0.10))
            # Traverse semantics (v7): climb + hold, then all wheels must leave
            # the platform far edge and be near/on the ground.  A walking robot
            # may lift one wheel, so do not require four simultaneous contacts.
            cond = (self.top_ok and lateral_ok and wheel_z is not None
                    and wheel_x is not None and x >= exit_x
                    and np.all(wheel_x >= float(hi) + 0.05)
                    and np.all(wheel_z <= 0.20)
                    and np.min(wheel_z) >= -0.10
                    and 0.30 <= z <= 0.80)
            self.exited = bool(self.exited or cond)
        elif kind == "stairs":
            top = float(self.j["top_z"])
            end_x = float(self.j["descend_end_x"])
            z_min = float(self.j.get("normal_z_min", 0.30))
            z_max = float(self.j.get("normal_z_max", 0.80))
            # Climb evidence: all four wheels on the top landing at some point.
            if (wheel_z is not None and np.all(wheel_z >= top - WHEEL_ON_SURFACE_TOL_M)
                    and x >= float(self.j["goal_x_min"])):
                self.top_seen = True
            # Terminal state: descended the far flight, all wheels on the ground,
            # and the trunk is back in its normal walking-height range.
            cond = (self.top_seen and lateral_ok and wheel_z is not None
                    and wheel_x is not None
                    and np.all(wheel_x >= end_x + STEP_DESCEND_CLEARANCE_M)
                    and np.all(wheel_z <= 0.20)
                    and np.min(wheel_z) >= -0.10
                    and x >= end_x + STEP_DESCEND_CLEARANCE_M
                    and z_min <= z <= z_max)
        elif kind == "slope":
            rise = float(self.j["rise_z"])
            top_cond = (x >= float(self.j["goal_x_min"]) and lateral_ok
                        and wheel_z is not None
                        and np.all((wheel_z >= rise - WHEEL_ON_SURFACE_TOL_M)
                                   & (wheel_z <= rise + 0.20)))
            if top_cond:
                self.top_seen = True
                self.top_held += dt
                if self.top_held >= float(self.j["hold_s"]) + TERMINAL_HOLD_EXTRA_S:
                    self.top_ok = True
            else:
                self.top_held = 0.0
            exit_x = float(self.j.get("exit_x_min", float(self.j["goal_x_min"]) + 2.5))
            cond = (self.top_ok and lateral_ok and wheel_z is not None
                    and wheel_x is not None and x >= exit_x
                    and np.all(wheel_x >= exit_x - 0.05)
                    and np.all(np.abs(wheel_z) <= float(self.j.get("exit_ground_tol", GROUND_WHEEL_TOL_M)))
                    and 0.30 <= z <= 0.80)
            self.exited = bool(self.exited or cond)
        elif kind == "wall":
            wx_wall = float(self.j["wall_x"])
            if wheel_x is not None and wheel_y is not None and wheel_z is not None:
                # Spec high-wall rule: every support point must cross the wall plane with y
                # inside the actual wall span; side/end crossings are a hard failure. No z
                # threshold is used (y span + MuJoCo collision exclude bypass).
                span = float(self.j.get("wall_span_half", WALL_SPAN_HALF_M))
                if self._prev_wx is None:
                    self._wall_crossed_mask = np.zeros(len(wheel_x), dtype=bool)
                else:
                    delta = wheel_x - self._prev_wx
                    crossing = ((self._prev_wx - wx_wall) * (wheel_x - wx_wall) <= 0) & (
                        np.abs(delta) < 1.0)
                    safe = np.abs(delta) > 1e-9
                    frac = np.where(safe, (wx_wall - self._prev_wx) / np.where(
                        safe, delta, np.ones_like(delta)), 0.0)
                    wy_at = self._prev_wy + (wheel_y - self._prev_wy) * frac
                    inside = np.abs(wy_at) <= span
                    self._wall_crossed_mask |= crossing & inside
                    if np.any(crossing & ~inside):
                        self._wall_side_bypass = True
                self._prev_wx = wheel_x.copy()
                self._prev_wy = wheel_y.copy()
                self.crossed = bool(np.all(self._wall_crossed_mask))
                # Post-crossing lane drift is not a side bypass: the y-span is
                # enforced at the wall plane above.  Requiring base y inside the
                # lane after crossing falsely failed real wall20 crossings that
                # drifted left after the plant (review batch 02).
                cond = (self.crossed and not self._wall_side_bypass
                        and x >= float(self.j["goal_x_min"])
                        and np.all(wheel_x >= wx_wall + WALL_BACK_CLEARANCE_M)
                        and np.all(wheel_z <= GROUND_WHEEL_TOL_M))
                if self._wall_side_bypass:
                    self.status, self.reason = "fail", "wall_side_bypass"
                    return self.status
                self.detail["wall_side_bypass"] = bool(self._wall_side_bypass)
            else:
                # Transitional fallback for traces without support y: legacy base-z proxy.
                if self.prev is not None and (self.prev - wx_wall) * (x - wx_wall) <= 0 \
                        and abs(x - self.prev) < 1.0 and lateral_ok:
                    self.crossed = True
                if abs(x - wx_wall) <= 0.8:
                    self.max_z_near = max(self.max_z_near, z)
                cond = (self.crossed and x >= float(self.j["goal_x_min"]) and lateral_ok
                        and self.max_z_near >= float(self.j["climb_z_min"])
                        and wheel_z is not None
                        and wheel_x is not None
                        and np.all(wheel_x >= wx_wall + WALL_BACK_CLEARANCE_M)
                        and np.all(wheel_z <= GROUND_WHEEL_TOL_M))
        elif kind == "gap":
            nx, fx = float(self.j["near_x"]), float(self.j["far_x"])
            depth = float(self.j["plateau_z"])
            # v7.5: the official trench level continues after the far plateau with a
            # 1 m exit ramp down to normal ground (world x = far_x + 1.0 .. far_x + 2.0;
            # the level's obstacle_bbox.max.x is exactly far_x + 2.0).  Crossing the
            # slot is necessary but not sufficient: the attempt only counts once the
            # robot has descended that ramp (all four wheels back on normal ground).
            ramp_end = float(self.j.get("exit_ramp_end", fx + 2.0))
            ground_tol = float(self.j.get("exit_ground_tol", GROUND_WHEEL_TOL_M))
            allowed = float(self.j.get("gap_max_drop_m", min(0.5 * depth, WHEEL_RADIUS_M)))
            floor_z = depth - allowed
            if self.prev is not None and abs(x - self.prev) < 1.0:
                if (self.prev - nx) * (x - nx) <= 0 and lateral_ok:
                    self.gap_near_crossed = True
                if (self.prev - fx) * (x - fx) <= 0 and lateral_ok:
                    self.gap_far_crossed = True
            if wheel_x is not None and wheel_z is not None:
                # Far-plateau evidence: all four wheels actually on the far plateau top.
                if (x > fx and np.all(wheel_x >= fx)
                        and np.all(np.abs(wheel_z - depth)
                                   <= WHEEL_ON_SURFACE_TOL_M + 0.02)):
                    self.top_seen = True
                in_slot = (wheel_x >= nx - 0.06) & (wheel_x <= fx + 0.06)
                if np.any(in_slot):
                    slot_z = float(np.min(wheel_z[in_slot]))
                    self.detail["min_wheel_z_slot"] = round(
                        min(self.detail["min_wheel_z_slot"], slot_z), 3)
                    if slot_z < floor_z - 0.01:
                        self.status, self.reason = "fail", f"gap:wheel_in_slot:z={slot_z:.2f}"
                        return self.status
            cond = (self.gap_near_crossed and self.gap_far_crossed and self.top_seen
                    and lateral_ok
                    and wheel_x is not None and wheel_z is not None
                    and x >= ramp_end + 0.10
                    and np.all(wheel_x >= ramp_end + 0.02)
                    and np.all(wheel_z <= ground_tol)
                    and float(np.min(wheel_z)) >= -0.10
                    and 0.30 <= z <= 0.80)
            self.exited = bool(self.exited or cond)
        elif kind == "plum":
            lo, hi = self.j["cube_x_range"]
            top_z = float(self.j["top_z"])
            # Cross the pile field: base over the cube window with no wheel bottom
            # below the "touched the ground" threshold (z_cross_min).
            if lo <= x <= hi and wheel_z is not None:
                if np.any(wheel_z < self.j["z_cross_min"]):
                    self.status, self.reason = "fail", (
                        f"dropped:z={float(np.min(wheel_z)):.2f}@x={x:.2f}")
                    return self.status
                self.top_seen = True
            # New owner-confirmed geometry: entry platform + 3 cubes + exit
            # platform, no ramps.  Reaching the exit platform is evidence only;
            # the top band is the tightened one (the old +0.20 m accepted a
            # robot perched above the platform as "on" it).
            if (x >= float(self.j["goal_x_min"]) and lateral_ok
                    and wheel_z is not None
                    and np.all((wheel_z >= top_z - WHEEL_ON_SURFACE_TOL_M)
                               & (wheel_z <= top_z + WHEEL_TOP_BAND_M))):
                self.exit_platform_seen = True
            # ``exit_ramp_end`` is the exit-platform far edge in the new geometry
            # (the manifest keeps the historical field name for compatibility).
            exit_edge = float(self.j["exit_ramp_end"])
            ground_tol = float(self.j["exit_ground_tol"])
            cond = (self.top_seen and self.exit_platform_seen and lateral_ok
                    and wheel_x is not None and wheel_z is not None
                    and x >= exit_edge + 0.10
                    and np.all(wheel_x >= exit_edge + 0.02)
                    and np.all(wheel_z <= ground_tol)
                    and float(np.min(wheel_z)) >= -0.10
                    and 0.30 <= z <= 0.80)
            self.exited = bool(self.exited or cond)
        else:
            raise ValueError(kind)

        self.last_t = self.t
        self.prev = x
        if side_active:
            cond = False
        if cond:
            self.held += dt
            if self.held >= float(self.j["hold_s"]) + TERMINAL_HOLD_EXTRA_S:
                self.status = "success"
        else:
            self.held = 0.0
        return self.status

    def _terminal_reason(self, fallback: str) -> str:
        """Give timeout/stopped trials a diagnostic reason instead of bare trace_end."""
        base = self.last_base
        wz = self.last_wheel_z
        kind = self.kind
        if kind == "platform":
            if not self.top_seen:
                return "not_on_platform"
            if not self.top_ok:
                return "top_not_stable"
            if not self.exited:
                return "not_exited_platform"
            if base is not None and abs(base[1]) > float(self.j["lane_half_width"]) + 0.02:
                return "out_of_bounds"
            return "not_exited_platform"
        if kind == "wall":
            return "not_crossed" if not self.crossed else "not_stable_after_crossing"
        if kind == "gap":
            if not (self.gap_near_crossed and self.gap_far_crossed):
                return "not_crossed"
            if not self.top_seen:
                return "not_on_far_plateau"
            return "not_descended"
        if kind == "stairs":
            # Distinguish "never reached the landing" from "reached the landing
            # but could not descend back to the ground" - both used to collapse
            # to the uninformative `not_completed`.
            if not self.top_seen:
                return "not_on_top_landing"
            return "not_descended"
        if kind == "slope":
            if not self.top_seen:
                return "not_on_top_landing"
            if not self.top_ok:
                return "top_not_stable"
            if not self.exited:
                return "not_exited_slope"
            return "not_exited_slope"
        if kind == "plum":
            if not self.top_seen:
                return "not_crossed"
            if not self.exit_platform_seen:
                return "not_on_exit_platform"
            if not self.exited:
                return "not_descended_exit_platform"
            return "not_descended_exit_platform"
        return fallback

    def finalize(self, reason: str = "timeout") -> dict:
        if self.status == "running":
            self.status = "fail"
            if self.side_since is not None:
                self.reason = "footprint_forbidden"
            elif not self.reason:
                self.reason = self._terminal_reason(reason)
        detail = {k: v for k, v in self.detail.items() if not k.startswith("_")}
        detail["top_seen"] = bool(self.top_seen)
        detail["exit_platform_seen"] = bool(self.exit_platform_seen)
        detail["exited"] = bool(self.exited)
        return {"status": self.status, "reason": self.reason, "detail": detail}
