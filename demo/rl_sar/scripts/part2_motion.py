"""Part 2 motion fixes, isolated from the recorded Part 1 demo controller.

Only Part 2 uses fixed-segment transit and simulation-timestamp timing. The
Part 1 route retains the implementation used by locked12_full40.
"""
from __future__ import annotations

import math
import os
import time

from part1_auto_route import RouteNode, TRANSFER_POLICY, wrap

FLAT_TRANSIT_POLICY = "flat"
FLAT_TRANSIT_SPEED = 0.50
FLAT_MIN_DISTANCE = 4.0
FLAT_BRAKE_RUNOUT = 1.20


def transit_corners(points, origin=None, tolerance=0.15):
    """Remove subdivision points on straight corridors; retain real corners.

    The optional measured origin may be slightly off the audited corridor.
    Only discard its first waypoint when the origin is already inside the
    corridor's position allowance and the waypoint lies ahead on that line.
    """
    corners = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        if corners and math.dist(corners[-1], point) < 1e-9:
            continue
        while len(corners) >= 2:
            a, b = corners[-2:]
            ux, uy = b[0] - a[0], b[1] - a[1]
            vx, vy = point[0] - b[0], point[1] - b[1]
            if abs(ux * vy - uy * vx) > 1e-8 or ux * vx + uy * vy <= 0.0:
                break
            corners.pop()
        corners.append(point)
    if origin is not None and len(corners) >= 2:
        a, b = corners[:2]
        ux, uy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(ux, uy)
        ox, oy = origin[0] - a[0], origin[1] - a[1]
        if (length > 0.0 and abs(ox * uy - oy * ux) / length <= tolerance
                and ox * ux + oy * uy <= 0.0):
            corners.pop(0)
    return corners


class Part2Motion(RouteNode):
    def __init__(self):
        self._exit_code = 0
        os.environ["PART1_ROUTE_STEPLOCK"] = "1"
        super().__init__()

    def _tick(self, fallback_s: float = 0.02) -> None:
        """Advance one control period, independently of eval publication rate."""
        if self._steplock:
            start_sim = self._sim_t
            duration = max(0.0, float(fallback_s))
            # Eval traces may arrive every 2 ms or every 20 ms.  Button holds,
            # command rates and settling durations must keep their sim-time
            # meaning at either rate, including with the realtime viewer.
            self._send_joy()
            last_sim = start_sim
            last_progress_wall = time.monotonic()
            while self._sim_t - start_sim < duration - 1e-9:
                # Clear before reading the timestamp so a final-frame event
                # cannot be lost between the readiness check and the wait.
                self._frame_evt.clear()
                now_sim = self._sim_t
                if now_sim - start_sim >= duration - 1e-9:
                    break
                if now_sim > last_sim:
                    last_sim = now_sim
                    last_progress_wall = time.monotonic()
                remaining_wall = 2.0 - (time.monotonic() - last_progress_wall)
                if remaining_wall <= 0.0:
                    raise RuntimeError("no /eval_state progress for 2 seconds")
                self._frame_evt.wait(timeout=min(0.1, remaining_wall))
            return
        self._hold(fallback_s)

    def _now(self) -> float:
        """Route clock: simulation time in step-lock, wall-clock otherwise.

        All control-loop deadlines use this clock so timeout decisions are tied
        to /eval_state frames (deterministic) rather than Python scheduling.
        """
        if self._steplock:
            return self._sim_t
        return time.monotonic()

    def _hold(self, seconds: float) -> None:
        """Hold the current output for `seconds`: sim-timed in step-lock
        (deterministic), wall-clock sleep otherwise (never recurse: the old
        implementation called itself and crashed any wall-clock route run)."""
        if not self._steplock:
            time.sleep(max(0.0, float(seconds)))
            return
        # Round fractional holds to complete control periods at every eval
        # rate, so a 150 ms button hold lasts 160 ms at both 2 ms and 20 ms
        # sampling and never publishes a faster final partial period.
        ticks = math.ceil(max(0.0, float(seconds)) / 0.02 - 1e-9)
        end_sim = self._sim_t + ticks * 0.02
        while self._sim_t < end_sim - 1e-9:
            self._tick()

    def transit_path(self, points, speed: float = 0.50) -> None:
        """Travel audited flat corridors, stopping only at their real corners.

        Long straight segments use the deployed N-W-wolf 16-DoF flat policy.
        Its braking/switching interval ends before each geometric corner;
        the locked connector completes the final 1.2 m and all short segments.
        Expert entry alignment remains a separate operation with its own gates.
        """
        p = self.pose()
        for target in transit_corners(points, (p.x, p.y)):
            p = self.pose()
            distance = math.hypot(target[0] - p.x, target[1] - p.y)
            if distance <= 0.15:
                continue
            if distance >= FLAT_MIN_DISTANCE:
                self._align_flat_transit(target)
                # The locked turn can move the centre slightly. Recompute the
                # straight segment from its stopped, measured pose.
                p = self.pose()
                distance = math.hypot(target[0] - p.x, target[1] - p.y)
                ux, uy = (target[0] - p.x) / distance, (target[1] - p.y) / distance
                brake_point = (target[0] - ux * FLAT_BRAKE_RUNOUT,
                               target[1] - uy * FLAT_BRAKE_RUNOUT)
                try:
                    self._transit_segment(brake_point, FLAT_TRANSIT_SPEED,
                                          policy=FLAT_TRANSIT_POLICY)
                finally:
                    # Flat's zero command can retain rolling motion. Reuse
                    # the measured braking/settling handover before approaching
                    # the corner, including when the flat segment has failed.
                    self.ensure_transfer(announce=True)
            self._transit_segment(target, min(abs(speed), 0.22))

    def _align_flat_transit(self, target):
        """Finish corridor turns in locked mode before loading flat.

        Flat has a stationary-yaw dead zone near the requested heading. Its
        job here is forward tracking; the measured stopped heading must be
        ready before the policy switch, including after locked-turn drift.
        """
        self.ensure_transfer()
        for attempt in range(4):
            p = self.pose()
            if self.fallen():
                break
            heading = math.atan2(target[1] - p.y, target[0] - p.x)
            err = wrap(heading - p.yaw)
            if abs(err) <= 0.08:
                return
            if attempt == 3:
                break
            timeout = max(15.0, abs(err) / 0.12 + 8.0)
            if not self.rotate_hold(heading, timeout_s=timeout):
                break
        self.stop(0.3)
        raise RuntimeError("Part2 locked heading failed before flat corridor " + str(target))

    def _transit_segment(self, target, speed, policy=None):
        p = self.pose()
        distance = math.hypot(target[0] - p.x, target[1] - p.y)
        timeout = max(20.0, 2.5 * distance / max(0.20, speed) + 12.0)
        if not self.goto(target, tol=0.15, timeout_s=timeout, speed=speed,
                         policy=policy):
            raise RuntimeError("Part2 transit failed before corridor corner " + str(target))

    def goto(self, target, tol: float = 0.12, timeout_s: float = 30.0,
             speed: float = 0.18, policy: str | None = None) -> bool:
        """Follow one straight segment and stop at its endpoint.

        Keep the segment direction fixed while correcting cross-track error.
        Point pursuit is unstable at a waypoint: a few centimetres of slip or
        stopping distance can put the point beside/behind the robot, asking the
        locked-wheel connector for a 90/180 degree turn on an otherwise straight
        transit.  Crossing the endpoint plane inside the tolerance completes the
        segment; missing it stops with failure instead of circling back.
        """
        active_policy = policy or TRANSFER_POLICY
        if policy is not None and policy != TRANSFER_POLICY:
            if policy != FLAT_TRANSIT_POLICY:
                raise ValueError("unsupported flat transit policy: " + policy)
            self.switch(policy)
        else:
            self.ensure_transfer()
        tx, ty = float(target[0]), float(target[1])
        tol, speed = max(0.001, float(tol)), abs(float(speed))
        self._cmd = f"goto(({tx:.2f},{ty:.2f}))"
        self._corr = {}
        p_cli = self.pose()
        target_yaw = math.atan2(ty - p_cli.y, tx - p_cli.x)
        ux, uy = math.cos(target_yaw), math.sin(target_yaw)
        self._set_move_cli(tx, ty, target_yaw)
        self._emit_plan()
        t0 = self._now()
        reached = False
        orienting = True
        anchor_x, anchor_y = p_cli.x, p_cli.y
        while self._now() - t0 < timeout_s:
            p = self.pose()
            if self.fallen():
                break
            ex, ey = tx - p.x, ty - p.y
            dist = math.hypot(ex, ey)
            if dist < tol:
                reached = True
                break
            if orienting:
                # Turning the locked connector can shift its centre by 15 cm.
                # Aim from the resulting position, then freeze that segment:
                # steering back to the pre-turn line cuts a short corner twice.
                bearing = math.atan2(ey, ex)
                err = wrap(bearing - p.yaw)
                if abs(err) > 0.10:
                    if active_policy == FLAT_TRANSIT_POLICY:
                        # Do not repeat flat's stalled in-place turn if the
                        # policy switch disturbed the pre-aligned heading.
                        break
                    wz_min, wz_max, wz_gain = (self._transfer_yaw_gains()
                        if active_policy == TRANSFER_POLICY else (0.12, 0.65, 1.5))
                    wz = math.copysign(
                        max(wz_min, min(wz_max, wz_gain * abs(err))), err)
                    self.command(0.0, 0.0, wz)
                    self._tick()
                    continue
                orienting = False
                anchor_x, anchor_y = p.x, p.y
                target_yaw = bearing
                ux, uy = math.cos(target_yaw), math.sin(target_yaw)
            remaining = ex * ux + ey * uy
            cross = -(p.x - anchor_x) * uy + (p.y - anchor_y) * ux
            if remaining <= 0.0 and abs(cross) <= tol:
                reached = True
                break
            if remaining <= 0.0:
                # A missed waypoint is a failed segment, never a command to
                # reverse the route and cross an obstacle from the wrong side.
                break
            # Bounded look-ahead steering corrects drift along the entire
            # segment, without letting the endpoint bearing swing sideways.
            if active_policy == FLAT_TRANSIT_POLICY:
                # Flat has little lateral response and weak yaw below 0.3.
                # Retain position-to-heading feedback as the primary lateral
                # correction; a one-metre lookahead keeps its turns shallow.
                lookahead, correction_limit = 1.0, 0.35
            else:
                lookahead = max(0.25, min(0.60, remaining))
                correction_limit = 0.55
            correction = max(-correction_limit, min(correction_limit,
                                                    math.atan2(-cross, lookahead)))
            err = wrap(target_yaw + correction - p.yaw)
            if abs(err) > 0.7:
                if active_policy == FLAT_TRANSIT_POLICY:
                    break
                wz_min, wz_max, wz_gain = (self._transfer_yaw_gains()
                    if active_policy == TRANSFER_POLICY else (0.12, 0.65, 1.5))
                wz = math.copysign(max(wz_min, min(wz_max, wz_gain * abs(err))), err)
                self.command(0.0, 0.0, wz)
            else:
                # The locked connector stops translating below ~0.10 m/s;
                # tapering to a tiny command stalls beside the waypoint while
                # its zero-command creep keeps accumulating lateral error.
                # Flat already stops at a brake point 1.2 m before the corner.
                # Keep its validated cruise command until that stopping gate;
                # slowing it first weakens lateral authority against crab.
                v = (speed if active_policy == FLAT_TRANSIT_POLICY
                     else min(speed, max(0.16, 0.5 * remaining)))
                vy = 0.0
                if active_policy == FLAT_TRANSIT_POLICY:
                    heading = wrap(p.yaw - target_yaw)
                    # Project the corridor-normal correction into the body.
                    # This assists actors with lateral authority, while yaw
                    # feedback above still closes position for flat's wheels.
                    vy = max(-0.20, min(0.20, -0.8 * cross * math.cos(heading)))
                    wz = max(-0.45, min(0.45, 2.0 * err))
                    if abs(err) > 0.06 and abs(wz) < 0.40:
                        wz = math.copysign(0.40, err)
                elif active_policy.startswith(("vt_track", "vt_locked")):
                    # The low-wz dead zone also hurts small heading corrections
                    # while driving (goto veered off the T-stairs approach).
                    wz = max(-0.40, min(0.40, 1.6 * err))
                    if abs(err) > 0.02 and abs(wz) < 0.14:
                        wz = math.copysign(0.14, err)
                else:
                    wz = max(-0.20, min(0.20, 1.0 * err))
                    if abs(err) > 0.02 and abs(wz) < 0.10:
                        wz = math.copysign(0.10, err)
                self.command(v, vy, wz)
            self._tick()
        self.stop(0.3)
        return reached and not self.fallen()
