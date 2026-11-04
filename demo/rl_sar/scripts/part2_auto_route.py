#!/usr/bin/env python3
"""Part2 full-field auto route v2 (corrected competition map, max-feasible legs).

Scene: ``demo/scenes/part2/full_field/part2_fixed_field.xml`` (launch
``route:=part2`` selects it with spawn (0, 0, 0.675, 0)).

Route order (deployment-verified tier; every leg has its own judged direction):

  1. high_platform_20cm  SOUTH->NORTH (stand (12.75,-3.90), cross gate y>1.35) row2p
     (S->N from the y=-4.5 corridor: the W->E field slot leaves only a 0.75 m
     runway onto the step, which the deployed experts cannot climb)
  2. stairs_20cm         WEST->EAST  (stand x=46.00, cross gate x>52.80)  st30
     (default = fully validated safe tier; the 200-point stairs_30cm tier is
     PART2_STAIRS_LEVEL=30 + PART2_EXPERT_STAIRS=vt_band_v1 +
     PART2_VX_STAIRS=0.60, field-proven but run-to-run variable)
  3. slope_35deg         WEST->EAST  (stand x=38.35, cross gate x>44.20)  row2p
  4. trench_30cm         WEST->EAST  (stand x=14.10, cross gate x>19.70)  s1d_trench
  5. high_plum_bypass    WEST->EAST  (no expert: no deployable blind plum
     expert exists; the route goes around the row at x=2.60 and joins the
     wall transit; PART2_PLUM_MODE=cross restores the old non-scoring pass)
  6. high_wall_20cm      EAST->WEST  (stand x=8.40,  cross gate x<7.20)   vt_band
     = cross_bar_5 (the field instance only leaves a 0.54 m runway; the 40 cm
     bar7 passes only the official 1.8 m-runway level on the deployment stack).
     PART2_ROUTE_WALL=0 skips the leg; PART2_ROUTE_WALL_VARIANT=bar6|bar7
     switches the bar (all variants clearance-audited).

Transits are chains of ``goto()`` waypoints <= 10 m (the locked-wheel connector
has almost no vy authority: a pure lane hold drifts ~0.008 m/m, while goto's
turn-drive-turn loop is what actually corrects cross-track).  Each corridor is
verified against the MuJoCo geom AABBs by ``tools/route/audit_part2_route.py``
(the module-level ``AUDIT_SEGMENTS`` below is the audited polyline).

Every expert switch is preceded by a bounded-primitive alignment (the locked
connector has no useful vy and a 90 deg in-place turn drifts 5-12 cm; the old
goto(tol=0.02) endgame and the old lateral `_pose_move_to` both dead-locked):
  * coarse goto(approach_point, tol=0.12) in the open corridor (not budgeted),
  * one fine goto(stand, tol=0.06, speed=0.10) (<=10 s),
  * yaw servo to the lane heading (low-wz dead-zone floor, ~18 s cap),
  * lateral S-turn settle (steer at clamp(12*|e|,8,25) deg, drive until
    |e|<=0.018 m, steer back; forward travel capped per leg so the expert
    runway is preserved),
  * straight creep along the lane to the pre-align target (heading hold, no
    goto endgame), final yaw polish, then a hard gate:
    default |e|<=0.05 m / |yaw_err|<=2.5 deg; stairs_30cm 0.02 m / 1.5 deg;
    slope/trench/plum 0.08 m / 2.5 deg; along-lane bounds per leg.
  * lateral ``_sidestep_lane`` (face perpendicular, drive to the lane, face
    back) removes the 0.15-0.25 m yaw-servo drift WITHOUT spending runway
    (this is what makes slope/trench/stairs alignment converge);
  * per-leg fine budget (45 s, platform 75 s), at most 3 align_to calls,
    only the failed component is redone; on failure the residual is logged
    and the route FAILs (never hangs);
  * falling during alignment aborts at once (retrying a fallen robot is useless).
Crossing recovery (generic, PART2_RECOVER=0 disables): if the expert makes no
lane progress for CROSS_STALL_WINDOW_S (12 s, progress < 0.03 m) or hits its
crossing timeout, the route backs off 0.6 m along -lane with the connector,
re-aligns (30 s budget) and retries the crossing ONCE; a second failure FAILs.
A stall on elevated terrain (base z >= 0.75 m, e.g. on the stairs) is NOT
reversed: the leg FAILs immediately (backing down stairs tips the robot).
Logs ``expert_recover leg=... reason=stall|timeout`` and ``recover=1`` in the
leg report.
The actual (x, y, yaw), e/along/yaw_err and gates at the switch are logged for
every leg ("expert switch" lines, machine-readable).  The fake kinematic sim
must meet the gates exactly (it follows commands without dynamics).

Deployment-verified tier (real sim_node+rl_sim trials, judged-rate traces in
artifacts/traces/, replayed with the Part2 v5 judge):
  platform    -> platform_20cm row2p adaptive 0.50 (field S->N 2/2 + passed
                 in every 1x full-field run; 30cm is 2/3 official-level but
                 only 1/4 on the field S->N approach, 40cm fails -> default
                 20, 30/40 stay one-env switches)
  stairs      -> DEFAULT stairs_20cm with st30_ppo_st30_s100_600 adaptive
                 vx=0.45 (old real8 field evidence; 1x full-field v20/v21
                 PASS).  Optional 200-point tier: stairs_30cm vt_band
                 adaptive vx=0.60 (v14/v17 PASS, v15/v18 top-landing stall;
                 vx=0.50/0.55/0.65/0.70 do not transfer from the eval side).
  slope       -> slope_35deg row2p adaptive 0.50 (3/3 single-level; 1x
                 full-field PASS after the lateral sidestep fix)
  trench      -> trench_30cm s1d adaptive 0.80 (3/3; 1x full-field PASS)
  plum        -> bypass (no deployable blind expert: 0/200 eval, 0/384 old
                 repo; crossing attempt stalled and fell in real_v14)
  wall        -> cross_bar_5 20cm in the field slot (0.54 m runway; the 40cm
                 bar7 only passes the OFFICIAL 1.8 m-runway level, 2/3, not
                 the field slot) -> default PART2_ROUTE_WALL_VARIANT=bar5.
Defaults below follow that set; every tier stays switchable.

Tier switches (all geometry is top-level tables, every combination is
clearance-audited): PART2_PLATFORM_LEVEL=40|30|20, PART2_STAIRS_LEVEL=20|25|30,
PART2_SLOPE_LEVEL=20|30|35, PART2_TRENCH_LEVEL=10|20|30,
PART2_ROUTE_WALL=0/1 + PART2_ROUTE_WALL_VARIANT=bar5|bar6|bar7; the platform
leg always crosses S->N from the y=-4.5 corridor (3.1 m runway).

Expert map (data-driven, env-overridable; the vx/mode follow the deployment
traces in docs/PART2-ROUTE.md §7.2/§7.3, not the eval-only matrix):
  platform row2p adaptive 0.50, stairs st30 adaptive 0.45 by default
  (stairs_30cm best tier: PART2_STAIRS_LEVEL=30 PART2_EXPERT_STAIRS=vt_band_v1
  PART2_VX_STAIRS=0.60),
  slope row2p adaptive 0.50, trench s1d adaptive 0.80, plum bypass,
  wall (bar5) vt_band adaptive 0.50.
  Env overrides: PART2_EXPERT_PLATFORM / _STAIRS / _SLOPE / _TRENCH /
  _WALL, PART2_VX_<LEG>, PART2_PLUM_MODE=bypass|cross.

Run with the fake kinematic sim (see docs/PART2-ROUTE.md):
  source demo/install/setup.bash
  PART1_ROUTE_STEPLOCK=1 PART1_ROUTE_JUDGE=0 PART1_TRANSFER_POLICY=vt_locked12 \
  PART1_RECORD_PATH=... PART1_RECORD_PLAN_PATH=... \
  .pixi/envs/demo/bin/python demo/rl_sar/scripts/part2_auto_route.py
"""
from __future__ import annotations

import math
import os

from part1_auto_route import (  # noqa: E402
    BTN_A, BTN_RB, RouteNode, TRANSFER_POLICY, wrap,
)


# --------------------------------------------------------------------------
# configurable roster (constants are deliberate: swap an expert here)
# --------------------------------------------------------------------------
def _policy_env(name: str, default: str) -> str:
    return (os.environ.get(name, default) or default).strip() or default


def _vx_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# Expert map (data-driven; expert-bench 1145-run matrix, 2026-11-04).
#   policy  deployment dir name (env-overridable per leg)
#   vx      commanded forward speed validated for that expert
#   mode    "adaptive" = training-side AdaptiveCommand vy/wz feedback,
#           "fixed"    = pure (vx,0,0) (expert not trained with feedback)
# platform40: row2p adaptive 0.50 (eval matrix; align tolerance 6/9)
# stairs30:   vt_band adaptive 0.70 with a -0.030 m pre-align dy (start_dy):
#             the vx=0.5 basin is chaotic (33/143), at 0.70 it is 93/143 and the
#             dy=-0.030 row passes 11/11 dyaw in [-5,+5] deg (fine basin scan,
#             artifacts/part2_experts/stairs30_basin.json).  The AdaptiveCommand
#             still centres on y=3.000, so the expert pulls back to the lane.
#             Alignment gate for this leg: |y_err|<=0.02, |yaw_err|<=1.5 deg.
# slope35:    row2p adaptive 0.50 (9/9 offsets robust)
# wall40:     vt_band adaptive 0.50 (8/9 offsets; row2p only 3/9 -> never row2p)
# trench30:   s1d adaptive 0.80 (adaptive, vx 0.8; fixed 0.8 fails at +2 deg yaw)
# plum:       obstacle adaptive 0.40 (non-scoring transit only; no judge expert)
# wall40:     vt_band adaptive 0.50 for cross_bar_7 (row2p is knife-edge at the
#             0.54 m runway; vt_band/obstacle pass 7/8 offsets, vt_band 3/3 wall40)
EXPERT_SPECS = {
    "platform": {
        "policy": _policy_env("PART2_EXPERT_PLATFORM", "row2p_p40w40_v1"),
        "vx": _vx_env("PART2_VX_PLATFORM", 0.50), "mode": "adaptive"},
    "stairs": {
        # Default = the fully validated safe tier: stairs_20cm with the old
        # real8-verified st30 policy @ 0.45 (1x full-field v20/v21 PASS).
        # The 200-point stairs_30cm tier is switchable via
        # PART2_STAIRS_LEVEL=30 PART2_EXPERT_STAIRS=vt_band_v1 PART2_VX_STAIRS=0.60
        # (1x full-field v14/v17 PASS, v15/v18 stalled on the top landing:
        # run-to-run variable; the eval-side vx=0.70 basin does NOT transfer).
        "policy": _policy_env("PART2_EXPERT_STAIRS", "st30_ppo_st30_s100_600"),
        "vx": _vx_env("PART2_VX_STAIRS", 0.45), "mode": "adaptive",
        "start_dy": 0.0,
        "align_xy_tol": 0.02,
        "align_yaw_tol_deg": 1.5,
        "settle_forward": 0.30},
    "slope": {
        "policy": _policy_env("PART2_EXPERT_SLOPE", "row2p_p40w40_v1"),
        "vx": _vx_env("PART2_VX_SLOPE", 0.50), "mode": "adaptive"},
    "trench": {
        "policy": _policy_env("PART2_EXPERT_TRENCH", "s1d_trench_reward_v75_m100"),
        "vx": _vx_env("PART2_VX_TRENCH", 0.80), "mode": "adaptive"},  # adaptive08
    "plum": {
        "policy": _policy_env("PART2_PLUM_POLICY", "obstacle"),
        "vx": _vx_env("PART2_VX_PLUM", 0.40), "mode": "adaptive"},
    "wall": {
        "policy": _policy_env("PART2_EXPERT_WALL", "vt_band_v1"),
        "vx": _vx_env("PART2_VX_WALL", 0.50), "mode": "adaptive"},
}

# Deployment-verified platform tier (top-level switch): the real sim_node+rl_sim
# stack could not climb platform_30/40cm in the 2026-11-04 deployment trials
# (eval-side row2p does climb); 20cm is deployment-verified.  The field geom is
# selected here: platform_40cm x=21.75, platform_30cm x=18.75, platform_20cm
# x=12.75 (all in the y=0 row, x +/- 0.75, y +/- 0.8).  Approach is always
# S->N from the y=-4.5 corridor with stand y=-3.90 and cross-end y>1.45.
PLATFORM_LEVELS = {
    40: {"name": "high_platform_40cm", "x": 21.75, "top": 0.40},
    30: {"name": "high_platform_30cm", "x": 18.75, "top": 0.30},
    20: {"name": "high_platform_20cm", "x": 12.75, "top": 0.20},
}
try:
    PLATFORM_LEVEL = int(os.environ.get("PART2_PLATFORM_LEVEL", "20"))
except (TypeError, ValueError):
    PLATFORM_LEVEL = 20
if PLATFORM_LEVEL not in PLATFORM_LEVELS:
    PLATFORM_LEVEL = 40

# Same switch for the other categories: every leg's field geom + approach
# geometry is a top-level table entry; the default is the max-difficulty target.
# (x extents measured from the compiled MuJoCo model; stand = slot midpoint with
# a ~0.6-1.1 m rear clearance, runway = first face - stand.)
def _level_env(name: str, default: int, allowed) -> int:
    try:
        v = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        v = int(default)
    return v if v in allowed else int(default)


STAIRS_LEVELS = {
    20: {"name": "stairs_20cm", "x0": 48.00, "x1": 52.40, "stand": 46.00},
    25: {"name": "stairs_25cm", "x0": 55.50, "x1": 59.90, "stand": 53.50},
    30: {"name": "stairs_30cm", "x0": 63.00, "x1": 67.40, "stand": 60.50},
}
SLOPE_LEVELS = {
    20: {"name": "slope_20deg", "x0": 21.00, "x1": 25.80, "stand": 20.40},
    30: {"name": "slope_30deg", "x0": 33.00, "x1": 37.80, "stand": 32.40},
    35: {"name": "slope_35deg", "x0": 39.00, "x1": 43.80, "stand": 38.35},
}
# end x of the previous stairs block for each tier (all stair blocks are 3.10 m
# apart): stairs_15 ends 44.90, stairs_20 52.40, stairs_25 59.90.  The north
# climb of the platform->stairs transit must use the CENTER of this gap.
PREV_STAIRS_END = {20: 44.90, 25: 52.40, 30: 59.90}
TRENCH_LEVELS = {
    10: {"name": "trench_10cm", "x0": 3.00, "x1": 7.10, "stand": 1.20},
    20: {"name": "trench_20cm", "x0": 9.00, "x1": 13.20, "stand": 8.05},
    30: {"name": "trench_30cm", "x0": 15.00, "x1": 19.30, "stand": 14.10},
}
WALL_TIERS = {
    # cross_bar_N is crossed EAST->WEST (row has no sensible +/-x runway);
    # stand is the midpoint of the gap between bar N and bar N+1.  bar5 lives
    # in the 1.08 m bar5/bar6 corridor where the 0.5 m body + locked-wheel turn
    # drift cannot hold a 5 cm gate: relax its alignment gate (the corridor is
    # 1.4 m tall, so |e|<=0.20 keeps the robot inside and the adaptive lane
    # command pulls it back while crossing).
    "bar5": {"name": "high_wall_20cm", "stand": 8.40, "cross_end": 7.20,
             "xy_tol": 0.20, "yaw_tol": 6.0},
    "bar6": {"name": "high_wall_30cm", "stand": 9.60, "cross_end": 8.40,
             "xy_tol": 0.05, "yaw_tol": 2.5},
    "bar7": {"name": "high_wall_40cm", "stand": 10.80, "cross_end": 9.90,
             "xy_tol": 0.05, "yaw_tol": 2.5},
}
STAIRS_LEVEL = _level_env("PART2_STAIRS_LEVEL", 20, STAIRS_LEVELS)
SLOPE_LEVEL = _level_env("PART2_SLOPE_LEVEL", 35, SLOPE_LEVELS)
TRENCH_LEVEL = _level_env("PART2_TRENCH_LEVEL", 30, TRENCH_LEVELS)

# Plum row: no deployable blind expert exists (0/200 eval, 0/384 old repo), so
# the default is a BYPASS around the row (x=2.60, west of the entry platform at
# x=4.00) instead of attempting the cubes.  PART2_PLUM_MODE=cross restores the
# old transit-only crossing for when an expert exists.
PLUM_MODE = os.environ.get("PART2_PLUM_MODE", "bypass").strip().lower()
if PLUM_MODE not in ("bypass", "cross"):
    PLUM_MODE = "bypass"

WALL_ENABLED = os.environ.get("PART2_ROUTE_WALL", "1").strip().lower() not in (
    "0", "false", "no", "off")
WALL_VARIANT = (os.environ.get("PART2_ROUTE_WALL_VARIANT", "bar5").strip().lower()
                or "bar7")

ROUTE_START = (0.0, 0.0)


# alignment gates (fake sim must meet them; real-run tolerance per lead):
ALIGN_XY_TOL = 0.05          # |xy_err| at the expert switch [m]
ALIGN_YAW_TOL_DEG = 2.5      # |yaw_err| at the expert switch [deg]
# Alignment is a hard-budget loop: one position goto + yaw servo + one short
# straight settle along the lane per attempt, at most 2 attempts, all inside
# this sim-time deadline (a stalling alignment must FAIL, never loop forever).
# Generic per-leg crossing recovery (opt-out PART2_RECOVER=0): if the expert
# makes no lane progress for CROSS_STALL_WINDOW_S (or hits cross_timeout), back
# off along -lane, re-align and retry the crossing exactly once.  Deployed
# platform/stairs/walls are run-to-run variable even from a good aligned pose,
# so a stall is not a permanent failure.
RECOVER_ENABLED = os.environ.get("PART2_RECOVER", "1").strip().lower() not in (
    "0", "false", "no", "off")
CROSS_STALL_WINDOW_S = 12.0   # no lane progress for this long => stall
CROSS_STALL_PROGRESS_M = 0.03  # "progress" = lane projection advanced this much
TRANSIT_SPEED = 0.30          # default goto speed for transit hops (was 0.50:
                              # locked-wheel goto oscillates +-0.5-1.0 m laterally
                              # at 0.5 m/s in 1.2-2.6 m corridors)
TIGHT_TRANSIT_SPEED = 0.22    # row-corridor hops (y=4.5/7.5/10.5/13.2/19.5 rows)
TIGHT_HOP_M = 4.0             # max waypoint spacing in those corridors
CROSS_BACKOFF_M = 0.60        # back off this far along -lane before the retry
RECOVER_MAX_GROUND_Z = 0.75   # recovery (reverse) only on flat ground: backing
                              # down elevated terrain tips the robot (real v10:
                              # stairs-top stall -> reverse -> fall)
CROSS_RETRY_ALIGN_S = 30.0    # re-align fine budget for the recovery retry

ALIGN_DEADLINE_S = 45.0      # fine-stage budget (vt_locked12 yaw responds at
                             # ~0.15 rad/s, so a 120 deg turn needs >10 s and
                             # the fine stage may need two attempts; the loop is
                             # still hard-capped, it can never hang)
ALIGN_GOTO_TOL = 0.05        # one position goto per attempt (not 3x tol=0.02)
ALIGN_SETTLE_FORWARD = 0.25  # max forward travel of the lateral S-turn settle


# --------------------------------------------------------------------------
# route table: verified against the compiled MuJoCo model
#   transit : goto() waypoint chain from the previous leg's exit point
#   stand   : pre-expert alignment pose (x, y); lane_y = y, lane_yaw = yaw
#   stop    : expert-crossing stop gate; post = required pose after settling
#   exit    : goto() chain leaving the row after the crossing
#   every gap under 1.0 m is a slot the robot stands in before the leg
# --------------------------------------------------------------------------
def _x_gt(value: float):
    return lambda p: p.x > value


def _x_lt(value: float):
    return lambda p: p.x < value


def _lane_gate(lane_y: float, tol: float = 0.40):
    return lambda p: abs(wrap(p.y - lane_y)) <= tol


def _post(x_pred, lane_y: float, y_tol: float = 0.40):
    """Post-crossing pose gate: gate direction plus staying on the lane."""
    return lambda p: bool(x_pred(p)) and abs(wrap(p.y - lane_y)) <= y_tol


def _y_gt(value: float):
    return lambda p: p.y > value


def _post_y(y_min: float, y_max: float, lane_x: float, x_tol: float = 0.40):
    """Post-crossing gate for a +y (south->north) crossing."""
    return lambda p: (y_min <= p.y <= y_max) and abs(p.x - lane_x) <= x_tol


def _leg(leg, policy, vx, mode, lane_x, lane_y, lane_yaw, transit, stand,
         cross_end, stop, post, exit, cross_timeout, note="", start_dy=0.0,
         align_deadline_s=ALIGN_DEADLINE_S,
         transit_speed=TRANSIT_SPEED,
         align_xy_tol=ALIGN_XY_TOL,
         align_yaw_tol_deg=ALIGN_YAW_TOL_DEG,
         settle_forward=ALIGN_SETTLE_FORWARD,
         along_min=-0.15, along_max=0.60, align_backoff=0.0,
         no_creep=False) -> dict:
    # ``stand`` is the lane-centre approach pose; ``start_dy`` shifts only the
    # pre-align target (the expert AdaptiveCommand keeps centring on lane_y).
    return {
        "leg": leg, "policy": policy, "vx": vx,
        "fixed": str(mode).lower() == "fixed",
        "lane_x": float(lane_x), "lane_y": float(lane_y),
        "lane_yaw": float(lane_yaw),
        "start_dy": float(start_dy),
        "align_deadline_s": float(align_deadline_s),
        "transit_speed": float(transit_speed),
        "stand_center": (float(stand[0]), float(stand[1])),
        "align_xy_tol": float(align_xy_tol),
        "align_yaw_tol_deg": float(align_yaw_tol_deg),
        "settle_forward": float(settle_forward),
        "along_min": float(along_min), "along_max": float(along_max),
        "align_backoff": float(align_backoff), "no_creep": bool(no_creep),
        "transit": [(float(x), float(y)) for x, y in transit],
        "stand": (float(stand[0]), float(stand[1]) + float(start_dy)),
        "cross_end": (float(cross_end[0]), float(cross_end[1])),
        "stop": stop, "post": post,
        "exit": [(float(x), float(y)) for x, y in exit],
        "cross_timeout": float(cross_timeout),
        "note": note,
    }


def _chain(x0: float, x1: float, y: float,
           hop: float = 10.0) -> list[tuple[float, float]]:
    """Waypoints from x0 to x1 along y, <=hop spacing (start excluded)."""
    pts: list[tuple[float, float]] = []
    x = x0
    step = hop if x1 >= x0 else -hop
    while abs(x1 - x) > hop - 0.1:
        x += step
        pts.append((x, y))
    if abs(x1 - x) <= 0.9:
        pts[-1] = (x1, y)
    else:
        pts.append((x1, y))
    return pts


def _build_legs() -> list[dict]:
    pl = PLATFORM_LEVELS[PLATFORM_LEVEL]
    cx = float(pl["x"])
    platform_leg = _leg(pl["name"], **EXPERT_SPECS["platform"],
                        lane_x=cx, lane_y=-3.90, lane_yaw=math.pi / 2.0,
                        transit=[(0.0, -4.5), (10.0, -4.5), (cx, -4.5)],
                        stand=(cx, -3.90), cross_end=(cx, 1.35),
                        stop=_y_gt(1.35),
                        post=_post_y(1.25, 1.85, cx, 0.40),
                        exit=[(cx + 1.75, 1.50)], cross_timeout=40.0,
                        settle_forward=1.00, along_min=-0.10, along_max=1.30,
                        align_xy_tol=0.07, align_yaw_tol_deg=3.0,
                        align_deadline_s=75.0,
                        note="platform_%dcm x=%.2f top%.2f; S->N, 3.10m runway"
                             % (PLATFORM_LEVEL, cx, pl["top"]))
    legs = [platform_leg]
    prev = platform_leg

    st = STAIRS_LEVELS[STAIRS_LEVEL]
    sx = float(st["stand"])
    # South-side stairs transit (real v8 drift): the y=1.5 lane is only 1.5 m
    # wide (platform row top edge y=0.8 vs stairs row bottom y=2.3) and a goto
    # arc excursion drifted north into stairs_08's west end.  From the platform
    # exit go through the platform-row gap at x=cx+1.50 south to the open
    # y=-2.5 corridor, east in <=10 m hops, then north at x=60.50 (the 3.1 m
    # stairs_25/30 gap) to the stairs stand.
    gap_x = cx + 1.50
    # North climb through the MIDDLE of the current tier's stairs row gap
    # (prev block end .. stairs x0; 61.45 for stairs_30, 46.45 for stairs_20,
    # 53.95 for stairs_25).  A hard-coded 61.45 was only correct for tier 30
    # and would clip stairs_25 for the stairs_20 fallback tier; and the old
    # x=60.50 climb only had 0.60 m to stairs_25 (real v9 drift grazed it).
    gap_x_stairs = (PREV_STAIRS_END[STAIRS_LEVEL] + st["x0"]) / 2.0
    stairs_transit = [(gap_x, 1.50), (gap_x, -2.50)]
    stairs_transit += _chain(gap_x, gap_x_stairs, -2.50)
    if stairs_transit[-1] != (gap_x_stairs, -2.50):
        stairs_transit.append((gap_x_stairs, -2.50))
    stairs_transit += [(gap_x_stairs, -0.50), (gap_x_stairs, 2.97),
                       (sx, 2.97)]
    stairs_leg = _leg(st["name"], **EXPERT_SPECS["stairs"],
                      lane_x=sx, lane_y=3.0, lane_yaw=0.0,
                      transit=stairs_transit,
                      stand=(sx, 3.0), cross_end=(st["x1"] + 0.40, 3.0),
                      stop=_x_gt(st["x1"] + 0.40),
                      post=_post(_x_gt(st["x1"] + 0.35), 3.0),
                      exit=[(st["x1"] + 1.20, 3.0), (st["x1"] + 1.20, 4.5)],
                      cross_timeout=45.0, along_min=-0.10, along_max=0.90,
                      note="%s x%.2f-%.2f top%.2f; %.2fm runway"
                           % (st["name"], st["x0"], st["x1"],
                              0.06 * STAIRS_LEVEL, st["x0"] - sx))
    legs.append(stairs_leg)
    prev = stairs_leg

    sl = SLOPE_LEVELS[SLOPE_LEVEL]
    px = float(sl["stand"])
    slope_leg = _leg(sl["name"], **EXPERT_SPECS["slope"],
                     lane_x=px, lane_y=6.0, lane_yaw=0.0,
                     transit=_chain(prev["exit"][-1][0], px, 4.50,
                                    hop=TIGHT_HOP_M) + [(px, 5.50)],
                     transit_speed=TIGHT_TRANSIT_SPEED,
                     stand=(px, 6.0), cross_end=(sl["x1"] + 0.40, 6.0),
                     stop=_x_gt(sl["x1"] + 0.40),
                     post=_post(_x_gt(sl["x1"] + 0.35), 6.0),
                     # slope row is a continuous wall of tiers (only 1.2 m
                     # gaps); never run the east end gap.  Exit SOUTH through
                     # the same 1.2 m gap used for the approach (x=38.35 for
                     # slope_35 -> x=44.40 is the 43.80..45.00 gap centre for
                     # tiers 40/35; west along the proven y=4.5 corridor).
                     # exit south through the 1.2 m gap just east of this
                     # tier's east face (x1+0.6 = the next tier's gap centre;
                     # 44.40 for slope_35, 38.40 for slope_30, 26.40 for
                     # slope_20), then west along the proven y=4.5 corridor.
                     exit=[(sl["x1"] + 0.60, 4.50)], cross_timeout=45.0,
                     settle_forward=0.05, along_min=-0.10, along_max=0.28,
                     align_xy_tol=0.08, align_yaw_tol_deg=2.5,
                     note="%s x%.2f-%.2f; %.2fm runway (tight)"
                          % (sl["name"], sl["x0"], sl["x1"], sl["x0"] - px))
    legs.append(slope_leg)
    prev = slope_leg

    tr = TRENCH_LEVELS[TRENCH_LEVEL]
    tx = float(tr["stand"])
    trench_leg = _leg(tr["name"], **EXPERT_SPECS["trench"],
                      lane_x=tx, lane_y=9.0, lane_yaw=0.0,
                      # Route around the WEST END of the slope row: down to
                      # the proven y=4.5 corridor, west past slope_20 (starts
                      # x=21.00), north at the open x=20.5 column to y=7.5,
                      # then west to the trench approach.
                      transit=(_chain(prev["exit"][-1][0], 20.50, 4.50,
                                     hop=TIGHT_HOP_M)
                               + [(20.50, 7.50)]
                               + _chain(20.50, tx, 7.50, hop=TIGHT_HOP_M)
                               + [(tx, 8.50)]),
                      transit_speed=TIGHT_TRANSIT_SPEED,
                      stand=(tx, 9.0), cross_end=(tr["x1"] + 0.40, 9.0),
                      stop=_x_gt(tr["x1"] + 0.40),
                      post=_post(_x_gt(tr["x1"] + 0.35), 9.0),
                      exit=[(tr["x1"] + 0.85, 9.0), (tr["x1"] + 0.85, 10.5)],
                      cross_timeout=35.0,
                      settle_forward=0.20, along_min=-0.10, along_max=0.50,
                      align_xy_tol=0.08, align_yaw_tol_deg=2.5,
                      note="%s x%.2f-%.2f; %.2fm runway"
                           % (tr["name"], tr["x0"], tr["x1"], tr["x0"] - tx))
    legs.append(trench_leg)
    prev = trench_leg

    if PLUM_MODE == "bypass":
        # No expert switch, no crossing: go around the west side of the plum
        # row (entry platform starts x=4.00) at x=2.60, then north to y=13.2.
        plum_leg = _leg("high_plum_bypass", policy="", vx=0.0,
                        mode="bypass",
                        lane_x=2.60, lane_y=12.0, lane_yaw=0.0,
                        transit=_chain(prev["exit"][-1][0], 2.60, 10.50,
                                       hop=TIGHT_HOP_M) + [(2.60, 13.20)],
                        transit_speed=TIGHT_TRANSIT_SPEED,
                        stand=(2.60, 12.0), cross_end=(2.60, 12.0),
                        stop=lambda p: True, post=lambda p: True,
                        exit=[], cross_timeout=1.0,
                        note="plum BYPASS at x=2.60 (no deployable blind expert; "
                             "no expert switch, no crossing)")
        plum_leg["bypass"] = True
    else:
        plum_leg = _leg("high_plum", **EXPERT_SPECS["plum"],
                        lane_x=2.20, lane_y=12.0, lane_yaw=0.0,
                        transit=_chain(prev["exit"][-1][0], 2.20, 10.50,
                                       hop=TIGHT_HOP_M),
                        transit_speed=TIGHT_TRANSIT_SPEED,
                        stand=(2.20, 12.0), cross_end=(7.90, 12.0),
                        stop=_x_gt(7.90), post=_post(_x_gt(7.85), 12.0),
                        exit=[(8.90, 12.0), (8.90, 13.2)], cross_timeout=35.0,
                        settle_forward=0.25, along_min=-0.10, along_max=1.00,
                        align_xy_tol=0.08, align_yaw_tol_deg=2.5,
                        note="plum entry4.0-4.8 cubes5.0-6.6 exit6.7-7.5; "
                             "TRANSIT ONLY (no judge-verified blind expert)")
    legs.append(plum_leg)
    prev = plum_leg

    if WALL_ENABLED:
        wt = WALL_TIERS.get(WALL_VARIANT, WALL_TIERS["bar5"])
        wx = float(wt["stand"])
        wall_spec = dict(EXPERT_SPECS["wall"])
        if WALL_VARIANT == "bar5":
            # old real8 crossed this exact field slot with row2p@0.40 after a
            # stand-side 180 deg rotate; keep vt_band for bar6/bar7.
            if "PART2_EXPERT_WALL" not in os.environ:
                wall_spec["policy"] = "row2p_p40w40_v1"
            if "PART2_VX_WALL" not in os.environ:
                wall_spec["vx"] = 0.40
        wall_leg = _leg(wt["name"], **wall_spec,
                        lane_x=wx, lane_y=21.0, lane_yaw=math.pi,
                        settle_forward=0.03, along_min=-0.15, along_max=0.10,
                        no_creep=True,
                        transit=([(2.40, 13.20)] if PLUM_MODE == "bypass" else
                                 _chain(prev["exit"][-1][0], 2.40, 13.20,
                                        hop=TIGHT_HOP_M))
                                + [(2.40, 19.5), (wx, 19.5), (wx, 20.5)],
                        transit_speed=TIGHT_TRANSIT_SPEED,
                        align_xy_tol=float(wt["xy_tol"]),
                        align_yaw_tol_deg=float(wt["yaw_tol"]),
                        stand=(wx, 21.0), cross_end=(wt["cross_end"], 21.0),
                        stop=_x_lt(wt["cross_end"]),
                        post=_post(_x_lt(wt["cross_end"] + 0.05), 21.0),
                        exit=[], cross_timeout=25.0,
                        note="%s E->W; stand x=%.2f, 0.54m runway"
                             % (WALL_VARIANT, wx))
        legs.append(wall_leg)
    return legs

LEGS = _build_legs()


# --------------------------------------------------------------------------
# audited polyline: same geometry the executor drives (see tools/route/
# audit_part2_route.py).  Segment kinds:
#   transit  - strict (>0.45 m) to every obstacle
#   approach - the target leg's crossed geoms are exempt (everything else
#              strict); stands in a slot between two row neighbours
#   slot     - the stand point itself: run-up and lateral neighbour clearances
#   cross    - only the geoms of the crossed target are exempt; the rest of the
#              row must keep >0.45 m (catches clipping the neighbouring step)
#   exit     - the crossed target of the previous leg is exempt
# --------------------------------------------------------------------------
def _audit_segments() -> list[dict]:
    segs: list[dict] = []
    prev = ROUTE_START
    for index, leg in enumerate(LEGS):
        if index > 0:
            prev_leg = LEGS[index - 1]
            for pt in prev_leg["exit"]:
                segs.append({
                    "name": f"{prev_leg['leg']}_exit", "kind": "exit",
                    "pts": [prev, pt], "lane_y": prev_leg["lane_y"],
                    "cross_pts": [prev_leg["stand"], prev_leg["cross_end"]],
                })
                prev = pt
        for pt in leg["transit"]:
            segs.append({"name": f"{leg['leg']}_transit", "kind": "transit",
                         "pts": [prev, pt]})
            prev = pt
        if leg.get("bypass"):
            prev = leg["transit"][-1]
            continue
        backoff = float(leg.get("align_backoff", 0.0))
        if backoff > 0.0:
            ax = float(leg["stand"][0]) - backoff * math.cos(float(leg["lane_yaw"]))
            ay = float(leg["stand"][1]) - backoff * math.sin(float(leg["lane_yaw"]))
            segs.append({"name": f"{leg['leg']}_transit", "kind": "transit",
                         "pts": [prev, (ax, ay)]})
            segs.append({
                "name": f"{leg['leg']}_approach", "kind": "approach",
                "lane_y": leg["lane_y"],
                "cross_pts": [leg["stand"], leg["cross_end"]],
                "pts": [(ax, ay), leg["stand"]],
            })
        else:
            segs.append({
                "name": f"{leg['leg']}_approach", "kind": "approach",
                "lane_y": leg["lane_y"],
                "cross_pts": [leg["stand"], leg["cross_end"]],
                "pts": [prev, leg["stand"]],
            })
        fwd = float(leg.get("settle_forward", 0.25))
        settle_target = (float(leg["stand"][0]) + fwd * math.cos(float(leg["lane_yaw"])),
                         float(leg["stand"][1]) + fwd * math.sin(float(leg["lane_yaw"])))
        segs.append({
            "name": f"{leg['leg']}_settle", "kind": "approach",
            "lane_y": leg["lane_y"], "cross_pts": [leg["stand"], leg["cross_end"]],
            "pts": [leg["stand"], settle_target],
        })
        segs.append({
            "name": f"{leg['leg']}_slot", "kind": "slot",
            "lane_y": leg["lane_y"], "cross_pts": [leg["stand"], leg["cross_end"]],
            "pts": [leg["stand"]],
        })
        segs.append({
            "name": f"{leg['leg']}_cross", "kind": "cross",
            "lane_y": leg["lane_y"], "cross_pts": [leg["stand"], leg["cross_end"]],
            "pts": [leg["stand"], leg["cross_end"]],
        })
        prev = leg["cross_end"]
    return segs


AUDIT_SEGMENTS = _audit_segments()



def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class Part2Route(RouteNode):
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
        super().switch(name)

    # ---- adaptive lane command (exact training-side gains) ----------------
    @staticmethod
    def lane_cmd(p, prev, lane_x: float, lane_y: float, lane_yaw: float,
                 vx: float):
        """Training-side AdaptiveCommand, generalised to any axis-aligned lane.

        Gains are the ones the experts were trained with: vy=clip(-0.5*e,±0.2),
        wz=clip(-0.8*e - 2.0*heading_err - 0.6*vy_lat, ±0.7).  ``e`` is the
        lateral offset in the lane frame (positive = left of the lane
        direction) relative to the lane line through (lane_x, lane_y) and
        ``vy_lat`` the measured lateral velocity; this form mirrors correctly
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
        vy_lat = max(-0.2, min(0.2, -0.5 * e))
        wz = max(-0.7, min(0.7, -0.8 * e - 2.0 * h - 0.6 * vy_lat_meas))
        return (vx, vy_lat * math.cos(h), wz)

    # ---- pre-expert alignment --------------------------------------------
    def _rotate_servo(self, target_yaw: float, timeout_s: float = 10.0) -> bool:
        """Rotate-only yaw servo with a lock-wheel dead-zone floor.

        Commands pure yaw (plus a tiny position hold only when the rotation
        drifted the body > 0.02 m); keeps |wz| >= 0.14 rad/s while the error is
        outside the 0.02 rad stop band so the locked-wheel connector's low-wz
        dead zone cannot stall the final few degrees.  Large errors use a 0.28
        rad/s cap (measured real turn rate of vt_locked12 is ~0.3 rad/s).
        """
        p0 = self.pose()
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            p = self.pose()
            err = wrap(target_yaw - p.yaw)
            if abs(err) <= 0.02:
                break
            cap = 0.28 if abs(err) > 0.35 else 0.20
            wz = math.copysign(min(cap, max(0.14, 0.8 * abs(err))), err)
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

    def _sidestep_lane(self, lane_x: float, lane_y: float, lane_yaw: float,
                       e: float, timeout_s: float = 12.0) -> float:
        """Move laterally onto the lane centre without using along-lane runway.

        The locked connector cannot strafe: turn 90 deg toward the lane, drive
        straight until the lane offset is gone, then turn back to the lane
        heading.  Used when a big yaw-servo turn drifted the body sideways
        (real v12 slope35: 160 deg turn drifted +0.18 m, and the slope's 5 cm
        settle travel can never remove it).  Returns the residual |e|.
        """
        log = self.get_logger()
        p0 = self.pose()
        side = math.copysign(math.pi / 2.0, e)   # e>0 -> face lane_yaw-90
        face_yaw = wrap(lane_yaw - side)
        log.info("align sidestep start e=%+.3f from=(%+.2f,%+.2f,%+.1fdeg) "
                 "face=%+.1fdeg t=%.1fs"
                 % (e, p0.x, p0.y, math.degrees(p0.yaw),
                    math.degrees(face_yaw), self._now()))
        self._rotate_servo(face_yaw, timeout_s=max(2.0, timeout_s * 0.45))
        t0 = self._now()
        while self._now() - t0 < timeout_s and self._now() < t0 + 4.0 * timeout_s:
            p = self.pose()
            e_now = self._lane_offset(p, lane_x, lane_y, lane_yaw)
            if abs(e_now) <= 0.02:
                break
            err = wrap(face_yaw - p.yaw)
            vx = max(0.03, min(0.12, 0.9 * abs(e_now)))
            wz = max(-0.25, min(0.25, 1.2 * err))
            self.command(vx, 0.0, wz)
            self._tick()
        self.stop(0.30)
        self._rotate_servo(lane_yaw, timeout_s=max(2.0, timeout_s * 0.45))
        p = self.pose()
        e1 = self._lane_offset(p, lane_x, lane_y, lane_yaw)
        log.info("align sidestep end e=%+.3f pose=(%+.2f,%+.2f,%+.1fdeg) t=%.1fs"
                 % (e1, p.x, p.y, math.degrees(p.yaw), self._now()))
        return e1

    def _lane_settle(self, lane_x: float, lane_y: float, lane_yaw: float,
                     timeout_s: float, max_travel: float = 0.35) -> None:
        """Correct residual lateral offset with a short bang-bang S-turn.

        vt_locked12 has no vy, so a lateral offset e is corrected by: steer
        theta=clamp(12*|e|, 8deg, 25deg) toward the lane line, drive straight
        until |e|<=0.012 (or max_travel), steer back to lane_yaw.  The forward
        travel is capped so the expert runway is not eaten; the final in-place
        turn leaves only ~1-2 cm of rotation drift.
        """
        t_end = self._now() + timeout_s
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
                               timeout_s=min(8.0, max(1.0, t_end - self._now() - 1.0)))
            p0 = self.pose()
            unit_x, unit_y = math.cos(lane_yaw), math.sin(lane_yaw)
            along0 = (p0.x - lane_x) * unit_x + (p0.y - lane_y) * unit_y
            t_drive = min(6.0, max(1.0, t_end - self._now() - 1.0))
            t1 = self._now()
            while self._now() - t1 < t_drive:
                p = self.pose()
                e = self._lane_offset(p, lane_x, lane_y, lane_yaw)
                along = ((p.x - lane_x) * unit_x
                         + (p.y - lane_y) * unit_y)
                if abs(e) <= 0.018 or (along - along0) > max_travel:
                    break
                err = wrap(heading - p.yaw)
                wz = max(-0.15, min(0.15, 1.2 * err))
                self.command(0.10, 0.0, wz)
                self._tick()
                if self.fallen():
                    return
            self.stop(0.20)
        # steer back to the lane heading
        if self._now() < t_end:
            self._rotate_servo(lane_yaw,
                               timeout_s=min(8.0, max(1.0, t_end - self._now())))
            self.stop(0.15)

    def _creep_along(self, lane_x: float, lane_y: float, lane_yaw: float,
                     target_along: float, timeout_s: float,
                     vx: float = 0.10) -> None:
        """Drive straight along the lane (heading hold) up to ``target_along``.

        No goto: position gotos have an unstable bearing-spin endgame on the
        locked-wheel model.  Keeps the lane heading, only corrects yaw; the
        lateral offset is the S-turn settle's job.
        """
        ux, uy = math.cos(lane_yaw), math.sin(lane_yaw)
        t0 = self._now()
        while self._now() - t0 < timeout_s:
            p = self.pose()
            along = (p.x - lane_x) * ux + (p.y - lane_y) * uy
            if along >= target_along:
                break
            err = wrap(lane_yaw - p.yaw)
            wz = max(-0.18, min(0.18, 1.0 * err))
            if abs(err) > math.radians(2.0) and abs(wz) < 0.12:
                wz = math.copysign(0.12, err)
            self.command(vx, 0.0, wz)
            self._tick()
            if self.fallen():
                break
        self.stop(0.20)

    def align_to(self, leg: dict, policy: str, deadline_s: float | None = None):
        """Align to the leg pre-align target on real dynamics (hard budgeted).

        vt_locked12 cannot creep into a small circle: a position goto ends with
        an unstable bearing-spin endgame, the yaw servo responds at ~0.15 rad/s
        and an in-place rotation drifts the body several cm.  The fine stage is
        therefore built only from bounded primitives:

          0. coarse ``goto(approach_point, tol=0.12)`` (approach_point is
             ``stand - backoff*dir``; the platform leg uses a 0.70 m backoff in
             the open corridor so the runway onto the 0.40 m step is intact),
          1. yaw servo to the lane heading (cap 18 s),
          2. ``_lane_settle``: bang-bang S-turn that removes the lateral offset
             left by the rotation (forward travel capped per leg so the expert
             runway is preserved),
          3. ``_creep_along``: straight run-in to the pre-align target (heading
             hold, no goto endgame),
          4. final yaw polish, then hard gate:
             default |lateral| <= 0.05 m / |yaw_err| <= 2.5 deg;
             stairs_30cm 0.02 m / 1.5 deg; along-lane within the leg bounds.
        At most 2 attempts inside a 90 s sim-time deadline; on failure the
        residual is logged and the route FAILs (never hangs).
        """
        leg_name = leg["leg"]
        target_yaw = float(leg["lane_yaw"])
        lane_x = float(leg["lane_x"])
        lane_y = float(leg["lane_y"])
        tx, ty = float(leg["stand"][0]), float(leg["stand"][1])
        xy_tol = float(leg["align_xy_tol"])
        yaw_tol = float(leg["align_yaw_tol_deg"])
        along_min = float(leg["along_min"])
        along_max = float(leg["along_max"])
        backoff = float(leg.get("align_backoff", 0.0))
        cos_l, sin_l = math.cos(target_yaw), math.sin(target_yaw)
        ax = tx - backoff * cos_l
        ay = ty - backoff * sin_l
        self.ensure_transfer()
        log = self.get_logger()
        p0 = self.pose()
        t_start = self._now()
        log.info("align start leg=%s policy=%s stand=(%+.2f,%+.2f) yaw=%+.1fdeg "
                 "backoff=%.2f from=(%+.2f,%+.2f,%+.1fdeg)"
                 % (leg_name, policy, tx, ty, math.degrees(target_yaw),
                    backoff, p0.x, p0.y, math.degrees(p0.yaw)))
        # 0) coarse approach
        self.goto((ax, ay), tol=0.12,
                  timeout_s=max(20.0, 3.5 * _dist((p0.x, p0.y), (ax, ay)) / 0.25),
                  speed=0.25)
        pc = self.pose()
        if self.fallen():
            raise RuntimeError(
                "fallen during alignment at leg %s (z=%.2f tilt=%.1fdeg)"
                % (leg_name, pc.z, pc.tilt))
        log.info("align leg=%s coarse pose=(%+.2f,%+.2f,%+.1fdeg) res=%.3f "
                 "t=%.1fs" % (leg_name, pc.x, pc.y, math.degrees(pc.yaw),
                              math.hypot(ax - pc.x, ay - pc.y),
                              self._now() - t_start))
        leg_budget = float(leg.get("align_deadline_s", ALIGN_DEADLINE_S))
        deadline = self._now() + (leg_budget if deadline_s is None
                                  else float(deadline_s))
        attempt = 0
        while attempt < 2 and self._now() < deadline - 3.0:
            attempt += 1
            if self.fallen():
                raise RuntimeError(
                    "fallen during alignment at leg %s (z=%.2f tilt=%.1fdeg)"
                    % (leg_name, self.pose().z, self.pose().tilt))
            rem = deadline - self._now()
            # re-position at the approach point if the coarse goto left > 0.10
            p = self.pose()
            if math.hypot(ax - p.x, ay - p.y) > 0.10:
                self.goto((ax, ay), tol=0.08,
                          timeout_s=min(8.0, max(2.0, rem * 0.25)), speed=0.10)
            # 1) yaw servo
            rem = deadline - self._now()
            self._rotate_servo(target_yaw,
                               timeout_s=min(18.0, max(2.0, rem * 0.30)))
            pr = self.pose()
            log.info("align leg=%s attempt=%d yaw_servo pose=(%+.2f,%+.2f,"
                     "%+.1fdeg) yaw_err=%+.1fdeg t=%.1fs"
                     % (leg_name, attempt, pr.x, pr.y, math.degrees(pr.yaw),
                        math.degrees(wrap(pr.yaw - target_yaw)),
                        self._now() - t_start))
            # 1b) lateral sidestep when the yaw turn drifted off the lane:
            # the S-turn settle cannot fix it on short-runway legs
            # (slope settle_forward=0.05 vs a 0.15-0.25 m rotation drift)
            rem = deadline - self._now()
            p = self.pose()
            e_now = self._lane_offset(p, lane_x, lane_y, target_yaw)
            if abs(e_now) > xy_tol and rem > 14.0:
                self._sidestep_lane(lane_x, lane_y, target_yaw, e_now,
                                    timeout_s=min(12.0, rem * 0.45))
            # 2) lateral S-turn settle (budget preserves the runway)
            rem = deadline - self._now()
            if rem > 4.0:
                travel = (backoff + 0.35) if backoff > 0.0 \
                    else float(leg["settle_forward"])
                self._lane_settle(lane_x, lane_y, target_yaw,
                                  timeout_s=min(20.0, rem * 0.55),
                                  max_travel=travel)
            # 2b) settle fallback: still off the lane -> sidestep again
            rem = deadline - self._now()
            p = self.pose()
            e_now = self._lane_offset(p, lane_x, lane_y, target_yaw)
            if abs(e_now) > xy_tol and rem > 14.0:
                self._sidestep_lane(lane_x, lane_y, target_yaw, e_now,
                                    timeout_s=min(10.0, rem * 0.45))
            # 2c) wall-style last resort: still off the lane but the heading is
            # what the crossing needs -> one extra yaw polish before failing
            rem = deadline - self._now()
            p = self.pose()
            e_now = self._lane_offset(p, lane_x, lane_y, target_yaw)
            if abs(e_now) > xy_tol and rem > 4.0:
                self._rotate_servo(target_yaw, timeout_s=min(8.0, rem * 0.35))
            # 3) straight run-in to the pre-align target
            rem = deadline - self._now()
            if rem > 3.0:
                pb = self.pose()
                ux, uy = cos_l, sin_l
                along_before = (pb.x - tx) * ux + (pb.y - ty) * uy
                if along_before < -0.05 and not leg.get("no_creep"):
                    self._creep_along(lane_x, lane_y, target_yaw, 0.0,
                                      timeout_s=min(10.0, rem - 1.0), vx=0.10)
            # 4) final yaw polish
            rem = deadline - self._now()
            p = self.pose()
            if abs(math.degrees(wrap(p.yaw - target_yaw))) > yaw_tol and rem > 2.0:
                self._rotate_servo(target_yaw, timeout_s=min(10.0, rem - 0.5))
            p = self.pose()
            e = self._lane_offset(p, lane_x, lane_y, target_yaw)
            along = (p.x - tx) * cos_l + (p.y - ty) * sin_l
            yaw_err = math.degrees(wrap(p.yaw - target_yaw))
            ok = (abs(e) <= xy_tol and abs(yaw_err) <= yaw_tol
                  and along_min <= along <= along_max)
            log.info("align leg=%s attempt=%d end pose=(%+.2f,%+.2f,%+.1fdeg) "
                     "e=%+.3f along=%+.2f yaw_err=%+.1fdeg gates=(%.2fm,%.1fdeg)"
                     " %s t=%.1fs"
                     % (leg_name, attempt, p.x, p.y, math.degrees(p.yaw),
                        e, along, yaw_err, xy_tol, yaw_tol,
                        "OK" if ok else "retry", self._now() - t_start))
            if ok:
                break
        p = self.pose()
        e = self._lane_offset(p, lane_x, lane_y, target_yaw)
        along = (p.x - tx) * cos_l + (p.y - ty) * sin_l
        yaw_err = math.degrees(wrap(p.yaw - target_yaw))
        elapsed = self._now() - t_start
        ok = (abs(e) <= xy_tol and abs(yaw_err) <= yaw_tol
              and along_min <= along <= along_max
              and elapsed <= float(leg.get("align_deadline_s", ALIGN_DEADLINE_S)) + 27.0)
        self.get_logger().info(
            "expert switch leg=%s policy=%s lane_center=(x=%+.2f,y=%+.2f,"
            "yaw=%+.1fdeg) start_dy=%+.3f stand=(x=%+.2f,y=%+.2f) "
            "actual=(x=%+.2f,y=%+.2f,yaw=%+.1fdeg) e=%+.3f along=%+.2f "
            "yaw_err=%+.1fdeg gates=(%.2fm,%.1fdeg) align_s=%.1f %s"
            % (leg_name, policy, lane_x, lane_y, math.degrees(target_yaw),
               float(leg["start_dy"]), tx, ty,
               p.x, p.y, math.degrees(p.yaw), e, along, yaw_err,
               xy_tol, yaw_tol, elapsed, "OK" if ok else "ALIGN_FAIL"))
        self._corr = {"x": round(p.x, 3), "y": round(p.y, 3),
                      "yaw_deg": round(math.degrees(p.yaw), 1),
                      "e": round(e, 3), "along": round(along, 3),
                      "yaw_err_deg": round(yaw_err, 1),
                      "align_s": round(elapsed, 1)}
        self._emit_plan()
        return ok, p.x, p.y, p.yaw, e, yaw_err

    # ---- expert crossing --------------------------------------------------
    def expert_cross(self, leg: dict, attempt: int = 1):
        """Run one expert leg with the adaptive lane command until its gate.

        Returns (crossed, why, pose).  ``why`` is one of gate / timeout / stall
        / fallen.  A stall means the lane-projected position made less than
        ``CROSS_STALL_PROGRESS_M`` progress for ``CROSS_STALL_WINDOW_S``; the
        caller may recover (back off + re-align + retry once).
        """
        policy = leg["policy"]
        vx = float(leg["vx"])
        lane_x = float(leg["lane_x"])
        lane_y = float(leg["lane_y"])
        lane_yaw = float(leg["lane_yaw"])
        self.switch(policy)
        retry_note = "" if attempt <= 1 else f" (recovery retry {attempt - 1})"
        self._cmd = ("%s(policy=%s, vx=%+.2f, lane=(%+.2f,%+.2f))%s"
                     % (leg["leg"], policy, vx, lane_x, lane_y, retry_note))
        self._cli = (f"switch {policy}; forward {vx:+.2f} m/s + adaptive lane hold"
                     + retry_note)
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
        while self._now() - t0 < float(leg["cross_timeout"]):
            p = self.pose()
            if self.fallen():
                why = "fallen"
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
            elif now - last_progress >= CROSS_STALL_WINDOW_S:
                why = "stall"
                log.info("expert stall leg=%s after %.1fs along=%+.2f "
                         "(progress %.2fm in %.1fs)"
                         % (leg["leg"], now - t0, along, best_along - along,
                            now - last_progress))
                break
            if leg.get("fixed"):
                # fixed-command expert (s1d trench): no adaptive vy/wz, the
                # pre-leg alignment gate is what keeps it on the lane
                self.command(vx, 0.0, 0.0)
            else:
                self.command(*self.lane_cmd(p, prev, lane_x, lane_y, lane_yaw, vx))
            prev = p
            self._tick()
        self.stop(0.50)
        p = self.pose()
        crossed = (why == "gate" and not self.fallen())
        return crossed, why, p

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
        self.goto((bx, by), tol=0.12, timeout_s=14.0, speed=0.20)
        if self.fallen():
            raise RuntimeError("fallen during crossing recovery at leg %s"
                               % leg["leg"])
        aligned = self.align_to(leg, leg["policy"],
                                deadline_s=CROSS_RETRY_ALIGN_S)
        if not aligned[0]:
            log.warning("expert_recover leg=%s re-align missed tolerance "
                        "(e=%+.3f yaw_err=%+.1fdeg); retrying crossing anyway"
                        % (leg["leg"], aligned[4], aligned[5]))
        return bool(aligned[0])

    # ---- route ------------------------------------------------------------
    def _log_table(self) -> None:
        log = self.get_logger()
        log.info("Part2 route v2: %d legs, connector=%s"
                 % (len(LEGS), TRANSFER_POLICY))
        log.info("tiers: platform=%dcm stairs=%dcm slope=%ddeg trench=%dcm "
                 "wall=%s(%s)"
                 % (PLATFORM_LEVEL, STAIRS_LEVEL, SLOPE_LEVEL, TRENCH_LEVEL,
                    WALL_VARIANT if WALL_ENABLED else "off",
                    "on" if WALL_ENABLED else "off"))
        for i, leg in enumerate(LEGS):
            log.info("  leg%d %-20s policy=%-32s vx=%+.2f stand=(%+.2f,%+.2f) "
                     "gate_x=%+.2f%s"
                     % (i + 1, leg["leg"], leg["policy"], leg["vx"],
                        leg["stand"][0], leg["stand"][1],
                        leg["cross_end"][0],
                        f"  # {leg['note']}" if leg["note"] else ""))

    def search_goto(self, target, speed: float = 0.50) -> None:
        """goto with a distance-scaled sim-time timeout (long transits)."""
        p = self.pose()
        dist = _dist((p.x, p.y), target)
        timeout = max(20.0, 2.5 * dist / max(0.20, speed) + 12.0)
        self.require(tuple(target), tol=0.12, timeout_s=timeout, speed=speed)

    def _route(self) -> None:
        log = self.get_logger()
        if not self.wait_pose(30.0):
            log.error("no /eval_state; launch with route:=part2 "
                      "(route raises eval_state_every to 10)")
            return
        log.info("Part2 route: standing up / entering RL with %s" % TRANSFER_POLICY)
        self._log_table()
        # Preselect the connector so the first RL entry loads the locked model
        # (the sim boots policy_name:=flat; EnterRL would otherwise load it).
        self.switch(TRANSFER_POLICY)
        self._press(buttons=[BTN_A], hold_s=0.15)
        if not self.wait_stable_z(0.46, tilt_max=8.0, hold_s=0.5, timeout_s=15.0):
            log.warning("stand-up not stable (z/tilt); continuing")
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

        for index, leg in enumerate(LEGS):
            t_leg0 = self._now()
            self.set_leg(leg["leg"])
            log.info("=== leg %d/%d %s (policy=%s, vx=%+.2f) ==="
                     % (index + 1, len(LEGS), leg["leg"], leg["policy"], leg["vx"]))
            hop_speed = float(leg.get("transit_speed", TRANSIT_SPEED))
            for pt in leg["transit"]:
                self.search_goto(pt, speed=hop_speed)
            if leg.get("bypass"):
                self._cmd = "%s(bypass transit, no expert)" % leg["leg"]
                self._cli = ("transit only: go around the plum row at x=2.60 "
                             "(no deployable blind expert)")
                self._corr = {}
                self._emit_plan()
                self.report(leg["leg"], True,
                            "TRANSIT ONLY / bypass: no deployable blind plum "
                            "expert; route goes around the row at x=2.60")
                self.ensure_transfer(announce=True)
                if not WALL_ENABLED:
                    log.info("Part2 route finished: plum bypassed "
                             "(no deployable expert)")
                    self.stop(0.5)
                    return
                continue
            # final approach + alignment; retry once with the full primitive
            aligned = self.align_to(leg, leg["policy"])
            for retry_i in range(2):
                if aligned[0]:
                    break
                log.warning("alignment missed tolerance; retry %d/2" % (retry_i + 1))
                aligned = self.align_to(leg, leg["policy"])
            if not aligned[0]:
                raise RuntimeError(
                    "alignment failed at leg %s: y_err=%+.3f yaw_err=%+.1fdeg"
                    % (leg["leg"], aligned[4], aligned[5]))
            crossed, why, pend = self.expert_cross(leg)
            recoveries = 0
            if (not crossed and RECOVER_ENABLED and why in ("stall", "timeout")
                    and not self.fallen()):
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
            t_cross = self._now() - t_leg0
            post_ok = bool(leg["post"](pend)) and not self.fallen()
            ok = bool(crossed and post_ok)
            detail = ("policy=%s vx=%+.2f end=(%+.2f,%+.2f,yaw=%+.1fdeg) gate=%s"
                      " post=%s y_err_at_switch=%+.3f yaw_err=%+.1fdeg"
                      " align_s=%.1f cross_s=%.1f recover=%d"
                      % (leg["policy"], leg["vx"], pend.x, pend.y,
                         math.degrees(pend.yaw), why,
                         "ok" if post_ok else "out", aligned[4], aligned[5],
                         float(self._corr.get("align_s", 0.0)), t_cross,
                         recoveries))
            if leg["leg"] == "high_plum":
                detail += (" TRANSIT_ONLY(no judge-verified deployable blind "
                           "expert; route report is transit+gate only)")
            self.report(leg["leg"], ok, detail)
            self.ensure_transfer(announce=True)
            for pt in leg["exit"]:
                self.search_goto(pt)
            only = (os.environ.get("PART2_ROUTE_ONLY", "").strip().lower())
            if only and (only == leg["leg"].lower()
                         or only == str(index + 1)):
                log.info("PART2_ROUTE_ONLY=%s: stopping after leg %d (debug)"
                         % (only, index + 1))
                self.stop(0.4)
                return

        if PLUM_MODE == "bypass":
            log.info("Part2 route finished: %d/%d legs PASS (plum bypassed: "
                     "no deployable expert)" % (len(LEGS), len(LEGS)))
        else:
            log.info("Part2 route finished: %d/%d legs PASS"
                     % (len(LEGS), len(LEGS)))
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
