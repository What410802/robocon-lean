"""The competition standard: machine-readable constants for both parts.

This module is the **single machine-readable source of truth** for scoring
numbers, attempt limits, difficulty curves and the specification-level robot
constraints.  The human-readable standard is ``eval/SPEC.md``; scene/judge
geometry lives in the generated Part 2 level manifest (``terrain/`` produces it)
and Part 1 scene geometry lives in ``demo/robots/<robot>/scene_part1.xml``.

Do not hard-code any number from here in judges/scoring/runner: import it.  The
judges keep their *geometry/tolerance* constants locally (they describe the
official field, not the score), but every score/curve/attempt constant must come
from this file.

Sources (see ``docs/RULES-BASIS.md`` for the evidence trail):
  * competition rules PDF ``队内赛.pdf`` (2026-10-20), which embeds the official
    25th V1.0 table 2/table 3 obstacle definitions and score columns;
  * local implementation choices (marked IMPL below; these are *stricter or
    operationalisation choices*, not official text and not externally confirmed).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Timing / sampling
# ---------------------------------------------------------------------------
PHYSICS_DT_S = 0.002            # official MJCF <option timestep="0.002">
CONTROL_DT_S = 0.005            # 200 Hz PD / command limiter
POLICY_DT_S = 0.020             # 50 Hz policy inference
JUDGE_FRAME_DT_S = PHYSICS_DT_S  # judged traces: one frame per 2 ms physics step
# Traces recorded at a coarser period (e.g. the legacy 20 ms /eval_state@10) are
# still replayable: the v5 judge advances hold timers from consecutive frame
# timestamps, never from a hard-coded frame count.

# ---------------------------------------------------------------------------
# Part 1 -- 足式障碍赛 (manual / fixed actions allowed), 900 points
# ---------------------------------------------------------------------------
PART1_OBSTACLES: tuple[str, ...] = (
    # Competition (course) order; the judge keeps per-obstacle zones.
    "slalom", "gravel_pit", "high_wall", "height_bar",
    "t_stairs", "slope", "bridge_blocks", "bridge_bands",
)
PART1_OBSTACLE_NAMES: dict[str, str] = {
    "slalom": "直角绕杆",
    "gravel_pit": "砂砾碎木坑",
    "high_wall": "高墙",
    "height_bar": "限高杆",
    "t_stairs": "T 字形台阶",
    "slope": "斜坡",
    "bridge_blocks": "木桥 A（方墩桥）",
    "bridge_bands": "木桥 B（窄带桥）",
}
PART1_OBSTACLE_SCORE = 100          # per-obstacle points (remote-control column)
PART1_RETURN_BONUS = 100            # +100 iff all 8 obstacles are FULL score and returned
PART1_MAX_SCORE = len(PART1_OBSTACLES) * PART1_OBSTACLE_SCORE + PART1_RETURN_BONUS  # 900
PART1_ATTEMPTS = 3                  # whole-run attempts (organiser Q1)
PART1_RECOVERY_CONSUMES_ATTEMPT = False   # pause/recover keeps the attempt
PART1_SKIP_OBSTACLE_CONSUMES_ATTEMPT = False

# T-stairs partial scores (2026-10-27: direction rule, not arm identity).
# up one arm + down another = 100; same-arm up+down = 50; one direction = 50.
PART1_T_STAIRS_SCORE = 100
PART1_T_STAIRS_PARTIAL_UP = 50      # one complete up traversal
PART1_T_STAIRS_PARTIAL_DOWN = 50    # one complete down traversal
PART1_T_STAIRS_SAME_ARM = 50        # up and down the same arm

# Recovery pose slack (implementation measurement slack; see SPEC Part 1).
PART1_RECOVERY_POS_TOL_M = 0.30
PART1_RECOVERY_YAW_TOL_RAD = 0.52

# Stable-stop timer (attempt clock stops after returning to the start area).
PART1_STOP_SPEED_MAX_MPS = 0.15
PART1_STOP_TILT_MAX_DEG = 10.0
PART1_STOP_HOLD_S = 0.5

# Slope success: cumulative distance along the 3 m long edge (official text).
PART1_SLOPE_MIN_DISTANCE_M = 1.0

# ---------------------------------------------------------------------------
# Part 2 -- 越障能力测试 (RL only), 1300 points
# ---------------------------------------------------------------------------
PART2_OBSTACLES: tuple[str, ...] = (
    "high_platform", "stairs", "slope", "high_wall", "trench", "plum",
)
PART2_OBSTACLE_NAMES: dict[str, str] = {
    "high_platform": "高台",
    "stairs": "楼梯",
    "slope": "斜坡一",
    "high_wall": "高墙",
    "trench": "沟壑",
    "plum": "梅花桩",
}
PART2_MAX_SCORE = 1300
PART2_ATTEMPTS_PER_OBSTACLE = 4
PART2_ROUNDING = "floor"            # every interpolated score is floored
PART2_RL_ONLY = True                # official: no fixed actions for Part 2

# Official score curves: (difficulty value, points). Values below the first
# point score 0; values above the last clamp to the last point; between points
# the score is linear and floored.  ``unit == "bool"`` obstacles use the single
# curve point (success -> points).
PART2_CURVES: dict[str, dict] = {
    "high_platform": {
        "unit": "cm",
        "points": ((10, 10), (20, 50), (30, 100), (40, 200)),
        "values": (10, 20, 30, 40),
    },
    "stairs": {
        "unit": "cm",
        "points": ((10, 10), (15, 50), (20, 100), (30, 200)),
        "values": (10, 15, 20, 30),
    },
    "slope": {
        "unit": "deg",
        "points": ((20, 10), (30, 50), (35, 100), (45, 200)),
        "values": (20, 30, 35, 45),
    },
    "high_wall": {
        "unit": "cm",
        "points": ((10, 0), (20, 50), (30, 100), (40, 200)),
        "values": (10, 20, 30, 40),
    },
    "trench": {
        "unit": "cm",
        # linear between the official endpoints: 10 cm = 0, 30 cm = 300.
        "points": ((10, 0), (30, 300)),
        "values": (10, 20, 30),
    },
    "plum": {
        "unit": "bool",
        "points": ((1, 200),),
        "values": (1,),
    },
}
PART2_OBSTACLE_MAX_SCORE: dict[str, int] = {
    oid: (spec["points"][-1][1] if spec["unit"] == "bool"
          else max(p for _, p in spec["points"]))
    for oid, spec in PART2_CURVES.items()
}
# Sanity: the official curve maxima sum to the part maximum.
if sum(PART2_OBSTACLE_MAX_SCORE.values()) != PART2_MAX_SCORE:
    raise AssertionError("Part 2 curve maxima must sum to 1300")

# Safety cap only (Part 2 is untimed by rule).  The generated level manifest
# carries the per-level value; this is the fallback/default for hand-written
# level entries and the documented cap for judged runs.
PART2_SAFETY_TIMEOUT_S = 120.0

# ---------------------------------------------------------------------------
# Robot / compliance constraints (official hardware limits, checked manually)
# ---------------------------------------------------------------------------
TORQUE_LIMITS_NM: dict[str, float] = {
    "hip": 25.0,
    "thigh": 25.0,
    "calf": 60.0,
}
SENSORS_ALLOWED: tuple[str, ...] = ("imu", "joint_encoder", "camera", "lidar")
SENSORS_FORBIDDEN: tuple[str, ...] = (
    "body_lin_vel", "heightmap", "foot_force", "foot_height", "body_height",
    "privileged_terrain_scan",
)
NAVIGATION_TRUTH_SCOPE = "world pose/velocity truth may only generate commands; never a network input"
PART1_FIXED_ACTIONS_ALLOWED = True
PART2_PERCEPTION = {
    "primary": "depth_camera",
    "radar": "not_used",
    "fallback": "blind (imu+encoder)",
}
# Task weights (the 30% defence is outside this spec).
COMPETITION_WEIGHTS = {"part1_part2": 0.70, "defence": 0.30}

# ---------------------------------------------------------------------------
# Structured views consumed by scoring / docs / tests
# ---------------------------------------------------------------------------
def part1() -> dict:
    """Part 1 view in the shape ``scoring.part1_*`` expects."""
    return {
        "max_score": PART1_MAX_SCORE,
        "attempts": {
            "count": PART1_ATTEMPTS,
            "recovery_consumes_attempt": PART1_RECOVERY_CONSUMES_ATTEMPT,
        },
        "obstacles": [
            {"id": oid,
             "name": PART1_OBSTACLE_NAMES[oid],
             "per_obstacle_score": PART1_OBSTACLE_SCORE,
             "partial_scores": ({"both_sides": PART1_T_STAIRS_SCORE,
                                 "one_side_up": PART1_T_STAIRS_PARTIAL_UP,
                                 "one_side_down": PART1_T_STAIRS_PARTIAL_DOWN}
                                if oid == "t_stairs" else None)}
            for oid in PART1_OBSTACLES
        ],
        "return_bonus": {"id": "return_to_start", "score": PART1_RETURN_BONUS,
                         "requires": "all_obstacles_full_score"},
    }


def part2() -> dict:
    """Part 2 view in the shape ``scoring``/``trace_part2_v5`` expect."""
    obstacles = []
    for oid in PART2_OBSTACLES:
        spec = PART2_CURVES[oid]
        obstacles.append({
            "id": oid,
            "name": PART2_OBSTACLE_NAMES[oid],
            "unit": spec["unit"],
            "max_score": PART2_OBSTACLE_MAX_SCORE[oid],
            "score_curve": [{"value": v, "points": p} for v, p in spec["points"]],
            "values": list(spec["values"]),
        })
    return {
        "max_score": PART2_MAX_SCORE,
        "attempts_per_obstacle": PART2_ATTEMPTS_PER_OBSTACLE,
        "rounding": PART2_ROUNDING,
        "rl_only": PART2_RL_ONLY,
        "safety_timeout_s": PART2_SAFETY_TIMEOUT_S,
        "obstacles": obstacles,
    }


def part2_obstacle(obstacle_id: str) -> dict:
    for obstacle in part2()["obstacles"]:
        if obstacle["id"] == obstacle_id:
            return obstacle
    raise KeyError(f"unknown Part 2 obstacle {obstacle_id!r}")


__all__ = [
    "PHYSICS_DT_S", "CONTROL_DT_S", "POLICY_DT_S", "JUDGE_FRAME_DT_S",
    "PART1_OBSTACLES", "PART1_OBSTACLE_NAMES", "PART1_OBSTACLE_SCORE",
    "PART1_RETURN_BONUS", "PART1_MAX_SCORE", "PART1_ATTEMPTS",
    "PART1_RECOVERY_CONSUMES_ATTEMPT", "PART1_SKIP_OBSTACLE_CONSUMES_ATTEMPT",
    "PART1_T_STAIRS_SCORE", "PART1_T_STAIRS_PARTIAL_UP",
    "PART1_T_STAIRS_PARTIAL_DOWN", "PART1_T_STAIRS_SAME_ARM",
    "PART1_RECOVERY_POS_TOL_M", "PART1_RECOVERY_YAW_TOL_RAD",
    "PART1_STOP_SPEED_MAX_MPS", "PART1_STOP_TILT_MAX_DEG", "PART1_STOP_HOLD_S",
    "PART1_SLOPE_MIN_DISTANCE_M",
    "PART2_OBSTACLES", "PART2_OBSTACLE_NAMES", "PART2_MAX_SCORE",
    "PART2_ATTEMPTS_PER_OBSTACLE", "PART2_ROUNDING", "PART2_RL_ONLY",
    "PART2_CURVES", "PART2_OBSTACLE_MAX_SCORE", "PART2_SAFETY_TIMEOUT_S",
    "TORQUE_LIMITS_NM", "SENSORS_ALLOWED", "SENSORS_FORBIDDEN",
    "NAVIGATION_TRUTH_SCOPE", "PART1_FIXED_ACTIONS_ALLOWED", "PART2_PERCEPTION",
    "COMPETITION_WEIGHTS",
    "part1", "part2", "part2_obstacle",
]
