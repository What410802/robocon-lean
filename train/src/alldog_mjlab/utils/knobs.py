"""Single registry for every environment variable read by the training code.

Why a registry instead of scattered ``os.environ.get`` calls:

* prefix encodes ownership — ``RC_`` = training/runtime contract shared by all
  robots and events, ``PART1_`` / ``PART2_`` = event-profile knobs, ``ROBOT_``
  must never appear (the robot is selected by the task id, not by env vars);
* a typo or an unregistered variable is a hard error instead of a silent
  fallback that changes a training recipe without a trace;
* every run can print the overrides that were actually in effect, so an
  artifact manifest can record them.

Values are read once at import time, matching the old behaviour (MjLab builds
configs at import): change the environment before launching the process.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class Knob:
    name: str
    scope: str  # "RC" | "PART1" | "PART2"
    default: object
    kind: str  # "float" | "int" | "bool" | "str"
    doc: str

def _k(name: str, scope: str, default: object, kind: str, doc: str) -> Knob:
    return Knob(name=name, scope=scope, default=default, kind=kind, doc=doc)

KNOBS: dict[str, Knob] = {
    # ---- training / runtime contract (shared) -------------------------
    "RC_PHYSICS_DT": _k("RC_PHYSICS_DT", "RC", 0.002, "float",
                        "MuJoCo physics substep [s]; official MJCF uses 0.002"),
    "RC_SIM_DECIMATION": _k("RC_SIM_DECIMATION", "RC", None, "int",
                            "derived = policy_period/physics_dt; override only to reproduce an old run"),
    "RC_CONTROL_PERIOD": _k("RC_CONTROL_PERIOD", "RC", 0.005, "float",
                            "deployment control-loop period [s] (organiser controller uses 5 ms)"),
    "RC_POLICY_PERIOD": _k("RC_POLICY_PERIOD", "RC", 0.020, "float",
                           "policy update period [s] (50 Hz); must be an integer multiple of both periods"),
    "RC_NCONMAX": _k("RC_NCONMAX", "RC", 1024, "int", "mujoco-warp contact capacity (memory budget)"),
    "RC_NJMAX": _k("RC_NJMAX", "RC", 4096, "int", "mujoco-warp constraint capacity (memory budget)"),
    "RC_CCD_ITERS": _k("RC_CCD_ITERS", "RC", 35, "int", "CCD iterations; 35 matches the judge deployment sim"),
    # PPO / HIM algorithm knobs
    "RC_ALGORITHM": _k("RC_ALGORITHM", "RC", "PPO", "str",
                       "PPO | fully-qualified PPO class path (e.g. alldog_mjlab.algorithms.him.ppo:HIMPPO)"),
    "RC_INIT_STD": _k("RC_INIT_STD", "RC", 1.0, "float", "Gaussian action std at init"),
    "RC_LEARN_STD": _k("RC_LEARN_STD", "RC", True, "bool", "learn the action std"),
    "RC_ENTROPY_COEF": _k("RC_ENTROPY_COEF", "RC", 0.01, "float", "PPO entropy coefficient"),
    "RC_EPOCHS": _k("RC_EPOCHS", "RC", 5, "int", "PPO learning epochs per update"),
    "RC_MINIBATCHES": _k("RC_MINIBATCHES", "RC", 4, "int", "PPO mini-batches per epoch"),
    "RC_NORM_ADV_PER_MB": _k("RC_NORM_ADV_PER_MB", "RC", False, "bool",
                             "normalize advantages per mini-batch"),
    "RC_LR": _k("RC_LR", "RC", 1.0e-3, "float", "initial learning rate"),
    "RC_LR_SCHEDULE": _k("RC_LR_SCHEDULE", "RC", "adaptive", "str", "adaptive | fixed"),
    "RC_RESUME_LOAD_CFG": _k("RC_RESUME_LOAD_CFG", "RC", None, "str",
                             "resume inheritance spec (e.g. actor); unset = full resume"),
    "RC_RESUME_LR": _k("RC_RESUME_LR", "RC", None, "float",
                       "override the learning rate restored from a checkpoint"),
    # guards / diagnostics
    "RC_REWARD_CLIP": _k("RC_REWARD_CLIP", "RC", 50.0, "float",
                         "per-step total reward clamp (<=0 disables)"),
    # Velocity-tracking recipe (flat baseline defaults; 2026-10-31 velocity tracking work).
    "RC_TRACK_LIN_W": _k("RC_TRACK_LIN_W", "RC", 1.0, "float",
                         "track_linear_velocity reward weight (frozen flat baseline = 1.0)"),
    "RC_TRACK_ANG_W": _k("RC_TRACK_ANG_W", "RC", 0.5, "float",
                         "track_angular_velocity reward weight (frozen flat baseline = 0.5)"),
    "RC_TRACK_SIGMA": _k("RC_TRACK_SIGMA", "RC", 0.25, "float",
                         "HIMLoco tracking reward exponential denominator (baseline = 0.25)"),
    "RC_TRACK_ERR_L2_W": _k("RC_TRACK_ERR_L2_W", "RC", 0.0, "float",
                            "explicit L2 planar tracking-error penalty weight (0 = off/legacy)"),
    "RC_POSE_CMD": _k("RC_POSE_CMD", "RC", False, "bool",
                      "flat task: use the low-speed pose-alignment command contract (vx/vy/wz full axes)"),
    "RC_POSE_VX_MIN": _k("RC_POSE_VX_MIN", "RC", -0.20, "float",
                         "pose command vx minimum (m/s)"),
    "RC_POSE_VX_MAX": _k("RC_POSE_VX_MAX", "RC", 0.50, "float",
                         "pose command vx maximum (m/s)"),
    "RC_POSE_VY_MAX": _k("RC_POSE_VY_MAX", "RC", 0.15, "float",
                         "pose command |vy| bound (m/s); 0 disables lateral sampling"),
    "RC_POSE_WZ_MAX": _k("RC_POSE_WZ_MAX", "RC", 0.50, "float",
                         "pose command |wz| bound (rad/s)"),
    "RC_POSE_STANDING_FRAC": _k("RC_POSE_STANDING_FRAC", "RC", 0.10, "float",
                                "pose contract: standing (zero-command) env fraction"),
    "RC_POSE_RESAMPLE_S": _k("RC_POSE_RESAMPLE_S", "RC", 5.0, "float",
                             "pose contract: command resampling period (s)"),
    "RC_POSE_LOW_FRAC": _k("RC_POSE_LOW_FRAC", "RC", 0.0, "float",
                           "pose contract: fraction of non-standing resamples drawn from the "
                           "low-speed box (0 = uniform envelope only, legacy)"),
    "RC_POSE_LOW_VX": _k("RC_POSE_LOW_VX", "RC", 0.15, "float",
                         "pose low-speed mixture |vx| bound (m/s)"),
    "RC_POSE_LOW_WZ": _k("RC_POSE_LOW_WZ", "RC", 0.15, "float",
                         "pose low-speed mixture |wz| bound (rad/s)"),
    "RC_POSE_EDGES": _k("RC_POSE_EDGES", "RC", False, "bool",
                        "flat task: swap the plane terrain for the platform-edge generator "
                        "(pose-edge robustness recipe)"),
    "RC_POSE_EDGE_ROWS": _k("RC_POSE_EDGE_ROWS", "RC", 6, "int",
                            "pose-edge terrain difficulty rows (height sweep resolution)"),
    "RC_POSE_EDGE_PROP": _k("RC_POSE_EDGE_PROP", "RC", 2.0, "float",
                            "pose-edge column spawn weight (vs the flat column)"),
    "RC_POSE_EDGE_FLAT_PROP": _k("RC_POSE_EDGE_FLAT_PROP", "RC", 1.0, "float",
                                 "pure-flat column spawn weight"),
    "RC_POSE_EDGE_H_MIN": _k("RC_POSE_EDGE_H_MIN", "RC", 0.05, "float",
                             "pose-edge platform top height at difficulty 0 (m)"),
    "RC_POSE_EDGE_H_MAX": _k("RC_POSE_EDGE_H_MAX", "RC", 0.20, "float",
                             "pose-edge platform top height at difficulty 1 (m)"),
    "RC_POSE_EDGE_ON_TOP_P": _k("RC_POSE_EDGE_ON_TOP_P", "RC", 0.45, "float",
                                "probability the platform covers the spawn origin (on-top mode)"),
    "RC_POSE_UPRIGHT_W": _k("RC_POSE_UPRIGHT_W", "RC", 1.2, "float",
                            "pose-edge recipe: upright (flat_orientation_l2) penalty weight; "
                            "round-1 showed the frozen -0.2 lets a persistent flat lean emerge"),
    "RC_POSE_BASEH_W": _k("RC_POSE_BASEH_W", "RC", 0.0, "float",
                          "pose-edge recipe: world-z base_height penalty weight; 0 removes the "
                          "term (it is incompatible with platform terrain — nominal stance on a "
                          "0.2 m platform reads as too high and collapses the gait)"),
    "RC_POSE_ROTDRIFT_W": _k("RC_POSE_ROTDRIFT_W", "RC", 0.0, "float",
                             "pose-edge recipe: rotation drift penalty weight — |v_xy| / |wz_cmd| "
                             "during pure-rotation commands (m per rad; 0 = off)"),
    "RC_POSE_DOFPOS_W": _k("RC_POSE_DOFPOS_W", "RC", 0.15, "float",
                           "pose-edge recipe: joint-space stance anchor — mean squared dof_pos "
                           "deviation from the default stance (ground-relative; matches the "
                           "experts' handover dof_pos distribution)"),
    "RC_POSE_SPAWN_XY": _k("RC_POSE_SPAWN_XY", "RC", 0.12, "float",
                           "pose-edge spawn xy randomization half-range (m; bounded so the "
                           "worst-case spawn footprint stays clear of platform sides)"),
    "RC_POSE_SPAWN_YAW": _k("RC_POSE_SPAWN_YAW", "RC", 0.35, "float",
                            "pose-edge spawn yaw randomization (rad)"),
    "RC_CMD_CURR_MAX_VX": _k("RC_CMD_CURR_MAX_VX", "RC", 2.0, "float",
                             "command-curriculum final |vx| bound (default = frozen baseline 2.0; "
                             "lower it to stop the curriculum expanding past the evaluated speeds)"),
    "RC_MIRROR_LOSS_COEFF": _k("RC_MIRROR_LOSS_COEFF", "RC", 0.0, "float",
                               "mirror-symmetry auxiliary loss coefficient (0 = off)"),
    "RC_ANOMALY_PROBE": _k("RC_ANOMALY_PROBE", "RC", False, "bool",
                           "install the anomaly probe (investigation instrument)"),
    "RC_PROBE_ACTION": _k("RC_PROBE_ACTION", "RC", 100.0, "float", "anomaly probe action threshold"),
    "RC_PROBE_OBS": _k("RC_PROBE_OBS", "RC", 200.0, "float", "anomaly probe observation threshold"),
    "RC_PROBE_JOINT_VEL": _k("RC_PROBE_JOINT_VEL", "RC", 100.0, "float", "anomaly probe joint velocity threshold"),
    # HIM (shared by both robots)
    "RC_HIM_INIT_STD": _k("RC_HIM_INIT_STD", "RC", 0.1, "float", "HIM action std at init / warm start"),
    "RC_HIM_FREEZE_STD": _k("RC_HIM_FREEZE_STD", "RC", None, "str",
                            "freeze HIM action std ('' = no; 'precheck' = near-zero)"),
    "RC_HIM_ANCHOR_ALPHA": _k("RC_HIM_ANCHOR_ALPHA", "RC", 0.0, "float",
                              "HIM teacher-anchor blend (0 = off; 1 = frozen at teacher)"),
    "RC_HIM_ESTIMATOR_LR": _k("RC_HIM_ESTIMATOR_LR", "RC", None, "float",
                              "estimator learning rate override (unset = configured value)"),
    "RC_HIM_OFFICIAL_WARMSTART": _k("RC_HIM_OFFICIAL_WARMSTART", "RC", None, "str",
                                    "organiser TorchScript checkpoint used to warm-start HIM"),
    # ---- Part 2 profile ------------------------------------------------
    "PART2_DR_EVENTS": _k("PART2_DR_EVENTS", "PART2", "payload_mass,base_com,push_robot", "str",
                            "comma-separated domain-randomisation events kept for Part 2 tasks"),
    "PART2_DEPLOY_SIM": _k("PART2_DEPLOY_SIM", "PART2", False, "bool",
                           "align training sim options with the judge/deployment physics"),
    "PART2_SUCCESS_HOLD_STEPS": _k("PART2_SUCCESS_HOLD_STEPS", "PART2", 20, "int",
                                   "success hold window [50 Hz steps] (judge: hold_s + 0.1)"),
    "PART2_LANE_HALF_WIDTH": _k("PART2_LANE_HALF_WIDTH", "PART2", 0.7, "float",
                                "scoring lane half width [m]"),
    "PART2_TRENCH_PEN_W": _k("PART2_TRENCH_PEN_W", "PART2", 0.0, "float",
                             "trench wheel-drop penalty weight"),
    "PART2_PLUM_FOOT_W": _k("PART2_PLUM_FOOT_W", "PART2", 0.0, "float",
                            "plum foothold potential weight"),
    "PART2_PLUM_JUDGE_PHYSICS": _k("PART2_PLUM_JUDGE_PHYSICS", "PART2", False, "bool",
                                   "plum: use judge/official MJCF solver options + spawn z=0.55 "
                                   "for training-env parity (P0 2026-10-31; changes dynamics, retrain)"),
    "PART2_CLEAR_W": _k("PART2_CLEAR_W", "PART2", 25.0, "float", "success bonus weight"),
    "PART2_OUT_OF_COURSE_MARGIN": _k("PART2_OUT_OF_COURSE_MARGIN", "PART2", 0.8, "float",
                                     "out-of-course termination x margin after the goal patch [m]"),
    "PART2_CLIMB_W": _k("PART2_CLIMB_W", "PART2", 0.0, "float", "climb-height reward weight"),
    "PART2_PROGRESS_W": _k("PART2_PROGRESS_W", "PART2", 8.0, "float", "course progress reward weight"),
    "PART2_GOAL_W": _k("PART2_GOAL_W", "PART2", 0.0, "float", "legacy goal reward weight"),
    "PART2_VX_MIN": _k("PART2_VX_MIN", "PART2", 0.40, "float",
                       "specialist/official command vx minimum (call-site defaults: exact 0.40, "
                       "official mixed 0.25, plum 0.25; traversal base 0.40)"),
    "PART2_VX_MAX": _k("PART2_VX_MAX", "PART2", 1.00, "float",
                       "specialist/official command vx maximum (call-site defaults: exact 1.00, "
                       "official mixed 1.00, plum 0.65; traversal base 1.00)"),
    "PART2_LATERAL_CMD": _k("PART2_LATERAL_CMD", "PART2", 0.10, "float",
                            "command vy bound (call-site defaults: exact 0.10, official mixed 0.15, "
                            "plum 0.05, traversal base 0.30)"),
    "PART2_YAW_CMD": _k("PART2_YAW_CMD", "PART2", 0.20, "float",
                        "command yaw-rate bound (call-site defaults: exact 0.20, official mixed 0.20, "
                        "plum 0.08, traversal base 1.00)"),
    "PART2_CMD_FORWARD_FRAC": _k("PART2_CMD_FORWARD_FRAC", "PART2", 1.0, "float",
                                 "fraction of envs with forward-only commands (vx=abs clamp>=0.3, vy=wz=0); "
                                 "default 1.0 = legacy full forward-only traversal"),
    "PART2_CMD_STANDING_FRAC": _k("PART2_CMD_STANDING_FRAC", "PART2", 0.0, "float",
                                  "fraction of envs with all-zero commands (standing); default 0.0 = legacy"),
    "PART2_TRAVERSAL_MIN_CMD_VX": _k("PART2_TRAVERSAL_MIN_CMD_VX", "PART2", 0.2, "float",
                                     "traversal-intent proxy for course reward/termination masks: "
                                     "cmd_vx > this value keeps the term active"),
    "PART2_CMD_RESAMPLE_S": _k("PART2_CMD_RESAMPLE_S", "PART2", 0.0, "float",
                               "override command resampling period [s] for Part2/obstacle tasks "
                               "(0 = keep per-task default e.g. 10 s; set >= episode length to keep "
                               "the sampled command/mode fixed for a whole episode)"),
    "PART2_FLAT_CMD_COLUMN": _k("PART2_FLAT_CMD_COLUMN", "PART2", False, "bool",
                                "add a 6th flat terrain column for full 3-axis command tracking in "
                                "blackw-official (obstacle columns keep the row2p traversal lane)"),
    "PART2_FLAT_COLUMN_PROP": _k("PART2_FLAT_COLUMN_PROP", "PART2", 1.67, "float",
                                 "spawn proportion of the flat command column (curriculum generator; "
                                 "obstacle columns use 1.0 each, so 1.67 gives ~25% flat envs)"),
    "PART2_FLAT_STANDING_FRAC": _k("PART2_FLAT_STANDING_FRAC", "PART2", 0.10, "float",
                                   "fraction of flat-column envs with all-zero (standing) commands"),
    "PART2_FLAT_VX_MIN": _k("PART2_FLAT_VX_MIN", "PART2", -0.6, "float",
                            "flat-column command vx minimum (negative = reverse samples)"),
    "PART2_FLAT_VX_MAX": _k("PART2_FLAT_VX_MAX", "PART2", 1.0, "float",
                            "flat-column command vx maximum"),
    "PART2_FLAT_LATERAL_CMD": _k("PART2_FLAT_LATERAL_CMD", "PART2", 0.25, "float",
                                 "flat-column command vy bound"),
    "PART2_FLAT_YAW_CMD": _k("PART2_FLAT_YAW_CMD", "PART2", 0.5, "float",
                             "flat-column command yaw-rate bound"),
    "BLACKW_ACTION_UNSIGNED": _k("BLACKW_ACTION_UNSIGNED", "BLACKW", False, "bool",
                                 "use +10 wheel scale on all legs (nwwolf PPO contract)"),
    "PART2_ADAPTIVE_CMD": _k("PART2_ADAPTIVE_CMD", "PART2", 0.0, "float",
                             "fraction of envs using state-feedback lane command (0=off, 1=all)"),
    "PART2_CMD_RANDOM": _k("PART2_CMD_RANDOM", "PART2", False, "bool",
                           "non-adaptive envs keep random vy/wz command instead of zero"),
    "PART2_DIFF_MIN": _k("PART2_DIFF_MIN", "PART2", None, "float", "terrain difficulty row minimum override"),
    "PART2_DIFF_MAX": _k("PART2_DIFF_MAX", "PART2", None, "float", "terrain difficulty row maximum override"),
    "PART2_MAX_ROW": _k("PART2_MAX_ROW", "PART2", None, "int", "max initial terrain row"),
    "BLACKW_WHEEL_DQ_CLIP": _k("BLACKW_WHEEL_DQ_CLIP", "BLACKW", 20.0, "float",
                               "clip wheel velocity targets to +/- this (default 20 = deployment command_dq_max; 0=off)"),
    "BLACKW_WHEEL_ACTION_SCALE": _k("BLACKW_WHEEL_ACTION_SCALE", "BLACKW", 10.0, "float",
                                    "wheel velocity target per unit action (deployment contract 10; raise for speed experiments)"),
    "PART2_STAIRS_WHEEL_Z_MIN": _k("PART2_STAIRS_WHEEL_Z_MIN", "PART2", -0.10, "float",
                                   "stairs exit wheel z lower bound (judge min_wheel_z >= -0.10)"),
    "PART2_SLOPE_TOP_HOLD_STEPS": _k("PART2_SLOPE_TOP_HOLD_STEPS", "PART2", 15, "int",
                                     "slope: steps all wheels must hold the top landing (judge hold_s+0.1)"),
    "PART2_SLOPE_EXIT_Z_TOL": _k("PART2_SLOPE_EXIT_Z_TOL", "PART2", 0.08, "float",
                                 "slope: |wheel_z| tolerance at exit (judge exit_ground_tol)"),
    "PART2_BASE_HEIGHT_W": _k("PART2_BASE_HEIGHT_W", "PART2", -4.0, "float",
                              "base_height L2 weight; -4 = legacy anti-crawl, 0 = allow tall climbs"),
    "PART2_UPRIGHT_W": _k("PART2_UPRIGHT_W", "PART2", -0.2, "float",
                          "world-up orientation penalty; 0 lets the trunk follow the slope"),
    "PART2_LIN_VEL_Z_W": _k("PART2_LIN_VEL_Z_W", "PART2", -2.0, "float",
                            "vertical base velocity penalty; lower it when climbing"),
    "PART2_BODY_ANG_VEL_W": _k("PART2_BODY_ANG_VEL_W", "PART2", -0.05, "float",
                               "roll/pitch rate penalty"),
    "PART2_NUM_ROWS": _k("PART2_NUM_ROWS", "PART2", 6, "int",
                        "terrain difficulty rows; 31 makes every official level an exact row (d=k/30)"),
    "PART2_SUBTERRAINS": _k("PART2_SUBTERRAINS", "PART2", None, "str",
                            "comma-separated obstacle categories trained by blackw-official"),
    "PART2_SPAWN_X": _k("PART2_SPAWN_X", "PART2", 0.0, "float", "reset x randomization half-range [m]"),
    "PART2_SPAWN_Y": _k("PART2_SPAWN_Y", "PART2", 0.0, "float", "reset y randomization half-range [m]"),
    "PART2_SPAWN_YAW": _k("PART2_SPAWN_YAW", "PART2", 0.0, "float", "reset yaw randomization half-range [rad]"),
    "PART2_RESET_ROOT_VEL": _k("PART2_RESET_ROOT_VEL", "PART2", 0.0, "float",
                               "reset root velocity half-range (0 matches the judge)"),
    "PART2_WORLD_FRICTION": _k("PART2_WORLD_FRICTION", "PART2", 0.0, "float",
                               "training-only terrain friction override (diagnostic; 0 = official)"),
}

def _env(name: str) -> tuple[str | None, Knob]:
    """Look up a registered knob; unset variables fall back to the registry default."""
    knob = KNOBS.get(name)
    if knob is None:
        raise KeyError(
            f"environment variable {name!r} is not registered in utils/knobs.py; "
            "add it there (scope/doc/default) instead of reading os.environ directly"
        )
    return os.environ.get(name), knob

def raw(name: str) -> str:
    value, knob = _env(name)
    if value is None:
        if knob.default is None:
            raise KeyError(f"required environment variable {name!r} is not set")
        return str(knob.default)
    return value

def optional(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None

def get_float(name: str, default: float | None = None) -> float:
    """Call-site default wins; the registry default is only a fallback/doc."""
    value, knob = _env(name)
    if value is None:
        return default if default is not None else knob.default  # type: ignore[return-value]
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not a float (knob: {knob.doc})") from exc

def get_int(name: str, default: int | None = None) -> int:
    value, knob = _env(name)
    if value is None:
        return default if default is not None else knob.default  # type: ignore[return-value]
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not an int (knob: {knob.doc})") from exc

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

def get_bool(name: str, default: bool | None = None) -> bool:
    value, knob = _env(name)
    if value is None:
        return bool(default) if default is not None else bool(knob.default)
    low = value.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"{name}={value!r} is not a bool (knob: {knob.doc})")

def get_str(name: str, default: str | None = None) -> str:
    value, knob = _env(name)
    if value is None:
        if default is not None:
            return default
        if knob.default is not None:
            return str(knob.default)
        raise KeyError(f"required environment variable {name!r} is not set")
    return value

def get_float_or_none(name: str) -> float | None:
    value = optional(name)
    return None if value is None else get_float(name)

def get_int_or_none(name: str) -> int | None:
    value = optional(name)
    return None if value is None else get_int(name)

def find_prefix(prefix: str) -> list[str]:
    """Return the registered/current variables that start with ``prefix``."""
    return sorted(k for k in os.environ if k.startswith(prefix))

def validate() -> list[str]:
    """Return unregistered variables that look like they belong to us."""
    known = set(KNOBS)
    unknown = sorted(
        key for key in os.environ
        if (key.startswith("RC_") or key.startswith("PART1_") or key.startswith("PART2_"))
        and key not in known
    )
    if unknown:
        print(
            "[knobs] WARNING: unregistered environment variables will be ignored: "
            + ", ".join(unknown),
            file=sys.stderr,
        )
    return unknown

def validate_config() -> list[str]:
    """Cross-knob consistency checks for the velocity-tracking training recipe.

    Returns a list of human-readable problems (empty = OK).  ``alldog_mjlab`` calls
    this at import time: MjLab imports the task packages through entry points and
    catches ``Exception`` while registering tasks, so an invalid combination raised
    inside a task builder would only surface as a confusing "task not registered"
    KeyError later.  Aborting the import here keeps every invalid recipe loud and
    before any GPU/terrain work.

    Only structural constraints are checked (ranges, fractions, mutually exclusive
    switches); per-term numeric ranges stay with the call sites that consume them.
    """
    problems: list[str] = []

    def _add(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def _unit(name: str) -> float:
        value = get_float(name)
        try:
            float(value)
        except (TypeError, ValueError):
            problems.append(f"{name} is not numeric")
            return 0.0
        _add(0.0 <= float(value) <= 1.0, f"{name}={value} must be in [0, 1]")
        return float(value)

    forward = _unit("PART2_CMD_FORWARD_FRAC")
    standing = _unit("PART2_CMD_STANDING_FRAC")
    adaptive = _unit("PART2_ADAPTIVE_CMD")
    _add(
        forward + standing <= 1.0 + 1e-9,
        "PART2_CMD_FORWARD_FRAC + PART2_CMD_STANDING_FRAC = "
        f"{forward + standing:.6f} > 1: the native command range must keep at "
        "least one env",
    )

    vx_min = get_float("PART2_VX_MIN")
    vx_max = get_float("PART2_VX_MAX")
    # Equal bounds are legitimate: fixed-speed training runs use them (e.g. the
    # S1 trench specialist PART2_VX_MIN=PART2_VX_MAX=1.0 and the tau gate 1.0/1.0).
    # Rejecting equality here made historical recipes unreproducible.
    _add(vx_min <= vx_max,
         f"PART2_VX_MIN={vx_min} must be <= PART2_VX_MAX={vx_max}")
    _add(get_float("PART2_LATERAL_CMD") >= 0.0, "PART2_LATERAL_CMD must be >= 0")
    _add(get_float("PART2_YAW_CMD") >= 0.0, "PART2_YAW_CMD must be >= 0")

    if (get_bool("PART2_CMD_RANDOM") or adaptive > 0.0) and optional(
        "PART2_CMD_FORWARD_FRAC"
    ) is not None and forward > 0.0:
        problems.append(
            "PART2_CMD_RANDOM=1 / PART2_ADAPTIVE_CMD>0 keep sampled vy/wz alive, "
            f"but PART2_CMD_FORWARD_FRAC={forward} would zero them for the "
            "forward-only envs; set PART2_CMD_FORWARD_FRAC=0 or remove the "
            "conflicting knob"
        )

    sigma = get_float("RC_TRACK_SIGMA")
    _add(sigma > 0.0, f"RC_TRACK_SIGMA={sigma} must be > 0 (it is a divisor)")
    _add(get_float("RC_TRACK_ERR_L2_W") >= 0.0,
         "RC_TRACK_ERR_L2_W must be >= 0 (0 = disabled)")
    max_abs_vx = get_float("RC_CMD_CURR_MAX_VX")
    _add(max_abs_vx >= 1.0,
         f"RC_CMD_CURR_MAX_VX={max_abs_vx} is below the command-curriculum initial "
         "range ±1.0; the curriculum ceiling would already exclude it")
    return problems

def overrides() -> dict[str, str]:
    """Every registered knob that is currently set (for run manifests)."""
    return {name: os.environ[name] for name in KNOBS if name in os.environ}
