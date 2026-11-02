"""blackW wheel-legged task assembly (first milestone: flat velocity tracking).

This module starts from the validated Black flat task and replaces exactly the
robot / wheels' action interface / observation contract / reset events. The
reward, command and termination wiring stay on the Black baseline so the first
blackW policy can be trained and evaluated end-to-end before adding competition
terrain.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from alldog_mjlab.robots.blackw import (
    BLACKW_FOOT_NAMES,
    BLACKW_JOINT_NAMES,
    BLACKW_WHEEL_JOINTS,
    get_blackw_locked_robot_cfg,
    get_blackw_robot_cfg,
)
from alldog_mjlab.tasks.velocity.common.course import (
    _goal_positions,
    apply_command_fractions,
    apply_command_resampling,
    apply_pose_command_contract,
    apply_traversal_base,
    competition_goal_reward,
)
from alldog_mjlab.tasks.velocity.common.velocity_env import (
    BLACK_COMMAND_NAME,
    build_velocity_env_cfg,
)
from alldog_mjlab.tasks.velocity.blackw.actions import (
    LockedWheelLegActionCfg,
    WheelLegActionCfg,
)
from alldog_mjlab.utils import knobs
from alldog_mjlab.tasks.velocity.profiles.part2.adaptive_command import (
    make_adaptive_command_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.part2.flat_column import (
    FlatCommandColumnTerrainCfg,
    make_column_mixed_command_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.pose import (
    apply_pose_edge_terrain,
    apply_pose_low_speed_mixture,
)
from alldog_mjlab.tasks.velocity.profiles.vt import apply_vt_contract
from alldog_mjlab.tasks.velocity.profiles.vt.locked import (
    all_feet_contact,
    feet_air_time,
    foot_lift,
    gait_phase,
    foot_clearance as locked_foot_clearance,
    foot_slip,
    stance_joint,
    wheel_axis_level,
    wheel_lock,
)
from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
    PART2_BASE_REF_Z,
    PART2_HOLD_STEPS,
    PART2_LANE_HALF_WIDTH,
    PART2_STEP_DESCEND_CLEARANCE_M,
    PART2_WALL_BACK_CLEARANCE_M,
    PART2_WALL_X,
    PART2_GROUND_WHEEL_TOL_M,
    PART2_WHEEL_RADIUS_M,
    PART2_WHEEL_SURFACE_TOL_M,
    _lane_deviation,
    _masked_out_of_course,
    _masked_part2_failure,
    _part2_state_buffers,
    _part2_success,
    _part2_success_streak,
    _part2_terrain_state,
    _platform_top_success,
    _platform_top_z,
)
from alldog_mjlab.tasks.velocity.profiles.part2.rewards import (
    PART2_TRAVERSAL_MIN_CMD_VX,
    _climb_height_reward,
    _course_progress_delta,
    _masked_competition_goal_reward,
    _masked_competition_progress_delta,
    _masked_course_climb_height_reward,
    _masked_part2_clear_reward,
    _part2_clear_reward,
    _platform_height_reward,
    _plum_foothold_reward,
    _trench_wheel_drop_penalty,
)
from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg

from alldog_mjlab.tasks.velocity.profiles.part2.terrain import (
    ExactHighPlatformTerrainCfg,
    ExactHighWallTerrainCfg,
    ExactPlumTerrainCfg,
    ExactSlopeTerrainCfg,
    ExactStairsTerrainCfg,
    ExactTrenchTerrainCfg,
)

BLACKW_ACTION_TERM_ORDER = tuple(f"wheel_leg_{leg.lower()}" for leg in BLACKW_FOOT_NAMES)
BLACKW_ACTOR_OBS_TERM_ORDER = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)
BLACKW_ACTOR_HISTORY_LENGTH = 6
BLACKW_ACTOR_OBS_SCALE = {
    "command": (2.0, 2.0, 0.25),
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "joint_pos": 1.0,
    "joint_vel": 0.05,
    "actions": 1.0,
}


def _wheel_leg_scale(leg: str) -> dict[str, float]:
    wheel_scale = knobs.get_float("BLACKW_WHEEL_ACTION_SCALE", 10.0)
    # The official blackW deployment contract mirrors the wheel action sign on the
    # right legs (rl_sar action_scale = -10 for FR/RR); training must use the same
    # convention or the official HIM warm start drives the right wheels backwards.
    # ``BLACKW_ACTION_UNSIGNED`` switches to the nwwolf PPO contract (+10 all four
    # wheels) so a nwwolf PPO checkpoint can be warm-started without the
    # previous-action observation mismatch.
    unsigned = knobs.get_bool("BLACKW_ACTION_UNSIGNED", False)
    return {
        f"{leg}_hip_joint": 0.25,
        f"{leg}_thigh_joint": 0.25,
        f"{leg}_calf_joint": 0.25,
        f"{leg}_wheel_joint": wheel_scale if unsigned or leg not in ("FR", "RR") else -wheel_scale,
    }


def _blackw_joint_pos_rel(env, asset_cfg: SceneEntityCfg, biased: bool = False):
    """Joint position error, with wheel positions forced to zero (official contract)."""
    obs = envs_mdp.joint_pos_rel(env, asset_cfg=asset_cfg, biased=biased)
    names = asset_cfg.joint_names
    wheel_cols = [i for i, name in enumerate(names) if name in BLACKW_WHEEL_JOINTS]
    if wheel_cols:
        obs = obs.clone()
        obs[:, wheel_cols] = 0.0
    return obs


def _override_robot(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.scene.entities = {"robot": get_blackw_robot_cfg()}


def _override_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
    sensors = []
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            continue  # blackW has no foot sites; clearance reward is removed below
        if sensor.name == "feet_ground_contact" and isinstance(sensor, ContactSensorCfg):
            sensor.primary = ContactMatch(
                mode="body",
                pattern=tuple(f"{leg}_foot" for leg in BLACKW_FOOT_NAMES),
                entity="robot",
            )
        sensors.append(sensor)
    cfg.scene.sensors = tuple(sensors)


def _override_actions(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.actions = {
        f"wheel_leg_{leg.lower()}": WheelLegActionCfg(
            entity_name="robot",
            actuator_names=tuple(
                f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf", "wheel")
            ),
            scale=_wheel_leg_scale(leg),
            preserve_order=True,
        )
        for leg in BLACKW_FOOT_NAMES
    }


def _override_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    actor = cfg.observations["actor"]
    actor.history_length = BLACKW_ACTOR_HISTORY_LENGTH
    actor.flatten_history_dim = True
    terms = actor.terms
    terms["joint_pos"] = replace(
        terms["joint_pos"],
        func=_blackw_joint_pos_rel,
        params={
            **terms["joint_pos"].params,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    terms["joint_vel"] = replace(
        terms["joint_vel"],
        func=envs_mdp.joint_vel_rel,
        params={
            **terms["joint_vel"].params,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            ),
        },
    )
    # blackW's IMU sensor is named imu_gyro, not the black XML's imu_ang_vel; use the
    # entity-state observations instead of the base config's builtin-sensor terms.
    terms["base_ang_vel"] = replace(
        terms["base_ang_vel"], func=envs_mdp.base_ang_vel, params={}
    )
    terms["projected_gravity"] = replace(
        terms["projected_gravity"], func=envs_mdp.projected_gravity, params={}
    )
    actor.terms = {name: terms[name] for name in BLACKW_ACTOR_OBS_TERM_ORDER}
    for name, scale in BLACKW_ACTOR_OBS_SCALE.items():
        actor.terms[name] = replace(actor.terms[name], scale=scale)

    joint_obs = ObservationTermCfg(
        func=envs_mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            )
        },
    )
    joint_vel_obs = ObservationTermCfg(
        func=envs_mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=BLACKW_JOINT_NAMES, preserve_order=True
            )
        },
    )
    cfg.observations["critic"] = replace(
        cfg.observations["critic"],
        terms={
            "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
            "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
            "joint_pos": joint_obs,
            "joint_vel": joint_vel_obs,
            "actions": ObservationTermCfg(func=envs_mdp.last_action),
            "command": ObservationTermCfg(
                func=envs_mdp.generated_commands,
                params={"command_name": "twist"},
            ),
            "height_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": "terrain_scan"},
                scale=0.2,
            ),
        },
    )


def _override_events(cfg: ManagerBasedRlEnvCfg) -> None:

    base_events = dict(cfg.events)
    reset_base = base_events.pop("reset_base")
    # Gate-matched spawn randomization (docs gate protocol: x +-0.10 m, y +-0.15 m,
    # yaw +-0.05 rad).  Disabled by default; set the env vars to enable.
    spawn_x = knobs.get_float("PART2_SPAWN_X", 0.0)
    spawn_y = knobs.get_float("PART2_SPAWN_Y", 0.0)
    spawn_yaw = knobs.get_float("PART2_SPAWN_YAW", 0.0)
    if max(spawn_x, spawn_y, spawn_yaw) > 0.0:
        params = dict(reset_base.params)
        pose_range = dict(params.get("pose_range") or {})
        pose_range.update({"x": (-spawn_x, spawn_x), "y": (-spawn_y, spawn_y),
                           "yaw": (-spawn_yaw, spawn_yaw)})
        params["pose_range"] = pose_range
        reset_base = replace(reset_base, params=params)
    # Judge spawns the robot with zero root velocity.  Match that by default;
    # override with PART2_RESET_ROOT_VEL for explicit DR experiments (audit P1-12).
    root_vel = knobs.get_float("PART2_RESET_ROOT_VEL", 0.0)
    if reset_base.params.get("velocity_range"):
        rv_params = dict(reset_base.params)
        rv_params["velocity_range"] = {
            k: (-root_vel, root_vel) for k in reset_base.params["velocity_range"]
        }
        reset_base = replace(reset_base, params=rv_params)
    events = {
        "reset_base": reset_base,
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.03, 0.03),
                "velocity_range": (-1.0, 1.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=BLACKW_JOINT_NAMES),
            },
        ),
    }
    # Keep the black competition task's domain-randomization terms.  The old code
    # replaced the whole dict and silently dropped them (audit P0-1, 2026-10-26).
    # Reset terms are superseded by the two above; the rest are DR.  Not every black
    # DR term is valid on blackW (foot collision geoms are named differently and the
    # action path is JointPosition rather than IdealPd), so keep a curated allowlist
    # and make it explicit instead of silently dropping anything.
    allowed = {
        name.strip() for name in knobs.get_str(
            "PART2_DR_EVENTS", "payload_mass,base_com,push_robot").split(",")
        if name.strip()
    }
    dropped = []
    for name, term in base_events.items():
        if name.startswith("reset_"):
            continue
        if name in allowed:
            events[name] = term
        else:
            dropped.append(name)
    cfg.events = events
    if dropped:
        print(f"[blackw] DR events dropped (unsupported on blackW): {sorted(dropped)}", flush=True)
    print(f"[blackw] DR events enabled: {sorted(allowed)}", flush=True)


def _course_climb_reward(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Climb-height shaping that only pays before the goal line is crossed."""
    import torch

    base = _climb_height_reward(env, asset_cfg=asset_cfg)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is None:
        return base
    goal, valid = goal_data
    root_x = env.scene[asset_cfg.name].data.root_link_pos_w[:, 0]
    mask = (root_x <= goal[:, 0]) & valid
    return base * mask.to(base.dtype)




def _override_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    # Foot clearance uses Black foot sites, which do not exist on blackW; the rest of
    # the frozen Black reward set is robot-agnostic.
    cfg.rewards.pop("foot_clearance", None)



# Part 2 training sweep defaults (the Part 1 competition generator previously
# supplied these; the exact/official tasks replace only the sub-terrains).
# 6 rows = the legacy sweep (d = 0,.2,.4,.6,.8,1.0).  Official levels are not
# all on that grid (platform30 is d=2/3, stairs20 d=1/2, slope35 d=3/5), so a
# mixed run never samples platform30 exactly.  PART2_NUM_ROWS=31 makes
# d = k/30, which represents every official difficulty exactly
# (platform 0,10,20,30; stairs20=15/30, stairs30=30/30; slope30=12/30,
# slope35=18/30; wall 0,10,20,30) and randomises the init level across all rows.
PART2_TRAIN_NUM_ROWS = knobs.get_int("PART2_NUM_ROWS", 6)
PART2_TRAIN_DIFFICULTY_RANGE = (0.0, 1.0)


def _apply_part2_command_cfg(
    cfg,
    *,
    vx_min_default: float,
    vx_max_default: float,
    lateral_default: float,
    yaw_default: float,
) -> None:
    """Apply the Part 2 command contract (ranges + native mix + adaptive/random).

    Every value comes from the registered knobs; ``*_default`` are the historical
    per-task values at the call site, so an unset knob reproduces the old cfg
    exactly (MjLab's ``rel_forward_envs`` / ``rel_standing_envs`` are set to the
    same 1.0 / 0.0 as before).  The 2026-10-31 velocity-tracking recipe reuses the
    same code path with forward/standing/native ranges:

    * ``PART2_CMD_FORWARD_FRAC`` / ``PART2_CMD_STANDING_FRAC`` -> effective sampler
      mix (via ``course.apply_command_fractions``);
    * ``PART2_ADAPTIVE_CMD`` / ``PART2_CMD_RANDOM`` -> state-feedback / random
      vy-wz command term, same precedence and defaults as the historical mixed
      builder (adaptive first; random-vy/wz switch otherwise).
    """
    command = cfg.commands[BLACK_COMMAND_NAME]
    vx_min = knobs.get_float("PART2_VX_MIN", vx_min_default)
    vx_max = knobs.get_float("PART2_VX_MAX", vx_max_default)
    # Equal bounds are legitimate: fixed-speed training (S1 trench
    # PART2_VX_MIN=PART2_VX_MAX=1.0, tau 1.0/1.0) samples a constant command.
    if vx_min > vx_max:
        raise ValueError(f"PART2_VX_MIN={vx_min} must be <= PART2_VX_MAX={vx_max}")
    command.ranges.lin_vel_x = (vx_min, vx_max)
    lateral = knobs.get_float("PART2_LATERAL_CMD", lateral_default)
    yaw = knobs.get_float("PART2_YAW_CMD", yaw_default)
    if lateral < 0.0 or yaw < 0.0:
        raise ValueError(
            f"PART2_LATERAL_CMD={lateral} / PART2_YAW_CMD={yaw} must be >= 0"
        )
    command.ranges.lin_vel_y = (-lateral, lateral)
    command.ranges.ang_vel_z = (-yaw, yaw)
    apply_command_fractions(command)
    apply_command_resampling(command)

    adaptive_frac = knobs.get_float("PART2_ADAPTIVE_CMD", 0.0)
    cmd_random = knobs.get_bool("PART2_CMD_RANDOM", False)
    if not math.isfinite(adaptive_frac) or not 0.0 <= adaptive_frac <= 1.0:
        raise ValueError(
            f"PART2_ADAPTIVE_CMD={adaptive_frac!r} must be a float in [0, 1]"
        )
    forward_explicit = knobs.optional("PART2_CMD_FORWARD_FRAC") is not None
    forward_frac = knobs.get_float("PART2_CMD_FORWARD_FRAC", 1.0)
    if (cmd_random or adaptive_frac > 0.0) and forward_explicit and forward_frac > 0.0:
        raise ValueError(
            "PART2_CMD_RANDOM=1 / PART2_ADAPTIVE_CMD>0 keep sampled vy/wz alive, "
            "but PART2_CMD_FORWARD_FRAC="
            f"{forward_frac} would zero them for the forward-only envs; set "
            "PART2_CMD_FORWARD_FRAC=0 or remove the conflicting knob"
        )
    if cmd_random:
        # Orthogonal switch independent of adaptive: keep the sampled vy/wz
        # commands (their values are also the tracking-reward targets), i.e. no
        # forward-only zeroing.
        command.rel_forward_envs = 0.0
    if adaptive_frac > 0.0:
        cfg.commands[BLACK_COMMAND_NAME] = make_adaptive_command_cfg(
            command,
            adaptive_frac=adaptive_frac,
            command_random=cmd_random,
        )


def _traversal_lane_termination(
    *, max_abs_y: float, flat_column: str | None = None
) -> TerminationTermCfg:
    """Lane-bypass termination that only kills traversal-intent envs.

    Mirrors the 2026-10-31 Phase-2 rule: the course lane applies to envs whose
    commanded vx is above the traversal proxy, so standing / negative-speed envs
    are not terminated by a rule that belongs to obstacle attempts.  Under the
    default forward-only mix every env passes the gate, so the default behaviour
    is unchanged.  Success-predicate lane semantics are untouched.

    ``flat_column`` additionally exempts the flat command-tracking terrain column
    (2026-10-31 flat-column recipe): those envs have no course/lane semantics.
    """
    return TerminationTermCfg(
        func=_lane_deviation,
        params={
            "max_abs_y": max_abs_y,
            "command_name": BLACK_COMMAND_NAME,
            "min_cmd_vx": PART2_TRAVERSAL_MIN_CMD_VX,
            "flat_column": flat_column,
        },
    )


def _traversal_part2_failure_termination() -> TerminationTermCfg:
    """Judge failure predicate applied only to traversal-intent envs (audit BL-3b).

    Phase 2 rehearses standing / reverse / large-lateral velocity commands on the
    obstacle terrain; the judge-mirrored ``_part2_failure`` includes the
    ``side_out`` course violation that would otherwise terminate every native env
    as soon as it turns.  The predicate itself (and success) is unchanged; only the
    training termination wiring is gated.  No-op under the default forward-only mix.
    """
    return TerminationTermCfg(
        func=_masked_part2_failure,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": BLACK_COMMAND_NAME,
            "min_cmd_vx": PART2_TRAVERSAL_MIN_CMD_VX,
        },
    )


def _traversal_out_of_course_termination(
    *, goal_name: str = "goal"
) -> TerminationTermCfg:
    """Course-exit termination gated by traversal intent (audit BL-3b companion).

    Same Phase-2 rationale as ``_traversal_part2_failure_termination``: a native
    velocity-rehearsal env must not be killed for drifting past the goal margin.
    The predicate is unchanged; only the training wiring is gated, and the default
    forward-only mix makes the gate a no-op.
    """
    return TerminationTermCfg(
        func=_masked_out_of_course,
        params={
            "goal_name": goal_name,
            "margin": knobs.get_float("PART2_OUT_OF_COURSE_MARGIN", 0.8),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": BLACK_COMMAND_NAME,
            "min_cmd_vx": PART2_TRAVERSAL_MIN_CMD_VX,
        },
    )


def _apply_part2_reward_weights(cfg) -> None:
    """Per-event overrides for the stability terms inherited from the flat task.

    The legacy ``base_height`` term (-4.0, target 0.43 m world z) was added to
    stop the Part1 "crawl under the bar" degeneracy, but on Part2 tall climbs
    (30 cm stairs, 45 deg slope) it penalises the root height the robot MUST
    gain: at a 1.8 m stair landing it is -4*(2.3-0.43)^2 per step.  The same
    holds for the world-up ``upright`` penalty (it fights slope-aligned posture)
    and ``lin_vel_z`` (climbing has positive vz).  Expose them as knobs so a
    climbing curriculum can zero them without touching the flat baseline.
    """
    overrides = {
        "base_height": knobs.get_float("PART2_BASE_HEIGHT_W", -4.0),
        "upright": knobs.get_float("PART2_UPRIGHT_W", -0.2),
        "lin_vel_z": knobs.get_float("PART2_LIN_VEL_Z_W", -2.0),
        "body_ang_vel": knobs.get_float("PART2_BODY_ANG_VEL_W", -0.05),
    }
    for term, weight in overrides.items():
        if term in cfg.rewards:
            cfg.rewards[term] = replace(cfg.rewards[term], weight=float(weight))


def _part2_base(play: bool):
    """Generic traversal base + the Part 2 six-row official-difficulty sweep."""
    cfg = _apply_blackw_overrides(
        apply_traversal_base(build_velocity_env_cfg(play=play, rough=True), play=play)
    )
    _apply_part2_reward_weights(cfg)
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = PART2_TRAIN_NUM_ROWS
    generator.difficulty_range = PART2_TRAIN_DIFFICULTY_RANGE
    return cfg

def _apply_blackw_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    _override_robot(cfg)
    _override_sensors(cfg)
    _override_actions(cfg)
    _override_observations(cfg)
    _override_events(cfg)
    _override_rewards(cfg)
    return cfg


def blackw_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _apply_blackw_overrides(build_velocity_env_cfg(play=play, rough=False))
    if knobs.get_bool("RC_POSE_CMD", False):
        apply_pose_command_contract(cfg)
        # Low-speed mixture sampling (dead-zone fix); inert unless
        # RC_POSE_LOW_FRAC > 0, and must run after the envelope contract above
        # so the copied ranges are the pose envelope ones.
        apply_pose_low_speed_mixture(cfg)
    if knobs.get_bool("RC_POSE_EDGES", False):
        # Platform-edge terrain + spawn randomization + climb-safe
        # terminations (pose-edge robustness recipe).
        apply_pose_edge_terrain(cfg, play=play)
    return cfg


def blackw_vt_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity-tracking primitive (Part 1 handover) on flat terrain.

    Explicit recipe in ``profiles/vt/assembly.py``: pose command envelope
    (vx/reverse/stop/low-speed/in-place yaw), tracking-dominant rewards, joint
    stance anchor, turn-drift penalty, RecoveryLoco-style random states/pushes
    and wheel-friction DR.  No env-var knobs: reproducible by task id.
    """
    cfg = _apply_blackw_overrides(build_velocity_env_cfg(play=play, rough=False))
    apply_vt_contract(cfg, play=play, edges=False)
    return cfg


def blackw_vt_edge_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity-tracking primitive + platform-edge terrain (Part 1 edge handover).

    Same command/reward contract as ``blackw-vt``; the terrain adds 0.05-0.20 m
    platform edges with on-top / beside spawns so the primitive can recover a
    dropped wheel under command (bridge platforms, stair lips).
    """
    cfg = _apply_blackw_overrides(build_velocity_env_cfg(play=play, rough=False))
    apply_vt_contract(cfg, play=play, edges=True)
    return cfg


def blackw_vt_locked_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity-tracking primitive on the 12-DoF locked-wheel (point-foot) robot.

    Wheel hinges are PD-held at 0 (``get_blackw_locked_robot_cfg``); the policy
    walks on hip/thigh/calf with a RecoveryLoco-style contact/gait reward set.
    The 16-action deployment contract is kept: the policy's 4 wheel outputs are
    ignored, so export/PolicyRunner only need a ``lock_wheels`` mode.
    """
    cfg = _apply_blackw_overrides(build_velocity_env_cfg(play=play, rough=False))
    cfg.scene.entities = {"robot": get_blackw_locked_robot_cfg()}
    cfg.actions = {
        f"wheel_leg_{leg.lower()}": LockedWheelLegActionCfg(
            entity_name="robot",
            actuator_names=tuple(
                f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf", "wheel")
            ),
            # The 4th (wheel) output is ignored by LockedWheelLegAction; a
            # single float is required here (tuple scales are unsupported).
            scale=0.25,
            preserve_order=True,
        )
        for leg in BLACKW_FOOT_NAMES
    }
    apply_vt_contract(cfg, play=play, edges=False)
    # Tracking is the primary task; gait shaping above is deliberately weak.
    cfg.rewards["track_linear_velocity"] = replace(
        cfg.rewards["track_linear_velocity"], weight=3.0)
    cfg.rewards["track_linear_velocity_l2"] = replace(
        cfg.rewards["track_linear_velocity_l2"], weight=-0.5)
    # Rolling-wheel oriented terms do not apply; the locked variant uses the
    # foot-contact/gait terms from profiles/vt/locked.py instead.
    cfg.rewards.pop("wheel_plane_level", None)
    cfg.rewards.pop("wheel_rotation_follow", None)
    locked_terms = {
        # vt_locked_v3 learned to step in place (foot_lift 1.4, tracking poor):
        # gait shaping must stay subordinate to command tracking.  Tracking is
        # raised for this task and the lift/contact-shaping weights reduced.
        "wheel_lock": (wheel_lock, -10.0, {}),
        "feet_air_time": (feet_air_time, 0.5, {}),
        "gait_phase": (gait_phase, 1.0, {}),
        "foot_lift": (foot_lift, 0.2, {}),
        "all_feet_contact": (all_feet_contact, -0.5, {}),
        "foot_clearance": (locked_foot_clearance, -0.05, {}),
        "foot_slip": (foot_slip, -8.0, {}),
        "stance_joint": (stance_joint, -0.5, {}),
        "wheel_axis_level": (wheel_axis_level, -1.0, {}),
    }
    for name, (func, weight, params) in locked_terms.items():
        cfg.rewards[name] = RewardTermCfg(func=func, weight=weight, params=params)
    # Explicit asset_cfgs are resolved by the reward manager; a function default
    # stays slice(None) and silently selects every geom/joint.
    cfg.rewards["foot_lift"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=tuple(f"{leg}_wheel_collision" for leg in BLACKW_FOOT_NAMES)
    )
    cfg.rewards["foot_clearance"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=tuple(f"{leg}_wheel_collision" for leg in BLACKW_FOOT_NAMES)
    )
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=tuple(f"{leg}_wheel_collision" for leg in BLACKW_FOOT_NAMES)
    )
    cfg.rewards["wheel_axis_level"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=tuple(f"{leg}_foot" for leg in BLACKW_FOOT_NAMES)
    )
    cfg.rewards["wheel_lock"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=BLACKW_WHEEL_JOINTS
    )
    cfg.rewards["stance_joint"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(".*hip_joint", ".*thigh_joint", ".*calf_joint")
    )
    # Walking envelope: lower top speed than the rolling primitive.
    cmd = cfg.commands[BLACK_COMMAND_NAME]
    cmd = replace(cmd, ranges=replace(cmd.ranges, lin_vel_x=(-0.30, 0.60),
                                      ang_vel_z=(-0.60, 0.60)))
    cfg.commands[BLACK_COMMAND_NAME] = cmd
    return cfg


def blackw_competition_env_cfg(
    play: bool = False, subterrains: tuple[str, ...] | None = None
) -> ManagerBasedRlEnvCfg:
    """Competition-proxy terrain (Part1/Part2 obstacles) with the blackW contract.

    ``subterrains`` restricts the generator to a subset (specialist policies), e.g.
    only the high-platform, slope or trench columns for the high Part2 tiers.
    """
    cfg = _part2_base(play)
    if subterrains and cfg.scene.terrain is not None:
        generator = cfg.scene.terrain.terrain_generator
        assert generator is not None
        for name in subterrains:
            if name not in generator.sub_terrains:
                raise ValueError(f"unknown blackW sub-terrain: {name}")
        generator.sub_terrains = {
            name: generator.sub_terrains[name] for name in subterrains
        }
    if cfg.scene.terrain is not None and not play:
        # Stage 2 targets the high-value Part2 tiers: initialise on rows 0..5
        # (high platform 30/40, slope 35/45, trench 20/30), not only the easy rows.
        cfg.scene.terrain.max_init_terrain_level = 5
    if not play:
        # A goal-x-only reward lets the policy walk around obstacle patches (y up to
        # several metres on real Part2 scenes). The judges score inside a narrow lane;
        # terminate lateral bypass so the policy must interact with the obstacle.
        # The traversal-intent gate keeps the rule off standing / reverse envs (no-op
        # under the default forward-only mix).
        cfg.terminations["lane_deviation"] = _traversal_lane_termination(
            max_abs_y=PART2_LANE_HALF_WIDTH
        )
    return cfg


def blackw_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return blackw_competition_env_cfg(play=play, subterrains=("platform",))


def blackw_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return blackw_competition_env_cfg(
        play=play, subterrains=("smooth_slope_up", "smooth_slope_down")
    )


def blackw_trench_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return blackw_competition_env_cfg(play=play, subterrains=("trench",))


# Exact specialist terrain class -> official Part2 judge kind.  The sub-terrain
# key IS the judge kind: ``_part2_terrain_state`` branches on the column name, so
# naming the single column ``"exact"`` made every part2_* term dead code.
_EXACT_TERRAIN_KIND: dict[str, str] = {
    "ExactHighPlatformTerrainCfg": "platform",
    "ExactHighWallTerrainCfg": "high_wall",
    "ExactSlopeTerrainCfg": "slope",
    "ExactStairsTerrainCfg": "stairs",
    "ExactTrenchTerrainCfg": "trench",
    "ExactPlumTerrainCfg": "plum",
}


def _apply_deployment_sim_options(cfg, *, force: bool = False) -> None:
    """Match the official Part2 scene ``<option>`` block used by the judge.

    The official ``ability_field/map.xml`` scenes run MuJoCo with elliptic cone,
    PGS solver, ``impratio=100``, Euler integrator, 100 solver iterations and 35
    CCD iterations.  MjLab defaults (newton / pyramidal / impratio=1 /
    implicitfast / 10 iterations) give a visibly different wheel-friction
    response on steeper slopes: a BC policy that cleared the 35 deg judge scene
    could not climb the same 35 deg slope in the training env, so PPO gradients
    pushed it away from the (already correct) deployment behaviour.
    """

    if not force and not knobs.get_bool("PART2_DEPLOY_SIM"):
        return
    sim = cfg.sim.mujoco
    sim.integrator = "euler"
    sim.cone = "elliptic"
    sim.jacobian = "sparse"
    # mujoco_warp does not implement mjSOL_PGS (the official scene uses PGS); keep
    # Newton but raise the iteration count / friction settings to match as closely
    # as the backend allows.
    sim.solver = "newton"
    sim.iterations = 100
    sim.ls_iterations = 50
    sim.ls_tolerance = 0.01
    sim.impratio = 100.0
    sim.ccd_iterations = knobs.get_int("RC_CCD_ITERS", 35)
    # The deployment scenes spawn the robot at the judge spawn height (0.5 m base),
    # not at the training init height (0.69 m).
    robot = cfg.scene.entities.get("robot")
    if robot is not None and getattr(robot, "init_state", None) is not None:
        robot.init_state.pos = (0.0, 0.0, 0.52)


def _exact_env_cfg(terrain_cfg, play: bool) -> ManagerBasedRlEnvCfg:

    cfg = _part2_base(play)
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    generator = cfg.scene.terrain.terrain_generator
    terrain_cfg.flat_patch_sampling = {
        "goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.3)
    }
    kind = _EXACT_TERRAIN_KIND.get(type(terrain_cfg).__name__)
    if kind is None:
        raise ValueError(f"unknown exact terrain cfg: {type(terrain_cfg).__name__}")
    generator.sub_terrains = {kind: terrain_cfg}
    _apply_deployment_sim_options(cfg)
    # Specialist runs fix a difficulty row so the whole batch trains one tier
    # (e.g. slope 45 deg, stairs 30 cm); default keeps the full 0..1 sweep.
    dmin = knobs.get_float("PART2_DIFF_MIN", 0.0)
    dmax = knobs.get_float("PART2_DIFF_MAX", 1.0)
    if (dmin, dmax) != (0.0, 1.0):
        generator.difficulty_range = (dmin, dmax)
    # Exact ramps/platforms create many wheel/edge contacts; blackW smoke hit
    # nconmax=172 on the 45-degree ramp.  Default is generous for single-line
    # experiments; when two training lines share one GPU, cap the warp capacity
    # (e.g. RC_NCONMAX=256, RC_NJMAX=1024) to fit a half-GPU memory budget.
    cfg.sim.nconmax = knobs.get_int("RC_NCONMAX", 1024)
    cfg.sim.njmax = knobs.get_int("RC_NJMAX", 4096)
    # CCD iterations come from _apply_deployment_sim_options() (judge value 35).
    # The old `cfg.sim.ccd_iterations = 0` here wrote a non-existent field, i.e. it
    # never disabled anything (audit P1-1, 2026-10-26).  Use RC_CCD_ITERS to
    # change it if a mujoco-warp CCD assert reappears.
    if not play:
        dmin = knobs.get_float("PART2_DIFF_MIN", 0.0)
        # Command profile for specialist runs (obstacle traversal is straight-lane).
        # Symmetric with the mixed/official task: same registered knobs, adaptive /
        # random switches and forward/standing mix; the call-site defaults keep the
        # historical 0.40-1.00 straight-lane contract when no knob is set.
        _apply_part2_command_cfg(
            cfg,
            vx_min_default=0.40,
            vx_max_default=1.00,
            lateral_default=0.10,
            yaw_default=0.20,
        )
        # Start exact tasks on the easy rows unless a fixed hard tier or an
        # explicit curriculum row is requested (PART2_MAX_ROW=0..num_rows-1).
        max_row = knobs.get_int("PART2_MAX_ROW", -1)
        if max_row >= 0:
            num_rows = int(generator.num_rows)
            cfg.scene.terrain.max_init_terrain_level = min(
                max_row, max(0, num_rows - 1))
        else:
            cfg.scene.terrain.max_init_terrain_level = (
                0 if dmin > 0.0 else 1)
        # Staged specialist curriculum: both bounds pin the gap/height range while
        # PART2_MAX_ROW caps the initially sampled difficulty rows.
        diff_lo = knobs.optional("PART2_DIFF_MIN")
        diff_hi = knobs.optional("PART2_DIFF_MAX")
        if diff_lo is not None and diff_hi is not None:
            generator.difficulty_range = (float(diff_lo), float(diff_hi))
        cfg.terminations["lane_deviation"] = _traversal_lane_termination(
            max_abs_y=PART2_LANE_HALF_WIDTH
        )
        cfg.terminations.pop("competition_goal_reached", None)
        # The base competition cfg ships competition_goal (weight 5.0, pays every
        # step the root is past the goal line) and a state-based
        # competition_progress (weight 1.5).  They must be removed here, not just
        # shadowed: a policy could farm them by camping past the obstacle line.
        cfg.rewards.pop("competition_goal", None)
        cfg.rewards.pop("climb_height", None)
        cfg.terminations["part2_success"] = TerminationTermCfg(
            func=_part2_success, params={"asset_cfg": SceneEntityCfg("robot")})
        cfg.terminations["part2_failure"] = _traversal_part2_failure_termination()
        cfg.rewards["competition_progress"] = RewardTermCfg(
            func=_course_progress_delta,
            weight=knobs.get_float("PART2_PROGRESS_W", 8.0),
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        goal_w = knobs.get_float("PART2_GOAL_W", 0.0)
        if goal_w > 0.0:
            cfg.rewards["competition_goal"] = RewardTermCfg(
                func=competition_goal_reward,
                weight=goal_w,
                params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
            )
        climb_w = knobs.get_float("PART2_CLIMB_W", 0.0)
        if climb_w > 0.0:
            cfg.rewards["climb_height"] = RewardTermCfg(
                func=_course_climb_reward,
                weight=climb_w,
            )
        cfg.terminations["part2_out_of_course"] = _traversal_out_of_course_termination()
        if knobs.get_float("PART2_CLEAR_W", 25.0) > 0.0:
            cfg.rewards["part2_clear"] = RewardTermCfg(
                func=_part2_clear_reward,
                weight=knobs.get_float("PART2_CLEAR_W", 25.0),
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
    return cfg


def blackw_stairs_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Exact official stairs task (full up + landing + down staircase)."""
    return _exact_env_cfg(ExactStairsTerrainCfg(), play)


def blackw_wall_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Exact official high-wall task (cross_bar 10..40 cm)."""
    return _exact_env_cfg(ExactHighWallTerrainCfg(), play)


def blackw_platform_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _exact_env_cfg(
        ExactHighPlatformTerrainCfg(side_walls=False, lane_half_width=0.70), play
    )
    if not play:
        cfg.rewards["platform_height"] = RewardTermCfg(
            func=_platform_height_reward, weight=50.0
        )
        # Goal-x alone is gameable (CCD-off penetration / walking around). Replace
        # the sparse x-goal with a real platform-top condition for the pilot.
        cfg.rewards.pop("competition_goal", None)
        cfg.terminations.pop("competition_goal_reached", None)
        # The exact task already carries the judge-mirrored ``part2_success`` from
        # ``_exact_env_cfg``.  Do NOT add the old proxy
        # ``official_platform_success`` here: it only checks base x/z and made the
        # policy look successful in training while eval (all four wheels + exit +
        # base-height band) said ``not_on_platform``.  Keep the strict term so
        # specialist fine-tuning optimises the thing the judge measures.
        cfg.terminations.pop("official_platform_success", None)
        # Pilot row 0 (10 cm) only.  Row promotion is a separate experiment and is
        # not implemented here; thresholds above are row-relative (audit P1-2).
        if cfg.scene.terrain is not None:
            cfg.scene.terrain.max_init_terrain_level = 0
        cfg.terminations["lane_deviation"].params["max_abs_y"] = PART2_LANE_HALF_WIDTH
    return cfg


def blackw_slope_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _exact_env_cfg(ExactSlopeTerrainCfg(), play)
    # Official slope is up ramp + landing + down ramp, i.e. 6.6 m of obstacle
    # from spawn; the default 8x8 patch puts the landing/down ramp over the
    # patch edge.  Use the same 16x8 patch the mixed task uses.
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.size = (16.0, 8.0)
    return cfg


def blackw_trench_exact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _exact_env_cfg(ExactTrenchTerrainCfg(), play)


def blackw_official_mixed_env_cfg(
    play: bool = False, subterrains: tuple[str, ...] | None = None
) -> ManagerBasedRlEnvCfg:
    """Official Part2 geometry mixed terrain (5 categories, no proxy shapes).

    Each column is one official obstacle type and each row is a difficulty step;
    ``PART2_MAX_ROW`` (default: hardest row) sets the initial difficulty row so a
    manual success-gated curriculum can start easy and promote.  The default
    command range covers the slow deployment speeds (0.25-1.0 m/s) at which the
    official blackW policy clears the platform / wall tiers; the 2026-10-31
    velocity-tracking Phase-2 mix is enabled through PART2_CMD_FORWARD_FRAC /
    PART2_CMD_STANDING_FRAC / PART2_VX_MIN/MAX / PART2_LATERAL_CMD /
    PART2_YAW_CMD, with course rewards and the lane termination gated to
    traversal-intent envs (cmd_vx > 0.2).
    """

    cfg = _part2_base(play)
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    sampling = {"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.3)}
    terrains = {
        "platform": ExactHighPlatformTerrainCfg(flat_patch_sampling=sampling),
        "high_wall": ExactHighWallTerrainCfg(flat_patch_sampling=sampling),
        "slope": ExactSlopeTerrainCfg(flat_patch_sampling=sampling),
        "stairs": ExactStairsTerrainCfg(flat_patch_sampling=sampling),
        "trench": ExactTrenchTerrainCfg(flat_patch_sampling=sampling),
    }
    if subterrains is None:
        env_sub = knobs.optional("PART2_SUBTERRAINS")
        if env_sub:
            subterrains = tuple(part.strip() for part in env_sub.split(",")
                                if part.strip())
    if subterrains:
        terrains = {k: v for k, v in terrains.items() if k in subterrains}
    # 2026-10-31 velocity-tracking Phase-2b: optional flat command-tracking column.
    # Command tracking and obstacle traversal then live on disjoint terrain in one
    # PPO batch (obstacle columns keep the row2p traversal contract; the flat
    # column gets full 3-axis/standing commands and no course rewards).
    flat_cmd_mode = knobs.get_bool("PART2_FLAT_CMD_COLUMN", False)
    if flat_cmd_mode:
        if "flat_cmd" in terrains:
            raise ValueError("PART2_FLAT_CMD_COLUMN: sub-terrain 'flat_cmd' name clash")
        terrains["flat_cmd"] = FlatCommandColumnTerrainCfg(
            proportion=knobs.get_float("PART2_FLAT_COLUMN_PROP", 1.67)
        )
    # The official trench (ramp + plateau + gap + ramp) is ~5.8 m long from the
    # spawn; the default 8 m patch would let it overflow into the next difficulty
    # row, so lengthen every patch to 16 m.  Obstacle offsets are spawn-relative
    # and unchanged.
    cfg.scene.terrain.terrain_generator.size = (16.0, 8.0)
    cfg.scene.terrain.terrain_generator.sub_terrains = terrains
    cfg.sim.nconmax = knobs.get_int("RC_NCONMAX", 256)
    cfg.sim.njmax = knobs.get_int("RC_NJMAX", 1024)
    # The mixed task used to skip this call, so PART2_DEPLOY_SIM=1 only affected
    # the exact specialist tasks: PPO/HIM trained with implicitfast/pyramidal/
    # impratio=1/iterations=10/ccd=50 while the judge runs elliptic/impratio=100/
    # iterations=100/ccd=35.  On slopes and trench edges that is a different
    # contact problem, so the deployment-sim knob must also cover this task.
    _apply_deployment_sim_options(cfg)
    # CCD comes from _apply_deployment_sim_options() (judge value 35); the old
    # `cfg.sim.ccd_iterations = 0` was a dead write (audit P1-1, 2026-10-26).
    if not play:
        num_rows = cfg.scene.terrain.terrain_generator.num_rows
        override_row = knobs.get_int_or_none("PART2_MAX_ROW")
        cfg.scene.terrain.max_init_terrain_level = (
            max(0, num_rows - 1) if override_row is None else override_row
        )
        # Fix the difficulty (e.g. train the hardest tiers only) when both bounds
        # are given; the part2 predicates read difficulty from the same range.
        diff_min = knobs.optional("PART2_DIFF_MIN")
        diff_max = knobs.optional("PART2_DIFF_MAX")
        if diff_min and diff_max:
            cfg.scene.terrain.terrain_generator.difficulty_range = (
                float(diff_min), float(diff_max))
            # Staged teacher runs pin the range but still cap initial difficulty
            # rows via PART2_MAX_ROW (e.g. start at row 0, then widen).
            cfg.scene.terrain.max_init_terrain_level = max(
                0, knobs.get_int("PART2_MAX_ROW", 0))
        cfg.terminations["lane_deviation"] = _traversal_lane_termination(
            max_abs_y=knobs.get_float("PART2_LANE_HALF_WIDTH", PART2_LANE_HALF_WIDTH),
            flat_column="flat_cmd" if flat_cmd_mode else None,
        )
        # Command profile: official-speed lane band plus the 2026-10-31
        # velocity-tracking Phase-2 mix.  All values are registered knobs; the
        # call-site defaults reproduce the historical 0.25-1.00 straight-lane
        # contract and forward-only sampler (1.0 / 0.0).  With
        # PART2_CMD_FORWARD_FRAC/PART2_CMD_STANDING_FRAC < 1 the remaining native
        # envs sample the full vx/vy/wz ranges so the same run also rehearses
        # velocity tracking (PART2_VX_MIN may go negative for reverse samples).
        if flat_cmd_mode:
            # Per-column command term: obstacle columns = row2p traversal lane
            # (vx sampled 0.25-1.0 + adaptive lane feedback); flat column = full
            # flat ranges from the PART2_FLAT_* knobs, including 0 / reverse /
            # standing.  The command term decides per env from terrain_types.
            base_cmd = cfg.commands[BLACK_COMMAND_NAME]
            base_cmd.ranges.lin_vel_x = (
                knobs.get_float("PART2_FLAT_VX_MIN", -0.6),
                knobs.get_float("PART2_FLAT_VX_MAX", 1.0),
            )
            flat_lateral = knobs.get_float("PART2_FLAT_LATERAL_CMD", 0.25)
            flat_yaw = knobs.get_float("PART2_FLAT_YAW_CMD", 0.5)
            base_cmd.ranges.lin_vel_y = (-flat_lateral, flat_lateral)
            base_cmd.ranges.ang_vel_z = (-flat_yaw, flat_yaw)
            cfg.commands[BLACK_COMMAND_NAME] = make_column_mixed_command_cfg(
                base_cmd,
                flat_column="flat_cmd",
                traversal_vx=(
                    knobs.get_float("PART2_VX_MIN", 0.25),
                    knobs.get_float("PART2_VX_MAX", 1.0),
                ),
                flat_standing_frac=knobs.get_float(
                    "PART2_FLAT_STANDING_FRAC", 0.10
                ),
                adaptive_lane=True,
            )
        else:
            _apply_part2_command_cfg(
                cfg,
                vx_min_default=0.25,
                vx_max_default=1.0,
                lateral_default=0.15,
                yaw_default=0.2,
            )
        # Obstacle traversal matters more than exact velocity tracking on a single
        # flat patch; keep tracking but add a dense climb signal and a bigger
        # sparse success bonus (the goal patch sits past every official obstacle).
        # The base cfg ships a state-based competition_progress (1.5) and a
        # per-step competition_goal (5.0); both pay while camping past an
        # obstacle line.  Drop them and re-add only potential-based variants, all
        # gated on the traversal-intent command mask (cmd_vx > 0.2) so standing /
        # reverse rehearsal envs do not farm course rewards.
        cfg.rewards.pop("competition_goal", None)
        cfg.rewards.pop("climb_height", None)
        cfg.rewards["competition_progress"] = RewardTermCfg(
            func=_masked_competition_progress_delta,
            weight=knobs.get_float("PART2_PROGRESS_W", 8.0),
            params={
                "goal_name": "goal",
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": BLACK_COMMAND_NAME,
            },
        )
        goal_w = knobs.get_float("PART2_GOAL_W", 0.0)
        if goal_w > 0.0:
            cfg.rewards["competition_goal"] = RewardTermCfg(
                func=_masked_competition_goal_reward,
                weight=goal_w,
                params={
                    "goal_name": "goal",
                    "asset_cfg": SceneEntityCfg("robot"),
                    "command_name": BLACK_COMMAND_NAME,
                },
            )
        climb_w = knobs.get_float("PART2_CLIMB_W", 0.0)
        if climb_w > 0.0:
            cfg.rewards["climb_height"] = RewardTermCfg(
                func=_masked_course_climb_height_reward,
                weight=climb_w,
                params={
                    "goal_name": "goal",
                    "asset_cfg": SceneEntityCfg("robot"),
                    "command_name": BLACK_COMMAND_NAME,
                },
            )
        cfg.terminations["part2_out_of_course"] = _traversal_out_of_course_termination()
        # Trench cleanup: the goal/progress rewards alone let a policy cross by
        # dropping into the slot and climbing out; the strict judge rejects that.
        trench_pen_w = knobs.get_float("PART2_TRENCH_PEN_W", 0.0)
        if trench_pen_w > 0.0 and "trench" in terrains:
            cfg.rewards["trench_wheel_drop"] = RewardTermCfg(
                func=_trench_wheel_drop_penalty,
                weight=-trench_pen_w,
                params={"trench_column": list(terrains).index("trench")},
            )
        # Judge-aligned terminal conditions: the old ``competition_goal_reached``
        # term ended the episode when the *trunk* crossed a line, which allowed
        # "wheels still on the obstacle" completions and trench cheating.  The
        # success term now requires all four wheels on the target support
        # surface, and a trench slot violation terminates the attempt as a
        # failure (rules: attempt invalid, same as the strict judge).  The
        # success/failure predicate lane semantics are unchanged by the
        # traversal-intent gate above.
        cfg.terminations.pop("competition_goal_reached", None)
        cfg.terminations["part2_success"] = TerminationTermCfg(
            func=_part2_success, params={"asset_cfg": SceneEntityCfg("robot")})
        cfg.terminations["part2_failure"] = _traversal_part2_failure_termination()
        clear_w = knobs.get_float("PART2_CLEAR_W", 25.0)
        if clear_w > 0.0:
            cfg.rewards["part2_clear"] = RewardTermCfg(
                func=_masked_part2_clear_reward,
                weight=clear_w,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "command_name": BLACK_COMMAND_NAME,
                },
            )
    return cfg


def blackw_plum_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Part2 tier-1 plum pile field: 24 piles + entry/exit platforms (200 pts).

    Uses the strict wheel-contact success/failure predicates: any wheel touching
    the ground between piles ends the attempt (judge rule).  v7.6 semantics: reaching
    the 0.8 m exit platform is only evidence; success requires descending the 1.2 m
    exit ramp and putting all four wheels back on normal ground past its end (mirrors
    the plum branch of ``eval/robocon_eval/part2/v5_judge.py``).  The dense
    progress/goal rewards plus the success bonus shape the crossing; the task has no
    difficulty levels (the official tier-1 field is fixed).
    """

    cfg = _part2_base(play)
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
    terrain = ExactPlumTerrainCfg()
    terrain.flat_patch_sampling = {
        "goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.3)
    }
    cfg.scene.terrain.terrain_generator.size = (16.0, 8.0)
    cfg.scene.terrain.terrain_generator.sub_terrains = {"plum": terrain}
    cfg.sim.nconmax = knobs.get_int("RC_NCONMAX", 512)
    cfg.sim.njmax = knobs.get_int("RC_NJMAX", 2048)
    # Judge plum_1 has seconds=18.0 (timeout is a hard fail); match it so training
    # cannot label success in the (18, 20] s window that the judge no longer accepts.
    cfg.episode_length_s = 18.0
    # Parity option (P0 2026-10-31): the judge plum scene runs the official MJCF
    # <option> block (euler / elliptic / newton / iterations=100 / impratio=100 /
    # ccd=35) and spawns at spawn_default_pose z=0.55.  By default this builder
    # keeps MjLab training physics (implicitfast / pyramidal / iterations=10 /
    # impratio=1 / ccd=50); P0 measured that this makes in-sim success only weakly
    # predictive of the judge (fixed-0.5 in-sim 3.8-5% vs judge pass; forcing the
    # judge options lifts it to 24-39%, and 98.4% at vx=0.7 with DR off).
    # PART2_PLUM_JUDGE_PHYSICS=1 switches this task to the judge options + spawn
    # z=0.55 for parity training/evaluation; the default stays unchanged because
    # existing checkpoints were trained under the old dynamics.
    if knobs.get_bool("PART2_PLUM_JUDGE_PHYSICS", False):
        _apply_deployment_sim_options(cfg, force=True)
        robot = cfg.scene.entities.get("robot")
        if robot is not None and getattr(robot, "init_state", None) is not None:
            # _apply_deployment_sim_options sets the mixed-task spawn (0.52); the
            # plum judge uses the default pose (0.55).
            robot.init_state.pos = (0.0, 0.0, 0.55)
    if not play:
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.terminations.pop("competition_goal_reached", None)
        cfg.terminations["lane_deviation"] = _traversal_lane_termination(
            max_abs_y=knobs.get_float("PART2_LANE_HALF_WIDTH", PART2_LANE_HALF_WIDTH),
        )
        cfg.terminations["part2_success"] = TerminationTermCfg(
            func=_part2_success, params={"asset_cfg": SceneEntityCfg("robot")})
        cfg.terminations["part2_failure"] = _traversal_part2_failure_termination()
        # Command branch symmetric with the mixed/exact builders (same registered
        # knobs: ranges, forward/standing mix, PART2_CMD_RANDOM / adaptive).  The
        # call-site defaults keep the historical plum 0.25-0.65 m/s straight field
        # with forward-only sampling when no knob is set.  NOTE: the plum
        # semantics (judge-mirrored strict predicates / teacher recipe) are NOT
        # wired to the new velocity-tracking recipe; this is code symmetry only.
        _apply_part2_command_cfg(
            cfg,
            vx_min_default=0.25,
            vx_max_default=0.65,
            lateral_default=0.05,
            yaw_default=0.08,
        )
        cfg.rewards.pop("competition_goal", None)
        cfg.rewards.pop("climb_height", None)
        cfg.rewards["competition_progress"] = RewardTermCfg(
            func=_course_progress_delta,
            weight=knobs.get_float("PART2_PROGRESS_W", 8.0),
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        foot_w = knobs.get_float("PART2_PLUM_FOOT_W", 0.0)
        if foot_w > 0.0:
            cfg.rewards["plum_foothold"] = RewardTermCfg(
                func=_plum_foothold_reward,
                weight=foot_w,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
        goal_w = knobs.get_float("PART2_GOAL_W", 0.0)
        if goal_w > 0.0:
            cfg.rewards["competition_goal"] = RewardTermCfg(
                func=competition_goal_reward,
                weight=goal_w,
                params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
            )
        climb_w = knobs.get_float("PART2_CLIMB_W", 0.0)
        if climb_w > 0.0:
            cfg.rewards["climb_height"] = RewardTermCfg(
                func=_course_climb_reward,
                weight=climb_w,
            )
        cfg.terminations["part2_out_of_course"] = _traversal_out_of_course_termination()
        if knobs.get_float("PART2_CLEAR_W", 25.0) > 0.0:
            cfg.rewards["part2_clear"] = RewardTermCfg(
                func=_part2_clear_reward,
                weight=knobs.get_float("PART2_CLEAR_W", 25.0),
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
    return cfg


def blackw_platform_easy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Platform-shape curriculum stage: 5->10 cm before the official 10 cm tier."""
    cfg = _exact_env_cfg(
        ExactHighPlatformTerrainCfg(
            platform_height_range=(0.05, 0.10),
            side_walls=False,
            lane_half_width=0.70,
        ),
        play,
    )
    if not play:
        cfg.rewards["platform_height"] = RewardTermCfg(
            func=_platform_height_reward, weight=50.0
        )
        cfg.rewards.pop("competition_goal", None)
        cfg.terminations.pop("competition_goal_reached", None)
        # Explicit PROXY: platform-easy 5-10 cm does not have a judge-mirrored
        # predicate; keep the old x/z proxy under a self-documenting name.
        cfg.terminations["proxy_platform_success"] = TerminationTermCfg(
            func=_platform_top_success
        )
        if cfg.scene.terrain is not None:
            cfg.scene.terrain.max_init_terrain_level = 0
        cfg.terminations["lane_deviation"].params["max_abs_y"] = PART2_LANE_HALF_WIDTH
    return cfg
