"""Course-traversal mechanics shared by all event profiles.

Originally part of the Part 1 competition builder; robot-agnostic, and also the
base mechanics for the Part 2 tasks (which replace the terrain and add the
judge-mirrored terminal predicates).  Terrain generators stay in
``profiles/<event>/terrain.py``; this module never touches ``cfg.scene.terrain``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from mjlab.envs import mdp
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.tasks.velocity.common.curriculums import CURRICULUM_TERM_NAME
from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_COMMAND_NAME
from alldog_mjlab.utils import knobs

"""Reward / termination terms for the competition-proxy obstacle courses.

The terrain generator marks a far-end goal flat patch (``flat_patches["goal"]``)
for the obstacle course sub-terrains.  Non-course sub-terrains get the spawn
origin as the fill value, so a goal is only treated as valid when it is at
least ``MIN_GOAL_DISTANCE_M`` ahead of the env origin along +x.  The reward is
sparse (1.0 on the step the robot root crosses the goal line) and a matching
termination ends the episode, giving PPO a task-success signal on top of the
velocity-tracking reward.
"""

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
MIN_GOAL_DISTANCE_M = 0.8


def _goal_positions(env, goal_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    terrain = env.scene.terrain
    if terrain is None or terrain.flat_patches is None:
        return None
    patches = terrain.flat_patches.get(goal_name)
    if patches is None:
        return None
    rows = terrain.terrain_levels
    cols = terrain.terrain_types
    goal = patches[rows, cols, 0]  # (num_envs, 3)
    origins = terrain.env_origins
    valid = (goal[:, 0] - origins[:, 0]) > MIN_GOAL_DISTANCE_M
    return goal, valid


def competition_goal_reward(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sparse task-success reward: 1.0 on the step the root crosses the goal line."""
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is None:
        return reward
    goal, valid = goal_data
    root_x = asset.data.root_link_pos_w[:, 0]
    reward[(root_x > goal[:, 0]) & valid] = 1.0
    return reward


def competition_goal_reached(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Termination: end the episode once the root crosses the goal line."""
    return competition_goal_reward(env, goal_name=goal_name, asset_cfg=asset_cfg) > 0.5


def competition_goal_progress(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense progress reward along +x towards the far-end goal (course columns only).

    ``progress = clip((x - origin_x - 0.5) / (goal_x - origin_x), 0, 1)`` so standing
    at spawn gives 0 and crossing the goal line gives 1.  Non-course columns get a
    goal patch filled with the spawn origin and are masked out.
    """
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is None:
        return reward
    goal, valid = goal_data
    origins = env.scene.terrain.env_origins
    span = (goal[:, 0] - origins[:, 0]).clamp(min=1e-3)
    progress = ((asset.data.root_link_pos_w[:, 0] - origins[:, 0] - 0.5) / span).clamp(0.0, 1.0)
    return torch.where(valid, progress, torch.zeros_like(progress))


def apply_command_fractions(
    command, *, forward_default: float = 1.0, standing_default: float = 0.0
) -> tuple[float, float]:
    """Set the native sampler fractions from the command-mix knobs.

    ``PART2_CMD_FORWARD_FRAC`` / ``PART2_CMD_STANDING_FRAC`` describe the
    *effective* per-resample assignment: forward-only envs (``vx=abs clamp>=0.3,
    vy=wz=0``) / standing envs (all-zero command) / native envs (full sampled
    range).  MjLab samples the forward and standing flags **independently** in
    ``UniformVelocityCommand._resample_command`` and a standing env then wins in
    ``_update_command``, so the forward flag must be conditioned on "not
    standing" (``rel_forward = forward / (1 - standing)``) for the realised mix
    to equal ``forward`` / ``standing`` / ``1-forward-standing``.

    Defaults (forward=1.0, standing=0.0) reproduce the historical forward-only
    traversal contract exactly.  Invalid combinations fail loudly instead of
    silently changing the recipe.
    """
    forward = knobs.get_float("PART2_CMD_FORWARD_FRAC", forward_default)
    standing = knobs.get_float("PART2_CMD_STANDING_FRAC", standing_default)
    for name, value in (
        ("PART2_CMD_FORWARD_FRAC", forward),
        ("PART2_CMD_STANDING_FRAC", standing),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}={value!r} must be a float in [0, 1]")
    if forward + standing > 1.0 + 1e-9:
        raise ValueError(
            "PART2_CMD_FORWARD_FRAC "
            f"({forward}) + PART2_CMD_STANDING_FRAC ({standing}) > 1: "
            "the native command range must keep at least one env"
        )
    rel_forward = 0.0 if standing >= 1.0 else forward / (1.0 - standing)
    command.rel_standing_envs = standing
    command.rel_forward_envs = rel_forward
    return forward, standing


def apply_command_resampling(command, default_s: float | None = None) -> None:
    """Optionally override the command resampling period.

    ``PART2_CMD_RESAMPLE_S=0`` (default) keeps the task's existing contract
    (10 s for the flat/traversal family).  Setting it >= the episode length makes
    the sampled command *and* the sampler mode (forward / standing / native) fixed
    for the whole episode, which matters for the velocity-tracking Phase 2 mix:
    with the 10 s default a 20 s traversal episode would switch mode mid-obstacle.
    """
    override = knobs.get_float("PART2_CMD_RESAMPLE_S", 0.0)
    if override <= 0.0:
        return
    if default_s is not None and abs(float(default_s) - override) < 1e-12:
        return
    command.resampling_time_range = (float(override), float(override))


def apply_pose_command_contract(cfg) -> None:
    """Low-speed pose-alignment command contract for the flat task.

    The default ``blackw-flat`` command sampler is the generic velocity task
    (wide ranges, forward-only/standing fractions, command curriculum).  A
    closed-loop position primitive needs a small command envelope whose full
    3-axis range is sampled for every native env, so this contract:

    * sets vx/vy/wz ranges to the low-speed operating envelope;
    * sets the forward-only fraction to 0 (vy/wz reach every native env);
    * keeps a configurable standing fraction for zero-command stopping;
    * fixes the resampling period and removes the command curriculum so the
      envelope cannot silently expand during training.

    Enabled with ``RC_POSE_CMD=1``; defaults are inert for existing runs.
    """
    command = cfg.commands[BLACK_COMMAND_NAME]
    vx_min = knobs.get_float("RC_POSE_VX_MIN", -0.20)
    vx_max = knobs.get_float("RC_POSE_VX_MAX", 0.50)
    vy_max = knobs.get_float("RC_POSE_VY_MAX", 0.15)
    wz_max = knobs.get_float("RC_POSE_WZ_MAX", 0.50)
    if vx_min > vx_max:
        raise ValueError(f"RC_POSE_VX_MIN={vx_min} must be <= RC_POSE_VX_MAX={vx_max}")
    if vy_max < 0.0 or wz_max < 0.0:
        raise ValueError("RC_POSE_VY_MAX / RC_POSE_WZ_MAX must be >= 0")
    command.ranges.lin_vel_x = (vx_min, vx_max)
    command.ranges.lin_vel_y = (-vy_max, vy_max)
    command.ranges.ang_vel_z = (-wz_max, wz_max)
    command.ranges.heading = None
    command.heading_command = False
    command.rel_heading_envs = 0.0
    standing = knobs.get_float("RC_POSE_STANDING_FRAC", 0.10)
    if not 0.0 <= standing <= 1.0:
        raise ValueError(f"RC_POSE_STANDING_FRAC={standing} must be in [0, 1]")
    command.rel_standing_envs = standing
    command.rel_forward_envs = 0.0
    resample = knobs.get_float("RC_POSE_RESAMPLE_S", 5.0)
    if resample > 0.0:
        command.resampling_time_range = (float(resample), float(resample))
    cfg.curriculum.pop(CURRICULUM_TERM_NAME, None)


def apply_traversal_base(cfg, *, play: bool) -> None:
    """Set the shared obstacle-traversal command/termination/reward contract."""
    # Obstacle courses are a traversal task, not random velocity tracking: the
    # command range is a slow forward-lane band and the command curriculum is
    # disabled (the terrain curriculum still runs).  The range/fraction values are
    # knob-controlled (defaults reproduce the historical contract exactly):
    #   * PART2_VX_MIN/MAX, PART2_LATERAL_CMD, PART2_YAW_CMD -> command ranges;
    #   * PART2_CMD_FORWARD_FRAC / PART2_CMD_STANDING_FRAC -> sampler mix
    #     (default forward=1.0 / standing=0.0 keeps the legacy forward-only
    #     traversal behaviour; with forward<1 the native envs actually see the
    #     sampled vy/wz, which the 2026-10-31 velocity-tracking Phase 2 uses).
    command = cfg.commands[BLACK_COMMAND_NAME]
    vx_min = knobs.get_float("PART2_VX_MIN", 0.40)
    vx_max = knobs.get_float("PART2_VX_MAX", 1.00)
    # Equal bounds are legitimate: fixed-speed training (S1 trench
    # PART2_VX_MIN=PART2_VX_MAX=1.0, tau 1.0/1.0) samples a constant command.
    if vx_min > vx_max:
        raise ValueError(f"PART2_VX_MIN={vx_min} must be <= PART2_VX_MAX={vx_max}")
    command.ranges.lin_vel_x = (vx_min, vx_max)
    # The historical comment claimed vy/yaw were "needed for Part 1 manual
    # driving", but rel_forward_envs=1.0 zeroed them for every env, so the
    # traversal tasks were forward-only in practice.  Keep the ranges (they feed
    # the native envs when PART2_CMD_FORWARD_FRAC < 1) and state the actual
    # contract: the sampler mix decides whether vy/wz survive.
    lateral = knobs.get_float("PART2_LATERAL_CMD", 0.30)
    yaw = knobs.get_float("PART2_YAW_CMD", 1.00)
    if lateral < 0.0 or yaw < 0.0:
        raise ValueError(
            f"PART2_LATERAL_CMD={lateral} / PART2_YAW_CMD={yaw} must be >= 0"
        )
    command.ranges.lin_vel_y = (-lateral, lateral)
    command.ranges.ang_vel_z = (-yaw, yaw)
    command.ranges.heading = None
    command.heading_command = False
    command.rel_heading_envs = 0.0
    apply_command_fractions(command)
    apply_command_resampling(command)
    cfg.curriculum.pop(CURRICULUM_TERM_NAME, None)
    if not play:
        cfg.curriculum.pop("terrain_levels", None)
    if not play:
        # Obstacle tasks require body/leg contact with walls, beams and stones
        # (climbing, crawling, pass-under).  The generic rough-task illegal
        # contact (trunk/thigh vs terrain) and stuck terms would abort exactly
        # those attempts, so replace them with a fall/orientation termination.
        cfg.terminations.pop("illegal_contact", None)
        cfg.terminations.pop("stuck", None)
        cfg.terminations["bad_orientation"] = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.05},
        )
        # Dense goal progress + sparse success bonus + success termination.
        # Anti-degeneracy: the first stage3 policy learned to crawl (base z 0.24,
        # tilt 22 deg) to squeeze under the low height bar, which then transferred
        # to every terrain and scored 0 in the real scenes.  Keep the base up.
        cfg.rewards["base_height"] = replace(
            cfg.rewards["base_height"], weight=-4.0
        )
        cfg.rewards["foot_clearance"] = replace(
            cfg.rewards["foot_clearance"], weight=-0.03
        )
        cfg.rewards["competition_progress"] = RewardTermCfg(
            func=competition_goal_progress,
            weight=1.5,
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        cfg.rewards["competition_goal"] = RewardTermCfg(
            func=competition_goal_reward,
            weight=5.0,
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        cfg.terminations["competition_goal_reached"] = TerminationTermCfg(
            func=competition_goal_reached,
            params={"goal_name": "goal", "asset_cfg": SceneEntityCfg("robot")},
        )
        # Competition terrain is contact-heavy (walls, beams, narrow bridges); a
        # single non-finite critic obs used to abort a 1024-env run at ~1.9k iter.
        # Sanitize non-finite observation values and print the offending term so
        # the underlying env bug (if any) is still diagnosable.
        for group_name in cfg.observations:
            cfg.observations[group_name].nan_policy = "warn"
    return cfg
