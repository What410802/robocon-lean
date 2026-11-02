"""Part 2 reward terms: progress / clearance / foothold shaping.

All terms are potential-difference or state-based (never raw distance farming);
the judge-mirrored termination predicates they reference live in ``predicates.py``.
"""

from __future__ import annotations

import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.tasks.velocity.common.course import (
    _goal_positions,
    competition_goal_progress,
    competition_goal_reward,
)
from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
    PART2_HOLD_STEPS,
    _part2_success_streak,
    _part2_wheel_lowest_points,
    _platform_top_z,
)
from alldog_mjlab.utils import knobs

# "Traversal intent" proxy of the 2026-10-31 velocity-tracking Phase 2 recipe:
# course rewards / lane termination stay active only for envs whose commanded
# body-frame vx is above this value.  Forward-only envs (abs clamp >= 0.3) and
# the official-mixed/native envs with vx > 0.2 keep the full course semantics;
# standing / negative-speed envs get pure velocity-tracking samples instead of
# being killed by the course lane rule.
PART2_TRAVERSAL_MIN_CMD_VX = knobs.get_float("PART2_TRAVERSAL_MIN_CMD_VX", 0.2)


def _traversal_command_mask(
    env,
    command_name: str = "twist",
    min_cmd_vx: float = PART2_TRAVERSAL_MIN_CMD_VX,
):
    """Per-env bool mask: env currently commands a traversal (cmd_vx > threshold)."""
    command = env.command_manager.get_command(command_name)
    return command[:, 0] > float(min_cmd_vx)


def _masked_competition_progress_delta(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
    command_name: str = "twist",
    min_cmd_vx: float = PART2_TRAVERSAL_MIN_CMD_VX,
):
    """Goal-progress shaping gated by the traversal-intent command mask."""
    progress = _course_progress_delta(
        env, goal_name=goal_name, asset_cfg=asset_cfg, scale=scale
    )
    return progress * _traversal_command_mask(env, command_name, min_cmd_vx).to(
        progress.dtype
    )


def _masked_competition_goal_reward(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "twist",
    min_cmd_vx: float = PART2_TRAVERSAL_MIN_CMD_VX,
):
    """Sparse goal-line bonus gated by the traversal-intent command mask."""
    reward = competition_goal_reward(env, goal_name=goal_name, asset_cfg=asset_cfg)
    return reward * _traversal_command_mask(env, command_name, min_cmd_vx).to(
        reward.dtype
    )


def _masked_part2_clear_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "twist",
    min_cmd_vx: float = PART2_TRAVERSAL_MIN_CMD_VX,
):
    """Judge-success bonus gated by the traversal-intent command mask."""
    reward = _part2_clear_reward(env, asset_cfg=asset_cfg)
    return reward * _traversal_command_mask(env, command_name, min_cmd_vx).to(
        reward.dtype
    )


def _masked_course_climb_height_reward(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "twist",
    min_cmd_vx: float = PART2_TRAVERSAL_MIN_CMD_VX,
):
    """Climb shaping before the goal line, gated by the traversal command mask.

    Mirrors ``blackw/env_cfgs.py::_course_climb_reward`` (goal-line gate) plus the
    Phase-2 traversal-intent gate, so standing / reverse envs get no climb reward.
    """
    base = _climb_height_reward(env, asset_cfg=asset_cfg)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is not None:
        goal, valid = goal_data
        root_x = env.scene[asset_cfg.name].data.root_link_pos_w[:, 0]
        base = base * ((root_x <= goal[:, 0]) & valid).to(base.dtype)
    return base * _traversal_command_mask(env, command_name, min_cmd_vx).to(base.dtype)


def _part2_clear_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """One-step success bonus: paid exactly when the held success predicate fires."""
    import torch

    streak = _part2_success_streak(env, asset_cfg)
    return (streak == PART2_HOLD_STEPS).to(torch.float32)


def _trench_wheel_drop_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    trench_column: int = -1,
    max_drop_m: float = 0.103,
    plateau_z: float = 0.30,
    gap_start_x: float = 3.5,
    wheel_radius: float = 0.103,
):
    """Penalize wheels sinking into the trench slot by more than the judge allows.

    The v7.1 Part 2 judge uses the explicit ``gap_max_drop_m`` z proxy
    (0.103 m below the 0.30 m plateau top) while a wheel's x is inside the slot.
    This term mirrors that rule as a dense training penalty, active only for
    trench-column envs.  Wheel height uses the same geometric lowest point as
    the success predicate, not body-origin minus radius.
    """
    import torch

    if trench_column < 0:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    origins = terrain.env_origins
    if isinstance(origins, torch.Tensor):
        origins = origins.to(env.device)
    else:
        origins = torch.as_tensor(origins, device=env.device)
    terrain_types = torch.as_tensor(terrain.terrain_types, device=env.device)
    terrain_levels = torch.as_tensor(terrain.terrain_levels, device=env.device)
    num_rows = int(terrain.cfg.terrain_generator.num_rows)
    body_names = asset.body_names
    wheel_ids = [body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
    pos = _part2_wheel_lowest_points(asset, wheel_ids, wheel_radius)  # (N, 4, 3)
    x_local = pos[..., 0] - origins[:, 0:1]
    z_bottom = pos[..., 2]
    difficulty = terrain_levels.to(torch.float32) / max(1.0, float(num_rows - 1))
    gap_w = 0.10 + 0.20 * difficulty                             # official 10..30 cm
    gap_lo = torch.full_like(gap_w, gap_start_x - 0.10)
    gap_hi = gap_start_x + gap_w + 0.30
    in_slot = (x_local >= gap_lo.unsqueeze(1)) & (x_local <= gap_hi.unsqueeze(1))
    allowed_drop = torch.full_like(gap_w, max_drop_m)
    allowed_floor = plateau_z - allowed_drop
    over = torch.clamp(allowed_floor.unsqueeze(1) - z_bottom, min=0.0)
    penalty = (over * in_slot.to(over.dtype)).sum(dim=1) / 4.0
    mask = (terrain_types == trench_column).to(penalty.dtype)
    return penalty * mask


def _platform_height_reward(env, clearance: float = 0.42,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Dense signal for lifting the root above the platform top (row-relative)."""
    import torch

    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(root_z - _platform_top_z(env) - clearance, min=0.0, max=0.2)


def _climb_height_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Dense reward for lifting the trunk above flat standing height.

    Applies to every official Part2 obstacle (platform, stairs, slope, trench
    plateau, wall): the robot starts on a flat patch, so any trunk height above
    the flat stance is evidence that it is climbing the obstacle.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(root_z - 0.55, min=0.0, max=0.60)


def _course_progress_delta(
    env,
    goal_name: str = "goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
):
    """Potential-based progress shaping: pay only the *increase* of goal progress.

    State-based progress (``competition_goal_progress``) pays every step while a
    policy stands still anywhere ahead of spawn, which is farmable; the delta
    form is a potential (Ng et al. 1999) and pays only actual advancement.
    """
    import torch

    progress = competition_goal_progress(env, goal_name=goal_name, asset_cfg=asset_cfg)
    goal_data = _goal_positions(env, goal_name)
    if goal_data is not None:
        goal, valid = goal_data
        root_x = env.scene[asset_cfg.name].data.root_link_pos_w[:, 0]
        progress = progress * ((root_x <= goal[:, 0]) & valid).to(progress.dtype)
    prev = getattr(env, "_course_prev_progress", None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros_like(progress)
    delta = torch.clamp(progress - prev, min=0.0)
    env._course_prev_progress = progress.detach()
    return delta * scale


def _plum_foothold_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_radius: float = 0.103,
    cube_x_lo: float = 2.70,
    cube_x_hi: float = 4.10,
):
    """Dense foothold shaping for the plum field: reward wheels standing on piles.

    Blind policies cannot see the gaps; a fixed periodic pile field still lets a
    timed gait work.  This term pays in proportion to how many wheels are on a
    pile top (wheel bottom in [0.14, 0.34] m) while the robot is over the field.
    Defaults span the owner-confirmed 3 cubes (x 2.7..4.1, see
    ``terrain/official/part2_terrain_spec.json``); the field is fixed, so this
    is not difficulty-interpolated.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    origins = terrain.env_origins
    origins = origins if isinstance(origins, torch.Tensor) else torch.as_tensor(
        origins, device=env.device)
    origins = origins.to(env.device)
    body_names = asset.body_names
    wheel_ids = [body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
    wpos = _part2_wheel_lowest_points(asset, wheel_ids, wheel_radius)
    wx = wpos[..., 0] - origins[:, 0:1]
    wz = wpos[..., 2]
    in_field = (wx >= cube_x_lo) & (wx <= cube_x_hi)
    on_pile = in_field & (wz >= 0.14) & (wz <= 0.34)
    frac = on_pile.to(torch.float32).mean(dim=1)
    # Potential (delta) form: only pay for NEW wheels landing on piles, so standing
    # still on a pile cannot farm the term (audit P0-3, 2026-10-26).
    prev = getattr(env, "_plum_prev_foothold", None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros_like(frac)
    delta = torch.clamp(frac - prev, min=0.0)
    env._plum_prev_foothold = frac.detach()
    return delta * in_field.any(dim=1).to(torch.float32)
