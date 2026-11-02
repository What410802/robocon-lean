"""Adaptive lane-centering velocity command for Part2 training.

The eval runner already has ``AdaptiveCommand`` (state-feedback vy/wz).  This
module makes the *training* command term produce the same feedback, so PPO/HIM
learns to track lane-centering commands instead of only forward-only
``(vx, 0, 0)``.

Contract notes:
- ``command[:, :2]`` is the body-frame linear target and ``command[:, 2]`` the
  yaw-rate target.  Both tracking rewards read this same command tensor, so
  when the feedback asks for ``wz=0.5`` the reward target is 0.5, not 0.
- y/yaw are terrain-relative: Part2 training uses a terrain generator, so the
  absolute qpos y is offset by the per-env terrain origin.  Official columns are
  aligned with +x, so the lane target is y_rel=0, yaw=0.
- The adaptive term is selected per env on command resample, so a run can mix
  adaptive and fixed/random envs (``PART2_ADAPTIVE_CMD`` in (0, 1]) without a
  separate task.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

# Same defaults as eval/robocon_eval/policy_runner.py AdaptiveCommand.
EVAL_K_Y = 0.8
EVAL_K_YAW = 2.0
EVAL_K_VY = 0.6
EVAL_K_VHEADING = 0.0
EVAL_VY_GAIN = 0.5
EVAL_MAX_VY = 0.2
EVAL_MAX_WZ = 0.7


@dataclass(kw_only=True)
class AdaptiveLaneCommandCfg(UniformVelocityCommandCfg):
    adaptive_frac: float = 1.0
    """Fraction of envs that receive the state-feedback command."""
    command_random: bool = False
    """For non-adaptive envs: keep random sampled vy/wz instead of forcing 0.

    This is one of the three orthogonal switches (adaptive / command random /
    state random).  With ``adaptive_frac=1`` it has no effect.
    """
    vy_gain: float = EVAL_VY_GAIN
    k_y: float = EVAL_K_Y
    k_yaw: float = EVAL_K_YAW
    k_vy: float = EVAL_K_VY
    k_vheading: float = EVAL_K_VHEADING
    max_vy: float = EVAL_MAX_VY
    max_wz: float = EVAL_MAX_WZ
    center_y: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "AdaptiveLaneCommand":
        return AdaptiveLaneCommand(self, env)


def make_adaptive_command_cfg(
    base: UniformVelocityCommandCfg,
    *,
    adaptive_frac: float,
    command_random: bool,
) -> AdaptiveLaneCommandCfg:
    """Copy the existing twist cfg and swap in the adaptive term."""
    return AdaptiveLaneCommandCfg(
        entity_name=base.entity_name,
        resampling_time_range=base.resampling_time_range,
        debug_vis=base.debug_vis,
        heading_command=base.heading_command,
        heading_control_stiffness=base.heading_control_stiffness,
        rel_standing_envs=base.rel_standing_envs,
        rel_heading_envs=base.rel_heading_envs,
        rel_world_envs=base.rel_world_envs,
        # Adaptive envs override vy/wz every step; non-adaptive envs need the
        # sampled values to survive, so the forward-only zeroing must be off.
        rel_forward_envs=0.0,
        init_velocity_prob=base.init_velocity_prob,
        ranges=base.ranges,
        adaptive_frac=float(adaptive_frac),
        command_random=bool(command_random),
    )


class AdaptiveLaneCommand(UniformVelocityCommand):
    cfg: AdaptiveLaneCommandCfg

    def __init__(self, cfg: AdaptiveLaneCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.is_adaptive_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if self.cfg.adaptive_frac <= 0.0:
            self.is_adaptive_env[env_ids] = False
        else:
            r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
            self.is_adaptive_env[env_ids] = r <= self.cfg.adaptive_frac
        if not self.cfg.command_random:
            fixed = env_ids[~self.is_adaptive_env[env_ids]]
            self.vel_command_b[fixed, 1] = 0.0
            self.vel_command_b[fixed, 2] = 0.0

    def _terrain_relative_y(self) -> torch.Tensor:
        y = self.robot.data.root_link_pos_w[:, 1]
        terrain = getattr(self._env.scene, "terrain", None)
        origins = getattr(terrain, "env_origins", None) if terrain is not None else None
        if origins is not None:
            y = y - origins[:, 1]
        return y

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # Base first: handles resampling bookkeeping, world/heading/standing
        # modes.  Adaptive feedback is applied afterwards so the base forward
        # env zeroing cannot overwrite it.
        super()._update_command(env_ids)
        if self.cfg.adaptive_frac <= 0.0:
            return
        mask = self.is_adaptive_env
        if not bool(mask.any()):
            return

        y = self._terrain_relative_y() - float(self.cfg.center_y)
        # Official terrain columns are aligned with +x; target heading is 0.
        yaw = self.robot.data.heading_w
        vx_b = self.robot.data.root_link_lin_vel_b[:, 0]
        vy_b = self.robot.data.root_link_lin_vel_b[:, 1]
        heading_err = torch.atan2(vy_b, torch.clamp(vx_b.abs(), min=0.2))

        vy = torch.clamp(-self.cfg.vy_gain * y, -self.cfg.max_vy, self.cfg.max_vy)
        wz = (
            -self.cfg.k_y * y
            - self.cfg.k_yaw * yaw
            - self.cfg.k_vy * vy_b
            - self.cfg.k_vheading * heading_err
        )
        wz = torch.clamp(wz, -self.cfg.max_wz, self.cfg.max_wz)

        self.vel_command_b[mask, 1] = vy[mask]
        self.vel_command_b[mask, 2] = wz[mask]

        if not self.cfg.command_random:
            fixed = ~mask
            self.vel_command_b[fixed, 1] = 0.0
            self.vel_command_b[fixed, 2] = 0.0
