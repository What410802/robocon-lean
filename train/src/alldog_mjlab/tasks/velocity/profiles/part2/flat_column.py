"""Flat command-tracking column + per-column mixed velocity command (blackW).

Why this module exists
----------------------
The 2026-10-31 velocity-tracking Phase-2 attempts that mixed full 3-axis commands
into the *obstacle* columns (``PART2_CMD_FORWARD_FRAC`` sampler mix) collapsed the
narrow obstacle skills: command-following envs on obstacle terrain produced
off-distribution states/falls and their PPO gradients corrupted the shared actor
(campaign coverage 10/21 -> 1-2/21 within 400 iterations).

The principled fix is to keep the two objectives on *disjoint terrain* in one PPO
batch:

* obstacle columns: unchanged traversal command contract (forward speed + the
  state-feedback lane centring used by the ``row2p`` recipe) -> skills preserved;
* one extra flat column ("flat_cmd"): full 3-axis commands incl. 0 / negative /
  standing, no course rewards/terminations -> velocity tracking samples are clean.

The command term below assigns commands per env from the env's terrain column at
every command resample.  Everything else (rewards/terminations) already keys off
the command tensor or the terrain column (goal patches only exist on obstacle
columns), so flat-column envs receive tracking reward + regularisers only.

Training-only code: deployment/eval contract (57x6 obs, 16 actions) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainOutput
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from alldog_mjlab.tasks.velocity.profiles.part2.terrain import _floor, _origin


# ---------------------------------------------------------------------------
# Terrain: one flat column for command tracking
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class FlatCommandColumnTerrainCfg(SubTerrainCfg):
    """A plain flat floor used only for velocity-command tracking episodes.

    No obstacles, no goal patch: the Part2 course rewards/terminations resolve to
    false on this column (``_goal_positions`` requires a goal patch ahead of the
    spawn point), so flat-column envs are pure tracking samples.
    """

    def function(self, difficulty: float, spec, rng) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        return TerrainOutput(origin=_origin(self.size), geometries=[_floor(body, self.size)])


# ---------------------------------------------------------------------------
# Command term
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ColumnMixedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Full-range commands on ``flat_column``; traversal lane commands elsewhere."""

    flat_column: str = "flat_cmd"
    traversal_vx: tuple[float, float] = (0.30, 1.00)
    """Sampled body-frame vx for obstacle-column envs (absolute value, clamp>=0.3)."""
    adaptive_lane: bool = True
    """Apply the eval/training AdaptiveLaneCommand vy/wz feedback on obstacle envs."""
    vy_gain: float = 0.5
    k_y: float = 0.8
    k_yaw: float = 2.0
    k_vy: float = 0.6
    k_vheading: float = 0.0
    max_vy: float = 0.2
    max_wz: float = 0.7
    center_y: float = 0.0

    def build(self, env):
        return ColumnMixedVelocityCommand(self, env)


class ColumnMixedVelocityCommand(UniformVelocityCommand):
    cfg: ColumnMixedVelocityCommandCfg

    def __init__(self, cfg: ColumnMixedVelocityCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self._flat_mask_cache: torch.Tensor | None = None
        self._flat_mask_key: tuple | None = None

    # -- helpers ---------------------------------------------------------
    def _terrain_flat_mask(self) -> torch.Tensor:
        """Per-env bool: env spawns on the flat command column.

        ``terrain_types`` is constant per env, so the mask is cached against the
        generator instance + column index.  Returns all-False when the task has
        no terrain or the configured column is not present (traversal-only).
        """
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or terrain.cfg.terrain_generator is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        generator: TerrainGenerator = terrain.cfg.terrain_generator
        columns = list(generator.sub_terrains.keys())
        if self.cfg.flat_column not in columns:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        index = columns.index(self.cfg.flat_column)
        key = (id(generator), index)
        if self._flat_mask_cache is None or self._flat_mask_key != key:
            types = torch.as_tensor(terrain.terrain_types, device=self.device)
            self._flat_mask_cache = types == index
            self._flat_mask_key = key
        return self._flat_mask_cache

    def _terrain_relative_y(self) -> torch.Tensor:
        y = self.robot.data.root_link_pos_w[:, 1]
        terrain = getattr(self._env.scene, "terrain", None)
        origins = getattr(terrain, "env_origins", None) if terrain is not None else None
        if origins is not None:
            y = y - origins[:, 1]
        return y

    # -- UniformVelocityCommand hooks ------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample full ranges globally, then overwrite obstacle-column envs."""
        super()._resample_command(env_ids)
        flat = self._terrain_flat_mask()[env_ids]
        obstacle = env_ids[~flat]
        if len(obstacle) == 0:
            return
        # Traversal lane: positive speed, zero lateral/yaw.  The adaptive lane
        # feedback in _update_command supplies vy/wz on top of this, matching the
        # row2p recipe (vx sampled in [vx_min, vx_max]).
        vx = torch.empty(len(obstacle), device=self.device).uniform_(
            float(self.cfg.traversal_vx[0]), float(self.cfg.traversal_vx[1])
        )
        vx = vx.abs().clamp(min=0.3)
        self.vel_command_b[obstacle, 0] = vx
        self.vel_command_b[obstacle, 1] = 0.0
        self.vel_command_b[obstacle, 2] = 0.0
        # Base sampler may have flagged these as standing; that would zero the
        # command every step in _update_command.  Flat-column envs keep their flag.
        self.is_standing_env[obstacle] = False
        self.is_forward_env[obstacle] = False

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if not self.cfg.adaptive_lane:
            return
        flat = self._terrain_flat_mask()
        obstacle = ~flat
        if not bool(obstacle.any()):
            return
        # State-feedback lane centring (same formula as AdaptiveLaneCommand /
        # eval AdaptiveCommand) only on obstacle columns.
        y = self._terrain_relative_y() - float(self.cfg.center_y)
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
        self.vel_command_b[obstacle, 1] = vy[obstacle]
        self.vel_command_b[obstacle, 2] = wz[obstacle]


def make_column_mixed_command_cfg(
    base: UniformVelocityCommandCfg,
    *,
    flat_column: str,
    traversal_vx: tuple[float, float],
    flat_standing_frac: float,
    adaptive_lane: bool = True,
) -> ColumnMixedVelocityCommandCfg:
    """Copy the existing twist cfg and swap in the per-column command term.

    ``base.ranges`` must be the **flat-column full ranges**; obstacle-column
    commands are sampled from ``traversal_vx`` inside the term.  ``flat_standing_frac``
    becomes the base ``rel_standing_envs`` (only flat-column envs keep the flag).
    """
    return ColumnMixedVelocityCommandCfg(
        entity_name=base.entity_name,
        resampling_time_range=base.resampling_time_range,
        debug_vis=base.debug_vis,
        heading_command=False,
        heading_control_stiffness=base.heading_control_stiffness,
        rel_standing_envs=float(flat_standing_frac),
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        init_velocity_prob=0.0,
        ranges=base.ranges,
        flat_column=flat_column,
        traversal_vx=traversal_vx,
        adaptive_lane=adaptive_lane,
    )
