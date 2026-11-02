"""Low-speed mixture sampling on top of the pose command envelope.

Why: the pose envelope (``RC_POSE_CMD``) samples vx/vy/wz *uniformly* over the
operating ranges, so the region where the measured dead zones live (|vx| <=
0.15 including reverse, |wz| <= 0.15) receives only ~20-30% of the command
mass.  Combined with the near-flat exponential tracking reward at small errors,
the policy rationally ignores tiny commands (vt_pose_v1 probe: -0.05 -> -0.007
m/s, +0.05 rad/s -> 0.000 rad/s).

Fix: keep the envelope ranges but, for a configurable fraction of non-standing
resamples, draw vx/wz from a small symmetric box around zero.  This multiplies
the low-speed sample mass by ~3x at ``low_frac=0.5`` without changing the
envelope itself.  vy is deliberately untouched (2026-11-01 direction: no vy
investment; the recipe sets ``RC_POSE_VY_MAX=0``).

Training-only code: the deployment/eval contract is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


@dataclass(kw_only=True)
class PoseEnvelopeVelocityCommandCfg(UniformVelocityCommandCfg):
	"""Pose envelope + low-speed mixture sampling."""

	low_frac: float = 0.5
	"""Fraction of non-standing resamples drawn from the low-speed box."""
	low_vx: float = 0.15
	"""Low-speed box |vx| bound [m/s] (covers the reverse dead zone)."""
	low_wz: float = 0.15
	"""Low-speed box |wz| bound [rad/s] (covers the in-place yaw dead zone)."""

	def __post_init__(self) -> None:
		super().__post_init__()
		for name, value in (
			("low_frac", self.low_frac),
			("low_vx", self.low_vx),
			("low_wz", self.low_wz),
		):
			if not 0.0 <= value:
				raise ValueError(f"{name}={value!r} must be >= 0")
		if self.low_frac > 1.0:
			raise ValueError(f"low_frac={self.low_frac!r} must be <= 1")

	def build(self, env):
		return PoseEnvelopeVelocityCommand(self, env)


class PoseEnvelopeVelocityCommand(UniformVelocityCommand):
	cfg: PoseEnvelopeVelocityCommandCfg

	def _resample_command(self, env_ids: torch.Tensor) -> None:
		super()._resample_command(env_ids)
		if self.cfg.low_frac <= 0.0:
			return
		low = torch.rand(len(env_ids), device=self.device) < float(self.cfg.low_frac)
		# Standing envs keep their zero command (zeroed every step anyway).
		low &= ~self.is_standing_env[env_ids]
		ids = env_ids[low]
		if len(ids) == 0:
			return
		r = torch.empty(len(ids), device=self.device)
		self.vel_command_b[ids, 0] = r.uniform_(
			-float(self.cfg.low_vx), float(self.cfg.low_vx)
		)
		self.vel_command_b[ids, 2] = r.uniform_(
			-float(self.cfg.low_wz), float(self.cfg.low_wz)
		)


def make_pose_envelope_command_cfg(
	base: UniformVelocityCommandCfg,
	*,
	low_frac: float,
	low_vx: float,
	low_wz: float,
) -> PoseEnvelopeVelocityCommandCfg:
	"""Copy the configured twist cfg and swap in the mixture command term.

	``base`` must already carry the pose envelope ranges/fractions (i.e. call
	this after ``apply_pose_command_contract``).
	"""
	return PoseEnvelopeVelocityCommandCfg(
		entity_name=base.entity_name,
		resampling_time_range=base.resampling_time_range,
		debug_vis=base.debug_vis,
		heading_command=base.heading_command,
		heading_control_stiffness=base.heading_control_stiffness,
		rel_standing_envs=base.rel_standing_envs,
		rel_heading_envs=base.rel_heading_envs,
		rel_world_envs=base.rel_world_envs,
		rel_forward_envs=base.rel_forward_envs,
		init_velocity_prob=base.init_velocity_prob,
		ranges=base.ranges,
		low_frac=float(low_frac),
		low_vx=float(low_vx),
		low_wz=float(low_wz),
	)
