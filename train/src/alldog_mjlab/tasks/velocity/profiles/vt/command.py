"""Command sampler for the velocity-tracking primitive.

The pose envelope (``POSE_CMD``) samples vx / wz independently over rectangles,
which makes a *pure* in-place turn (vx=0, wz != 0) very rare: |vx| < 0.04 has
probability ~4% inside the envelope, so a position-loop harness that says
"turn in place now" is an out-of-distribution request.  The same is true for
pure translation (wz=0 while |vx| is meaningful) and for reverse-then-stop.

This sampler keeps the envelope + low-speed mixture and adds explicit mode
masses:

* ``pure_yaw_frac``: vx=0, vy=0, wz ~ +-U(wz_min, wz_max);
* ``pure_vx_frac``: vx ~ U(vx_min, vx_max), vy=0, wz=0;
* the rest: the normal envelope / low-speed mixture.

Training-only code: the deployment contract is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from alldog_mjlab.tasks.velocity.profiles.pose.command import (
	PoseEnvelopeVelocityCommand,
	PoseEnvelopeVelocityCommandCfg,
	make_pose_envelope_command_cfg,
)


@dataclass(kw_only=True)
class VTPrimitiveVelocityCommandCfg(PoseEnvelopeVelocityCommandCfg):
	pure_yaw_frac: float = 0.35
	"""Fraction of non-standing resamples that become a pure in-place turn."""
	pure_vx_frac: float = 0.15
	"""Fraction of non-standing resamples that become pure translation."""
	pure_yaw_wz_min: float = 0.15
	"""Minimum |wz| for a pure turn [rad/s] (above the low-speed dead zone)."""
	pure_yaw_wz_max: float = 0.60
	"""Maximum |wz| for a pure turn [rad/s]."""

	def __post_init__(self) -> None:
		super().__post_init__()
		for name, value in (
			("pure_yaw_frac", self.pure_yaw_frac),
			("pure_vx_frac", self.pure_vx_frac),
		):
			if not 0.0 <= value <= 1.0:
				raise ValueError(f"{name}={value!r} must be in [0, 1]")
		if self.pure_yaw_frac + self.pure_vx_frac > 1.0:
			raise ValueError("pure_yaw_frac + pure_vx_frac must be <= 1")
		if not 0.0 <= self.pure_yaw_wz_min <= self.pure_yaw_wz_max:
			raise ValueError("pure_yaw_wz_min/max must satisfy 0 <= min <= max")

	def build(self, env):
		return VTPrimitiveVelocityCommand(self, env)


class VTPrimitiveVelocityCommand(PoseEnvelopeVelocityCommand):
	cfg: VTPrimitiveVelocityCommandCfg

	def _resample_command(self, env_ids: torch.Tensor) -> None:
		super()._resample_command(env_ids)
		n = len(env_ids)
		if n == 0:
			return
		u = torch.rand(n, device=self.device)
		standing = self.is_standing_env[env_ids]
		non_standing = ~standing
		yaw_mass = float(self.cfg.pure_yaw_frac)
		vx_mass = float(self.cfg.pure_vx_frac)
		yaw_mask = non_standing & (u < yaw_mass)
		vx_mask = non_standing & (u >= yaw_mass) & (u < yaw_mass + vx_mass)

		ids = env_ids[yaw_mask]
		if len(ids):
			sign = torch.where(
				torch.rand(len(ids), device=self.device) < 0.5, -1.0, 1.0
			)
			self.vel_command_b[ids, 0] = 0.0
			self.vel_command_b[ids, 1] = 0.0
			self.vel_command_b[ids, 2] = sign * torch.empty(
				len(ids), device=self.device
			).uniform_(self.cfg.pure_yaw_wz_min, self.cfg.pure_yaw_wz_max)

		ids = env_ids[vx_mask]
		if len(ids):
			self.vel_command_b[ids, 0] = torch.empty(
				len(ids), device=self.device
			).uniform_(float(self.cfg.ranges.lin_vel_x[0]), float(self.cfg.ranges.lin_vel_x[1]))
			self.vel_command_b[ids, 1] = 0.0
			self.vel_command_b[ids, 2] = 0.0


def make_vt_command_cfg(base: PoseEnvelopeVelocityCommandCfg) -> VTPrimitiveVelocityCommandCfg:
	"""Copy a configured pose-envelope command cfg into the VT mode sampler."""
	return VTPrimitiveVelocityCommandCfg(
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
		low_frac=base.low_frac,
		low_vx=base.low_vx,
		low_wz=base.low_wz,
		pure_yaw_frac=0.50,
		pure_vx_frac=0.10,
		pure_yaw_wz_min=0.10,
		pure_yaw_wz_max=0.60,
	)


__all__ = [
	"VTPrimitiveVelocityCommand",
	"VTPrimitiveVelocityCommandCfg",
	"make_vt_command_cfg",
	"make_pose_envelope_command_cfg",
]
