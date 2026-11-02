"""Pose-primitive reward terms (blackW pose-edge recipe)."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET = SceneEntityCfg(
	"robot", joint_names=(".*hip_joint", ".*thigh_joint", ".*calf_joint")
)
"""Leg joints only: the wheel hinges are unbounded rotations whose positions
are meaningless for stance (r5 first launch measured -2.3/step because wheel
spin alone dominated the term)."""


def joint_pos_deviation_l2(
	env,
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
) -> torch.Tensor:
	"""Mean squared leg-joint deviation from the default stance.

	Ground-relative stance anchor for the pose-edge recipe: the world-z
	``base_height`` term is incompatible with platform terrain (standing at
	nominal leg extension on a 0.2 m platform reads as +0.2 m "too high", so
	the r2/r4 rounds collapsed the stance into a ~0.27 m crouch everywhere,
	including flat ground).  Anchoring in joint space instead keeps the
	nominal leg extension on any surface, which is also exactly the
	handover-compatibility requirement: the obstacle experts read absolute
	``dof_pos`` observations and were trained from the default stance.
	"""
	asset = env.scene[asset_cfg.name]
	ids = asset_cfg.joint_ids
	dev = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
	return torch.mean(torch.square(dev), dim=1)


def rotation_drift_rate(
	env,
	command_name: str = "twist",
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
) -> torch.Tensor:
	"""Planar drift speed per commanded turn rate for pure-rotation commands.

	Directly optimises the wheel-rectangle drift per radian turned (the
	support-rectangle criterion): during (0, 0, wz) commands the rolling-wheel
	in-place turn couples rotation with translation (no-strafe kinematics),
	and the generic tracking terms penalise that drift only weakly (measured
	0.1-0.3 m per 90 deg across model generations).  This term returns
	|v_xy| / max(|wz_cmd|, floor) for envs commanded a pure rotation and zero
	otherwise, so the policy pays in exactly the units that matter.
	"""
	asset = env.scene[asset_cfg.name]
	command = env.command_manager.get_command(command_name)
	rot = command[:, 2].abs() > 0.10
	lin = (command[:, 0].abs() + command[:, 1].abs()) < 0.05
	mask = rot & lin
	speed = asset.data.root_link_lin_vel_b[:, :2].norm(dim=1)
	scale = command[:, 2].abs().clamp(min=0.10)
	return torch.where(mask, speed / scale, torch.zeros_like(speed))
