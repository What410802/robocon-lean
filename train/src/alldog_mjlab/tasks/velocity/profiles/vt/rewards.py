"""Reward terms specific to the velocity-tracking primitive (blackW).

Shares the pose profile's stance anchor semantics; adds the turn-drift term used
by the position-loop harness requirement "turn in place without translating
away".
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET = SceneEntityCfg("robot")


def rotation_drift_rate_masked(
	env,
	command_name: str = "twist",
	yaw_sigma: float = 0.02,
	speed_floor: float = 0.10,
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
) -> torch.Tensor:
	"""Drift speed per commanded yaw rate, gated by yaw-tracking accuracy.

	For pure-rotation commands (``vx=vy=0``, ``|wz| > 0.1``) returns
	``gate * |v_xy| / max(|wz_cmd|, speed_floor)`` where
	``gate = exp(-(wz_cmd - wz_actual)^2 / yaw_sigma)``.

	Why the gate: the plain ratio (pose profile ``rotation_drift_rate``) can be
	cheated by not turning at all (``|v_xy|=0`` pays no penalty), and with weak
	yaw tracking it made the policy prefer "stand still" over "turn slowly".
	Gating on yaw accuracy keeps the drift gradient exactly where the robot is
	actually executing the turn, so the optimum becomes "turn accurately *and*
	without translation".
	"""

	asset = env.scene[asset_cfg.name]
	command = env.command_manager.get_command(command_name)
	assert command is not None, f"Command '{command_name}' not found."
	rot = command[:, 2].abs() > 0.10
	lin = (command[:, 0].abs() + command[:, 1].abs()) < 0.05
	mask = rot & lin
	wz_err2 = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
	gate = torch.exp(-wz_err2 / max(float(yaw_sigma), 1e-6))
	speed = asset.data.root_link_lin_vel_b[:, :2].norm(dim=1)
	scale = command[:, 2].abs().clamp(min=float(speed_floor))
	return torch.where(mask, gate * speed / scale, torch.zeros_like(speed))


def turn_in_place_hold(
	env,
	command_name: str = "twist",
	yaw_sigma: float = 0.02,
	speed_sigma: float = 0.01,
	asset_cfg: SceneEntityCfg = _DEFAULT_ASSET,
) -> torch.Tensor:
	"""Positive "turn without translating" shaping, gated on yaw accuracy.

	For pure-rotation commands returns
	``gate * exp(-|v_xy|^2 / speed_sigma)`` where
	``gate = exp(-(wz_cmd - wz_actual)^2 / yaw_sigma)``.

	This is a positive reward (not a ratio penalty), so it has a usable
	gradient at the small drift speeds that matter for the 1.0 m platform
	handover; the gate prevents "stand still, don't turn" from collecting it.
	"""
	asset = env.scene[asset_cfg.name]
	command = env.command_manager.get_command(command_name)
	assert command is not None, f"Command '{command_name}' not found."
	rot = command[:, 2].abs() > 0.10
	lin = (command[:, 0].abs() + command[:, 1].abs()) < 0.05
	mask = rot & lin
	wz_err2 = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
	yaw_gate = torch.exp(-wz_err2 / max(float(yaw_sigma), 1e-6))
	speed2 = asset.data.root_link_lin_vel_b[:, :2].square().sum(dim=1)
	hold = torch.exp(-speed2 / max(float(speed_sigma), 1e-6))
	return torch.where(mask, yaw_gate * hold, torch.zeros_like(hold))


# Four contact points / leg plane (RecoveryLoco point-foot rewards adapted to
# wheel-leg): the wheels must follow the commanded body rotation without lag,
# stay in one plane parallel to the trunk, and not twist.  These use the wheel
# collision geom names added to the training MJCF (names only, no physics).
WHEEL_COLLISION_GEOMS = (
	"FL_wheel_collision",
	"FR_wheel_collision",
	"RL_wheel_collision",
	"RR_wheel_collision",
)
_WHEEL_CFG = SceneEntityCfg("robot", geom_names=WHEEL_COLLISION_GEOMS)


def _wheel_frame(env, asset_cfg: SceneEntityCfg):
	asset = env.scene[asset_cfg.name]
	return asset, asset_cfg.geom_ids


def wheel_plane_level(
	env,
	command_name: str = "twist",
	asset_cfg: SceneEntityCfg = _WHEEL_CFG,
	sigma: float = 0.0004,
) -> torch.Tensor:
	"""Reward the four wheel centres lying in one body-parallel plane.

	Body-frame wheel-centre z spread is exactly "四个落足点所在平面与机身平齐";
	unlike a world-z spread it stays valid when the trunk is pitched/rolled on
	an obstacle.  Near standing damping => spread ~0, reward ~1; a lifted or
	misplaced wheel increases the spread and loses reward.  Split out from the
	old point-foot ``feet_air_time``/``collision`` terms because wheel-legs have
	no swing phase, but the *contact plane consistency* they implied remains.
	"""
	asset, geom_ids = _wheel_frame(env, asset_cfg)
	command = env.command_manager.get_command(command_name)
	assert command is not None, f"Command '{command_name}' not found."
	rot = command[:, 2].abs() > 0.10
	lin = (command[:, 0].abs() + command[:, 1].abs()) < 0.05
	standing = (command[:, :3].abs().sum(dim=1) < 0.05)
	active = (rot & lin) | standing
	pos_w = asset.data.geom_pos_w[:, geom_ids, :]
	root_pos = asset.data.root_link_pos_w[:, None, :]
	root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, pos_w.shape[1], -1)
	rel = pos_w - root_pos
	from mjlab.utils.lab_api.math import quat_apply_inverse

	body_pos = quat_apply_inverse(root_quat, rel)
	z = body_pos[..., 2]
	spread = z.max(dim=1).values - z.min(dim=1).values
	level = torch.exp(-torch.square(spread) / max(float(sigma), 1e-9))
	# Only when rotating in place / standing: while translating, a four-wheel
	# coplanarity bonus made the policy drive slowly (vt_v7_600 translate ratio
	# 0.63 vs 0.75+ on v1).  The user's requirement is about the turn/handover
	# state, not straight-line speed.
	return torch.where(active, level, torch.zeros_like(level))


def wheel_rotation_follow(
	env,
	command_name: str = "twist",
	sigma: float = 0.02,
	asset_cfg: SceneEntityCfg = _WHEEL_CFG,
) -> torch.Tensor:
	"""Reward wheel-centre velocities matching the rigid commanded yaw.

	For pure-yaw commands the rigid-body field about the trunk is
	``v_i = wz_cmd * (-(y_i-y_root), +(x_i-x_root))``.  A wheel that lags or
	slips has a large deviation, so this is the wheel-leg analogue of
	RecoveryLoco's foot-velocity/contact tracking: all four contact points follow
	the rotation instead of the trunk twisting relative to them.
	"""
	asset, geom_ids = _wheel_frame(env, asset_cfg)
	command = env.command_manager.get_command(command_name)
	assert command is not None, f"Command '{command_name}' not found."
	rot = command[:, 2].abs() > 0.10
	lin = (command[:, 0].abs() + command[:, 1].abs()) < 0.05
	mask = rot & lin
	# Target the *commanded* yaw rate.  (Using the actual trunk rate was tried in
	# vt_v8: it let the policy stand still and collect the follow reward, because
	# with wz_actual~0 the target wheel velocities are ~0 too.  Command tracking
	# keeps the rotation incentive; the probe reports both variants.)
	wz = command[:, 2]
	pos_w = asset.data.geom_pos_w[:, geom_ids, :2]
	root_xy = asset.data.root_link_pos_w[:, None, :2]
	rel = pos_w - root_xy
	target = torch.stack((-wz[:, None] * rel[..., 1], wz[:, None] * rel[..., 0]), dim=-1)
	vel = asset.data.geom_lin_vel_w[:, geom_ids, :2]
	err = torch.sum(torch.square(vel - target), dim=(1, 2))
	return torch.where(mask, torch.exp(-err / max(float(sigma), 1e-9)), torch.zeros_like(err))


__all__ = ["rotation_drift_rate_masked", "turn_in_place_hold",
           "wheel_plane_level", "wheel_rotation_follow"]

