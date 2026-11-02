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


__all__ = ["rotation_drift_rate_masked", "turn_in_place_hold"]
