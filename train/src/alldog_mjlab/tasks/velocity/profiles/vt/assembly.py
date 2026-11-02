"""Cfg-level contract for the blackW velocity-tracking primitive task.

Explicit recipe (no env-var knobs: the task id ``blackw-vt`` is reproducible by
construction).  Values/negative results are documented in
``docs/TRAINING.md`` and ``docs/VT-RECIPE.md`` at the repo root.

Built on top of ``build_velocity_env_cfg(rough=False)`` + the blackW overrides,
so the actor/critic contract, sensors, actions and physics stay identical to
every other blackW task (57x6 obs, 16 actions, 2 ms physics).
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.tasks.velocity.black.rewards import track_linear_velocity_xy_l2
from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_COMMAND_NAME
from alldog_mjlab.tasks.velocity.profiles.pose.assembly import apply_pose_edge_terrain
from alldog_mjlab.tasks.velocity.profiles.pose.command import (
	make_pose_envelope_command_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.vt.command import make_vt_command_cfg
from alldog_mjlab.tasks.velocity.profiles.pose.rewards import joint_pos_deviation_l2
from alldog_mjlab.tasks.velocity.profiles.vt.rewards import (
	rotation_drift_rate_masked,
	turn_in_place_hold,
)

# ---------------------------------------------------------------------------
# Command envelope: what the position-loop harness actually sends in Part 1.
# ---------------------------------------------------------------------------
VX_MIN = -0.40
VX_MAX = 1.00
VY_MAX = 0.0          # direction 2026-11-01: no lateral investment (wheel-leg weak axis)
WZ_MAX = 0.65
STANDING_FRAC = 0.10
RESAMPLE_S = 5.0
LOW_FRAC = 0.50       # fraction of non-standing resamples from the low-speed box
LOW_VX = 0.12         # covers reverse / slow creep
LOW_WZ = 0.20         # covers in-place yaw dead zone

# ---------------------------------------------------------------------------
# Reward contract (tracking-dominant + stance + turn-drift).
# ---------------------------------------------------------------------------
TRACK_LIN_W = 2.0
TRACK_ANG_W = 2.5
TRACK_LIN_SIGMA = 0.25
TRACK_ANG_SIGMA = 0.08
TRACK_ERR_L2_W = 0.30
UPRIGHT_W = 1.2
DOFPOS_W = 0.80
ROTDRIFT_W = 1.00
ROTDRIFT_YAW_SIGMA = 0.02
ROTDRIFT_SPEED_FLOOR = 0.10
TURN_HOLD_W = 0.0  # disabled: vt_v3 negative result (under-turn / wander)
TURN_HOLD_YAW_SIGMA = 0.02
TURN_HOLD_SPEED_SIGMA = 0.01

# ---------------------------------------------------------------------------
# Robustness (RecoveryLoco-style random states / pushes / friction DR).
# ---------------------------------------------------------------------------
RESET_POSE_X = 0.20
RESET_POSE_Y = 0.20
RESET_POSE_ROLLPITCH = 0.10
RESET_VEL_XY = 0.30
RESET_VEL_ROLLPITCH = 0.15
RESET_VEL_YAW = 0.50
RESET_JOINT_POS = 0.08
RESET_JOINT_VEL = 1.0
PUSH_INTERVAL_S = (5.0, 12.0)
PUSH_VEL_XY = 0.80
WHEEL_FRICTION_RANGE = (0.40, 1.20)

LEG_JOINT_NAMES = (".*hip_joint", ".*thigh_joint", ".*calf_joint")
WHEEL_COLLISION_GEOMS = (
	"FL_wheel_collision",
	"FR_wheel_collision",
	"RL_wheel_collision",
	"RR_wheel_collision",
)


def _configure_command(cfg) -> None:
	command = cfg.commands[BLACK_COMMAND_NAME]
	command.ranges.lin_vel_x = (VX_MIN, VX_MAX)
	command.ranges.lin_vel_y = (-VY_MAX, VY_MAX)
	command.ranges.ang_vel_z = (-WZ_MAX, WZ_MAX)
	command.ranges.heading = None
	command.heading_command = False
	command.rel_heading_envs = 0.0
	command.rel_forward_envs = 0.0
	command.rel_standing_envs = STANDING_FRAC
	command.resampling_time_range = (RESAMPLE_S, RESAMPLE_S)
	# No command curriculum: the envelope above is the deployment envelope.
	cfg.curriculum.pop("command", None)
	envelope = make_pose_envelope_command_cfg(
		command,
		low_frac=LOW_FRAC,
		low_vx=LOW_VX,
		low_wz=LOW_WZ,
	)
	# Explicit pure-yaw / pure-vx mode masses: the position loop asks for
	# "turn in place" and "drive straight" far more often than the independent
	# envelope sampler produces them.
	cfg.commands[BLACK_COMMAND_NAME] = make_vt_command_cfg(envelope)


def _configure_rewards(cfg) -> None:
	track_lin = cfg.rewards["track_linear_velocity"]
	cfg.rewards["track_linear_velocity"] = replace(
		track_lin,
		weight=TRACK_LIN_W,
		params={**track_lin.params, "sigma": TRACK_LIN_SIGMA},
	)
	# Sharper angular sigma: exp(-err^2/sigma) is nearly flat at small yaw
	# errors for sigma=0.25, which made "ignore low wz commands" cheap.  With
	# sigma=0.08 a 0.2 rad/s command executed as 0 scores 0.61 instead of 0.85,
	# so accurate in-place turning pays.
	track_ang = cfg.rewards["track_angular_velocity"]
	cfg.rewards["track_angular_velocity"] = replace(
		track_ang,
		weight=TRACK_ANG_W,
		params={**track_ang.params, "sigma": TRACK_ANG_SIGMA},
	)
	cfg.rewards["track_linear_velocity_l2"] = RewardTermCfg(
		func=track_linear_velocity_xy_l2,
		weight=-TRACK_ERR_L2_W,
		params={"command_name": BLACK_COMMAND_NAME},
	)
	# Stronger upright than the frozen flat baseline (pose recipe r2: 1.2).
	cfg.rewards["upright"] = replace(cfg.rewards["upright"], weight=-UPRIGHT_W)
	# Joint-space stance anchor: ground-relative and equal to the obstacle
	# experts' dof_pos distribution at handover (world-z base_height discouraged
	# a crouch in the pose recipe; removed here for the same reason).
	cfg.rewards.pop("base_height", None)
	cfg.rewards["pose_dofpos"] = RewardTermCfg(
		func=joint_pos_deviation_l2,
		weight=-DOFPOS_W,
		params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
	)
	# "Turn without translating": penalise |v_xy| / |wz_cmd| for pure-yaw
	# commands, gated on yaw-tracking accuracy (otherwise "don't turn" is free).
	cfg.rewards["pose_rotdrift"] = RewardTermCfg(
		func=rotation_drift_rate_masked,
		weight=-ROTDRIFT_W,
		params={
			"command_name": BLACK_COMMAND_NAME,
			"yaw_sigma": ROTDRIFT_YAW_SIGMA,
			"speed_floor": ROTDRIFT_SPEED_FLOOR,
		},
	)
	# Positive companion: rewards being still *while actually turning*.
	# Disabled by default: the vt_v3 run showed it can trade away low-wz yaw
	# tracking (the policy wanders instead of completing the turn).  Kept as an
	# explicit knob for a later stepping-turn experiment.
	if TURN_HOLD_W > 0.0:
		cfg.rewards["pose_turn_hold"] = RewardTermCfg(
			func=turn_in_place_hold,
			weight=TURN_HOLD_W,
			params={
				"command_name": BLACK_COMMAND_NAME,
				"yaw_sigma": TURN_HOLD_YAW_SIGMA,
				"speed_sigma": TURN_HOLD_SPEED_SIGMA,
			},
		)


def _configure_events(cfg, *, edges: bool) -> None:
	reset_base = cfg.events["reset_base"]
	params = dict(reset_base.params)
	# RecoveryLoco-style random states: random yaw over the full circle + small
	# roll/pitch, so the primitives starts from any handover orientation.  On the
	# edge terrain the pose range is owned by apply_pose_edge_terrain (it must keep
	# on-top spawns inside the platform footprint), so only velocities are widened.
	if not edges:
		pose_range = dict(params.get("pose_range") or {})
		pose_range.update(
			{
				"x": (-RESET_POSE_X, RESET_POSE_X),
				"y": (-RESET_POSE_Y, RESET_POSE_Y),
				"roll": (-RESET_POSE_ROLLPITCH, RESET_POSE_ROLLPITCH),
				"pitch": (-RESET_POSE_ROLLPITCH, RESET_POSE_ROLLPITCH),
				"yaw": (-math.pi, math.pi),
			}
		)
		params["pose_range"] = pose_range
	params["velocity_range"] = {
		"x": (-RESET_VEL_XY, RESET_VEL_XY),
		"y": (-RESET_VEL_XY, RESET_VEL_XY),
		"z": (0.0, 0.0),
		"roll": (-RESET_VEL_ROLLPITCH, RESET_VEL_ROLLPITCH),
		"pitch": (-RESET_VEL_ROLLPITCH, RESET_VEL_ROLLPITCH),
		"yaw": (-RESET_VEL_YAW, RESET_VEL_YAW),
	}
	cfg.events["reset_base"] = replace(reset_base, params=params)

	reset_joints = cfg.events["reset_robot_joints"]
	joint_params = dict(reset_joints.params)
	joint_params["position_range"] = (-RESET_JOINT_POS, RESET_JOINT_POS)
	joint_params["velocity_range"] = (-RESET_JOINT_VEL, RESET_JOINT_VEL)
	cfg.events["reset_robot_joints"] = replace(reset_joints, params=joint_params)

	if "push_robot" in cfg.events:
		push = cfg.events["push_robot"]
		push_params = dict(push.params)
		push_params["velocity_range"] = {
			"x": (-PUSH_VEL_XY, PUSH_VEL_XY),
			"y": (-PUSH_VEL_XY, PUSH_VEL_XY),
			"z": (0.0, 0.0),
			"roll": (0.0, 0.0),
			"pitch": (0.0, 0.0),
			"yaw": (0.0, 0.0),
		}
		cfg.events["push_robot"] = replace(
			push, interval_range_s=PUSH_INTERVAL_S, params=push_params
		)


def _configure_wheel_friction(cfg) -> None:
	"""Wheel-friction domain randomization.

	The generic ``foot_friction`` event targets ``<leg>_foot_collision`` geoms,
	which do not exist on blackW; the wheel collision geoms are now named
	(``<leg>_wheel_collision``), so retarget the same DR event instead of
	dropping it.  Only added to this task (other blackW tasks stay untouched).
	"""
	cfg.events["wheel_friction"] = EventTermCfg(
		func=dr.geom_friction,
		mode="startup",
		params={
			"asset_cfg": SceneEntityCfg("robot", geom_names=WHEEL_COLLISION_GEOMS),
			"operation": "abs",
			"ranges": WHEEL_FRICTION_RANGE,
		},
	)


def apply_vt_contract(cfg, *, play: bool, edges: bool = False) -> None:
	"""Apply the complete velocity-tracking primitive recipe to a blackW flat cfg."""
	# Edge variant first: it replaces terrain / spawn / terminations and installs
	# its reward terms, which we then overwrite with the explicit VT weights.
	if edges:
		apply_pose_edge_terrain(cfg, play=play)
	_configure_command(cfg)
	_configure_rewards(cfg)
	if not play:
		_configure_events(cfg, edges=edges)
		_configure_wheel_friction(cfg)
		if edges:
			# Edge mounting needs leg/trunk contact with the platform; the default
			# flat-task illegal_contact / stuck terminations abort exactly those
			# attempts. apply_pose_edge_terrain already swapped them for the
			# traversal fall guard + OOB truncation.
			pass
	else:
		# Never run DR / random resets in play: the rollout must be deterministic.
		for name in ("wheel_friction",):
			cfg.events.pop(name, None)
