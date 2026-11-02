"""Cfg-level wiring for the pose-edge flat-task recipe (blackW).

Called from ``blackw_flat_env_cfg``:

* ``apply_pose_low_speed_mixture`` — swap the twist command term for the
  low-speed mixture sampler (``RC_POSE_LOW_FRAC > 0``); inert by default;
* ``apply_pose_edge_terrain`` — replace the plane terrain with the
  flat + platform-edge generator (``RC_POSE_EDGES=1``), add spawn
  randomization around the patch centre, and fix the terminations for edge
  climbing (the flat task's trunk/thigh illegal-contact and no-progress stuck
  terms would abort exactly the leg-assisted edge mounts this recipe must
  learn; replace them with the traversal task's fall guard and an
  out-of-bounds truncation).
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import mdp
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_COMMAND_NAME
from alldog_mjlab.tasks.velocity.profiles.pose.command import (
	make_pose_envelope_command_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.pose.rewards import (
	joint_pos_deviation_l2,
	rotation_drift_rate,
)
from alldog_mjlab.tasks.velocity.profiles.pose.terrain import (
	pose_edge_terrain_generator_cfg,
)
from alldog_mjlab.utils import knobs


def apply_pose_low_speed_mixture(cfg) -> None:
	"""Swap the twist term for the low-speed mixture sampler (knob-gated)."""
	frac = knobs.get_float("RC_POSE_LOW_FRAC", 0.0)
	if frac <= 0.0:
		return
	if frac > 1.0:
		raise ValueError(f"RC_POSE_LOW_FRAC={frac} must be <= 1")
	base = cfg.commands[BLACK_COMMAND_NAME]
	cfg.commands[BLACK_COMMAND_NAME] = make_pose_envelope_command_cfg(
		base,
		low_frac=frac,
		low_vx=knobs.get_float("RC_POSE_LOW_VX", 0.15),
		low_wz=knobs.get_float("RC_POSE_LOW_WZ", 0.15),
	)


def apply_pose_edge_terrain(cfg, *, play: bool) -> None:
	"""Platform-edge terrain + spawn randomization + climb-safe terminations."""
	assert cfg.scene.terrain is not None
	generator = pose_edge_terrain_generator_cfg()
	cfg.scene.terrain.terrain_type = "generator"
	cfg.scene.terrain.terrain_generator = generator
	# Static random row assignment (the terrain_levels curriculum stays popped
	# by the flat task): every height row is sampled from iter 0.
	cfg.scene.terrain.max_init_terrain_level = generator.num_rows - 1

	# Spawn randomization around the patch centre.  xy must stay inside the
	# on-top platform footprint (0.6/0.9 half extents - 0.25 placement offset
	# leaves >= 0.35 m margin) and keep the worst-case floor-spawn footprint
	# (xy rand + front-wheel reach ~= 0.35 m) clear of the beside-mode platform
	# edge (clearance >= 0.38 m); yaw variety gives varied edge approach angles.
	spawn_xy = knobs.get_float("RC_POSE_SPAWN_XY", 0.12)
	spawn_yaw = knobs.get_float("RC_POSE_SPAWN_YAW", 0.35)
	if not (0.0 <= spawn_xy <= 0.20):
		raise ValueError(f"RC_POSE_SPAWN_XY={spawn_xy} must be in [0, 0.20]")
	if not (0.0 <= spawn_yaw <= 3.15):
		raise ValueError(f"RC_POSE_SPAWN_YAW={spawn_yaw} must be in [0, pi]")
	reset_base = cfg.events["reset_base"]
	params = dict(reset_base.params)
	pose_range = dict(params.get("pose_range") or {})
	pose_range.update(
		{"x": (-spawn_xy, spawn_xy), "y": (-spawn_xy, spawn_xy), "yaw": (-spawn_yaw, spawn_yaw)}
	)
	params["pose_range"] = pose_range
	cfg.events["reset_base"] = replace(reset_base, params=params)

	if not play:
		# Platform edges add wheel/leg-edge contacts that the flat-plane default
		# per-world nconmax (35) cannot hold; reuse the shared capacity knob with
		# a recipe default (the Part2 obstacle fields run 1024).
		cfg.sim.nconmax = knobs.get_int("RC_NCONMAX", 256)
		# Round-1 negative result: with the frozen flat-task upright weight
		# (-0.2) the shared actor generalized the edge-mount forward lean into a
		# persistent ~15 deg flat-ground pitch (probe tilt 1.6 -> 15 deg, vx
		# overshoot +17..25%).  A persistent lean must cost more than the climb
		# benefit it buys near edges, while mount transients (~1 s) still pay
		# off against the tracking differential, so scale the upright term.
		upright_w = knobs.get_float("RC_POSE_UPRIGHT_W", 1.2)
		if upright_w < 0.0:
			raise ValueError(f"RC_POSE_UPRIGHT_W={upright_w} must be >= 0")
		if "upright" in cfg.rewards:
			cfg.rewards["upright"] = replace(
				cfg.rewards["upright"], weight=-float(upright_w)
			)
		# Round-2 route negative result: the frozen base_height weight (-1.0)
		# let the stance collapse to a ~0.27 m crouch on flat ground (nominal
		# 0.48 m); handing over to an obstacle expert from that state jerks the
		# robot (measured 0.16 m + 14 deg in the first 0.5 s after the switch)
		# and derails the crossing.  Raising its weight (r4, -4.0) did NOT
		# fix it because the term targets *world* z: standing at nominal leg
		# extension on a 0.2 m platform reads as +0.2 m too high, so the
		# optimal compromise is still a low stance.  Replace it with a joint-
		# space anchor (ground-relative by construction, and equal to the
		# experts' dof_pos input distribution at the handover).
		baseh_w = knobs.get_float("RC_POSE_BASEH_W", 0.0)
		if baseh_w < 0.0:
			raise ValueError(f"RC_POSE_BASEH_W={baseh_w} must be >= 0")
		if "base_height" in cfg.rewards:
			if baseh_w > 0.0:
				cfg.rewards["base_height"] = replace(
					cfg.rewards["base_height"], weight=-float(baseh_w)
				)
			else:
				cfg.rewards.pop("base_height")
		dofpos_w = knobs.get_float("RC_POSE_DOFPOS_W", 0.15)
		if dofpos_w < 0.0:
			raise ValueError(f"RC_POSE_DOFPOS_W={dofpos_w} must be >= 0")
		rotdrift_w = knobs.get_float("RC_POSE_ROTDRIFT_W", 0.0)
		if rotdrift_w < 0.0:
			raise ValueError(f"RC_POSE_ROTDRIFT_W={rotdrift_w} must be >= 0")
		if rotdrift_w > 0.0:
			cfg.rewards["pose_rotdrift"] = RewardTermCfg(
				func=rotation_drift_rate,
				weight=-float(rotdrift_w),
				params={"command_name": BLACK_COMMAND_NAME},
			)
		if dofpos_w > 0.0:
			cfg.rewards["pose_dofpos"] = RewardTermCfg(
				func=joint_pos_deviation_l2,
				weight=-float(dofpos_w),
				params={
					"asset_cfg": SceneEntityCfg(
						"robot",
						joint_names=(".*hip_joint", ".*thigh_joint", ".*calf_joint"),
					)
				},
			)
		# Edge mounts need trunk/thigh contact with the platform and can dwell
		# without planar progress; the flat task's illegal_contact (trunk+thigh
		# vs terrain, 1 N) and stuck terms would terminate exactly those
		# attempts.  Mirror the traversal contract: fall guard + OOB truncation.
		cfg.terminations.pop("illegal_contact", None)
		cfg.terminations.pop("stuck", None)
		cfg.terminations["bad_orientation"] = TerminationTermCfg(
			func=mdp.bad_orientation,
			params={"limit_angle": 1.05},
		)
		cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
			func=velocity_mdp.out_of_terrain_bounds,
			time_out=True,
		)
