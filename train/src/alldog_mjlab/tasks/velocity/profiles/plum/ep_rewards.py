"""Extreme-Parkour reward table for the plum teacher (ported from ~/RoboCon).

The old repo's only successful plum teacher used the EP table from
arXiv:2309.14341 verbatim (parkour ``LeggedRobotCfg.rewards.scales``) with
``only_positive`` clipping, instead of Part-2 progress bonuses.  Two hand-written
positive-only tables in this repo scored 0/178 and 0/4000 because their dominant
term (monotone ``min(wheel_x)`` progress) is support-blind: vaulting over the
field paid the same as stepping on it, and the policy learned to fly.

Reward-scale convention: EP multiplies ``scales`` by ``dt`` in its own reward
normalisation; MjLab's RewardManager applies ``raw * weight * policy_dt``
(0.02 s), so the EP scales are used verbatim as MjLab weights.

Adaptations to this repo / this field (all deliberate, none silent):

* ``tracking_goal_vel`` / ``tracking_yaw`` -- EP projects velocity onto the
  direction of the next waypoint; this field is straight +x, so the waypoint
  system collapses to ``min(vx_world, cmd_vx) / cmd_vx`` and target yaw 0.  The
  command still exists as a driver but never enters the actor observation.
* ``feet_edge`` -- EP uses a heightfield edge mask.  This field is analytic box
  geometry, so the rim distance is computed from ``spec.py`` cube centres.  Only
  the *official* layout has a meaningful rim; with the cube-size curriculum
  (``cube_half`` scaled) the margin is measured against the scaled cube.
* ``feet_stumble`` -- EP flags ``|F_xy| > 4|F_z|`` (hitting a vertical face);
  reused unchanged on the wheel contact sensor.
* ``collision`` -- EP counts non-wheel bodies with contact force > 0.1 N.  The
  strict termination already ends the episode on the first such contact; the
  penalty still shapes the step *before* termination and is kept for fidelity.
* ``dof_error`` / ``hip_pos`` -- the unbounded wheel-roll joints are excluded
  (their angle has no meaningful default); EP's point-foot robot has none.
* The success bonus is added separately by the teacher builder (``part2_clear``),
  because EP's table alone is a pure locomotion prior with no terminal signal.
"""

from __future__ import annotations

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as envs_mdp
from mjlab.utils.lab_api.math import wrap_to_pi

from alldog_mjlab.tasks.velocity.black.rewards import (
    angular_velocity_xy_l2,
    dof_acc_l2,
    vertical_linear_velocity_l2,
)
from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_ILLEGAL_CONTACT_SENSOR
from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
    PART2_WHEEL_RADIUS_M,
    _part2_wheel_lowest_points,
    _plum_top_z,
)

from .spec import SPEC

#: EP parkour scales, verbatim (see module docstring).
EP_SCALES = {
    "tracking_goal_vel": 1.5,
    "tracking_yaw": 0.5,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "dof_acc": -2.5e-7,
    "collision": -10.0,
    "action_rate": -0.1,
    "delta_torques": -1e-7,
    "torques": -1e-5,
    "hip_pos": -0.5,
    "dof_error": -0.04,
    "feet_stumble": -1.0,
    "feet_edge": -1.0,
}

_ASSET = SceneEntityCfg("robot")
_WHEEL_NAMES = ("FL", "FR", "RL", "RR")
_FEET_SENSOR = "feet_ground_contact"
_EDGE_MARGIN = 0.05          # EP edge_width_thresh
_ON_CUBE_TOL = 0.06          # wheel bottom band around the cube top


def _origins(env) -> torch.Tensor:
    origins = env.scene.terrain.env_origins
    return origins.to(env.device) if isinstance(origins, torch.Tensor) else torch.as_tensor(
        origins, device=env.device)


def _wheel_geometry(env, asset_cfg: SceneEntityCfg = _ASSET):
    """Env-local wheel x/y and world z of the four wheel lowest points."""
    asset = env.scene[asset_cfg.name]
    ids = [asset.body_names.index(f"{leg}_foot") for leg in _WHEEL_NAMES]
    pos = _part2_wheel_lowest_points(asset, ids, PART2_WHEEL_RADIUS_M)
    origins = _origins(env)
    return pos[..., 0] - origins[:, 0:1], pos[..., 1] - origins[:, 1:2], pos[..., 2]


def _root_yaw(asset) -> torch.Tensor:
    w, x, y, z = [asset.data.root_link_quat_w[:, i] for i in range(4)]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Tracking (EP waypoint system reduced to this straight +x field)
# ---------------------------------------------------------------------------


def tracking_goal_vel(
    env, command_name: str = "twist", asset_cfg: SceneEntityCfg = _ASSET
) -> torch.Tensor:
    """``min(<v, +x>, cmd_vx) / cmd_vx``: 1 when tracking, 0 standing, <0 backward."""
    asset = env.scene[asset_cfg.name]
    cmd_vx = env.command_manager.get_command(command_name)[:, 0]
    return torch.minimum(asset.data.root_link_lin_vel_w[:, 0], cmd_vx) / (cmd_vx + 1e-5)


def tracking_yaw(
    env, asset_cfg: SceneEntityCfg = _ASSET, min_speed_frac: float = 0.35
) -> torch.Tensor:
    """``exp(-|yaw|)``, gated on actually moving forward.

    Ungated this term is the idle optimum: a stationary, forward-facing robot
    collects ``exp(0) = 1`` every step, and with the ``only_positive`` clip the
    penalties that should deter standing still are erased.  Measured on this
    field: 21 s of standing still returned **+11.8** while a successful crossing
    paid +10 -- i.e. idling strictly out-earned solving the task, and the v3b run
    converged to 1061-step episodes with zero successes.

    The gate uses the robot's own forward speed relative to the fixed command, so
    a robot that is stationary, or moving backward, gets nothing.
    """
    from .teacher import COMMAND_VX

    asset = env.scene[asset_cfg.name]
    gate = 0.5 * (COMMAND_VX[0] + COMMAND_VX[1]) * min_speed_frac
    moving = (asset.data.root_link_lin_vel_w[:, 0] > gate).float()
    return moving * torch.exp(-_root_yaw(asset).abs())


# ---------------------------------------------------------------------------
# Regularisation (EP implementations; the shared blackW ones match exactly)
# ---------------------------------------------------------------------------


def _leg_joint_ids(asset) -> list[int]:
    return [i for i, name in enumerate(asset.joint_names) if "wheel" not in name]


def torques(env, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    return torch.sum(torch.square(env.scene[asset_cfg.name].data.qfrc_actuator), dim=1)


def delta_torques(env, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    tau = env.scene[asset_cfg.name].data.qfrc_actuator.detach()
    prev = getattr(env, "_plum_ep_prev_torques", None)
    if prev is None or prev.shape != tau.shape:
        prev = torch.zeros_like(tau)
    prev = prev.clone()
    prev[env.episode_length_buf <= 1] = 0.0
    out = torch.sum(torch.square(tau - prev), dim=1)
    env._plum_ep_prev_torques = tau.detach().clone()
    return out


def hip_pos(env, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = [i for i, name in enumerate(asset.joint_names) if name.endswith("_hip_joint")]
    q = asset.data.joint_pos[:, ids]
    return torch.sum(torch.square(q - asset.data.default_joint_pos[:, ids]), dim=1)


def dof_error(env, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _leg_joint_ids(asset)
    q = asset.data.joint_pos[:, ids]
    return torch.sum(torch.square(q - asset.data.default_joint_pos[:, ids]), dim=1)


def collision(
    env, sensor_name: str = BLACK_ILLEGAL_CONTACT_SENSOR, force_threshold: float = 0.1
) -> torch.Tensor:
    """Count non-wheel bodies pressing on terrain (EP ``_reward_collision``)."""
    sensor = env.scene[sensor_name]
    force = sensor.data.force
    assert force is not None
    num_primary = len(sensor.primary_names)
    slots = force.shape[1] // max(1, num_primary)
    force = force.view(force.shape[0], num_primary, slots, 3)
    per_body = (torch.norm(force, dim=-1) > force_threshold).any(dim=-1)
    return per_body.sum(dim=-1).to(torch.float32)


def feet_stumble(env, sensor_name: str = _FEET_SENSOR) -> torch.Tensor:
    """Wheel hitting a vertical face: ``|F_xy| > 4 |F_z|`` (EP)."""
    force = env.scene[sensor_name].data.force
    assert force is not None
    horizontal = torch.norm(force[..., :2], dim=-1)
    return torch.any(horizontal > 4.0 * torch.abs(force[..., 2]), dim=-1).to(torch.float32)


def feet_edge(
    env,
    asset_cfg: SceneEntityCfg = _ASSET,
    edge_margin: float = _EDGE_MARGIN,
) -> torch.Tensor:
    """Wheels resting on a cube top within ``edge_margin`` of its rim.

    The cube half-width is read per env from the terrain curriculum, so the rim
    stays where the geometry actually is while the cubes shrink.
    """
    found = env.scene[_FEET_SENSOR].data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    from .curriculum import cube_half_per_env

    # [B, 1, 1] to broadcast over (wheels, cubes).
    half = cube_half_per_env(env).view(-1, 1, 1)
    wx, wy, wz = _wheel_geometry(env, asset_cfg)
    # The cube top is the row's barrier height, not the official 0.20 m.
    clear = _plum_top_z(env).view(-1, 1)
    on_cube = (wz - clear).abs() <= _ON_CUBE_TOL
    centres = torch.tensor(SPEC.cube_centres, device=env.device, dtype=wx.dtype)
    dx = wx.unsqueeze(-1) - centres[:, 0]
    dy = wy.unsqueeze(-1) - centres[:, 1]
    inside = (dx.abs() <= half) & (dy.abs() <= half)
    rim = torch.minimum(half - dx.abs(), half - dy.abs())
    at_rim = (inside & (rim < edge_margin)).any(dim=-1)
    return ((found > 0) & on_cube & at_rim).to(torch.float32).sum(dim=-1)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def build_ep_rewards() -> dict[str, RewardTermCfg]:
    """The 14-term EP table with this field's tracking/edge adaptations."""
    return {
        "tracking_goal_vel": RewardTermCfg(
            func=tracking_goal_vel, weight=EP_SCALES["tracking_goal_vel"],
            params={"command_name": "twist", "asset_cfg": _ASSET}),
        "tracking_yaw": RewardTermCfg(
            func=tracking_yaw, weight=EP_SCALES["tracking_yaw"],
            params={"asset_cfg": _ASSET}),
        "lin_vel_z": RewardTermCfg(
            func=vertical_linear_velocity_l2, weight=EP_SCALES["lin_vel_z"],
            params={"asset_cfg": _ASSET}),
        "body_ang_vel": RewardTermCfg(
            func=angular_velocity_xy_l2, weight=EP_SCALES["ang_vel_xy"],
            params={"asset_cfg": _ASSET}),
        "upright": RewardTermCfg(
            func=envs_mdp.flat_orientation_l2, weight=EP_SCALES["orientation"]),
        "dof_acc": RewardTermCfg(
            func=dof_acc_l2, weight=EP_SCALES["dof_acc"], params={"asset_cfg": _ASSET}),
        "collision": RewardTermCfg(
            func=collision, weight=EP_SCALES["collision"]),
        "action_rate": RewardTermCfg(
            func=envs_mdp.action_rate_l2, weight=EP_SCALES["action_rate"]),
        "delta_torques": RewardTermCfg(
            func=delta_torques, weight=EP_SCALES["delta_torques"], params={"asset_cfg": _ASSET}),
        "torques": RewardTermCfg(
            func=torques, weight=EP_SCALES["torques"], params={"asset_cfg": _ASSET}),
        "hip_pos": RewardTermCfg(
            func=hip_pos, weight=EP_SCALES["hip_pos"], params={"asset_cfg": _ASSET}),
        "dof_error": RewardTermCfg(
            func=dof_error, weight=EP_SCALES["dof_error"], params={"asset_cfg": _ASSET}),
        "feet_stumble": RewardTermCfg(
            func=feet_stumble, weight=EP_SCALES["feet_stumble"]),
        "feet_edge": RewardTermCfg(
            func=feet_edge, weight=EP_SCALES["feet_edge"],
            params={"asset_cfg": _ASSET}),
    }


def build_reward_table(clear_weight: float = 500.0) -> dict[str, RewardTermCfg]:
    """EP table + the sparse success bonus.

    The bonus is the one non-EP term: EP's table is a pure locomotion prior with
    no terminal signal, and both previous scratch runs sat at ``part2_success=0``
    for their whole training.  500 * 0.02 = +10 on the success step, matching the
    old repo's ``success_bonus500``.
    """
    from alldog_mjlab.tasks.velocity.profiles.part2.rewards import _part2_clear_reward

    table = build_ep_rewards()
    if clear_weight > 0.0:
        table["part2_clear"] = RewardTermCfg(
            func=_part2_clear_reward, weight=float(clear_weight),
            params={"asset_cfg": _ASSET})
    return table

