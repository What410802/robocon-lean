"""Locked-wheel (12-DoF point-foot) rewards for the blackW handover task.

The handover prototype locks the four wheel joints at q=0 and walks on the
remaining 12 leg DoF, i.e. a standard point-foot quadruped whose feet are
cylindrical wheels.  The RecoveryLoco principle carries over, but the wheel-leg
geometry needs explicit terms:

* ``wheel_lock``          - keep the wheel hinge at its locked angle;
* ``feet_air_time``        - point-foot swing/contact rhythm;
* ``foot_clearance``       - lift the cylindrical foot while swinging;
* ``foot_slip``            - contact foot must not translate (no lag/slip);
* ``stance_joint``         - penalise leg-joint deviation from the standard
                             stance (outward/fore-aft foot expansion);
* ``wheel_axis_level``     - keep the wheel plane vertical (spin axis horizontal).

All positions/velocities come from ``EntityData``/the existing
``feet_ground_contact`` ContactSensor; no physics or MJCF changes are needed.
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.robots.blackw import BLACKW_FOOT_NAMES, BLACKW_WHEEL_JOINTS

CONTACT_SENSOR = "feet_ground_contact"

_WHEEL_GEOM_NAMES = tuple(f"{leg}_wheel_collision" for leg in BLACKW_FOOT_NAMES)
_FOOT_BODY_NAMES = tuple(f"{leg}_foot" for leg in BLACKW_FOOT_NAMES)
_WHEEL_CFG = SceneEntityCfg("robot", geom_names=_WHEEL_GEOM_NAMES)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=BLACKW_WHEEL_JOINTS)
_LEG_JOINT_NAMES = (".*hip_joint", ".*thigh_joint", ".*calf_joint")
_LEG_CFG = SceneEntityCfg("robot", joint_names=_LEG_JOINT_NAMES)

# Joint deviation weights: hip roll (outward foot expansion) and thigh
# (fore/aft extension) are the directions the user wants discouraged most.
_JOINT_WEIGHTS = {"hip": 2.0, "thigh": 1.5, "calf": 1.0}


def _contact(env) -> torch.Tensor:
    sensor = env.scene.sensors[CONTACT_SENSOR]
    found = sensor.data.found
    return found > 0.0


def wheel_lock(env, asset_cfg: SceneEntityCfg = _WHEEL_JOINT_CFG) -> torch.Tensor:
    """Penalise wheel-joint angle/velocity (the lock must actually hold)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, ids]
    dq = asset.data.joint_vel[:, ids]
    return torch.sum(q * q + 0.02 * dq * dq, dim=1)


def feet_air_time(env, target_air_s: float = 0.20, max_air_s: float = 0.45) -> torch.Tensor:
    """Reward stance feet whose previous swing was long enough (RecoveryLoco)."""
    sensor = env.scene.sensors[CONTACT_SENSOR]
    contact = sensor.data.found > 0.0
    air = sensor.data.last_air_time
    reward = torch.clamp(air - target_air_s, min=0.0, max=max_air_s) * contact
    return torch.sum(reward, dim=1)


def foot_clearance(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    target_height: float = 0.15,
) -> torch.Tensor:
    """Penalise a swinging foot that stays too low; planted feet are ignored."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.geom_ids
    pos_w = asset.data.geom_pos_w[:, ids, :]
    vel_w = asset.data.geom_lin_vel_w[:, ids, :]
    contact = _contact(env)
    height_err = torch.square(pos_w[..., 2] - target_height)
    lateral_speed = torch.linalg.vector_norm(vel_w[..., :2], dim=-1)
    return torch.sum(height_err * lateral_speed * (~contact), dim=1)


def foot_slip(env, asset_cfg: SceneEntityCfg = _WHEEL_CFG) -> torch.Tensor:
    """Penalise horizontal velocity of feet that are in contact."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.geom_ids
    vel_w = asset.data.geom_lin_vel_w[:, ids, :2]
    contact = _contact(env)
    return torch.sum(torch.square(vel_w) * contact.unsqueeze(-1), dim=(1, 2))


def stance_joint(env, asset_cfg: SceneEntityCfg = _LEG_CFG) -> torch.Tensor:
    """Weighted leg-joint deviation from the standard stance.

    The weights emphasise hip (outward) and thigh (fore/aft) deviations, so
    "step wider/longer than the default stance" is penalised more than calf
    flexion.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    all_names = getattr(asset, "joint_names", ())
    if isinstance(ids, slice):
        id_list = list(range(len(all_names)))[ids]
    else:
        id_list = list(ids)
    names = [all_names[i] for i in id_list]
    dev = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    weights = torch.ones((len(ids),), dtype=dev.dtype, device=dev.device)
    for i, name in enumerate(names):
        for key, weight in _JOINT_WEIGHTS.items():
            if key in name:
                weights[i] = weight
                break
    return torch.sum(torch.square(dev) * weights, dim=1) / torch.sum(weights)


def gait_phase(
    env,
    period_s: float = 0.5,
    stance_frac: float = 0.55,
) -> torch.Tensor:
    """Trot contact-pattern reward for the point-foot gait.

    RecoveryLoco's feet_air_time only rewards air time; a from-scratch policy
    can satisfy it by lifting one foot while standing (vt_locked_v3/v4).  This
    term compares actual contact with a diagonal trot schedule (FL+RR vs FR+RL)
    only while a motion command is active, giving a direct gradient toward
    alternation and forward stepping.
    """
    sensor = env.scene.sensors[CONTACT_SENSOR]
    contact = sensor.data.found > 0.0
    names = list(getattr(sensor, "primary_names", ()))
    # Foot column order is the contact-sensor primary order; map each leg.
    cols = {leg: i for i, leg in enumerate(("FL", "FR", "RL", "RR"))}
    for i, name in enumerate(names):
        for leg in cols:
            if name.startswith(leg):
                cols[leg] = i
                break
    phase = (env.episode_length_buf.to(contact.dtype) * float(env.step_dt) / max(period_s, 1e-6)) % 1.0
    stance_a = phase < stance_frac           # FL + RR
    desired = torch.zeros_like(contact)
    desired[:, cols["FL"]] = stance_a
    desired[:, cols["RR"]] = stance_a
    desired[:, cols["FR"]] = ~stance_a
    desired[:, cols["RL"]] = ~stance_a
    match = (contact == desired).to(torch.float32)
    match_mean = match.mean(dim=1)
    moving = env.command_manager.get_command("twist")[:, :3].abs().sum(dim=1) > 0.05
    return torch.where(moving, match_mean, torch.zeros_like(match_mean))


def foot_lift(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    base_height: float = 0.105,
    lift_range: float = 0.08,
) -> torch.Tensor:
    """Reward lifting any wheel off the ground (direct swing-phase signal).

    ``feet_air_time`` only pays after a foot has landed, which is a sparse
    signal for a from-scratch locked-wheel gait (vt_locked_v1/v2 never lifted a
    foot).  This term gives immediate gradient toward swing while being
    self-limiting: once the wheel is above ``lift_range`` it stops paying.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.geom_ids
    z = asset.data.geom_pos_w[:, ids, 2]
    lift = torch.clamp(z - base_height, min=0.0, max=lift_range)
    return torch.sum(lift / max(lift_range, 1e-9), dim=1)


def all_feet_contact(env) -> torch.Tensor:
    """Penalise four-wheel support while a motion command is active."""
    contact = _contact(env)
    moving = env.command_manager.get_command("twist")[:, :3].abs().sum(dim=1) > 0.05
    return torch.where(moving, contact.all(dim=1).to(contact.dtype), torch.zeros_like(contact[:, 0]))


def wheel_axis_level(env, asset_cfg: SceneEntityCfg = _FOOT_BODY_CFG) -> torch.Tensor:
    """Keep the wheel spin axis horizontal => the wheel plane stays vertical."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    quat = asset.data.body_link_quat_w[:, ids, :]  # w, x, y, z
    # local x axis transformed by the body quaternion: z component = axis_z.
    w, x, y, z = quat.unbind(dim=-1)
    axis_z = 2.0 * (x * z + w * y)
    return torch.sum(torch.square(axis_z), dim=1)


__all__ = [
    "all_feet_contact",
    "feet_air_time",
    "foot_clearance",
    "gait_phase",
    "foot_slip",
    "stance_joint",
    "wheel_axis_level",
    "foot_lift",
    "wheel_lock",
]
