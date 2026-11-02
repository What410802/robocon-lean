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

# Nominal wheel-centre (x, y) in the trunk yaw frame for the *default joint
# pose* (compiled MJCF forward kinematics, hip +/-0.05).  The deployed expert
# policies converge to this stance at zero command (measured on the
# deployment-contract runner: rectangle centre ~(0, 0), half extents
# ~0.21 x 0.23), so this is the correct handover target.  It is deliberately
# NOT the locked robot's passive PD equilibrium (which settles ~+0.05 m
# forward); targeting that offset would hand the expert a shifted stance.
LOCKED_NOMINAL_FEET = {
    "FL": (0.2006, 0.2277),
    "FR": (0.2006, -0.2277),
    "RL": (-0.2246, 0.2277),
    "RR": (-0.2246, -0.2277),
}
# Trunk-centred support box at zero command (user requirement: the four
# footholds stay inside a trunk-centred rectangle).  The box grows with the
# commanded stance excursion (see ``support_box_half_extents``) so a normal
# stride is not punished while a stopped handover stays tight.
LOCKED_CENTER = (0.0, 0.0)
LOCKED_HALF_X = 0.2250
LOCKED_HALF_Y = 0.2350
LOCKED_RECT_SLIP = 0.015
LOCKED_RECT_CAP = 0.30
LOCKED_STANCE_PERIOD_S = 0.32
LOCKED_RECT_SIGMA = 0.02
# Extra slack used by the termination check only: the training penalty starts
# at the box itself; the episode is killed once a foot is clearly outside.
LOCKED_RECT_TERM_MARGIN = 0.045


def support_box_half_extents(
    cmd: torch.Tensor,
    period_s: float = LOCKED_STANCE_PERIOD_S,
    hx0: float = LOCKED_HALF_X,
    hy0: float = LOCKED_HALF_Y,
    slip: float = LOCKED_RECT_SLIP,
    cap: float = LOCKED_RECT_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Command-dependent half extents of the trunk-centred contact box.

    During one stance phase a contact point moves roughly ``0.5 * T * |v|``
    relative to the trunk (and ``0.5 * T * |wz| * r`` when turning), so the box
    starts at the tight zero-command rectangle and grows with the commanded
    speed.  This keeps the low-speed/handover box tight (good initial state for
    the obstacle expert) while allowing long strides at higher speed instead of
    fighting them with a static rectangle.

    ``cmd`` is body-frame ``[vx, vy, wz]`` with shape ``[..., 3]``.
    """
    cmd = cmd[..., :3]
    vx = cmd[..., 0].abs()
    vy = cmd[..., 1].abs()
    wz = cmd[..., 2].abs()
    hx = hx0 + slip + 0.5 * period_s * (vx + wz * hy0)
    hy = hy0 + slip + 0.5 * period_s * (vy + wz * hx0)
    return hx.clamp(max=cap), hy.clamp(max=cap)


def _contact_legs(env) -> torch.Tensor:
    """Contact matrix reordered to ``BLACKW_FOOT_NAMES`` order."""
    sensor = env.scene.sensors[CONTACT_SENSOR]
    found = sensor.data.found > 0.0
    names = list(getattr(sensor, "primary_names", ()))
    cols: dict[str, int] = {}
    for i, name in enumerate(names):
        for leg in BLACKW_FOOT_NAMES:
            if name.startswith(leg):
                cols[leg] = i
                break
    if len(cols) != len(BLACKW_FOOT_NAMES):
        # Fallback: sensor pattern is already in foot order.
        return found
    return torch.stack([found[:, cols[leg]] for leg in BLACKW_FOOT_NAMES], dim=1)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _feet_body_xy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Wheel-centre XY in the trunk yaw frame, shape [B, 4]."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.geom_ids
    pos_w = asset.data.geom_pos_w[:, ids, :]
    root = asset.data.root_link_pos_w
    yaw = _yaw_from_quat(asset.data.root_link_quat_w)
    dx = pos_w[..., 0] - root[:, 0:1]
    dy = pos_w[..., 1] - root[:, 1:2]
    c = torch.cos(-yaw).unsqueeze(1)
    s = torch.sin(-yaw).unsqueeze(1)
    x = c * dx - s * dy
    y = s * dx + c * dy
    return torch.stack((x, y), dim=-1)


def _nominal_xy(env) -> torch.Tensor:
    vals = torch.tensor(
        [LOCKED_NOMINAL_FEET[leg] for leg in BLACKW_FOOT_NAMES],
        dtype=torch.float32, device=env.device,
    )  # [4, 2]
    return vals


def command_support_box(
    env, command_name: str = "twist"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env half extents ``(hx, hy)`` of the current command's support box."""
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"Command '{command_name}' not found."
    return support_box_half_extents(cmd)


def _dynamic_foot_target(env, command_name: str = "twist") -> torch.Tensor:
    """Per-foot touchdown target in the trunk yaw frame, shape ``[B, 4, 2]``.

    Standard Raibert term: nominal corner + half a stance phase of commanded
    body motion, including lateral ``vy`` and the yaw rotation field.  The
    target is clamped inside the command-dependent support box so the foothold
    stays inside the trunk-centred rectangle.
    """
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"Command '{command_name}' not found."
    nominal = _nominal_xy(env).unsqueeze(0)  # [1, 4, 2]
    T = LOCKED_STANCE_PERIOD_S
    vx = cmd[:, 0].unsqueeze(-1)  # [B, 1]
    vy = cmd[:, 1].unsqueeze(-1)
    wz = cmd[:, 2].unsqueeze(-1)
    tx = nominal[..., 0] + 0.5 * T * (vx - wz * nominal[..., 1])
    ty = nominal[..., 1] + 0.5 * T * (vy + wz * nominal[..., 0])
    hx, hy = support_box_half_extents(cmd)
    tx = torch.clamp(tx, min=-hx.unsqueeze(-1), max=hx.unsqueeze(-1))
    ty = torch.clamp(ty, min=-hy.unsqueeze(-1), max=hy.unsqueeze(-1))
    return torch.stack((tx, ty), dim=-1)


def _landing_mask(env) -> torch.Tensor:
    sensor = env.scene.sensors[CONTACT_SENSOR]
    contact = _contact_legs(env)
    names = list(getattr(sensor, "primary_names", ()))
    cols = {}
    for i, name in enumerate(names):
        for leg in BLACKW_FOOT_NAMES:
            if name.startswith(leg):
                cols[leg] = i
                break
    if len(cols) == len(BLACKW_FOOT_NAMES):
        contact_time = sensor.data.current_contact_time[:, [cols[leg] for leg in BLACKW_FOOT_NAMES]]
        return contact & (contact_time <= 2.5 * float(env.step_dt))
    return contact & (sensor.data.current_contact_time <= 2.5 * float(env.step_dt))


def foot_landing_error(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    sigma: float = 0.03,
) -> torch.Tensor:
    """Penalise a new touchdown away from the command-dependent foot target.

    The target is the nominal trunk-frame corner plus half a stance phase of
    commanded motion (``_dynamic_foot_target``), clamped inside the
    command-dependent support box.  At zero command this reduces to the tight
    nominal rectangle, which is what the expert handover needs; at higher speed
    it follows the natural stride instead of dragging the feet back.
    """
    xy = _feet_body_xy(env, asset_cfg)
    err = (xy - _dynamic_foot_target(env)) / max(sigma, 1e-6)
    penalty = torch.clamp(torch.sum(torch.square(err), dim=-1), max=5.0)
    return torch.sum(penalty * _landing_mask(env).to(penalty.dtype), dim=1)


def foot_swing_target(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    sigma: float = 0.06,
) -> torch.Tensor:
    """Pull swing feet toward the command-dependent target (not a static corner)."""
    xy = _feet_body_xy(env, asset_cfg)
    err = (xy - _dynamic_foot_target(env)) / max(sigma, 1e-6)
    penalty = torch.clamp(torch.sum(torch.square(err), dim=-1), max=5.0)
    contact = _contact_legs(env)
    return torch.sum(penalty * (~contact).to(penalty.dtype), dim=1)


def support_extent(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    sigma: float = LOCKED_RECT_SIGMA,
) -> torch.Tensor:
    """Penalise contacting feet outside the command-dependent trunk-centred box.

    At zero command the box is the tight nominal rectangle around the trunk
    origin; it grows with ``|vx|, |vy|, |wz|`` (see ``support_box_half_extents``)
    so stepping at speed is not punished.  Applied per contacting foot rather
    than to the AABB, so one splayed foot cannot hide behind the others.
    """
    xy = _feet_body_xy(env, asset_cfg)
    contact = _contact_legs(env)
    hx, hy = command_support_box(env)
    ex = torch.relu(xy[..., 0].abs() - hx.unsqueeze(1)) / max(sigma, 1e-6)
    ey = torch.relu(xy[..., 1].abs() - hy.unsqueeze(1)) / max(sigma, 1e-6)
    penalty = torch.clamp(ex * ex + ey * ey, max=5.0)
    return torch.sum(penalty * contact.to(penalty.dtype), dim=1)


def support_area(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    margin_frac: float = 0.10,
) -> torch.Tensor:
    """Penalise a contact rectangle whose axis-aligned area exceeds the box."""
    xy = _feet_body_xy(env, asset_cfg)
    contact = _contact_legs(env)
    hx, hy = command_support_box(env)
    x = torch.where(contact, xy[..., 0].abs(), torch.zeros_like(xy[..., 0]))
    y = torch.where(contact, xy[..., 1].abs(), torch.zeros_like(xy[..., 1]))
    area = 4.0 * x.max(dim=1).values * y.max(dim=1).values
    nominal = 4.0 * hx * hy
    excess = torch.clamp(area - nominal * (1.0 + margin_frac), min=0.0)
    return torch.clamp((excess / (nominal * margin_frac)) ** 2, max=5.0)


def support_center(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    sigma: float = 0.04,
) -> torch.Tensor:
    """Penalise the contact centroid drifting away from the trunk origin.

    Active only for (near-)zero commands: during walking the centroid naturally
    moves with the stance, but at the handover moment it must be centred so the
    expert starts from a symmetric stance.
    """
    xy = _feet_body_xy(env, asset_cfg)
    contact = _contact_legs(env).to(xy.dtype)
    count = contact.sum(dim=1).clamp_min(1.0)
    mean = (xy * contact.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
    centre = torch.tensor(LOCKED_CENTER, dtype=mean.dtype, device=mean.device)
    penalty = torch.clamp(torch.sum(torch.square((mean - centre) / sigma), dim=1), max=5.0)
    cmd = env.command_manager.get_command("twist")
    still = (torch.linalg.vector_norm(cmd[:, :2], dim=1) < 0.03) & (cmd[:, 2].abs() < 0.03)
    return torch.where(still, penalty, torch.zeros_like(penalty))


def handover_ready(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    corner_sigma: float = 0.06,
    center_sigma: float = 0.05,
) -> torch.Tensor:
    """Zero-command penalty for a stance the obstacle expert cannot inherit well.

    When the command is ~0, all four wheels must be in contact, each close to
    its nominal trunk-frame corner, and the contact centroid close to the trunk
    origin.  This is the explicit "hand the expert a good initial state" term.
    """
    cmd = env.command_manager.get_command("twist")
    still = (torch.linalg.vector_norm(cmd[:, :2], dim=1) < 0.03) & (cmd[:, 2].abs() < 0.03)
    contact = _contact_legs(env)
    xy = _feet_body_xy(env, asset_cfg)
    nominal = _nominal_xy(env).unsqueeze(0)
    corner = torch.clamp(
        torch.sum(torch.square((xy - nominal) / max(corner_sigma, 1e-6)), dim=-1), max=5.0)
    corner_pen = torch.sum(corner, dim=1)
    missing = (~contact).sum(dim=1).to(corner.dtype) * 2.0
    count = contact.sum(dim=1).clamp_min(1).to(xy.dtype)
    mean = (xy * contact.unsqueeze(-1)).sum(dim=1) / count.unsqueeze(-1)
    centre_pen = torch.clamp(
        torch.sum(torch.square(mean / max(center_sigma, 1e-6)), dim=1), max=5.0)
    return torch.where(still, 0.5 * corner_pen + missing + centre_pen,
                       torch.zeros_like(corner_pen))

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


def _leg_field(env, field: str) -> torch.Tensor:
    """Reorder a per-primary contact-sensor field into BLACKW_FOOT_NAMES order."""
    sensor = env.scene.sensors[CONTACT_SENSOR]
    vals = getattr(sensor.data, field)
    cols = _sensor_cols(env)
    if len(cols) != len(BLACKW_FOOT_NAMES):
        return vals
    return torch.stack([vals[:, cols[leg]] for leg in BLACKW_FOOT_NAMES], dim=1)


def feet_air_time_cmd(
    env,
    target_air_s: float = 0.16,
    max_air_s: float = 0.30,
) -> torch.Tensor:
    """Landing-event swing reward, active only while a motion command is given.

    This replaces the old continuous, ungated ``foot_lift`` reward that allowed
    a one-leg statue: the reward is only paid when a foot lands after a swing
    and only under a non-zero command.  ``foot_stance_time`` prevents the other
    three feet from staying planted forever.
    """
    contact = _contact_legs(env)
    air = _leg_field(env, "last_air_time")
    moving = _moving(env).to(air.dtype)
    reward = torch.clamp(air - target_air_s, min=0.0, max=max_air_s) * contact.to(air.dtype)
    return torch.sum(reward, dim=1) * moving


def foot_stance_time(
    env,
    max_stance_s: float = 0.60,
    sigma: float = 0.20,
    per_foot_cap: float = 2.0,
    total_cap: float = 3.0,
) -> torch.Tensor:
    """Penalise wheels that stay planted too long while a command is active.

    Stateless equivalent of a per-foot duty-cycle bound; together with
    ``foot_air_time_penalty`` it forces all four feet to participate in the
    stepping cycle instead of a three-legged stance.  Both the per-foot and the
    total penalty are capped: without a cap a standing policy accumulates an
    unbounded penalty (the first S1 run reached -2.5e3/s and destroyed the value
    function) instead of producing a usable "step now" gradient.
    """
    contact = _contact_legs(env)
    stance_t = _leg_field(env, "current_contact_time")
    moving = _moving(env)
    excess = torch.relu(stance_t - max_stance_s) / max(sigma, 1e-6)
    per_foot = torch.clamp(excess * excess, max=per_foot_cap)
    total = torch.sum(per_foot * contact.to(per_foot.dtype)
                      * moving.unsqueeze(1).to(per_foot.dtype), dim=1)
    return torch.clamp(total, max=total_cap)


def foot_air_time_penalty(
    env,
    max_air_s: float = 0.40,
    sigma: float = 0.15,
    per_foot_cap: float = 2.0,
    total_cap: float = 2.0,
) -> torch.Tensor:
    """Penalise a wheel that dangles in the air while a command is active."""
    contact = _contact_legs(env)
    air_t = _leg_field(env, "current_air_time")
    moving = _moving(env)
    excess = torch.relu(air_t - max_air_s) / max(sigma, 1e-6)
    per_foot = torch.clamp(excess * excess, max=per_foot_cap)
    total = torch.sum(per_foot * (~contact).to(per_foot.dtype)
                      * moving.unsqueeze(1).to(per_foot.dtype), dim=1)
    return torch.clamp(total, max=total_cap)


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
    """Small, command-gated swing bootstrap reward.

    Guarded against the historical "one-leg statue" cheat: only paid while a
    motion command is active, only for feet that are actually off the ground,
    only when at least two other wheels still support the body, and normalised
    by the number of feet so a single lifted foot cannot dominate.  This is
    intentionally a *small* bootstrap term; the real stepping gradient comes
    from ``foot_stance_time`` / ``feet_air_time`` / ``progress`` / ``raibert``.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.geom_ids
    z = asset.data.geom_pos_w[:, ids, 2]
    contact = _contact_legs(env)
    lift = torch.clamp(z - base_height, min=0.0, max=lift_range) / max(lift_range, 1e-9)
    moving = _moving(env)
    support = contact.sum(dim=1) >= 2
    active = moving & support
    gated = lift * (~contact).to(lift.dtype)
    return torch.sum(gated, dim=1) / 4.0 * active.to(lift.dtype)


def all_feet_contact(env) -> torch.Tensor:
    """Penalise four-wheel support while a motion command is active."""
    contact = _contact_legs(env)
    moving = _moving(env)
    return torch.where(moving, contact.all(dim=1).to(contact.dtype),
                       torch.zeros_like(contact[:, 0]))


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
    "command_support_box",
    "feet_air_time",
    "feet_air_time_cmd",
    "foot_air_time_penalty",
    "foot_landing_error",
    "foot_clearance",
    "gait_phase",
    "foot_slip",
    "foot_stance_time",
    "handover_ready",
    "stance_joint",
    "support_area",
    "support_box_half_extents",
    "support_center",
    "support_extent",
    "track_linear_velocity_rel",
    "track_yaw_velocity_rel",
    "wheel_axis_level",
    "foot_lift",
    "wheel_lock",
]


# ---------------------------------------------------------------------------
# Port of the N-W-wolf point-foot black recipe (black_config.py).
# The recipe is the reference for the from-scratch 12-DoF gait: termination
# penalty, strong orientation/height shaping, phase-based clearance/trot,
# Raibert foothold planning, hip/impact/slip terms.  Our locked-wheel variant
# swaps the point feet for the wheel-centre contact point.
# ---------------------------------------------------------------------------

def _sensor_cols(env) -> dict[str, int]:
    sensor = env.scene.sensors[CONTACT_SENSOR]
    names = list(getattr(sensor, "primary_names", ()))
    cols: dict[str, int] = {}
    for i, name in enumerate(names):
        for leg in BLACKW_FOOT_NAMES:
            if name.startswith(leg):
                cols[leg] = i
                break
    return cols


def _contact_force_legs(env) -> torch.Tensor:
    sensor = env.scene.sensors[CONTACT_SENSOR]
    force = sensor.data.force
    cols = _sensor_cols(env)
    if len(cols) != len(BLACKW_FOOT_NAMES):
        return force
    return torch.stack([force[:, cols[leg], :] for leg in BLACKW_FOOT_NAMES], dim=1)


def _phase(env, cycle_s: float = 0.8) -> torch.Tensor:
    return (env.episode_length_buf.to(torch.float32) * float(env.step_dt)
            % cycle_s) / cycle_s


def _stance_mask(env, cycle_s: float = 0.8) -> torch.Tensor:
    phase = _phase(env, cycle_s)
    sin_pos = torch.sin(2.0 * torch.pi * phase)
    return torch.stack((sin_pos >= 0.0, sin_pos < 0.0), dim=1)  # [B,2]


def _moving(env) -> torch.Tensor:
    """Command is non-trivial.  Threshold matches the no-dead-zone target."""
    cmd = env.command_manager.get_command("twist")[:, :3]
    return (torch.linalg.vector_norm(cmd[:, :2], dim=1) > 0.03) | (torch.abs(cmd[:, 2]) > 0.03)


def termination_penalty(env) -> torch.Tensor:
    """Terminal reward (-100 weight): falls must not be a reward escape."""
    return env.reset_terminated.to(torch.float32)


def trot(env, cycle_s: float = 0.8) -> torch.Tensor:
    """Diagonal trot contact-schedule reward without the all-stance loophole.

    The old formula gave an all-four-contact robot 0.75 of the reward, so
    standing was nearly as good as trotting.  Here the reward is
    ``1 - mean(|contact - schedule|)`` (all-stance = 0.5, perfect trot = 1.0)
    and all-stance is explicitly excluded, leaving a usable stepping gradient.
    """
    force_z = torch.abs(_contact_force_legs(env)[..., 2])
    prob = torch.sigmoid((force_z - 5.0) * 0.5)
    c = {leg: prob[:, i] for i, leg in enumerate(BLACKW_FOOT_NAMES)}
    mask = _stance_mask(env, cycle_s).to(torch.float32)
    desired = {
        "FL": mask[:, 0],
        "RR": mask[:, 0],
        "FR": mask[:, 1],
        "RL": mask[:, 1],
    }
    match = torch.stack([1.0 - torch.abs(c[leg] - desired[leg])
                         for leg in BLACKW_FOOT_NAMES], dim=1)
    score = match.mean(dim=1)
    all_stance = (prob > 0.5).all(dim=1).to(score.dtype)
    moving = _moving(env).to(score.dtype)
    return score * (1.0 - all_stance) * moving


def foot_clearance_phase(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    cycle_s: float = 0.8,
    clearance: float = 0.08,
    stance_floor: float = 0.103,
) -> torch.Tensor:
    """Phase-based wheel-centre height: stance on ground, swing ~clearance."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2]
    height = z - stance_floor
    sin_val = torch.sin(2.0 * torch.pi * (_phase(env, cycle_s).unsqueeze(1)
                                          + torch.tensor([0.0, 0.0, 0.0, 0.0],
                                                         device=height.device)))
    # FL/RR swing on one half cycle, FR/RL on the other.
    offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=height.device)
    sin_val = torch.sin(2.0 * torch.pi * ((_phase(env, cycle_s).unsqueeze(1) + offsets) % 1.0))
    stance = sin_val > 0.0
    swing_target = -sin_val * clearance
    stance_err = torch.relu(height - 0.02)
    swing_low = torch.relu(swing_target - height)
    swing_high = torch.relu(height - (swing_target + 0.04))
    err = torch.where(stance, stance_err, swing_low + swing_high)
    moving = _moving(env).to(torch.float32)
    return torch.sum(err, dim=1) * moving


def raibert(
    env,
    asset_cfg: SceneEntityCfg = _WHEEL_CFG,
    sigma: float = 0.06,
) -> torch.Tensor:
    """Swing-foot foothold reward around the command-dependent target.

    Unlike the old version this includes lateral ``vy`` and uses the same
    stance-excursion model as the support box, so the target stays inside the
    box and the target itself scales with the full command vector.
    """
    moving = _moving(env)
    if not bool(moving.any()):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    xy = _feet_body_xy(env, asset_cfg)
    target = _dynamic_foot_target(env)
    swing = (~_contact_legs(env)).to(xy.dtype)
    err = torch.sum(torch.square((xy - target) / sigma), dim=-1)
    reward = torch.exp(-err) * swing
    return torch.sum(reward, dim=1) * moving.to(reward.dtype)

def track_linear_velocity_rel(
    env,
    command_name: str = "twist",
    eps: float = 0.03,
) -> torch.Tensor:
    """Relative planar tracking penalty (0 = perfect, ~1 = standing still).

    The exponential tracker ``exp(-err^2/0.25)`` gives standing at 0.1 m/s
    ~96% of its reward, which is the mechanical cause of the low-speed dead
    zone.  This term normalises the error by the command magnitude, so "not
    moving" costs ~1 at every speed; it is a penalty (negative weight).
    """
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"Command '{command_name}' not found."
    cmd_xy = cmd[:, :2]
    err = torch.sum(torch.square(cmd_xy - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    denom = torch.sum(torch.square(cmd_xy), dim=1) + eps * eps
    return torch.clamp(err / denom, max=4.0)


def track_yaw_velocity_rel(
    env,
    command_name: str = "twist",
    eps: float = 0.10,
) -> torch.Tensor:
    """Relative yaw-rate tracking penalty (0 = perfect, ~1 = not turning)."""
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"Command '{command_name}' not found."
    err = torch.square(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    denom = torch.square(cmd[:, 2]) + eps * eps
    return torch.clamp(err / denom, max=4.0)


def hip_pos(env, asset_cfg: SceneEntityCfg = _LEG_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    names = [asset.joint_names[i] for i in ids]
    hip_ids = [i for i, n in zip(ids, names) if "hip" in n]
    pen = torch.sum(torch.abs(asset.data.joint_pos[:, hip_ids]
                              - asset.data.default_joint_pos[:, hip_ids]), dim=1)
    cmd = env.command_manager.get_command("twist")
    straight = (torch.abs(cmd[:, 1]) < 0.1) & (torch.abs(cmd[:, 2]) < 0.1)
    return pen * torch.where(straight, 1.0, 0.2)


def all_joint_pos(env, asset_cfg: SceneEntityCfg = _LEG_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return torch.sum(torch.square(asset.data.joint_pos[:, ids]
                                 - asset.data.default_joint_pos[:, ids]), dim=1)


def stand_still(env, asset_cfg: SceneEntityCfg = _LEG_CFG) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    cmd = env.command_manager.get_command("twist")
    still = (torch.linalg.vector_norm(cmd[:, :2], dim=1) < 0.03) & (torch.abs(cmd[:, 2]) < 0.03)
    pos_err = torch.sum(torch.abs(asset.data.joint_pos[:, ids]
                                  - asset.data.default_joint_pos[:, ids]), dim=1)
    vel_err = torch.sum(torch.abs(asset.data.joint_vel[:, ids]), dim=1)
    return (pos_err + 0.05 * vel_err) * still.to(pos_err.dtype)


def progress(env) -> torch.Tensor:
    """Normalised body progress along the command (0..1).

    The old term was bounded by ``|cmd|``, so at 0.1 m/s its maximum reward was
    0.1 and could not compete with any gait risk.  Normalising by the command
    magnitude gives a full 0..1 gradient at every speed, which is required to
    remove the low-speed dead zone.
    """
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command("twist")
    cmd_xy = cmd[:, :2]
    norm = torch.linalg.vector_norm(cmd_xy, dim=1)
    moving = norm > 0.03
    direction = cmd_xy / torch.clamp(norm.unsqueeze(1), min=1e-6)
    speed = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
    ratio = torch.clamp(speed / torch.clamp(norm, min=0.02), 0.0, 1.0)
    return ratio * moving.to(ratio.dtype)


def foot_slip_point(env, asset_cfg: SceneEntityCfg = _WHEEL_CFG) -> torch.Tensor:
    """Point-foot recipe: sqrt(horizontal foot speed) while in contact."""
    asset = env.scene[asset_cfg.name]
    vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
    speed = torch.linalg.vector_norm(vel, dim=-1)
    contact = _contact_legs(env).to(speed.dtype)
    return torch.sum(torch.sqrt(speed) * contact, dim=1)


def foot_impact_vel(env, asset_cfg: SceneEntityCfg = _WHEEL_CFG) -> torch.Tensor:
    """Penalise excessive downward velocity on the first touchdown."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, 2]
    impact = torch.clamp(-vz - 0.2, min=0.0)
    landing = _landing_mask(env).to(impact.dtype)
    return torch.sum(torch.square(impact) * landing, dim=1)
