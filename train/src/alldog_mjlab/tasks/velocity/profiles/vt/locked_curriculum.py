"""Staged curriculum for the locked-wheel (12-DoF point-foot) task.

A from-scratch locked-wheel gait is prone to two local optima: skating on the
locked wheels (no swing) and stepping in place (swing without translation).
The curriculum isolates those skills before the full handover envelope:

    S0 stand/lock  ->  S1 low-speed straight  ->  S2 pure in-place yaw
    -> S3 combined  ->  S4 full handover envelope (+ full DR)

Each stage is a separate task id (``blackw-vt-locked-s0`` ... ``s4``); the
final ``blackw-vt-locked`` task is S4.  The robot, action term, observations and
deployment contract are identical across stages; only command ranges, reward
weights, reset/DR strength and (optionally) push perturbations are scheduled.

Early stages also shrink the random-state reset ranges: the full RecoveryLoco
reset (tilt 0.1 rad, 0.3 m/s) is a recovery test, not an entry condition for a
robot that cannot stand yet.  S4 restores the exact VT reset contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from alldog_mjlab.tasks.velocity.common.velocity_env import BLACK_COMMAND_NAME
from alldog_mjlab.tasks.velocity.profiles.vt.command import VTPrimitiveVelocityCommandCfg

STAGE_NAMES = ("stand", "straight", "pure_yaw", "combined", "full")


@dataclass(frozen=True)
class LockedStage:
    name: str
    vx: tuple[float, float]
    vy: tuple[float, float]
    wz: tuple[float, float]
    pure_yaw_frac: float
    pure_yaw_vy_frac: float
    pure_vx_frac: float
    pure_yaw_wz: tuple[float, float]
    standing_frac: float
    low_frac: float
    # reward weights
    tracking_lin: float
    tracking_ang: float
    foot_landing: float
    foot_swing: float
    support_extent: float
    support_area: float
    support_center: float
    foot_lift: float
    gait_phase: float
    feet_air_time: float
    foot_clearance: float
    foot_slip: float
    stance_joint: float
    wheel_axis: float
    push: bool
    reset_scale: float


STAGES: tuple[LockedStage, ...] = (
    LockedStage(
        name="stand", vx=(-0.02, 0.08), vy=(-0.03, 0.03), wz=(0.0, 0.0),
        pure_yaw_frac=0.0, pure_yaw_vy_frac=0.0, pure_vx_frac=0.10, pure_yaw_wz=(0.0, 0.0),
        standing_frac=0.65, low_frac=0.10,
        tracking_lin=1.5, tracking_ang=1.0,
        foot_landing=0.0, foot_swing=0.0, support_extent=0.0,
        support_area=0.0, support_center=0.0,
        foot_lift=0.0, gait_phase=0.0, feet_air_time=0.0,
        foot_clearance=0.0, foot_slip=-2.0, stance_joint=-2.0,
        wheel_axis=-1.0, push=False, reset_scale=0.25,
    ),
    LockedStage(
        name="straight", vx=(0.02, 0.25), vy=(-0.05, 0.05), wz=(-0.05, 0.05),
        pure_yaw_frac=0.0, pure_yaw_vy_frac=0.0, pure_vx_frac=0.70, pure_yaw_wz=(0.0, 0.0),
        standing_frac=0.05, low_frac=0.30,
        tracking_lin=2.0, tracking_ang=1.0,
        # Placement/support shaping is present from S1 but the values below are
        # *magnitudes*; apply_locked_stage installs them with a negative sign
        # (the functions return penalties).
        foot_landing=0.30, foot_swing=0.20, support_extent=1.00,
        support_area=0.20, support_center=0.50,
        foot_lift=0.20, gait_phase=0.50, feet_air_time=0.50,
        foot_clearance=-3.0, foot_slip=-2.0, stance_joint=-0.3,
        wheel_axis=-1.0, push=False, reset_scale=0.00,
    ),
    LockedStage(
        name="pure_yaw", vx=(-0.03, 0.10), vy=(-0.10, 0.10), wz=(-0.35, 0.35),
        pure_yaw_frac=0.50, pure_yaw_vy_frac=0.30, pure_vx_frac=0.05, pure_yaw_wz=(0.05, 0.35),
        standing_frac=0.05, low_frac=0.40,
        tracking_lin=1.0, tracking_ang=3.0,
        foot_landing=0.50, foot_swing=0.20, support_extent=1.50,
        support_area=0.30, support_center=0.80,
        foot_lift=0.10, gait_phase=0.60, feet_air_time=0.50,
        foot_clearance=-4.0, foot_slip=-8.0, stance_joint=-0.3,
        wheel_axis=-1.0, push=False, reset_scale=0.50,
    ),
    LockedStage(
        name="combined", vx=(-0.20, 0.40), vy=(-0.20, 0.20), wz=(-0.45, 0.45),
        pure_yaw_frac=0.30, pure_yaw_vy_frac=0.20, pure_vx_frac=0.30, pure_yaw_wz=(0.05, 0.45),
        standing_frac=0.05, low_frac=0.40,
        tracking_lin=3.0, tracking_ang=3.0,
        foot_landing=0.50, foot_swing=0.20, support_extent=1.50,
        support_area=0.30, support_center=0.80,
        foot_lift=0.05, gait_phase=0.50, feet_air_time=0.50,
        foot_clearance=-6.0, foot_slip=-8.0, stance_joint=-0.3,
        wheel_axis=-1.0, push=True, reset_scale=0.80,
    ),
    LockedStage(
        name="full", vx=(-0.35, 0.55), vy=(-0.25, 0.25), wz=(-0.65, 0.65),
        pure_yaw_frac=0.25, pure_yaw_vy_frac=0.20, pure_vx_frac=0.30, pure_yaw_wz=(0.05, 0.65),
        standing_frac=0.05, low_frac=0.40,
        tracking_lin=3.0, tracking_ang=3.0,
        foot_landing=0.50, foot_swing=0.20, support_extent=1.50,
        support_area=0.30, support_center=0.80,
        foot_lift=0.05, gait_phase=0.50, feet_air_time=0.50,
        foot_clearance=-8.0, foot_slip=-8.0, stance_joint=-0.3,
        wheel_axis=-1.0, push=True, reset_scale=1.00,
    ),
)


def stage_index(stage: int | str | None) -> int | None:
    if stage is None:
        return None
    if isinstance(stage, str):
        if stage not in STAGE_NAMES:
            raise ValueError(f"unknown locked stage {stage!r}; expected {STAGE_NAMES}")
        return STAGE_NAMES.index(stage)
    idx = int(stage)
    if not 0 <= idx < len(STAGES):
        raise ValueError(f"locked stage index {idx} out of range")
    return idx


def _scale_reset_contract(cfg, scale: float) -> None:
    """Shrink random-state reset ranges for early curriculum stages."""
    if scale >= 1.0:
        return  # S4 keeps the exact RecoveryLoco VT reset contract
    if "reset_base" in cfg.events:
        ev = cfg.events["reset_base"]
        params = dict(ev.params)
        pose = dict(params.get("pose_range") or {})
        pose.update({
            "x": (-0.20 * scale, 0.20 * scale),
            "y": (-0.20 * scale, 0.20 * scale),
            "roll": (-0.10 * scale, 0.10 * scale),
            "pitch": (-0.10 * scale, 0.10 * scale),
            "yaw": (-math.pi, math.pi),
        })
        vel = dict(params.get("velocity_range") or {})
        vel.update({
            "x": (-0.30 * scale, 0.30 * scale),
            "y": (-0.30 * scale, 0.30 * scale),
            "z": (0.0, 0.0),
            "roll": (-0.15 * scale, 0.15 * scale),
            "pitch": (-0.15 * scale, 0.15 * scale),
            "yaw": (-0.50 * scale, 0.50 * scale),
        })
        params["pose_range"] = pose
        params["velocity_range"] = vel
        cfg.events["reset_base"] = replace(ev, params=params)
    if "reset_robot_joints" in cfg.events:
        ev = cfg.events["reset_robot_joints"]
        params = dict(ev.params)
        params["position_range"] = (-0.08 * scale, 0.08 * scale)
        params["velocity_range"] = (-1.0 * scale, 1.0 * scale)
        cfg.events["reset_robot_joints"] = replace(ev, params=params)


# Stage-controlled sampler parameters that are not part of LockedStage.
# (low_vx/low_wz were previously fixed at 0.12/0.20 for every stage, which
# injected commands outside the stage envelope; low_wz=0 on the stand stage is
# required because its yaw range is exactly zero.)
_STAGE_SAMPLER = {
    0: dict(low_vx=0.00, low_wz=0.00, resample_s=5.0),
    1: dict(low_vx=0.10, low_wz=0.05, resample_s=4.0),
    2: dict(low_vx=0.08, low_wz=0.10, resample_s=4.0),
    3: dict(low_vx=0.10, low_wz=0.15, resample_s=4.0),
    4: dict(low_vx=0.10, low_wz=0.20, resample_s=4.0),
}

# Per-stage gait/reward schedule.  Positive values are rewards, negative values
# are penalties; every weight is applied explicitly (the LockedStage placement
# fields are magnitudes and are negated below because those functions return
# penalties).
_STAGE_GAIT = {
    0: dict(air_cmd=0.0, trot=0.0, clear=0.0, impact=0.0, progress=0.0,
            raibert=0.0, gait_phase=0.0, stance_time=0.0, air_pen=0.0,
            handover=0.0, rel_lin=0.0, rel_yaw=0.0, all_feet=0.0),
    1: dict(air_cmd=0.5, trot=0.5, clear=-3.0, impact=-3.0, progress=1.5,
            raibert=0.5, gait_phase=0.5, stance_time=-1.0, air_pen=-1.0,
            handover=-0.5, rel_lin=-1.5, rel_yaw=-0.5, all_feet=-1.0),
    2: dict(air_cmd=0.5, trot=0.7, clear=-4.0, impact=-4.0, progress=1.0,
            raibert=1.0, gait_phase=0.6, stance_time=-1.2, air_pen=-1.2,
            handover=-0.8, rel_lin=-1.5, rel_yaw=-1.0, all_feet=-0.8),
    3: dict(air_cmd=0.5, trot=0.8, clear=-6.0, impact=-6.0, progress=1.0,
            raibert=1.0, gait_phase=0.5, stance_time=-1.2, air_pen=-1.2,
            handover=-0.8, rel_lin=-1.8, rel_yaw=-1.2, all_feet=-0.8),
    4: dict(air_cmd=0.5, trot=0.8, clear=-8.0, impact=-8.0, progress=1.0,
            raibert=1.0, gait_phase=0.5, stance_time=-1.2, air_pen=-1.2,
            handover=-0.8, rel_lin=-1.8, rel_yaw=-1.2, all_feet=-0.8),
}


def apply_locked_stage(cfg, stage: int | str) -> None:
    """Apply command envelope + reward schedule for one curriculum stage."""
    idx = stage_index(stage)
    assert idx is not None
    s = STAGES[idx]
    samp = _STAGE_SAMPLER[idx]
    gait = _STAGE_GAIT[idx]

    cmd = cfg.commands[BLACK_COMMAND_NAME]
    assert isinstance(cmd, VTPrimitiveVelocityCommandCfg)
    new_ranges = replace(cmd.ranges, lin_vel_x=s.vx, lin_vel_y=s.vy,
                         ang_vel_z=s.wz)
    cfg.commands[BLACK_COMMAND_NAME] = replace(
        cmd,
        ranges=new_ranges,
        pure_yaw_frac=s.pure_yaw_frac,
        pure_yaw_vy_frac=s.pure_yaw_vy_frac,
        pure_vx_frac=s.pure_vx_frac,
        pure_yaw_wz_min=s.pure_yaw_wz[0],
        pure_yaw_wz_max=s.pure_yaw_wz[1],
        rel_standing_envs=s.standing_frac,
        low_frac=s.low_frac,
        low_vx=samp["low_vx"],
        low_wz=samp["low_wz"],
        resampling_time_range=(samp["resample_s"], samp["resample_s"]),
    )

    r = cfg.rewards
    r["track_linear_velocity"] = replace(r["track_linear_velocity"], weight=s.tracking_lin)
    r["track_angular_velocity"] = replace(r["track_angular_velocity"], weight=s.tracking_ang)
    # Relative tracking penalties are the low-speed dead-zone fix: standing
    # still costs ~1.0 at every command magnitude.
    r["track_linear_velocity_rel"] = replace(
        r["track_linear_velocity_rel"], weight=gait["rel_lin"])
    r["track_yaw_velocity_rel"] = replace(
        r["track_yaw_velocity_rel"], weight=gait["rel_yaw"])
    r["lin_vel_z"] = replace(r["lin_vel_z"], weight=-1.5)
    r["body_ang_vel"] = replace(r["body_ang_vel"], weight=-0.05)
    r["upright"] = replace(r["upright"], weight=-3.0)
    r["base_height"] = replace(r["base_height"], weight=-3.0)
    r["action_rate_l2"] = replace(r["action_rate_l2"], weight=-0.3)
    r["smoothness"] = replace(r["smoothness"], weight=-0.01)
    r["termination"] = replace(r["termination"], weight=-100.0)

    r["feet_air_time"] = replace(r["feet_air_time"], weight=gait["air_cmd"])
    r["trot"] = replace(r["trot"], weight=gait["trot"])
    r["foot_clearance"] = replace(r["foot_clearance"], weight=gait["clear"])
    r["foot_impact_vel"] = replace(r["foot_impact_vel"], weight=gait["impact"])
    r["progress"] = replace(r["progress"], weight=gait["progress"])
    r["raibert"] = replace(r["raibert"], weight=gait["raibert"])
    r["gait_phase"] = replace(r["gait_phase"], weight=gait["gait_phase"])
    r["foot_stance_time"] = replace(r["foot_stance_time"], weight=gait["stance_time"])
    r["foot_air_time_penalty"] = replace(
        r["foot_air_time_penalty"], weight=gait["air_pen"])
    r["handover_ready"] = replace(r["handover_ready"], weight=gait["handover"])
    r["hip_pos"] = replace(r["hip_pos"], weight=-0.5)
    r["all_joint_pos"] = replace(r["all_joint_pos"], weight=-0.001)
    r["stand_still"] = replace(r["stand_still"], weight=-1.0)
    r["foot_slip"] = replace(r["foot_slip"], weight=s.foot_slip)
    r["foot_lift"] = replace(r["foot_lift"], weight=s.foot_lift)
    r["all_feet_contact"] = replace(r["all_feet_contact"], weight=gait["all_feet"])
    r["wheel_lock"] = replace(r["wheel_lock"], weight=-5.0)
    r["wheel_axis_level"] = replace(r["wheel_axis_level"], weight=-1.0)
    # Placement/support penalties: LockedStage fields are magnitudes; the
    # underlying functions return >=0 penalties, so the sign MUST be negative.
    r["foot_landing_error"] = replace(r["foot_landing_error"], weight=-s.foot_landing)
    r["foot_swing_target"] = replace(r["foot_swing_target"], weight=-s.foot_swing)
    r["support_extent"] = replace(r["support_extent"], weight=-s.support_extent)
    r["support_area"] = replace(r["support_area"], weight=-s.support_area)
    r["support_center"] = replace(r["support_center"], weight=-s.support_center)
    if not s.push:
        cfg.events.pop("push_robot", None)
    _scale_reset_contract(cfg, s.reset_scale)

    # Task-priority terminations.  S0/S1 must be allowed to discover a gait:
    # the earlier locked_s1_vy run had 98.6% of episodes terminated by this
    # gate at a 4.9 s average length, which prevented any stride accumulation.
    # S2 gets a loose gate; S3/S4 restore the strict scheduler.
    if "locked_gait_fail" in cfg.terminations:
        from mjlab.managers import TerminationTermCfg  # local to keep cfg stage self-contained
        term = cfg.terminations["locked_gait_fail"]
        if idx <= 1:
            cfg.terminations.pop("locked_gait_fail", None)
        else:
            loose = idx == 2
            params = dict(term.params)
            params.update({
                "lin_abs": 0.20 if loose else 0.10,
                "lin_frac": 0.80 if loose else 0.50,
                "ang_abs": 0.20 if loose else 0.10,
                "ang_frac": 0.80 if loose else 0.50,
                "track_timeout": 1.50 if loose else 0.80,
                "static_timeout": 1.50 if loose else 0.80,
                "air_timeout": 1.50 if loose else 0.80,
                "stance_timeout": 2.00 if loose else 1.20,
                "rect_timeout": 1.00 if loose else 0.50,
                "rect_x_margin": 0.08 if loose else 0.045,
                "rect_y_margin": 0.08 if loose else 0.045,
            })
            cfg.terminations["locked_gait_fail"] = TerminationTermCfg(
                func=term.func, params=params)


__all__ = ["STAGES", "STAGE_NAMES", "LockedStage", "apply_locked_stage", "stage_index"]
