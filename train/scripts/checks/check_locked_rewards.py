#!/usr/bin/env python3
"""Pre-flight audit for the locked-wheel handover task.

Builds every locked curriculum stage and checks the reward/termination contract
that the S1 training runs depend on:

* penalty-shaped functions must have non-positive weights (the historical bug was
  five placement/support terms with *positive* weights, i.e. rewards for splaying
  the feet outside the box);
* reward-shaped functions must have non-negative weights;
* the continuous ungated ``foot_lift`` bootstrap must stay small (<= 0.5);
* the support box must be trunk-centred (no +0.05 m offset) and command-growing
  (covered by the unit test, re-checked here for the stage weight signs);
* S0/S1 must not carry the strict gait-failure termination (it used to truncate
  every episode at ~4.9 s before a gait could form); S2+ must carry it;
* the straight stage command envelope must be positive-only and not inject
  commands from the low-speed mixture outside its range.

Usage:
    PYTHONPATH=train/src .pixi/envs/train/bin/python \
        train/scripts/checks/check_locked_rewards.py
"""

from __future__ import annotations

import sys

from alldog_mjlab.tasks.velocity.blackw.env_cfgs import blackw_vt_locked_env_cfg

REWARD_FUNCS = {
    "track_linear_velocity",
    "track_angular_velocity",
    "feet_air_time_cmd",
    "trot",
    "gait_phase",
    "foot_lift",
    "progress",
    "raibert",
}
PENALTY_FUNCS = {
    "track_linear_velocity_l2",
    "track_linear_velocity_rel",
    "track_yaw_velocity_rel",
    "vertical_linear_velocity_l2",
    "angular_velocity_xy_l2",
    "flat_orientation_l2",
    "base_height_l2_flat",
    "action_rate_l2",
    "action_acc_l2",
    "joint_pos_deviation_l2",
    "rotation_drift_rate_masked",
    "termination_penalty",
    "foot_clearance_phase",
    "foot_impact_vel",
    "foot_slip_point",
    "foot_landing_error",
    "foot_swing_target",
    "support_extent",
    "support_area",
    "support_center",
    "handover_ready",
    "foot_stance_time",
    "foot_air_time_penalty",
    "all_feet_contact",
    "hip_pos",
    "all_joint_pos",
    "stand_still",
    "wheel_lock",
    "wheel_axis_level",
}


def fname(term) -> str:
    return getattr(term.func, "__name__", str(term.func))


def main() -> int:
    problems: list[str] = []
    for stage in range(5):
        cfg = blackw_vt_locked_env_cfg(stage=stage)
        w = {k: float(v.weight) for k, v in cfg.rewards.items()}
        for name, term in cfg.rewards.items():
            f = fname(term)
            weight = float(term.weight)
            if f in REWARD_FUNCS and weight < 0.0:
                problems.append(f"stage {stage}: reward {name} ({f}) has weight {weight}")
            if f in PENALTY_FUNCS and weight > 0.0:
                problems.append(f"stage {stage}: penalty {name} ({f}) has weight {weight}")
        if w.get("foot_lift", 0.0) > 0.5:
            problems.append(f"stage {stage}: continuous foot_lift weight {w['foot_lift']} > 0.5")
        if stage <= 1:
            if "locked_gait_fail" in cfg.terminations:
                problems.append(f"stage {stage}: strict gait termination must be disabled")
        else:
            if "locked_gait_fail" not in cfg.terminations:
                problems.append(f"stage {stage}: gait-failure termination missing")
        # Straight stage: positive-only vx, and the low-speed mixture must not
        # be allowed to produce out-of-range commands (the sampler clips too).
        if stage == 1:
            vx = cfg.commands["twist"].ranges.lin_vel_x
            if vx[0] <= 0.0:
                problems.append(f"stage 1: vx range {vx} is not positive-only")
            low_vx = cfg.commands["twist"].low_vx
            if low_vx > max(abs(v) for v in vx) + 1e-9:
                problems.append(f"stage 1: low_vx {low_vx} exceeds vx range {vx}")

    if problems:
        print("LOCKED REWARD AUDIT: FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("LOCKED REWARD AUDIT: OK (5 stages, sign + termination + envelope checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
