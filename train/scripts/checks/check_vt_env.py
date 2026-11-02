#!/usr/bin/env python3
"""Validate the velocity-tracking primitive task wiring (blackw-vt / blackw-vt-edge).

Modes
-----
config (default): build both env configs and assert the recipe is intact
    (command envelope + mode masses, reward weights, reset/push/friction events,
    terminations, unchanged robot/observation/action contract).
build: additionally construct the MuJoCo/Warp environment and step it a few
    times (GPU; use a small num_envs).

Usage:
    PYTHONPATH=train/src python train/scripts/checks/check_vt_env.py
    PYTHONPATH=train/src python train/scripts/checks/check_vt_env.py --build
"""
from __future__ import annotations

import argparse
import sys

from alldog_mjlab.tasks.velocity.blackw.env_cfgs import (
    blackw_vt_edge_env_cfg,
    blackw_vt_env_cfg,
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_config() -> None:
    from alldog_mjlab.tasks.velocity.profiles.vt import assembly
    from alldog_mjlab.tasks.velocity.profiles.vt.command import (
        VTPrimitiveVelocityCommandCfg,
    )

    for edges in (False, True):
        cfg = blackw_vt_edge_env_cfg() if edges else blackw_vt_env_cfg()
        cmd = cfg.commands["twist"]
        _check(isinstance(cmd, VTPrimitiveVelocityCommandCfg),
               f"{edges=}: command term is {type(cmd).__name__}")
        _check(cmd.ranges.lin_vel_x == (assembly.VX_MIN, assembly.VX_MAX),
               f"{edges=}: vx range {cmd.ranges.lin_vel_x}")
        _check(cmd.ranges.lin_vel_y == (0.0, 0.0),
               f"{edges=}: vy must stay 0 (direction: no lateral investment)")
        _check(cmd.ranges.ang_vel_z == (-assembly.WZ_MAX, assembly.WZ_MAX),
               f"{edges=}: wz range {cmd.ranges.ang_vel_z}")
        _check(cmd.rel_forward_envs == 0.0 and cmd.rel_heading_envs == 0.0,
               f"{edges=}: forward/heading env masks must be off")
        _check(abs(cmd.rel_standing_envs - assembly.STANDING_FRAC) < 1e-9,
               f"{edges=}: standing frac {cmd.rel_standing_envs}")
        _check("command" not in cfg.curriculum, f"{edges=}: command curriculum must be off")

        weights = {k: v.weight for k, v in cfg.rewards.items()}
        _check(weights["track_linear_velocity"] == assembly.TRACK_LIN_W, "track lin w")
        _check(weights["track_angular_velocity"] == assembly.TRACK_ANG_W, "track ang w")
        _check(weights["track_linear_velocity_l2"] == -assembly.TRACK_ERR_L2_W, "track l2")
        _check(weights["upright"] == -assembly.UPRIGHT_W, "upright w")
        _check("pose_dofpos" in weights and "pose_rotdrift" in weights, "pose rewards")
        _check("base_height" not in cfg.rewards, "base_height must be removed (crouch fix)")

        reset = cfg.events["reset_base"].params
        vel = reset["velocity_range"]
        _check(vel["x"] == (-assembly.RESET_VEL_XY, assembly.RESET_VEL_XY), "reset vx")
        _check(vel["yaw"] == (-assembly.RESET_VEL_YAW, assembly.RESET_VEL_YAW), "reset vyaw")
        if not edges:
            pose = reset["pose_range"]
            _check(pose["yaw"] == (-3.141592653589793, 3.141592653589793), "full yaw RSI")
        _check("wheel_friction" in cfg.events, "wheel friction DR missing")
        _check("push_robot" in cfg.events, "push DR missing")
        if edges:
            _check(cfg.scene.terrain.terrain_type == "generator", "edge terrain missing")
            _check("illegal_contact" not in cfg.terminations, "edge must allow contact")
        print(f"[OK] {'edge' if edges else 'flat'} VT cfg: "
              f"{len(cfg.rewards)} rewards, {len(cfg.events)} events, "
              f"wz=+-{assembly.WZ_MAX}, pure_yaw={cmd.pure_yaw_frac}")


def check_build(num_envs: int) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    cfg = blackw_vt_env_cfg()
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.reset()
        actions = torch.zeros((num_envs, 16), device="cuda:0")
        for _ in range(3):
            env.step(actions)
        print(f"[OK] blackw-vt env built and stepped: nq={env.sim.model.nq} "
              f"nu={env.sim.model.nu} num_envs={num_envs}")
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="also build/step the env (GPU)")
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()
    check_config()
    if args.build:
        check_build(args.num_envs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
