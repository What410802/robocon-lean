#!/usr/bin/env python3
"""Evaluate a locked-wheel checkpoint in the training env (deterministic).

Reports, per command: body-frame velocity response, yaw rate, base height,
contacts per frame, support-box violation and footprint radius.  This is the
connector acceptance probe (velocity following + trunk-centred support box),
not a deployment-contract rollout; use the export/PolicyRunner probe after
export for sim2sim.

Usage:
  PYTHONPATH=train/src .pixi/envs/train/bin/python \
    train/scripts/checks/eval_locked_checkpoint.py \
      --checkpoint logs/rsl_rl/black_velocity/<run>/model_399.pt \
      --task blackw-vt-locked-s1 --steps 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from alldog_mjlab.robots.blackw.blackw_constants import BLACKW_FOOT_NAMES
from alldog_mjlab.tasks.velocity.profiles.vt.locked import (
    _contact_legs,
    support_box_half_extents,
)


def load_actor(path: str, device: str) -> torch.nn.Module:
    """Load the MLP body from a checkpoint without hard-coding 16-DoF dims."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("mlp."): v
          for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")}
    layers: list[torch.nn.Module] = []
    i = 0
    while f"{i}.weight" in sd:
        weight = sd[f"{i}.weight"]
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        layers.append(torch.nn.ELU())
        i += 2
    layers.pop()  # no activation after the output layer
    model = torch.nn.Sequential(*layers)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="blackw-vt-locked-s1")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    commands = [
        (0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.30, 0.0, 0.0),
        (0.50, 0.0, 0.0),
        (-0.20, 0.0, 0.0),
        (0.0, 0.20, 0.0),
        (0.0, 0.0, 0.30),
        (0.0, 0.0, -0.30),
        (0.20, 0.10, 0.30),
    ]

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    obs, _ = env.reset()
    cterm = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    foot_ids = [body_names.index(f"{leg}_foot") for leg in BLACKW_FOOT_NAMES]
    actor = load_actor(args.checkpoint, args.device)
    action_dim = int(env.action_manager.total_action_dim)
    dev = torch.device(args.device)

    print(f"{'cmd':>18s} {'vx':>7s} {'vy':>7s} {'wz':>7s} {'z':>6s} "
          f"{'contacts':>8s} {'box_viol':>8s} {'foot_r':>7s}")
    for c in commands:
        env.reset()
        cmd = torch.tensor([c], dtype=torch.float32, device=dev)
        for _ in range(args.warmup):
            cterm.vel_command_b[:] = cmd
            obs, *_ = env.step(torch.zeros(1, action_dim, device=dev))
        vs, ws, zs, contacts, viol, radii = [], [], [], [], [], []
        for _ in range(args.steps):
            cterm.vel_command_b[:] = cmd
            with torch.no_grad():
                a = actor(obs["actor"])
            obs, *_ = env.step(a)
            vs.append(robot.data.root_link_lin_vel_b[0, :2].detach().cpu().numpy())
            ws.append(float(robot.data.root_link_ang_vel_b[0, 2]))
            zs.append(float(robot.data.root_link_pos_w[0, 2]))
            cont = _contact_legs(env)[0]
            contacts.append(float(cont.sum()))
            # Feet XY in trunk-yaw frame.
            pos = robot.data.body_link_pos_w[0, foot_ids, :]
            root = robot.data.root_link_pos_w[0]
            w, x, y, z = robot.data.root_link_quat_w[0].unbind()
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            dx = pos[:, 0] - root[0]
            dy = pos[:, 1] - root[1]
            c_, s_ = torch.cos(-yaw), torch.sin(-yaw)
            fx = c_ * dx - s_ * dy
            fy = s_ * dx + c_ * dy
            hx, hy = support_box_half_extents(cmd.to(dtype=torch.float32), )
            hx, hy = float(hx[0]), float(hy[0])
            selected = cont.to(torch.bool)
            if bool(selected.any()):
                sx = torch.abs(fx[selected]).max().item()
                sy = torch.abs(fy[selected]).max().item()
                viol.append(max(sx - hx, sy - hy, 0.0))
                radii.append(float(torch.sqrt(fx[selected] ** 2 + fy[selected] ** 2).max()))
            else:
                viol.append(0.0)
                radii.append(0.0)
        v = np.mean(vs, axis=0)
        print(f"{str(c):>18s} {v[0]:+7.3f} {v[1]:+7.3f} {np.mean(ws):+7.3f} "
              f"{np.mean(zs):6.3f} {np.mean(contacts):8.2f} "
              f"{np.mean(viol):8.3f} {np.mean(radii):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
