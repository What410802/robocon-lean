#!/usr/bin/env python3
"""Collect distillation rollouts: (depth, proprio, GT fine scan, teacher action).

Default: the frozen teacher drives the env.  With ``--student`` (a previous
``student.pt``) the student drives the env while the teacher still labels each
state -- closed-loop DAgger coverage of the student's own state distribution
(the old repo's first, offline-only round scored just 5/20 and 12/20).

Run (train env, GPU):
    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
      train/scripts/plum/collect.py --teacher <model_xxx.pt> \
      --out artifacts/plum/data/round0 [--student artifacts/plum/student.pt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import add_depth_camera
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim
from alldog_mjlab.tasks.velocity.profiles.plum.student import (
    build_scan_cnn, build_teacher_mlp, student_action,
)
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import plum_teacher_env_cfg


def _obs(reset_or_step):
    """VecEnv wrappers return either an obs dict or (obs, extras)."""
    return reset_or_step[0] if isinstance(reset_or_step, tuple) else reset_or_step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, help="native PPO checkpoint .pt")
    ap.add_argument("--student", default=None,
                    help="scan-predicting student.pt (OLD design: depth -> predicted "
                         "scan -> frozen teacher). Not used by the depth-actor path.")
    ap.add_argument("--depth-policy", default=None,
                    help="CNNModel checkpoint (.pt with actor_state_dict) that DRIVES "
                         "the env for a DAgger round; the teacher still labels.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--difficulty", type=float, default=None,
                    help="pin every terrain row to this difficulty (1.0 = the "
                         "official 0.40 m gap). Default keeps the play-mode "
                         "spread over all six curriculum rows, which teaches "
                         "behaviour that does not transfer to the official row.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    configure_torch_backends()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = build_teacher_mlp(ckpt["actor_state_dict"], device)

    # Optional student rollout policy (frozen teacher + CNN-predicted scan).
    cnn = head = None
    if args.student:
        s = torch.load(args.student, map_location="cpu", weights_only=False)
        cnn, head = build_scan_cnn(fine_scan_dim(), device)
        cnn.load_state_dict(s["cnn_state"])
        head.load_state_dict(s["scan_head"])
        cnn.eval()
        head.eval()
        print(f"[plum/collect] DAgger round: student drives, teacher labels "
              f"({args.student})", flush=True)

    depth_actor = None
    if args.depth_policy:
        from rsl_rl.models.cnn_model import CNNModel
        from tensordict import TensorDict
        from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (
            DEPTH_CNN_CFG,
        )
        from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG

    cfg = plum_teacher_env_cfg(play=True)
    add_depth_camera(cfg)
    if args.difficulty is not None:
        gen = cfg.scene.terrain.terrain_generator
        gen.difficulty_range = (args.difficulty, args.difficulty)
        print(f"[plum/collect] every row pinned to difficulty "
              f"{args.difficulty:g}", flush=True)
    cfg.scene.num_envs = args.num_envs
    print(f"[plum/collect] spawn pose_range="
          f"{cfg.events['reset_base'].params.get('pose_range')}", flush=True)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None), clip_actions=None)
    obs = _obs(env.reset())
    for group in ("actor", "fine_scan", "camera"):
        if group not in obs:
            raise RuntimeError(f"missing observation group {group!r}: {list(obs)}")

    if args.depth_policy:
        from rsl_rl.models.cnn_model import CNNModel
        from tensordict import TensorDict
        from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (
            DEPTH_CNN_CFG,
        )
        from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
        ck = torch.load(args.depth_policy, map_location="cpu", weights_only=False)
        st = ck.get("actor_state_dict", ck)
        sample = TensorDict({"actor": obs["actor"][:2], "camera": obs["camera"][:2]},
                            batch_size=[2])
        depth_actor = CNNModel(
            sample, {"actor": ("actor", "camera"), "critic": ("critic",)},
            "actor", 16,
            hidden_dims=tuple(BLACK_CONFIG.policy.actor_hidden_dims),
            activation=BLACK_CONFIG.policy.activation,
            obs_normalization=False, cnn_cfg=dict(DEPTH_CNN_CFG),
        ).to(device)
        # A BC checkpoint carries the Gaussian's std; the rollout actor has no
        # distribution, so drop those keys ("drop the Gaussian std").
        st = {k: v for k, v in st.items() if not k.startswith("distribution.")}
        depth_actor.load_state_dict(st, strict=True)
        depth_actor.eval()
        print(f"[plum/collect] DAgger: depth actor drives ({args.depth_policy})",
              flush=True)

    total = 0
    for it in range(args.iters):
        depths, proprios, scans, actions = [], [], [], []
        for _ in range(args.steps):
            with torch.no_grad():
                proprio = obs["actor"]
                scan = obs["fine_scan"]
                depth = obs["camera"]
                # Teacher label always uses the GROUND-TRUTH privileged scan.
                teacher_action = teacher(torch.cat([proprio, scan], dim=-1))
                if depth_actor is not None:
                    td = TensorDict({"actor": proprio, "camera": depth},
                                    batch_size=[proprio.shape[0]])
                    action = depth_actor(td, stochastic_output=False)
                elif cnn is None:
                    action = teacher_action
                else:
                    action, _ = student_action(teacher, cnn, head, proprio, depth)
                action = torch.nan_to_num(action)
            depths.append(depth.detach().to(torch.float16).cpu())
            proprios.append(proprio.detach().to(torch.float16).cpu())
            scans.append(scan.detach().to(torch.float16).cpu())
            actions.append(teacher_action.detach().to(torch.float16).cpu())
            obs = _obs(env.step(action))
        np.savez_compressed(
            args.out / f"shard_{it:04d}.npz",
            depth=torch.stack(depths).numpy(),
            proprio=torch.stack(proprios).numpy(),
            gt_scan=torch.stack(scans).numpy(),
            teacher_action=torch.stack(actions).numpy(),
        )
        outcomes = {}
        tm_done = None
        # On-policy outcomes (DAgger only: with --depth-policy the STUDENT
        # drives, so these counts describe the state distribution being
        # relabelled, which is what decides whether another round is useful).
        try:
            tm = env.unwrapped.termination_manager
            done_mask = env.unwrapped.reset_buf if hasattr(env.unwrapped, 'reset_buf') else None
        except Exception:
            tm = None
        total += args.num_envs * args.steps
        print(f"[plum/collect] iter {it}: total_frames={total}", flush=True)
    env.close()

    meta = {
        "teacher": str(args.teacher),
        "student": args.student,
        "frames": total,
        "num_envs": args.num_envs,
        "iters": args.iters,
        "steps": args.steps,
        "scan_dim": fine_scan_dim(),
        "proprio_dim": int(proprios[0].shape[-1]),
        "seed": args.seed,
        "no_camera_dr": True,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"[plum/collect] DONE {total} frames -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
