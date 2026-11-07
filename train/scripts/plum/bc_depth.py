#!/usr/bin/env python3
"""Behaviour-cloning boot of the DEPTH ACTOR from teacher actions.

Unlike ``distill.py`` (depth -> predicted 1764-D fine scan -> frozen teacher),
this trains the deployable actor directly:

    action = MLP(proprio(324), CNN(depth 1x48x64))

on ``(depth, proprio, teacher_action)`` shards written by ``collect.py``.  The
teacher MLP is never part of the student; it only supplies labels, so a later
PPO fine-tune is free to leave the teacher's behaviour behind.

Run (train env, GPU; collection must have used --difficulty 1.0):
    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
      train/scripts/plum/bc_depth.py --data artifacts/plum/data/bc_row1 \
      --out artifacts/plum/bc_depth_v1.pt [--init-from <blind policy.pt>]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from alldog_mjlab.algorithms.privileged_warm_start import (
    warm_start_teacher_from_blind,
)
from alldog_mjlab.tasks.velocity.common.rl_cfg import black_actor_distribution_cfg
from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG
from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (
    CNN_LATENT_DIM, DEPTH_CNN_CFG,
)
from alldog_mjlab.tasks.velocity.profiles.plum.runner import _load_blind_state
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (
    ACTOR_HISTORY, COMMAND_VX, PROPRIO_DIM,
)


def load_shards(data_dirs):
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    files = []
    for data_dir in data_dirs:
        found = sorted(glob.glob(str(data_dir / "shard_*.npz")))
        if not found:
            raise SystemExit(f"no shard_*.npz under {data_dir}")
        files.extend(found)
    depth, proprio, act = [], [], []
    for f in files:
        with np.load(f) as d:
            # collect.py stores (steps, envs, ...); flatten both leading dims.
            dep = torch.from_numpy(d["depth"]).reshape(-1, *d["depth"].shape[2:])
            pro = torch.from_numpy(d["proprio"]).reshape(-1, d["proprio"].shape[-1])
            tea = torch.from_numpy(d["teacher_action"]).reshape(
                -1, d["teacher_action"].shape[-1])
        depth.append(dep); proprio.append(pro); act.append(tea)
    return (torch.cat(depth).float(), torch.cat(proprio).float(),
            torch.cat(act).float())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="one or more collect.py output dirs (merged in order)")
    ap.add_argument("--resume", default=None,
                    help="previous depth CNNModel checkpoint; loaded strictly, so "
                         "the CNN encoder carries over (unlike --init-from)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init-from", default=None,
                    help="blind or previous depth checkpoint for the actor's "
                         "proprio columns (CNN columns stay at their init)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dev = args.device

    depth, proprio, act = load_shards(args.data)
    n = depth.shape[0]
    print(f"[bc/depth] frames={n} proprio={proprio.shape[1]} "
          f"depth={tuple(depth.shape[1:])} action={act.shape[1]}", flush=True)
    if proprio.shape[1] != PROPRIO_DIM:
        raise SystemExit(f"proprio dim {proprio.shape[1]} != {PROPRIO_DIM}")
    print(f"[bc/depth] teacher action: mean {act.mean():+.3f} std {act.std():.3f} "
          f"absmax {act.abs().max():.3f}", flush=True)

    # Model: exactly the PPO actor, built the same way the runner builds it.
    from tensordict import TensorDict
    from rsl_rl.models.cnn_model import CNNModel

    sample = TensorDict({"actor": proprio[:2].to(dev),
                         "camera": depth[:2].to(dev)}, batch_size=[2])
    actor = CNNModel(
        sample, {"actor": ("actor", "camera"), "critic": ("critic",)},
        obs_set="actor", output_dim=act.shape[1],
        hidden_dims=tuple(BLACK_CONFIG.policy.actor_hidden_dims),
        activation=BLACK_CONFIG.policy.activation,
        obs_normalization=False,
        distribution_cfg=dict(black_actor_distribution_cfg()),
        cnn_cfg=dict(DEPTH_CNN_CFG),
    ).to(dev)
    print(f"[bc/depth] actor: MLP in {actor.mlp[0].in_features} "
          f"({PROPRIO_DIM} proprio + {CNN_LATENT_DIM} CNN latent)", flush=True)

    if args.resume:
        state = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        state = state.get("actor_state_dict", state)
        # Keep distribution keys: this actor is built WITH the Gaussian, so a
        # BC checkpoint carries distribution.std_param.  (collect.py strips it
        # for its driver actor, which has no distribution; that is not this.)
        actor.load_state_dict(state, strict=True)
        print(f"[bc/depth] resumed actor from {args.resume} (CNN + MLP)", flush=True)
    if args.init_from and not args.resume:
        state = _load_blind_state(Path(args.init_from))
        vx = 0.5 * (COMMAND_VX[0] + COMMAND_VX[1])
        warm_start_teacher_from_blind(
            actor, state, history=ACTOR_HISTORY, scan_dim=CNN_LATENT_DIM,
            fixed_command=(vx, 0.0, 0.0), skip_prefixes=("cnns.",))
        print(f"[bc/depth] proprio columns initialised from {args.init_from}",
              flush=True)

    idx = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    best = float("inf")
    best_state = None
    for ep in range(args.epochs):
        actor.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            td = TensorDict({"actor": proprio[b].to(dev),
                             "camera": depth[b].to(dev)},
                            batch_size=[len(b)])
            pred = actor(td, stochastic_output=False)
            loss = torch.nn.functional.mse_loss(pred, act[b].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        actor.eval()
        with torch.no_grad():
            mse = 0.0
            for i in range(0, len(val), 2048):
                b = val[i:i + 2048]
                td = TensorDict({"actor": proprio[b].to(dev),
                                 "camera": depth[b].to(dev)},
                                batch_size=[len(b)])
                pred = actor(td, stochastic_output=False)
                mse += float(torch.nn.functional.mse_loss(pred, act[b].to(dev))) * len(b)
            mse /= max(1, len(val))
        rmse = mse ** 0.5
        print(f"[bc/depth] ep{ep:02d} train_mse={tot / len(tr):.3e} "
              f"val_action_rmse={rmse:.4f}", flush=True)
        if mse < best:
            best = mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in actor.state_dict().items()}

    torch.save({
        "actor_state_dict": best_state,
        "iter": args.epochs,
        "bc": {"frames": n, "val_action_rmse": best ** 0.5,
               "data": str(args.data), "init_from": args.init_from, "resume": args.resume},
    }, args.out)
    (args.out.parent / (args.out.stem + ".json")).write_text(json.dumps({
        "frames": n, "val_action_mse": best,
        "val_action_rmse": best ** 0.5,
        "teacher_action_std": float(act.std()),
        "data": str(args.data), "init_from": args.init_from, "resume": args.resume,
        "cnn_latent_dim": CNN_LATENT_DIM,
    }, indent=1) + "\n")
    print(f"[bc/depth] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
