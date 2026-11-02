#!/usr/bin/env python3
"""Behaviour-clone the locked-wheel teacher into the 342 -> 16 PPO actor.

Why: the locked-wheel gait has never been observed to emerge from the S0
stand-only checkpoint (S1 stays in a standing/skating local optimum with mean
action std collapsing to 0.10).  The N-W-wolf teacher already executes a
12-DoF locked-wheel stepping gait in the same observation contract once wrapped
to 57x6/16-action.  We clone it into the PPO actor as an initialisation, then
let PPO fine-tune with the corrected command distribution and support-box
rewards.

The script trains the exact rsl_rl MLP shape (512-256-128 ELU) on collected
(obs, act) pairs, then writes a PPO checkpoint by taking an existing checkpoint
as a template and replacing only ``actor_state_dict`` (and resetting the action
std for exploration).  Critic/optimizer are kept from the template so the
checkpoint can be loaded with ``--agent.resume True``.

Usage:
  .pixi/envs/train/bin/python train/scripts/teacher/bc_warmstart_locked.py \
    --data artifacts/teacher/locked_demos.npz \
    --template logs/rsl_rl/black_velocity/<s0-run>/model_149.pt \
    --out    logs/rsl_rl/black_velocity/<s0-run>/model_bc.pt \
    --epochs 60 --init-std 0.30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from alldog_mjlab.utils.observation_contract import (
    BLACKW_DEPLOY_LAYOUT,
    BLACKW_FRAME_DIM,
)


class ActorMLP(nn.Module):
    """Same shape/activation as the rsl_rl MLPModel actor (342 -> 16)."""

    def __init__(self, in_dim: int = 342, out_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def deploy_to_native_permutation(history: int = 6, frame_dim: int = BLACKW_FRAME_DIM,
                                 term_layout=BLACKW_DEPLOY_LAYOUT) -> np.ndarray:
    """Permutation p with deployment[j] = native[p[j]].

    Training history (MjLab native) is term-major oldest->newest; the deployment
    contract is frame-major newest->oldest.  The first Linear layer must therefore
    have columns permuted; otherwise the actor reads shuffled history and the
    cloned gait collapses (observed as instant illegal-contact falls in S1).
    """
    p = np.zeros(frame_dim * history, dtype=np.int64)
    native_block = 0
    for term_dim, frame_offset in term_layout:
        for frame in range(history):
            native_frame = history - 1 - frame
            for d in range(term_dim):
                p[frame * frame_dim + frame_offset + d] = (
                    native_block + native_frame * term_dim + d)
        native_block += term_dim * history
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-std", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.data)
    obs = torch.from_numpy(data["obs"]).float()
    act = torch.from_numpy(data["act"]).float()
    print(f"data: obs {tuple(obs.shape)} act {tuple(act.shape)}")

    model = ActorMLP(obs.shape[1], act.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(obs)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = obs[idx].to(device)
            yb = act[idx].to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}/{args.epochs}: mse {total / n:.5f}")

    model.eval()
    with torch.no_grad():
        pred = model(obs.to(device)).cpu()
    err = (pred - act).abs()
    print(f"final |err| mean {float(err.mean()):.4f} p95 {float(err.quantile(0.95)):.4f} "
          f"max {float(err.max()):.4f}; per-joint mean "
          f"{np.round(err.mean(dim=0).numpy(), 4)}")

    ckpt = torch.load(args.template, map_location="cpu", weights_only=False)
    state = ckpt["actor_state_dict"]
    sd = dict(model.state_dict())
    # Keys in the template are already mlp.0/.2/.4/.6 (the rsl_rl MLPModel
    # names its Sequential "mlp"), so the state_dict keys map 1:1.
    for k, v in sd.items():
        if k in state and state[k].shape == v.shape:
            state[k] = v.clone()
        else:
            print(f"WARNING: keep template weight for {k}")
    # BC data came from the deployment contract (frame-major newest->oldest),
    # but the PPO actor consumes MjLab native history (term-major
    # oldest->newest).  Permute the first Linear columns instead of writing a
    # policy that only works with reordered observations.
    perm = deploy_to_native_permutation()
    inv = torch.tensor(np.argsort(perm))
    state["mlp.0.weight"] = state["mlp.0.weight"][:, inv].clone()
    std = state["distribution.std_param"]
    state["distribution.std_param"] = torch.full_like(std, args.init_std)
    ckpt["actor_state_dict"] = state
    ckpt["iter"] = 0
    out = Path(args.out)
    torch.save(ckpt, out)
    print(f"saved BC checkpoint: {out} (std={args.init_std})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
