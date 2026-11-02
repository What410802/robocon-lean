#!/usr/bin/env python3
"""Collect (obs, action) demonstrations from the locked-wheel teacher policy.

The N-W-wolf teacher walks on blackW with the four wheels locked (0.25 m/s and
above, and lateral/combined commands).  It is a 12-DoF / 45x6 policy wrapped
into our 57x6 / 16-action deployment contract by
``tools/teacher/wrap_nwwolf_locked.py``.  The student used by the locked VT
task has exactly the 342 -> 16 contract, so these pairs are a direct BC
initialisation: the student starts with a working stepping primitive instead of
having to discover it from a standing-only S0 checkpoint (the failure mode of
the first S1 runs).

Data are collected on the deployment-contract CPU MuJoCo runner (same obs
terms/scales and 50 Hz loop as deployment), then saved as npz.

Usage:
  MUJOCO_GL=egl PYTHONPATH=eval .pixi/envs/eval/bin/python \
    train/scripts/teacher/collect_locked_teacher_demos.py \
      --teacher demo/rl_sar/policy/blackW/nwwolf_locked16 \
      --out artifacts/teacher/locked_demos.npz --episodes 60
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT / "eval"))
from robocon_eval.policy_runner import PolicyRunner  # noqa: E402

HISTORY = 6
OBS_DIM = 57
ACT_DIM = 16


def sample_command(rng: np.random.Generator) -> np.ndarray:
    """Commands the teacher is known to execute (its low-speed/pure-yaw dead
    zone is deliberately not sampled; PPO/S2 extends the low-speed range)."""
    if rng.random() < 0.12:  # stand, for the stop/handover regime
        return np.zeros(3, dtype=np.float64)
    vx = float(rng.uniform(0.25, 0.75))
    vy = float(rng.choice([0.0, 0.0, 0.15, -0.15, 0.30, -0.30]))
    wz = float(rng.choice([0.0, 0.0, 0.0, 0.20, -0.20, 0.40, -0.40]))
    return np.array([vx, vy, wz], dtype=np.float64)


def episode(runner: PolicyRunner, rng: np.random.Generator, steps: int,
            obs_out: list, act_out: list, cmd_out: list) -> int:
    cmd = sample_command(rng)
    runner.reset()
    runner.settle(1.0)
    hist: list[np.ndarray] = []
    kept = 0
    for _ in range(steps):
        obs = runner._obs(cmd)
        if not hist:
            hist = [obs.copy() for _ in range(HISTORY)]
        else:
            hist.append(obs.copy())
            hist = hist[-HISTORY:]
        policy_input = np.concatenate(list(reversed(hist))).astype(np.float32)
        with torch.no_grad():
            act = runner.policy(
                torch.from_numpy(policy_input).unsqueeze(0)).squeeze(0).numpy()
        act = np.asarray(act, dtype=np.float64)
        runner.apply_action(act, cmd)
        runner.last_action = act.copy()
        fr = runner.frame(cmd)
        if fr["tilt_deg"] > 35.0 or float(fr["base"][2]) < 0.25:
            break
        obs_out.append(policy_input)
        act_out.append(act.astype(np.float32))
        cmd_out.append(cmd.astype(np.float32))
        kept += 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    help="policy dir under demo/rl_sar/policy/blackW/ (or absolute)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    teacher = Path(args.teacher)
    if not teacher.is_absolute():
        teacher = ROOT / "demo/rl_sar/policy/blackW" / teacher.name
    rng = np.random.default_rng(args.seed)
    runner = PolicyRunner(
        "blackW", 1, teacher,
        scene=ROOT / "demo/robots/blackW/smoke_flat.xml",
        settle_s=1.0, seed=args.seed,
    )
    obs_out: list[np.ndarray] = []
    act_out: list[np.ndarray] = []
    cmd_out: list[np.ndarray] = []
    for ep in range(args.episodes):
        kept = episode(runner, rng, args.steps, obs_out, act_out, cmd_out)
        if (ep + 1) % 10 == 0:
            print(f"episode {ep + 1}/{args.episodes}: kept {kept}, total {len(obs_out)}")
    obs = np.stack(obs_out).astype(np.float32)
    act = np.stack(act_out).astype(np.float32)
    cmd = np.stack(cmd_out).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs, act=act, cmd=cmd)
    print(f"saved {out}: obs {obs.shape}, act {act.shape}, cmd {cmd.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
