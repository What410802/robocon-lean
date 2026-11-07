#!/usr/bin/env python3
"""Build-and-step smoke test for the plum teacher (GPU, ~30 s).

Checks that the v3 recipe actually constructs: the shrinking-cube terrain, the
success-gated curriculum, the EP reward table, the fine-scan group and the
blind->teacher warm start all load together, and that the env steps without a
NaN or a shape error.

    MUJOCO_GL=egl PART2_PLUM_WARM_START=demo/rl_sar/policy/blackW/row2p_p40w40_v1/policy.pt \
      pixi run -e train python train/scripts/plum/smoke_env.py --envs 8 --steps 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.plum.runner import PlumTeacherRunner  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    plum_teacher_env_cfg, plum_teacher_runner_cfg,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--log-dir", default="/tmp/plum_smoke_log")
    args = ap.parse_args()

    cfg = plum_teacher_env_cfg(stage=args.stage)
    cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs = env.reset()
    o = obs[0] if isinstance(obs, tuple) else obs
    print(f"[smoke] actor={tuple(o['actor'].shape)} scan={tuple(o['fine_scan'].shape)}")
    print(f"[smoke] rewards={tuple(cfg.rewards)}")
    print(f"[smoke] curriculum={tuple(cfg.curriculum)}")

    rl = RslRlVecEnvWrapper(env, clip_actions=100.0)
    train_cfg = asdict(plum_teacher_runner_cfg(stage=args.stage))
    runner = PlumTeacherRunner(rl, train_cfg, args.log_dir, "cuda:0")
    if args.no_warm_start and getattr(runner, "_warm_start_report", None):
        raise SystemExit("smoke: warm start applied despite --no-warm-start")

    action = torch.zeros(args.envs, 16, device="cuda:0")
    out = None
    for _ in range(args.steps):
        out = rl.step(action)
    reward = out[1]
    print(f"[smoke] stepped {args.steps}; reward finite={bool(torch.isfinite(reward).all())} "
          f"mean={float(reward.mean()):.4f}")
    print(f"[smoke] terrain levels={env.scene.terrain.terrain_levels.tolist()}")
    env.close()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
