#!/usr/bin/env python3
"""Per-term return of standing still for one full episode.

``diag_reward_hack.py`` shows idling returns ~+11.7 vs +10.0 for success.  This
splits that return by reward term to name the culprit.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/diag_idle_terms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.plum.teacher import plum_teacher_env_cfg  # noqa: E402


def main() -> int:
    cfg = plum_teacher_env_cfg(stage=0)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    steps = int(env.max_episode_length)
    sums: dict[str, float] = {}
    dt = 0.02
    for _ in range(steps):
        out = env.step(torch.zeros(1, 16, device="cuda:0"))
        terms = getattr(env.reward_manager, "_episode_sums", {})
        if _ == 0:
            print("terms:", sorted(k for k in env.reward_manager.active_terms))
        if bool(out[2][0]):
            break
    # measure directly: one step of each active term, with the idle action
    env.reset()
    out = env.step(torch.zeros(1, 16, device="cuda:0"))
    per = {}
    for name in env.reward_manager.active_terms:
        cfg_t = env.reward_manager.get_term_cfg(name)
        val = cfg_t.func(env, **cfg_t.params)
        per[name] = float(val[0]) * cfg_t.weight * dt
    total = sum(per.values())
    for k in sorted(per, key=lambda k: -abs(per[k])):
        print(f"  {k:20s} {per[k]:+8.5f} /step   x{steps} = {per[k]*steps:+8.2f}")
    print(f"  {'TOTAL':20s} {total:+8.5f} /step   x{steps} = {total*steps:+8.2f}")
    print(f"  success bonus (once): +10.00")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
