#!/usr/bin/env python3
"""Why does the plum curriculum never promote?

Wraps ``plum_difficulty_curriculum`` and records, per reset, how many envs the
success predicate marks, how the up/down streaks evolve, and whether promotion
fires.  Run a short GPU rollout; nothing is trained.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/diag_curriculum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.plum import curriculum as C  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import plum_teacher_env_cfg  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (  # noqa: E402
    _part2_success, _part2_failure,
)

orig = C.plum_difficulty_curriculum
stats = {"calls": 0, "success_total": 0, "failure_total": 0, "up_total": 0, "down_total": 0}


def wrapped(env, env_ids=None, **kw):
    s = _part2_success(env)
    f = _part2_failure(env)
    out = orig(env, env_ids, **kw)
    stats["calls"] += 1
    stats["success_total"] += int(s.sum())
    stats["failure_total"] += int(f.sum())
    if out:
        stats["up_total"] += int(out.get("up", 0))
        stats["down_total"] += int(out.get("down", 0))
    if stats["calls"] % 20 == 0:
        lv = env.scene.terrain.terrain_levels
        print(f"call {stats['calls']:4d} succ={stats['success_total']:4d} "
              f"fail={stats['failure_total']:5d} up={stats['up_total']:3d} "
              f"down={stats['down_total']:3d} levels={lv.unique().tolist()} "
              f"max={int(lv.max())}", flush=True)
    return out


C.plum_difficulty_curriculum = wrapped
import alldog_mjlab.tasks.velocity.profiles.plum.teacher as T
T.plum_difficulty_curriculum = wrapped

cfg = plum_teacher_env_cfg(stage=0)
cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
# Random actions terminate episodes quickly, which is what triggers resets and
# hence the curriculum term (zero actions never reset in 400 steps).
for _ in range(400):
    env.step(torch.randn(64, 16, device="cuda:0") * 1.5)
print("streak buffer sample:", getattr(env, "_plum_up_streak", None) is not None)
env.close()
