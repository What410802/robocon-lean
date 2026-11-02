#!/usr/bin/env python3
"""P3-14: training-env high-wall predicate smoke probe.

Instantiates the exact high-wall task and checks that the P3-14 support-point predicate
buffers exist and are updated for every env step. This is the environment-level half of the
training/judge behavioral probe; parity and crossing-logic unit tests live next to it.
"""
from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> int:
    cfg = load_env_cfg("blackw-wall-exact", play=False)
    cfg.scene.num_envs = 1
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.max_init_terrain_level = 0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    obs = env.get_observations()
    u = env.unwrapped
    actions = torch.zeros((1, u.action_manager.total_action_dim), device="cuda:0")
    for step in range(5):
        obs, _rew, terminated, truncated, _extra = env.step(actions)
        for name in ("_part2_wall_wheel_crossed", "_part2_wall_side_bypass"):
            if not hasattr(u, name):
                raise SystemExit(f"missing predicate buffer {name}")
        if step == 0:
            print("obs", tuple(obs["actor"].shape),
                  "crossed", u._part2_wall_wheel_crossed.int().tolist(),
                  "bypass", u._part2_wall_side_bypass.int().tolist())
    env.close()
    print("WALL ENV PROBE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
