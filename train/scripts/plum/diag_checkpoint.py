#!/usr/bin/env python3
"""Roll a trained plum teacher checkpoint and report where episodes end.

Unlike ``diag_illegal.py`` (which rolls the *warm-started* policy), this loads a
training checkpoint and prints, per episode, the terminating terms, the base
position and all four wheel lowest points.  Use it to see whether the policy
fails on the entry step, on a cube gap, or on the exit.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/diag_checkpoint.py \
        --checkpoint <model_1999.pt> --difficulty 1.0 --episodes 8
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
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (  # noqa: E402
    PART2_WHEEL_RADIUS_M, _part2_wheel_lowest_points,
)
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    TASK_ID,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1250)
    args = ap.parse_args()

    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False   # read the pre-reset terminal state
    gen = cfg.scene.terrain.terrain_generator
    gen.difficulty_range = (args.difficulty, args.difficulty)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = MjlabOnPolicyRunner(wrapped, asdict(load_rl_cfg(TASK_ID)), None, "cuda:0")
    runner.load(args.checkpoint, load_cfg={"actor": True})
    policy = runner.alg.get_policy()

    asset = env.scene["robot"]
    tm = env.termination_manager
    ids = [asset.body_names.index(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
    origins = env.scene.terrain.env_origins

    for ep in range(args.episodes):
        obs = env.reset()
        o = obs[0] if isinstance(obs, tuple) else obs
        ended = False
        for step in range(args.max_steps):
            with torch.no_grad():
                action = policy(o, stochastic_output=False)
            out = env.step(action)
            o = out[0] if isinstance(out, tuple) else out
            if bool(out[2][0]):
                ended = True
                fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
                w = _part2_wheel_lowest_points(asset, ids, PART2_WHEEL_RADIUS_M)[0]
                rx = float(asset.data.root_link_pos_w[0, 0] - origins[0, 0])
                rz = float(asset.data.root_link_pos_w[0, 2])
                print(f"ep{ep}: {step + 1:4d} steps root_x={rx:+.2f} root_z={rz:.2f} "
                      f"terms={fired}")
                print(f"       wheels x={[round(float(v - origins[0, 0]), 2) for v in w[:, 0]]} "
                      f"z={[round(float(v), 2) for v in w[:, 2]]}")
                break
        if not ended:
            w = _part2_wheel_lowest_points(asset, ids, PART2_WHEEL_RADIUS_M)[0]
            rx = float(asset.data.root_link_pos_w[0, 0] - origins[0, 0])
            print(f"ep{ep}: TIMEOUT root_x={rx:+.2f} "
                  f"wheels z={[round(float(v), 2) for v in w[:, 2]]}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
