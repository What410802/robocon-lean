#!/usr/bin/env python3
"""Measure each candidate warm-start policy on the plum entry staircase.

All deployed blackW policies share the 342-D blind layout, so any of them can be
copied into the teacher's proprio columns.  They differ in what they were trained
for and in their command range, so pick the prior by measurement, not by name.

Reports, per policy: how far the base travels on the easy (difficulty 0) field in
6 s, and whether it ends with a legal-contact termination.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/sweep_warm_start.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

from alldog_mjlab.algorithms.privileged_warm_start import (  # noqa: E402
    warm_start_teacher_from_blind,
)
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.runner import _load_blind_state  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    ACTOR_HISTORY, COMMAND_VX, TASK_ID, plum_teacher_env_cfg,
)
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

POLICY_DIR = ROOT / "demo/rl_sar/policy/blackW"
CANDIDATES = (
    "row2p_p40w40_v1",
    "st30_ppo_st30_s100_600",
    "s1d_trench_reward_v75_m100",
    "vt_band_v1",
    "vt_edge_v1",
    "vt_pose_v1",
    "vt_track_v2",
)


def run_one(name: str, steps: int = 300) -> None:
    path = POLICY_DIR / name / "policy.pt"
    if not path.is_file():
        print(f"{name:28s} MISSING")
        return
    # Training config, difficulty pinned: play=True spawns on a RANDOM terrain
    # patch, so a "dx=0 at spawn height" result there means the robot was dropped
    # onto a staircase tread, not that the policy failed (measured: every policy
    # except one showed dx~0 and an immediate termination).
    cfg = load_env_cfg(TASK_ID, play=False)
    cfg.scene.num_envs = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs = env.reset()
    o = obs[0] if isinstance(obs, tuple) else obs
    from rsl_rl.models.mlp_model import MLPModel
    model = MLPModel(
        obs=o, obs_groups={"actor": ("actor", "fine_scan"), "critic": ("critic",)},
        obs_set="actor", output_dim=16, hidden_dims=(512, 256, 128),
        activation="elu", obs_normalization=False,
    ).to("cuda:0")
    warm_start_teacher_from_blind(
        model, _load_blind_state(path), history=ACTOR_HISTORY,
        scan_dim=fine_scan_dim(), fixed_command=(0.5 * sum(COMMAND_VX), 0.0, 0.0))
    asset = env.scene["robot"]
    x0 = float(asset.data.root_link_pos_w[0, 0] - env.scene.terrain.env_origins[0, 0])
    term = ""
    for _ in range(steps):
        with torch.no_grad():
            act = model(o, stochastic_output=False)
        out = env.step(act)
        o = out[0] if isinstance(out, tuple) else out
        if bool(out[2][0]):
            fired = [n for n in env.termination_manager.active_terms
                     if bool(env.termination_manager.get_term(n)[0])]
            term = ",".join(fired)
            break
    x = float(asset.data.root_link_pos_w[0, 0] - env.scene.terrain.env_origins[0, 0])
    z = float(asset.data.root_link_pos_w[0, 2])
    print(f"{name:28s} dx={x - x0:+.2f} m  root_z={z:.2f}  end={term or 'still walking'}")
    env.close()


def main() -> int:
    for name in CANDIDATES:
        run_one(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
