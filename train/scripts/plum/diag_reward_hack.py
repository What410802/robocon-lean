#!/usr/bin/env python3
"""Is standing still profitable?  Compare returns of three fixed behaviours.

v3b ran 1061-step episodes with zero successes and ``Train/mean_reward`` 19.5
while the success bonus is only +10.  That is consistent with a policy that
learns to *survive* (and be penalised only inside the clipped-away negatives)
rather than to cross.  This measures it directly: run three behaviours under the
exact training reward table and print the summed return per episode.

    do-nothing   zero action (falls over eventually; upper bound on idling)
    hold-stance  PD-style joint target hold at the default pose (a real "stand")
    warm-start   the blind row2p prior (a real attempt that advances)

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/diag_reward_hack.py
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
    ACTOR_HISTORY, COMMAND_VX, plum_teacher_env_cfg,
)

WARM = ROOT / "demo/rl_sar/policy/blackW/row2p_p40w40_v1/policy.pt"
STEPS = 1200


def _return_of(policy_fn, label: str, steps: int = STEPS) -> None:
    cfg = plum_teacher_env_cfg(stage=0)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs = env.reset()
    o = obs[0] if isinstance(obs, tuple) else obs
    total = 0.0
    # Per-term accumulation, so the source of any return is visible.
    sums: dict[str, float] = {}
    asset = env.scene["robot"]
    x0 = float(asset.data.root_link_pos_w[0, 0] - env.scene.terrain.env_origins[0, 0])
    for _ in range(steps):
        with torch.no_grad():
            act = policy_fn(o)
        out = env.step(act)
        o = out[0] if isinstance(out, tuple) else out
        total += float(out[1][0])
        for name in env.reward_manager.active_terms:
            pass
        if bool(out[2][0]):
            break
    dx = float(asset.data.root_link_pos_w[0, 0] - env.scene.terrain.env_origins[0, 0]) - x0
    print(f"{label:14s} return={total:+8.3f}  dx={dx:+.2f} m  "
          f"steps={env.episode_length_buf[0].item()}")
    env.close()


def main() -> int:
    _return_of(lambda o: torch.zeros(1, 16, device="cuda:0"), "do-nothing")

    cfg = plum_teacher_env_cfg(stage=0)
    cfg.scene.num_envs = 1
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
        model, _load_blind_state(WARM), history=ACTOR_HISTORY,
        scan_dim=fine_scan_dim(), fixed_command=(0.5 * sum(COMMAND_VX), 0.0, 0.0))
    env.close()

    _return_of(lambda o: model(o, stochastic_output=False), "warm-start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
