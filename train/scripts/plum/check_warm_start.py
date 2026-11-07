#!/usr/bin/env python3
"""Does the warm-started teacher reproduce the blind policy's behaviour?

The teacher env is the plum field, but its spawn is on the flat runway, so for the
first ~2 s it is a pure locomotion problem identical to what row2p was trained on.
If the warm start is correct, the teacher's base x must advance at roughly the
commanded speed before it reaches the entry platform at x=1.8 m.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/check_warm_start.py
"""

from __future__ import annotations

import argparse
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
    ACTOR_HISTORY, COMMAND_VX, PROPRIO_DIM, plum_teacher_env_cfg,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warm", default=str(
        ROOT / "demo/rl_sar/policy/blackW/row2p_p40w40_v1/policy.pt"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--envs", type=int, default=1)
    args = ap.parse_args()

    cfg = plum_teacher_env_cfg(stage=0, play=True)
    cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs = env.reset()
    o = obs[0] if isinstance(obs, tuple) else obs

    # Build the teacher's actor exactly as rsl_rl would, then warm-start it.
    from rsl_rl.models.mlp_model import MLPModel
    model = MLPModel(
        obs=o, obs_groups={"actor": ("actor", "fine_scan"), "critic": ("critic",)},
        obs_set="actor", output_dim=16, hidden_dims=(512, 256, 128),
        activation="elu", obs_normalization=False,
    ).to("cuda:0")
    vx = 0.5 * sum(COMMAND_VX)
    warm_start_teacher_from_blind(
        model, _load_blind_state(Path(args.warm)), history=ACTOR_HISTORY,
        scan_dim=fine_scan_dim(), fixed_command=(vx, 0.0, 0.0))
    print(f"[check] input={model.obs_dim} (expect {PROPRIO_DIM + fine_scan_dim()}), "
          f"groups={model.obs_groups}")

    asset = env.scene["robot"]
    x0 = asset.data.root_link_pos_w[:, 0].clone()
    for step in range(args.steps):
        with torch.no_grad():
            action = model(o)
        out = env.step(action)
        o = out[0] if isinstance(out, tuple) else out
        if step % 20 == 19:
            x = asset.data.root_link_pos_w[:, 0]
            z = asset.data.root_link_pos_w[:, 2]
            vx_now = asset.data.root_link_lin_vel_w[:, 0]
            print(f"  step {step + 1:3d}  x={float(x.mean()):+.3f} "
                  f"(d={float((x - x0).mean()):+.3f}) z={float(z.mean()):.3f} "
                  f"vx={float(vx_now.mean()):+.2f}")
    adv = float((asset.data.root_link_pos_w[:, 0] - x0).mean())
    print(f"[check] advanced {adv:+.3f} m in {args.steps * 0.02:.1f} s "
          f"(target ~{vx * args.steps * 0.02:+.2f} m)")
    env.close()
    return 0 if adv > 0.2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
