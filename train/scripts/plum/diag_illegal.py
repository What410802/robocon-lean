#!/usr/bin/env python3
"""Why do episodes end after ~30 steps?  Roll the warm-started policy and report.

The training logs show episodes ending at ~30 steps (0.6 s) with
``illegal_contact`` firing ~33 times per 1024-env batch, while a play-mode probe
of the same policy advances 2.3 m cleanly.  This runs the SAME policy in the
TRAINING config (noise + spawn randomisation + all terminations) and prints the
terminating term and the base position for each episode.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/diag_illegal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from rsl_rl.models.mlp_model import MLPModel  # noqa: E402

from alldog_mjlab.algorithms.privileged_warm_start import (  # noqa: E402
    warm_start_teacher_from_blind,
)
from alldog_mjlab.tasks.velocity.common.velocity_env import (  # noqa: E402
    BLACK_ILLEGAL_CONTACT_SENSOR,
)
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.runner import _load_blind_state  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    ACTOR_HISTORY, COMMAND_VX, plum_teacher_env_cfg,
)

WARM = ROOT / "demo/rl_sar/policy/blackW/row2p_p40w40_v1/policy.pt"


def main() -> int:
    cfg = plum_teacher_env_cfg(stage=0)
    cfg.scene.num_envs = 1
    # Read the pre-reset terminal state: with auto_reset the env has already
    # respawned by the time ``env.step`` returns, which silently reports
    # x=0/z=spawn/force=0 (this burned one diagnostic round).
    cfg.auto_reset = False
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs = env.reset()
    o = obs[0] if isinstance(obs, tuple) else obs

    model = MLPModel(
        obs=o, obs_groups={"actor": ("actor", "fine_scan"), "critic": ("critic",)},
        obs_set="actor", output_dim=16, hidden_dims=(512, 256, 128),
        activation="elu", obs_normalization=False,
    ).to("cuda:0")
    warm_start_teacher_from_blind(
        model, _load_blind_state(WARM), history=ACTOR_HISTORY,
        scan_dim=fine_scan_dim(), fixed_command=(0.5 * sum(COMMAND_VX), 0.0, 0.0))

    sensor = env.scene[BLACK_ILLEGAL_CONTACT_SENSOR]
    asset = env.scene["robot"]
    tm = env.termination_manager
    print(f"noise enabled: {cfg.observations['actor'].enable_corruption}")
    print(f"solver: cone={cfg.sim.mujoco.cone} iter={cfg.sim.mujoco.iterations}")

    origins = env.scene.terrain.env_origins
    for ep in range(6):
        fresh = env.reset()[0]
        o = fresh
        for step in range(300):
            with torch.no_grad():
                obs_td = o
                action = model(obs_td, stochastic_output=False)
            out = env.step(action)
            o = out[0] if isinstance(out, tuple) else out
            dones = out[2]
            if bool(dones[0]):
                fired = [n for n in tm.active_terms if bool(tm.get_term(n)[0])]
                x = float(asset.data.root_link_pos_w[0, 0] - origins[0, 0])
                z = float(asset.data.root_link_pos_w[0, 2])
                ids = [asset.body_names.index(f"{leg}_foot")
                       for leg in ("FL", "FR", "RL", "RR")]
                from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
                    PART2_WHEEL_RADIUS_M, _part2_wheel_lowest_points)
                wz = _part2_wheel_lowest_points(asset, ids, PART2_WHEEL_RADIUS_M)[0, :, 2]
                wx = _part2_wheel_lowest_points(asset, ids, PART2_WHEEL_RADIUS_M)[0, :, 0]
                print(f"      wheels x={[round(float(v), 2) for v in wx.tolist()]} "
                      f"z={[round(float(v), 2) for v in wz.tolist()]}")
                mag = torch.norm(sensor.data.force[0], dim=-1)
                body = ""
                if float(mag.max()) > 1.0:
                    num_primary = len(sensor.primary_names)
                    slots = sensor.data.force.shape[1] // max(1, num_primary)
                    slot = int(mag.argmax())
                    body = sensor.primary_names[slot // slots]
                print(f"  ep{ep}: {step + 1:3d} steps  x={x:+.2f} z={z:.2f}  "
                      f"terms={fired}  contact={body} "
                      f"force={float(mag.max()):.0f}N")
                break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
