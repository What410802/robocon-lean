#!/usr/bin/env python3
"""Score a depth ACTOR (CNNModel) in-sim on the plum field.

The privileged teacher is evaluated by ``diag_contact_bodies.py``; a depth actor
needs the camera group instead, so this is the matching harness:

    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
      train/scripts/plum/eval_depth_actor.py --checkpoint <depth.pt> \
      --difficulty 1.0 [--std 0.0] [--envs 256] [--steps 320]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from rsl_rl.models.cnn_model import CNNModel  # noqa: E402

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.depth import (  # noqa: E402
    plum_depth_env_cfg,
)
from alldog_mjlab.tasks.velocity.profiles.plum.depth_camera import (  # noqa: E402
    DEPTH_CNN_CFG, add_depth_camera,
)
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    plum_teacher_env_cfg,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--std", type=float, default=0.0)
    ap.add_argument("--stage", type=int, default=0, choices=(0, 1, 2),
                    help="spawn envelope: 0/1 on the entry platform, 2 the "
                         "judge's ground start (scene x=0, base z=0.55)")
    ap.add_argument("--difficulty", type=float, default=1.0)
    args = ap.parse_args()

    cfg = plum_depth_env_cfg(stage=args.stage)
    cfg.scene.num_envs = args.envs
    gen = cfg.scene.terrain.terrain_generator
    gen.difficulty_range = (args.difficulty, args.difficulty)
    cfg.curriculum = {}
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    o = env.reset()[0]

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    st = ck.get("actor_state_dict", ck)
    sample = TensorDict({"actor": o["actor"][:2], "camera": o["camera"][:2]},
                        batch_size=[2])
    actor = CNNModel(
        sample, {"actor": ("actor", "camera"), "critic": ("critic",)},
        "actor", 16, hidden_dims=tuple(BLACK_CONFIG.policy.actor_hidden_dims),
        activation=BLACK_CONFIG.policy.activation, obs_normalization=False,
        cnn_cfg=dict(DEPTH_CNN_CFG),
    ).to("cuda:0")
    missing = actor.load_state_dict(st, strict=False).missing_keys
    actor.eval()
    print(f"[eval/depth] {args.checkpoint} loaded (missing: {missing or 'none'})",
          flush=True)

    alive = torch.ones(args.envs, dtype=torch.bool, device="cuda:0")
    counts: dict[tuple, int] = {}
    loci: dict[tuple, list] = {}
    for step in range(args.steps):
        with torch.no_grad():
            td = TensorDict({"actor": o["actor"], "camera": o["camera"]},
                            batch_size=[args.envs])
            act = actor(td, stochastic_output=False)
            if args.std > 0:
                act = act + args.std * torch.randn_like(act)
        out = env.step(act)
        o = out[0] if isinstance(out, tuple) else out
        done = out[2] | out[3]
        if bool(done.any()):
            idx = torch.nonzero(done).flatten()
            tm = env.termination_manager
            from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
                PART2_WHEEL_RADIUS_M, _part2_wheel_lowest_points,
            )
            asset = env.scene["robot"]
            wpos = _part2_wheel_lowest_points(
                asset, [asset.body_names.index(f"{leg}_foot")
                        for leg in ("FL", "FR", "RL", "RR")],
                PART2_WHEEL_RADIUS_M)
            origins = env.scene.terrain.env_origins
            rx = (asset.data.root_link_pos_w[:, 0] - origins[:, 0]).cpu()
            wmin = wpos[..., 2].min(dim=1).values.cpu()
            for k in range(len(idx)):
                j = int(idx[k])
                terms = tuple(n for n in tm.active_terms if bool(tm.get_term(n)[j]))
                counts[terms] = counts.get(terms, 0) + 1
                loci.setdefault(terms, []).append((float(rx[j]), float(wmin[j])))
            fresh, _ = env.reset(env_ids=idx)
            for key in o:
                # A subset reset may return full-size tensors (mjlab) or the
                # subset only; handle both, as diag_contact_bodies.py does.
                o[key][idx] = (fresh[key][idx]
                               if fresh[key].shape[0] == args.envs else fresh[key])
    env.close()
    total = sum(counts.values())
    succ = sum(v for k, v in counts.items() if "part2_success" in k)
    print(f"episodes {total} | success {succ} = {100 * succ / max(1, total):.1f}%")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        xs = sorted(x for x, _ in loci.get(k, []))
        zs = sorted(z for _, z in loci.get(k, []))
        med = (f"  rx median {xs[len(xs)//2]:+.2f}  wz_min median {zs[len(zs)//2]:+.3f}"
               if xs else "")
        print(f"  {v:5d}  {k}{med}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
