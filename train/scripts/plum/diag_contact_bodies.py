#!/usr/bin/env python3
"""Which body ends the episode, and where?  Mirrors the training config.

The 2048-env run ended ~95 % of episodes on ``illegal_contact`` at an average
length of ~48 steps (1 s).  ``diag_illegal.py`` uses deterministic actions and
sees no contact at all, so the cause is action noise or the early gait, not the
spawn.  This runs many envs with the warm-started policy PLUS Gaussian noise
(init_std), records every illegal-contact termination with the body name, the
env-local base x and the wheel heights, and prints the distribution.

    MUJOCO_GL=egl pixi run -e train python train/scripts/plum/diag_contact_bodies.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
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
from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (  # noqa: E402
    PART2_WHEEL_RADIUS_M, _part2_wheel_lowest_points,
)
from alldog_mjlab.tasks.velocity.profiles.plum.fine_scan import fine_scan_dim  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.runner import _load_blind_state  # noqa: E402
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import (  # noqa: E402
    ACTOR_HISTORY, COMMAND_VX, plum_teacher_env_cfg,
)

WARM = ROOT / "demo/rl_sar/policy/blackW/row2p_p40w40_v1/policy.pt"
WHEELS = ("FL", "FR", "RL", "RR")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--std", type=float, default=0.10)
    ap.add_argument("--difficulty", type=float, default=None,
                    help="pin the curriculum row (default: sample the ladder)")
    ap.add_argument("--stage", type=int, default=0, choices=(0, 1),
                    help="spawn envelope stage (1 = wider handover range)")
    ap.add_argument("--spawn-dx", type=float, default=None,
                    help="pin the spawn x offset from the platform centre (m)")
    ap.add_argument("--spawn-yaw-deg", type=float, default=0.0,
                    help="pin the spawn yaw (deg) when --spawn-dx is given")
    ap.add_argument("--spawn-z-off", type=float, default=None,
                    help="pin the spawn z OFFSET from the platform spawn "
                         "(m; the eval ground start is -0.20)")
    ap.add_argument("--zero-scan", action="store_true",
                    help="zero the 1764-D fine scan: does the policy use perception?")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="evaluate a training checkpoint instead of the warm start")
    args = ap.parse_args()

    cfg = plum_teacher_env_cfg(stage=args.stage)
    cfg.scene.num_envs = args.envs
    if args.spawn_dx is not None:
        ev = dict(cfg.events)
        rb = ev["reset_base"]
        pose = dict(dict(rb.params).get("pose_range") or {})
        import math as _m
        yr = _m.radians(args.spawn_yaw_deg)
        pose.update({"x": (args.spawn_dx, args.spawn_dx), "y": (0.0, 0.0),
                     "yaw": (yr, yr)})
        if args.spawn_z_off is not None:
            pose["z"] = (args.spawn_z_off, args.spawn_z_off)
        ev["reset_base"] = __import__("dataclasses").replace(
            rb, params={**dict(rb.params), "pose_range": pose})
        cfg.events = ev
        print(f"[diag] pinned spawn dx={args.spawn_dx:+.2f} m, dy=0, "
              f"yaw={args.spawn_yaw_deg:+.1f} deg", flush=True)
    if args.difficulty is not None:
        gen = cfg.scene.terrain.terrain_generator
        gen.difficulty_range = (args.difficulty, args.difficulty)
        cfg.curriculum = {}
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    o = env.reset()[0]

    model = MLPModel(
        obs=o, obs_groups={"actor": ("actor", "fine_scan"), "critic": ("critic",)},
        obs_set="actor", output_dim=16, hidden_dims=(512, 256, 128),
        activation="elu", obs_normalization=False,
    ).to("cuda:0")
    if args.checkpoint is None:
        warm_start_teacher_from_blind(
            model, _load_blind_state(WARM), history=ACTOR_HISTORY,
            scan_dim=fine_scan_dim(), fixed_command=(0.5 * sum(COMMAND_VX), 0.0, 0.0))
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = {k: v for k, v in ckpt["actor_state_dict"].items()
                 if not k.startswith("distribution.")}
        missing = model.load_state_dict(state, strict=False).missing_keys
        print(f"checkpoint iteration {ckpt.get('iter')} loaded "
              f"(missing: {missing or 'none'})")
    model.eval()

    sensor = env.scene[BLACK_ILLEGAL_CONTACT_SENSOR]
    asset = env.scene["robot"]
    tm = env.termination_manager
    origins = env.scene.terrain.env_origins
    foot_ids = [asset.body_names.index(f"{leg}_foot") for leg in WHEELS]

    alive = torch.ones(args.envs, dtype=torch.bool, device="cuda:0")
    spawn = {}

    def _record_spawn(ids):
        """Spawn pose per env, in terrain-local coords (x offset, y, yaw)."""
        pos = asset.data.root_link_pos_w[ids].clone()
        quat = asset.data.root_link_quat_w[ids].clone()
        for n, j in enumerate(ids.tolist()):
            w, qx, qy, qz = quat[n].tolist()
            import math
            yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy),
                                         1 - 2 * (qy * qy + qz * qz)))
            spawn[j] = (float(pos[n, 0] - origins[j, 0]),
                        float(pos[n, 1] - origins[j, 1]), yaw)

    _record_spawn(torch.arange(args.envs, device="cuda:0"))
    fails: list[dict] = []
    levels = Counter()
    bodies = Counter()
    # auto_reset=False: a terminated env must be reset explicitly before the next
    # step, so terminated envs are recorded and immediately respawned.
    for step in range(args.steps):
        with torch.no_grad():
            obs_in = dict(o)
            if args.zero_scan:
                obs_in["fine_scan"] = torch.zeros_like(o["fine_scan"])
            act = model(obs_in, stochastic_output=False)
            act = act + args.std * torch.randn_like(act)
        out = env.step(act)
        o = out[0] if isinstance(out, tuple) else out
        done = out[2] | out[3]
        newly = done  # every done env is respawned below
        if bool(newly.any()):
            idx = torch.nonzero(newly).flatten()
            force = sensor.data.force
            n_primary = len(sensor.primary_names)
            slots = force.shape[1] // max(1, n_primary)
            # [B, P*slots, 3] -> [B, P, slots] (primary-major layout).
            mag = torch.norm(force[idx], dim=-1).view(len(idx), n_primary, slots).amax(-1)
            fired = {n for n in tm.active_terms if bool(tm.get_term(n)[idx].any())}
            wz = _part2_wheel_lowest_points(asset, foot_ids, PART2_WHEEL_RADIUS_M)[idx, :, 2]
            wx = (_part2_wheel_lowest_points(asset, foot_ids, PART2_WHEEL_RADIUS_M)[idx, :, 0]
                  - origins[idx, 0:1])
            rx = asset.data.root_link_pos_w[idx, 0] - origins[idx, 0]
            if "illegal_contact" in fired:
                slot = int(mag.max(dim=-1).indices[0].item())
                bodies[sensor.primary_names[slot // slots]] += 1
            for k in range(len(idx)):
                fails.append({
                    "step": step, "rx": float(rx[k]), "wz_min": float(wz[k].min()),
                    "spawn": spawn.get(int(idx[k])),
                    "wz": [round(float(v), 3) for v in wz[k]],
                    "wx_max": float(wx[k].max()),
                    "terms": [n for n in tm.active_terms
                              if bool(tm.get_term(n)[idx[k]])],
                })
            levels[int(env.scene.terrain.terrain_levels[idx[0]])] += 1
            # Respawn the terminated envs before the next step (auto_reset=False).
            fresh, _ = env.reset(env_ids=idx)
            _record_spawn(idx)
            for key in o:
                o[key][idx] = (fresh[key][idx] if fresh[key].shape[0] == args.envs
                               else fresh[key])
            if len(fails) >= args.envs * 3:
                break

    print(f"episodes ended: {len(fails)} / {args.envs} in {step + 1} steps")
    print(f"contact bodies: {dict(bodies)}")
    print(f"levels: {dict(levels)}")
    counts = Counter(tuple(f["terms"]) for f in fails)
    for terms, n in counts.most_common():
        sub = [f for f in fails if tuple(f["terms"]) == terms]
        print(f"  {n:4d}  {terms}  rx median {sorted(x['rx'] for x in sub)[len(sub) // 2]:+.2f}"
              f"  wz_min median {sorted(x['wz_min'] for x in sub)[len(sub) // 2]:+.3f}")
        sp = [x["spawn"] for x in sub if x.get("spawn")]
        if sp:
            # ``spawn`` entries are dicts (dx, dy, yaw), not tuples.
            sx = sorted(v["dx"] for v in sp); sy = sorted(v["dy"] for v in sp)
            syaw = sorted(v["yaw"] for v in sp)
            print(f"        spawn median dx={sx[len(sx)//2]:+.3f} m "
                  f"dy={sy[len(sy)//2]:+.3f} m yaw={syaw[len(syaw)//2]:+.1f} deg "
                  f"| range dx[{sx[0]:+.2f},{sx[-1]:+.2f}] dy[{sy[0]:+.2f},{sy[-1]:+.2f}] "
                  f"yaw[{syaw[0]:+.1f},{syaw[-1]:+.1f}]")
    if fails:
        lo = [f for f in fails if f["rx"] < 1.8]
        print(f"\nrx < 1.8 (before entry, i.e. on the runway): {len(lo)}")
        print(f"rx in [1.8, 2.7]: {sum(1 for f in fails if 1.8 <= f['rx'] < 2.7)}")
        print(f"rx in [2.7, 4.5]: {sum(1 for f in fails if 2.7 <= f['rx'] <= 4.5)}")
    sp = [f for f in fails if f.get("spawn")]
    if sp:
        print("\nspawn dx bins (front of platform = larger dx):")
        lo_sp = min(v["dx"] for v in sp); hi_sp = max(v["dx"] for v in sp)
        step = (hi_sp - lo_sp) / 4.0
        edges = [lo_sp + i * step for i in range(5)] + [hi_sp + 1e-6]
        for lo, hi in zip(edges, edges[1:]):
            sel = [f for f in sp if lo <= f["spawn"]["dx"] < hi]
            ok = sum(1 for f in sel if "part2_success" in f["terms"])
            n = len(sel)
            print(f"  [{lo:.2f},{hi:.2f}): n={n:4d} ok={ok:4d} fail={n-ok:3d} "
                  f"fail_rate={100*(n-ok)/max(1,n):5.1f}%")
        print("spawn |yaw| bins (deg):")
        for lo, hi in [(0,2),(2,4),(4,6),(6,8),(8,10),(10,13),(13,17),(17,21)]:
            sel = [f for f in sp if lo <= abs(f["spawn"]["yaw"]) < hi]
            ok = sum(1 for f in sel if "part2_success" in f["terms"])
            n = len(sel)
            print(f"  [{lo},{hi}): n={n:4d} ok={ok:4d} fail={n-ok:3d} "
                  f"fail_rate={100*(n-ok)/max(1,n):5.1f}%")
        print("spawn |dy| bins (cm):")
        for lo, hi in [(0,4),(4,8),(8,13)]:
            sel = [f for f in sp if lo <= abs(f["spawn"]["dy"])*100 < hi]
            ok = sum(1 for f in sel if "part2_success" in f["terms"])
            n = len(sel)
            print(f"  [{lo},{hi}): n={n:4d} ok={ok:4d} fail={n-ok:3d} "
                  f"fail_rate={100*(n-ok)/max(1,n):5.1f}%")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
