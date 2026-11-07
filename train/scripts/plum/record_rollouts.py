#!/usr/bin/env python3
"""Record representative teacher rollouts and group them by failure mode.

Runs the frozen teacher in the training env with ``auto_reset=False`` so the
terminal state is the real pre-reset state.  For every episode it writes a
replay-compatible JSONL (base/quat/q/feet) plus per-step diagnostics, then
renders one representative mp4 per termination group with the repo's existing
video tool.  The output is meant for visual review of typical failure modes.

Run (train env; video rendering uses the eval env's python):
    MUJOCO_GL=egl PYTHONPATH=train/src .pixi/envs/train/bin/python \
      train/scripts/plum/record_rollouts.py --checkpoint <model_2999.pt> \
      --out artifacts/plum/rollouts --episodes 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from alldog_mjlab.tasks.velocity.profiles.part2.predicates import (
    PART2_WHEEL_RADIUS_M, _part2_wheel_lowest_points,
)
from alldog_mjlab.tasks.velocity.profiles.plum.student import build_teacher_mlp
from alldog_mjlab.tasks.velocity.profiles.plum.teacher import plum_teacher_env_cfg

ROOT = Path(__file__).resolve().parents[3]
EVAL_PY = ROOT / ".pixi" / "envs" / "eval" / "bin" / "python"
VIDEO_TOOL = ROOT / "tools" / "video" / "replay_route_video.py"
PLUM_SCENE = ROOT / "demo" / "scenes" / "part2" / "generated" / "part2_off_plum_1.xml"
WHEELS = ("FL", "FR", "RL", "RR")
_ASSET = SceneEntityCfg("robot")


def _state(env, asset, t: float) -> dict:
    """One replay frame + diagnostics, in SCENE coordinates.

    ``env_origins`` is a whole-patch offset (e.g. [24, 0, 0]) that changes per
    env build; the generated scene XML and the judge use scene coordinates with
    the spawn at x=0, so every world position must be de-offset for the video.
    ``t`` is the episode-relative time (env.common_step_counter is cumulative
    across episodes under auto_reset=False).
    """
    origin = env.scene.terrain.env_origins[0]
    pos = asset.data.root_link_pos_w[0]
    quat = asset.data.root_link_quat_w[0]
    ids = [asset.body_names.index(f"{leg}_foot") for leg in WHEELS]
    wpos = _part2_wheel_lowest_points(asset, ids, PART2_WHEEL_RADIUS_M)[0]  # [4, 3]
    feet = {f"{leg}_foot": [float(p[i] - origin[i]) for i in range(3)]
            for leg, p in zip(WHEELS, wpos)}
    wx = wpos[:, 0] - origin[0]
    wy = wpos[:, 1] - origin[1]
    wz = wpos[:, 2]
    w, x, y, z = [float(v) for v in quat]
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return {
        "t": t,
        "base": [float(pos[i] - origin[i]) for i in range(3)],
        "quat": [w, x, y, z],
        "q": [float(v) for v in asset.data.joint_pos[0]],
        "feet": feet,
        # diagnostics (ignored by the video tool)
        "root_x": float(pos[0] - origin[0]),
        "root_z": float(pos[2]),
        "abs_y": float(abs(pos[1] - origin[1])),
        "yaw_deg": float(yaw),
        "wheel_z_min": float(wz.min()),
        "wheel_z_max": float(wz.max()),
        "wheel_x_max": float(wx.max()),
        "wheel_x_min": float(wx.min()),
        "wheel_abs_y_max": float(wy.abs().max()),
    }


def classify(terms: list[str], final: dict) -> str:
    """Split the raw termination into a reviewable failure mode."""
    if "part2_success" in terms:
        return "success"
    if "part2_out_of_course" in terms:
        return "overshoot_beyond_goal"
    if "illegal_contact" in terms:
        # Strict body contact.  Where it happens tells the story: before the
        # platform face means the entry step, inside the cube window means a
        # cube rim/leg hit, after the field means the exit lip.
        x = final["root_x"]
        if x < 1.8:
            return "contact_on_runway"
        if x < 2.6:
            return "contact_at_entry_face"
        if x <= 4.55:
            return "contact_in_cube_field"
        return "contact_at_exit"
    if "part2_failure" in terms or "lane_deviation" in terms:
        if final["wheel_z_min"] < 0.05 and 2.55 <= final["root_x"] <= 4.55:
            return "dropped_wheel_in_field"
        if final["abs_y"] > 0.70 or final["wheel_abs_y_max"] > 0.70:
            return "lateral_drift"
        return "other_field_failure"
    if "bad_orientation" in terms or "bad_sim_state" in terms:
        return "instability"
    return "timeout_or_other"


def _record_one(env, teacher, max_steps: int, std: float = 0.0):
    asset = env.scene["robot"]
    tm = env.termination_manager
    obs, _ = env.reset()
    frames = []
    terminated_at = None
    for step in range(max_steps):
        frames.append(_state(env, asset, step * 0.02))
        with torch.no_grad():
            action = teacher(torch.cat([obs["actor"], obs["fine_scan"]], dim=-1))
            if std > 0.0:
                action = action + std * torch.randn_like(action)
        obs, _rew, terminated, truncated, _info = env.step(torch.nan_to_num(action))
        done = bool(terminated[0].item() or truncated[0].item())
        if done:
            # auto_reset=False: tm.get_term still holds the pre-reset values.
            fired = [name for name in tm.active_terms
                     if bool(tm.get_term(name)[0].item())]
            end = _state(env, asset, (step + 1) * 0.02)
            terminated_at = {"step": step, "terms": fired, "state": end,
                             "mode": classify(fired, end)}
            break
    return frames, terminated_at


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--std", type=float, default=0.0,
                    help="Gaussian action noise (0 = deterministic); set to the trained action std to sample failures, e.g. 0.59")
    ap.add_argument("--difficulty", type=float, default=1.0,
                    help="curriculum row to roll on (1.0 = official field)")
    ap.add_argument("--scene", type=Path, default=None,
                    help="replay scene matching --difficulty "
                         "(default: the official plum scene for 1.0, otherwise "
                         "generated by scripts/plum/make_scene.py)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher = build_teacher_mlp(ckpt["actor_state_dict"], args.device)

    # Use the TRAIN cfg: blackw_plum_env_cfg(play=True) does not register the
    # part2_success / part2_out_of_course / lane_deviation terminations, so a
    # play-mode rollout cannot be compared with the gate's play=False result.
    cfg = plum_teacher_env_cfg(play=False)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    # Roll on ONE curriculum row so the video shows the terrain the agent sees.
    # The training cfg samples rows from the curriculum, which would make the
    # replay scene wrong for most episodes.
    gen = cfg.scene.terrain.terrain_generator
    gen.difficulty_range = (args.difficulty, args.difficulty)
    cfg.scene.terrain.max_init_terrain_level = 0
    # The curriculum term would move envs off the pinned row after the first
    # reset; freeze it for a controlled recording.
    cfg.curriculum = {}
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode=None)
    if args.scene is not None:
        replay_scene = args.scene
    elif abs(args.difficulty - 1.0) < 1e-9:
        replay_scene = PLUM_SCENE
    else:
        replay_scene = (ROOT / "demo/scenes/part2/generated"
                        / f"plum_curriculum_d{args.difficulty:.2f}.xml")
        subprocess.run(
            [sys.executable, str(ROOT / "train/scripts/plum/make_scene.py"),
             "--difficulty", str(args.difficulty), "--out", str(replay_scene)],
            check=True, capture_output=True)
    print(f"[plum/rollout] difficulty={args.difficulty} scene={replay_scene.name}",
          flush=True)

    asset = env.scene["robot"]
    episodes = []
    for i in range(args.episodes):
        frames, term = _record_one(env, teacher, args.max_steps, args.std)
        tag = f"ep_{i:02d}"
        (args.out / f"{tag}.jsonl").write_text(
            "\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
        group = term["mode"] if term else "no_termination"
        rec = {
            "tag": tag, "frames": len(frames), "group": group,
            "terms": term["terms"] if term else [],
            "final": term["state"] if term else frames[-1],
            "max_root_x": max(f["root_x"] for f in frames),
        }
        episodes.append(rec)
        print(f"[plum/rollout] {tag}: {len(frames)} steps, group={group}, "
              f"max_x={rec['max_root_x']:.2f}, "
              f"final_z={rec['final']['root_z']:.2f}", flush=True)
    env.close()

    # Group by dominant termination and pick one representative per group.
    by_group: dict[str, list] = defaultdict(list)
    for e in episodes:
        by_group[e["group"]].append(e)
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "group_counts": dict(Counter(e["group"] for e in episodes)),
        "groups": {},
    }
    for group, recs in sorted(by_group.items()):
        rep = min(recs, key=lambda r: abs(r["max_root_x"] -
                                         np.median([x["max_root_x"] for x in recs])))
        video = args.out / f"{group}.mp4"
        if not args.no_video and EVAL_PY.is_file() and VIDEO_TOOL.is_file():
            cmd = [str(EVAL_PY), str(VIDEO_TOOL),
                   str(args.out / f"{rep['tag']}.jsonl"), str(video),
                   "--scene", str(replay_scene)]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  env={**__import__("os").environ, "MUJOCO_GL": "egl"})
            video_ok = proc.returncode == 0
            if not video_ok:
                print(f"[plum/rollout] video failed for {group}: "
                      f"{proc.stderr[-300:]}", flush=True)
            else:
                # Episodes terminate early (50-110 steps = 1-2 s at 25 fps);
                # slow down 3x so the failure is actually reviewable.
                raw = args.out / f"{group}_raw.mp4"
                video.replace(raw)
                slow = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                     "-filter:v", "setpts=3*PTS", "-r", "25", str(video)],
                    capture_output=True, text=True)
                if slow.returncode != 0:
                    print(f"[plum/rollout] slow-mo failed for {group}: "
                          f"{slow.stderr[-300:]}", flush=True)
                    raw.replace(video)
                else:
                    raw.unlink(missing_ok=True)
        else:
            video_ok = False
        summary["groups"][group] = {
            "count": len(recs),
            "representative": rep["tag"],
            "video": str(video) if video_ok else None,
            "median_max_root_x": float(np.median([r["max_root_x"] for r in recs])),
            "median_final_root_z": float(np.median([r["final"]["root_z"] for r in recs])),
            "median_wheel_z_min": float(np.median([r["final"]["wheel_z_min"] for r in recs])),
            "median_abs_y": float(np.median([r["final"]["abs_y"] for r in recs])),
            "terms": Counter(t for r in recs for t in r["terms"]),
        }
    summary["groups"] = {k: {**v, "terms": dict(v["terms"])}
                         for k, v in summary["groups"].items()}
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[plum/rollout] wrote {args.out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
