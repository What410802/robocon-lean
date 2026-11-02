#!/usr/bin/env python3
"""Check Part1/Part2 spawn poses against the blackW model (support height + penetration).

Reads the generated Part 2 level manifest + scenes from the new single repo:
``demo/scenes/part2/part2_official_levels.json`` and
``demo/scenes/part2/generated/``; the Part 1 spawn comes from the official
Part 1 scene (skipped with a warning if that scene is not loadable in the current
migration state).  The output JSON goes to the git-ignored ``logs/`` directory.

Run (repository root):

    /home/bis/RoboCon/.pixi/envs/eval/bin/python train/scripts/checks/check_spawns.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # train/scripts/checks -> repo root
PART2_JSON = ROOT / "demo" / "scenes" / "part2" / "part2_official_levels.json"
PART2_SCENE_DIR = ROOT / "demo" / "scenes" / "part2" / "generated"
PART1_SCENE_DIR = ROOT / "demo" / "scenes" / "part1" / "official"
PART1 = (3.7, -9.0, None, 0.0)  # x, y, z filled from support height
CLEARANCE = 0.005


def wheel_offset(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Root-height minus the lowest wheel point, at the model's default joint pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qpos[2] = 3.0
    mujoco.mj_forward(model, data)
    foot = [g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name.endswith("_foot")
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
    bottom = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in foot)
    return 3.0 - bottom


def support_height(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float) -> float:
    """Highest static surface at (x, y); robot moved away so it cannot block the ray."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qpos[0] = 1000.0
    mujoco.mj_forward(model, data)
    pnt = np.array([x, y, 5.0], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, np.ones(6, dtype=np.uint8), 1, -1, gid)
    return max(0.0, 5.0 - dist) if dist >= 0.0 else 0.0


def penetration(model: mujoco.MjModel, data: mujoco.MjData, spawn: tuple[float, ...]) -> tuple[float, int]:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = spawn[:3]
    yaw = spawn[3]
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    robot = set()
    for b in range(1, model.nbody):
        if model.body_parentid[b] == 0:
            stack = [b]
            while stack:
                cur = stack.pop()
                robot.add(cur)
                stack += [c for c in range(1, model.nbody) if model.body_parentid[c] == cur]
    worst = 0.0
    n = 0
    for c in data.contact[:data.ncon]:
        if (model.geom_bodyid[c.geom1] in robot) != (model.geom_bodyid[c.geom2] in robot):
            worst = min(worst, float(c.dist))
            n += 1
    return worst, n


def main() -> int:
    rows = []
    offsets = {}
    part2 = json.loads(PART2_JSON.read_text())
    scenes = []
    part1_scene = PART1_SCENE_DIR / "scene_terrain.xml"
    try:
        mujoco.MjModel.from_xml_path(str(part1_scene))
        scenes.append(("part1", part1_scene, PART1, False))
    except Exception as exc:  # noqa: BLE001 - migration-state dependency, skip
        print(f"[warn] skipping Part 1 spawn check ({part1_scene}): {exc}")
    scenes += [
        (e["key"], PART2_SCENE_DIR / e["scene"],
         tuple(e.get("spawn") or e.get("start")), True)
        for e in part2["levels"]
    ]
    for key, scene_path, start, is_part2 in scenes:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        scene = scene_path.name
        data = mujoco.MjData(model)
        if scene not in offsets:
            offsets[scene] = wheel_offset(model, data)
        off = offsets[scene]
        support = support_height(model, data, start[0], start[1])
        suggested = (start[0], start[1], support + off + CLEARANCE, start[3])
        pen, ncon = penetration(model, data, suggested)
        start_pen = None if start[2] is None else penetration(model, data, (start[0], start[1], start[2], start[3]))[0]
        rows.append({"key": key, "scene": scene, "start": list(start), "support": round(support, 3),
                     "start_penetration": None if start_pen is None else round(start_pen, 4),
                     "suggested": [round(v, 3) for v in suggested],
                     "z_delta": None if start[2] is None else round(start[2] - suggested[2], 3),
                     "worst_contact_dist": round(pen, 4),
                     "robot_contacts": ncon})
        start_z = "n/a" if start[2] is None else f"{start[2]:.3f}"
        delta = "n/a" if rows[-1]['z_delta'] is None else f"{rows[-1]['z_delta']:+.3f}"
        start_pen_s = "n/a" if start_pen is None else f"{start_pen:.4f}"
        print(f"{key:24s} support={support:.3f} start_z={start_z} (start_pen {start_pen_s}) "
              f"-> spawn={suggested[2]:.3f} (delta {delta}) "
              f"spawn_pen={pen:.4f} contacts={ncon}")
    out = ROOT / "logs" / "spawn_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    bad = [r for r in rows if r["worst_contact_dist"] < -0.001]
    print(f"penetrating spawns: {len(bad)} / {len(rows)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
