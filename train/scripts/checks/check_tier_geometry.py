"""Verify training exact terrains keep the same geometric *structure* across
difficulty and only scale their dimensions (no qualitative change of tiers)."""

from __future__ import annotations

import argparse
import json

import mujoco

from mjlab.terrains.terrain_generator import TerrainGenerator
from mjlab.tasks.registry import load_env_cfg

BUILDERS = {
    "platform": ("blackw-platform-exact", None),
    "wall": ("blackw-wall-exact", None),
    "slope": ("blackw-slope-exact", None),
    "stairs": ("blackw-stairs-exact", None),
    "trench": ("blackw-trench-exact", None),
    "plum": ("blackw-plum", None),
}


def dump(task: str, difficulty: float) -> dict:
    env_cfg = load_env_cfg(task, play=False)
    gen_cfg = env_cfg.scene.terrain.terrain_generator
    gen_cfg.num_rows = 1
    gen_cfg.difficulty_range = (difficulty, difficulty)
    gen = TerrainGenerator(gen_cfg, device="cpu")
    spec = mujoco.MjSpec()
    gen.compile(spec)
    model = spec.compile()
    geoms = []
    for gid in range(model.ngeom):
        geoms.append({
            "type": int(model.geom_type[gid]),
            "size": [round(float(v), 4) for v in model.geom_size[gid]],
            "pos": [round(float(v), 4) for v in model.geom_pos[gid]],
            "quat": [round(float(v), 4) for v in model.geom_quat[gid]],
        })
    # bbox of all geoms (position +- size is enough for boxes/planes; it is a
    # structural summary, not an exact AABB for rotated geoms).
    xs, zs = [], []
    for g in geoms:
        xs.extend([g["pos"][0] - g["size"][0], g["pos"][0] + g["size"][0]])
        zs.extend([g["pos"][2] - g["size"][2], g["pos"][2] + g["size"][2]])
    return {
        "n_geoms": len(geoms),
        "types": [g["type"] for g in geoms],
        "signature": [(g["type"], round(g["size"][1], 3)) for g in geoms],
        "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                       for i in range(model.ngeom)],
        "z_max_bbox": round(max(zs), 4),
        "geom_types_ordered": [g["type"] for g in geoms],
        "x_span": [round(min(xs), 3), round(max(xs), 3)],
        "geoms": geoms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    names = [args.only] if args.only else list(BUILDERS)
    out = {}
    for name in names:
        task = BUILDERS[name][0]
        low = dump(task, 0.0)
        high = dump(task, 1.0)
        same_types = low["types"] == high["types"]
        same_count = low["n_geoms"] == high["n_geoms"]
        out[name] = {
            "task": task,
            "n_geoms": [low["n_geoms"], high["n_geoms"]],
            "same_geom_type_sequence": same_types,
            "same_geom_count": same_count,
            "z_max_bbox": [low["z_max_bbox"], high["z_max_bbox"]],
            "x_span": [low["x_span"], high["x_span"]],
            "low_geom_types": low["types"],
            "high_geom_types": high["types"],
            "qualitative_change": not (same_types and same_count),
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
