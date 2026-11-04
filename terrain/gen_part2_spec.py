#!/usr/bin/env python3
"""Derive the Part 2 terrain spec (single source) from the official map.xml.

Why this exists: the training terrains (``train/src/alldog_mjlab/tasks/velocity/
profiles/part2/terrain.py``) and the generated judge scenes must use the *same*
obstacle dimensions.  Hand-copying the numbers out of ``terrain.zip/map.xml``
produced geometry drift before (the HIM branch still had an older slope ramp).
This tool parses the official map once and writes a small JSON spec that the
training terrain builder and the level-scene generator load; the spec is
committed with the source sha256, so a re-generated spec can be diffed and any
dimension change is visible in review.

The judge protocol constants (1.8 m runway between spawn and obstacle near face,
0.7 m scoring-lane half width, 0.50 m trunk height on flat) are part of the
organiser's evaluation scene generation, not of the map geometry; they are
recorded in the spec as ``protocol``.

The official ``map.xml`` carries the *old* plum field (two staggered rows of 12
cubes + entry/exit ramps), which the rules PDF itself says is wrong.  The
``plum`` section is therefore taken from ``terrain/gen_plum.py`` (the PDF-figure
geometry, single source) instead of parsed from the map; every other category is
copied from the official map geometry.

Output: ``terrain/official/part2_terrain_spec.json`` (committed single source).

Usage (from anywhere; numpy is the only third-party dependency)::

    python terrain/gen_part2_spec.py
    python terrain/gen_part2_spec.py --map /path/to/map.xml --out /tmp/spec.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

import gen_plum

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = ROOT / "terrain" / "official" / "map.xml"
DEFAULT_OUT = ROOT / "terrain" / "official" / "part2_terrain_spec.json"

RUNWAY_M = 1.8
LANE_HALF_WIDTH = 0.7
SPAWN_Z = 0.50

_MESHES: dict[str, np.ndarray] = {}


def _parse_attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', tag))


def _parse_floats(text: str) -> np.ndarray:
    return np.fromstring(text.replace(",", " "), sep=" ")


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _parse_xml(map_path: Path) -> tuple[list[str], list[str]]:
    text = map_path.read_text(encoding="utf-8")
    asset = re.search(r"<asset>(.*?)</asset>", text, re.S).group(1)
    mesh_lines = re.findall(r"<mesh [^>]*/>", asset)
    for m in mesh_lines:
        a = _parse_attrs(m)
        if "vertex" in a:
            _MESHES[a["name"]] = _parse_floats(a["vertex"]).reshape(-1, 3)
    return mesh_lines, re.findall(r"<geom [^>]*/>", text)


def _geom_bbox(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bbox of one geom in the official map frame."""
    a = _parse_attrs(tag)
    pos = _parse_floats(a["pos"]) if "pos" in a else np.zeros(3)
    if len(pos) < 3:
        pos = np.zeros(3)
    rot = np.eye(3)
    if "euler" in a:
        rot = _rot_z(float(a["euler"].split()[2]))
    if a["type"] == "mesh":
        v = _MESHES[a["mesh"]] @ rot.T + pos
        return v.min(axis=0), v.max(axis=0)
    size = _parse_floats(a["size"])
    if a["type"] == "box":
        half = size[:3]
    elif a["type"] == "cylinder":
        half = np.array([size[0], size[0], size[1]])
    elif a["type"] == "sphere":
        half = np.full(3, size[0])
    else:
        raise ValueError(f"unsupported geom type {a['type']}")
    return pos - half, pos + half


def _select(geoms: list[str], prefixes: tuple[str, ...]) -> list[str]:
    out = []
    for g in geoms:
        name = _parse_attrs(g)["name"]
        if any(name == p or name.startswith(p) for p in prefixes):
            out.append(g)
    return out


def _bbox(name: str, geoms: list[str]) -> tuple[np.ndarray, np.ndarray]:
    for g in geoms:
        if _parse_attrs(g)["name"] == name:
            return _geom_bbox(g)
    raise KeyError(name)


def _union(geoms: list[str]) -> tuple[np.ndarray, np.ndarray]:
    boxes = [_geom_bbox(g) for g in geoms]
    return (np.min([b[0] for b in boxes], axis=0),
            np.max([b[1] for b in boxes], axis=0))


def _round(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def build_spec(map_path: Path) -> dict:
    mesh_lines, geom_lines = _parse_xml(map_path)

    # ---- high platform: one solid box per tier, width/depth constant ----------
    platform_levels = [10, 20, 30, 40]
    platform_tiers = []
    for cm in platform_levels:
        geoms = _select(geom_lines, (f"platform_{cm:02d}cm",))
        bmin, bmax = _union(geoms)
        platform_tiers.append({
            "level_cm": cm,
            "top_m": _round(bmax[2]),
            "half_x_m": _round((bmax[0] - bmin[0]) / 2.0),
            "half_y_m": _round((bmax[1] - bmin[1]) / 2.0),
        })
    p0, p1 = platform_tiers[0], platform_tiers[-1]
    if len({(t["half_x_m"], t["half_y_m"]) for t in platform_tiers}) != 1:
        raise ValueError("official platform tiers do not share footprint")
    high_platform = {
        "levels": platform_tiers,
        "height_range_m": [p0["top_m"], p1["top_m"]],
        "half_x_m": p0["half_x_m"],
        "half_y_m": p0["half_y_m"],
        "center_x_m": _round(RUNWAY_M + p0["half_x_m"]),
    }

    # ---- slope: wedge mesh (run x rise) + landing box -------------------------
    slope_levels = [20, 30, 35, 45]
    slope_tiers = []
    for deg in slope_levels:
        up = _select(geom_lines, (f"slope_{deg:02d}deg_up",))
        landing = _select(geom_lines, (f"slope_{deg:02d}deg_landing",))
        bmin_up, bmax_up = _union(up)
        bmin_l, bmax_l = _union(landing)
        run = float(bmax_up[0] - bmin_up[0])
        rise = float(bmax_up[2])
        slope_tiers.append({
            "level_deg": deg,
            "run_m": _round(run),
            "rise_m": _round(rise),
            "angle_deg": _round(math.degrees(math.atan2(rise, run)), 4),
            "width_m": _round(bmax_up[1] - bmin_up[1]),
            "landing_half_x_m": _round((bmax_l[0] - bmin_l[0]) / 2.0),
        })
    if len({(t["run_m"], t["width_m"], t["landing_half_x_m"]) for t in slope_tiers}) != 1:
        raise ValueError("official slope tiers do not share ramp/landing shape")
    slope = {
        "levels": slope_tiers,
        "deg_range": [slope_tiers[0]["angle_deg"], slope_tiers[-1]["angle_deg"]],
        "ramp_horizontal_m": slope_tiers[0]["run_m"],
        "ramp_width_m": slope_tiers[0]["width_m"],
        "landing_half_x_m": slope_tiers[0]["landing_half_x_m"],
        "ramp_start_x_m": RUNWAY_M,
    }

    # ---- trench: raised plateau + entry/exit wedge ramps + slot gap ----------
    trench_levels = [10, 20, 30]
    trench_tiers = []
    for cm in trench_levels:
        near = _select(geom_lines, (f"trench_{cm:02d}cm_near",))
        far = _select(geom_lines, (f"trench_{cm:02d}cm_far",))
        entry = _select(geom_lines, (f"trench_{cm:02d}cm_entry",))
        _, nb = _union(near)
        fb, fmax = _union(far)
        entry_min, entry_max = _union(entry)
        trench_tiers.append({
            "level_cm": cm,
            "gap_m": _round(float(fb[0] - nb[0])),
            "plateau_m": _round(float(nb[2])),
            "near_half_x_m": _round((nb[0] - _union(near)[0][0]) / 2.0),
            "channel_half_y_m": _round((nb[1] - _union(near)[0][1]) / 2.0),
            "ramp_run_m": _round(float(entry_max[0] - entry_min[0])),
        })
    if len({(t["plateau_m"], t["near_half_x_m"], t["channel_half_y_m"],
             t["ramp_run_m"]) for t in trench_tiers}) != 1:
        raise ValueError("official trench tiers do not share plateau shape")
    trench = {
        "levels": trench_tiers,
        "gap_range_m": [trench_tiers[0]["gap_m"], trench_tiers[-1]["gap_m"]],
        "plateau_m": trench_tiers[0]["plateau_m"],
        "near_half_x_m": trench_tiers[0]["near_half_x_m"],
        "channel_half_y_m": trench_tiers[0]["channel_half_y_m"],
        "ramp_run_m": trench_tiers[0]["ramp_run_m"],
        "start_x_m": RUNWAY_M,
    }

    # ---- high wall: cross_bar boxes, level 10 uses the judge's 10 cm override -
    wall_field = {10: ("cross_bar_3", 0.10), 20: ("cross_bar_5", None),
                  30: ("cross_bar_6", None), 40: ("cross_bar_7", None)}
    wall_tiers = []
    for cm, (name, override) in wall_field.items():
        geoms = _select(geom_lines, (name,))
        bmin, bmax = _union(geoms)
        top = override if override is not None else float(bmax[2])
        wall_tiers.append({
            "level_cm": cm,
            "top_m": _round(top),
            "half_y_m": _round((bmax[1] - bmin[1]) / 2.0),
            "thickness_m": _round(bmax[0] - bmin[0]),
        })
    if len({(t["half_y_m"], t["thickness_m"]) for t in wall_tiers}) != 1:
        raise ValueError("official cross_bar tiers do not share thickness/width")
    high_wall = {
        "levels": wall_tiers,
        "height_range_m": [wall_tiers[0]["top_m"], wall_tiers[-1]["top_m"]],
        "half_y_m": wall_tiers[0]["half_y_m"],
        "thickness_m": wall_tiers[0]["thickness_m"],
        "center_x_m": _round(RUNWAY_M + wall_tiers[0]["thickness_m"] / 2.0),
    }

    # ---- stairs: 6 up + landing + 6 down -------------------------------------
    stairs_levels = [10, 15, 20, 30]
    stairs_tiers = []
    for cm in stairs_levels:
        geoms = _select(geom_lines, (f"stairs_{cm:02d}cm_",))
        up = [g for g in geoms if "_up_" in _parse_attrs(g)["name"]]
        landing = _select(geoms, (f"stairs_{cm:02d}cm_landing",))
        lb, lmax = _union(landing)
        step_boxes = [_geom_bbox(g) for g in up]
        stairs_tiers.append({
            "level_cm": cm,
            "num_steps": len(up),
            "tread_m": _round(float(step_boxes[0][1][0] - step_boxes[0][0][0])),
            "half_width_m": _round(float(step_boxes[0][1][1] - step_boxes[0][0][1]) / 2.0),
            "landing_half_x_m": _round(float((lmax[0] - lb[0]) / 2.0)),
            "total_height_m": _round(float(lmax[2])),
        })
    if len({(t["num_steps"], t["tread_m"], t["half_width_m"],
             t["landing_half_x_m"]) for t in stairs_tiers}) != 1:
        raise ValueError("official stairs tiers do not share step/landing shape")
    stairs = {
        "levels": stairs_tiers,
        "total_height_range_m": [stairs_tiers[0]["total_height_m"],
                                 stairs_tiers[-1]["total_height_m"]],
        "num_steps": stairs_tiers[0]["num_steps"],
        "tread_m": stairs_tiers[0]["tread_m"],
        "half_width_m": stairs_tiers[0]["half_width_m"],
        "landing_half_x_m": stairs_tiers[0]["landing_half_x_m"],
        "start_x_m": RUNWAY_M,
    }

    # ---- plum: rebuilt from the rules-PDF figure (NOT the official map) ------
    # The map's plum field (ramps + cubes in two rows) is the one the PDF says has
    # wrong parameters; ``gen_plum.py`` is the PDF-figure geometry source.
    plum = gen_plum.plum_section()
    if not plum.get("no_ramps"):
        raise ValueError("plum section must be the no-ramp PDF layout")
    if abs(plum["start_x_m"] - RUNWAY_M) > 1e-9:
        raise ValueError(
            f"plum entry platform must start at the {RUNWAY_M} m runway, "
            f"got {plum['start_x_m']}")

    spec = {
        "schema_version": 1,
        "source": {
            "file": map_path.name,
            "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            "bytes": map_path.stat().st_size,
        },
        "protocol": {
            "runway_m": RUNWAY_M,
            "lane_half_width_m": LANE_HALF_WIDTH,
            "spawn_trunk_z_m": SPAWN_Z,
            "note": "Judge scene convention: obstacle near face sits runway_m ahead of the spawn; the partial_difficulty rows interpolate between the official tier endpoints.",
        },
        "categories": {
            "high_platform": high_platform,
            "slope": slope,
            "trench": trench,
            "high_wall": high_wall,
            "stairs": stairs,
            "plum": plum,
        },
    }
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP,
                        help=f"official terrain/map.xml (default: {DEFAULT_MAP})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output spec json (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    spec = build_spec(args.map)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")
    for name, cat in spec["categories"].items():
        print(f"  {name}: {json.dumps(cat, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
