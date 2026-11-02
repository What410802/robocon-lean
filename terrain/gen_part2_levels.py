#!/usr/bin/env python3
"""Generate the **official** Part 2 capability levels from the official ability field.

The organiser's ``terrain.zip`` contains ``map.xml``: the official Part 2 ability
field (one row per obstacle type, one column per difficulty).  The competition
PDF allows a custom map as long as the obstacle *types* match the provided
field, so local evaluation uses one isolated scene per official difficulty
level: the official geom definitions are copied verbatim from ``map.xml`` (same
sizes, same mesh vertices) and only translated so that each obstacle starts
1.8 m ahead of the spawn point.

Only the official levels are generated (no ``plum_reg_*`` or other non-official
entries):

    high_platform 10/20/30/40 cm, stairs 10/15/20/30 cm, slope 20/30/35/45 deg,
    high_wall 10/20/30/40 cm, trench 10/20/30 cm, plum 1 (3-pile layout) = 20.

The ``plum`` obstacle uses the team owner's no-ramp 3-pile geometry from
``terrain/gen_plum.py`` (single source) instead of the old map plum field.

Outputs (relative to the repository root)::

    demo/scenes/part2/generated/part2_off_<key>.xml    level scenes (MuJoCo)
    demo/scenes/part2/part2_official_levels.json       runtime manifest

Scene robots: every scene includes
``../../../robots/blackW/blackW_description.xml`` (relative to the generated/
directory, i.e. ``demo/robots/blackW/blackW_description.xml``).  MuJoCo resolves
the included model's ``meshdir="assets"`` against the *main scene file's*
directory, so the generator also ensures ``demo/scenes/part2/generated/assets``
is a symlink to ``demo/robots/blackW/assets``.

``seconds`` is the per-level runtime safety cap only (not a scoring time limit).

Usage (from anywhere; numpy is the only third-party dependency)::

    python terrain/gen_part2_levels.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

import gen_plum

ROOT = Path(__file__).resolve().parent.parent  # repository root
MAP_XML = ROOT / "terrain" / "official" / "map.xml"
SCENE_DIR = ROOT / "demo" / "scenes" / "part2" / "generated"
OUT_JSON = ROOT / "demo" / "scenes" / "part2" / "part2_official_levels.json"

# Include path relative to the generated scene directory
# (demo/scenes/part2/generated -> ../../../ = demo/, where robots/ lives).
ROBOT_INCLUDE = "../../../robots/blackW/blackW_description.xml"
# MuJoCo resolves the included model's `meshdir="assets"` against the main scene
# file's directory, so the generated/ directory needs an assets symlink.
ROBOT_ASSETS_LINK = "../../../robots/blackW/assets"

RUNWAY_M = 1.8          # obstacle near face stays this far ahead of the spawn
SPAWN_RAW_Z = 0.675     # sim_node start:=raw (straight legs) wheel-bottom ~0
SPAWN_DEFAULT_Z = 0.550  # default pose (used by the eval runner)
SPAWN_Z = SPAWN_RAW_Z   # launch consumes `spawn` (raw start)
BASE_REF_Z = 0.50       # judge: trunk height on flat ground
# Per-level runtime safety cap only; the judge decides success, not the clock.
SECONDS = 120.0

_MESHES: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Official map.xml parsing (identical to the migration reference implementation)
# ---------------------------------------------------------------------------
def _parse_attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', tag))


def _parse_floats(text: str) -> np.ndarray:
    return np.fromstring(text.replace(",", " "), sep=" ")


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _parse_xml() -> tuple[list[str], list[str]]:
    text = MAP_XML.read_text(encoding="utf-8")
    asset = re.search(r"<asset>(.*?)</asset>", text, re.S).group(1)
    meshes = re.findall(r"<mesh [^>]*/>", asset)
    for m in meshes:
        attrs = _parse_attrs(m)
        if "vertex" in attrs:
            _MESHES[attrs["name"]] = _parse_floats(attrs["vertex"]).reshape(-1, 3)
    return meshes, re.findall(r"<geom [^>]*/>", text)


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
        raise ValueError(f"unsupported geom type in official map: {a['type']}")
    return pos - half, pos + half


def _select_geoms(geom_lines: list[str], prefixes: tuple[str, ...]) -> list[str]:
    out = []
    for g in geom_lines:
        name = _parse_attrs(g)["name"]
        if any(name == p or name.startswith(p) for p in prefixes):
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Official level list (20 levels, one per official difficulty)
# ---------------------------------------------------------------------------
def _specs() -> list[dict]:
    """One entry per *official* difficulty level (no non-official rows)."""
    specs: list[dict] = []
    for h in (10, 20, 30, 40):
        specs.append({"category": "high_platform", "level": h, "unit": "cm",
                      "prefixes": (f"platform_{h:02d}cm",), "kind": "platform"})
    for h in (10, 15, 20, 30):
        specs.append({"category": "stairs", "level": h, "unit": "cm",
                      "prefixes": (f"stairs_{h:02d}cm_",), "kind": "stairs"})
    for deg in (20, 30, 35, 45):
        specs.append({"category": "slope", "level": deg, "unit": "deg",
                      "prefixes": (f"slope_{deg:02d}deg_",), "kind": "slope"})
    # cross_bar_1..8 = 4/8/12/16/20/30/40/50 cm tall, 1.4 m x 0.12 m.
    wall_field = {20: "cross_bar_5", 30: "cross_bar_6", 40: "cross_bar_7"}
    for h in (10, 20, 30, 40):
        specs.append({"category": "high_wall", "level": h, "unit": "cm",
                      "prefixes": (wall_field.get(h, "cross_bar_3"),), "kind": "wall",
                      "override_h": 0.10 if h == 10 else None})
    for w in (10, 20, 30):
        specs.append({"category": "trench", "level": w, "unit": "cm",
                      "prefixes": (f"trench_{w:02d}cm_",), "kind": "gap"})
    # Plum: owner-confirmed no-ramp 3-pile layout (terrain/gen_plum.py), not the
    # map's old ramp/staggered-row field.
    specs.append({"category": "plum", "level": 1, "unit": "bool",
                  "prefixes": (), "kind": "plum"})
    return specs


def _override_wall_height(geom_lines: list[str], height: float) -> list[str]:
    out = []
    for g in geom_lines:
        a = _parse_attrs(g)
        size = _parse_floats(a["size"])
        size[2] = height / 2.0
        pos = _parse_floats(a["pos"])
        pos[2] = height / 2.0
        g = re.sub(r'pos="[^"]*"', f'pos="{pos[0]} {pos[1]} {pos[2]:.6f}"', g)
        g = re.sub(r'size="[^"]*"', f'size="{size[0]} {size[1]} {size[2]:.6f}"', g)
        out.append(g)
    return out


def _rel(name: str, geoms: list[str]) -> np.ndarray:
    """World-frame bbox min of the first geom whose name contains ``name``."""
    for g in geoms:
        if name in _parse_attrs(g)["name"]:
            return _geom_bbox(g)[0]
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Judge dicts (training/judge contract; mirrors the eval judge schema)
# ---------------------------------------------------------------------------
def _judge(kind: str, geoms: list[str], bmin: np.ndarray, bmax: np.ndarray,
           override_h: float | None) -> dict:
    lane = {"lane_half_width": 0.7, "hold_s": 0.20}
    if kind == "platform":
        return {"kind": "platform", "top_z": float(bmax[2]),
                "x_range": [round(float(bmin[0] + 0.20), 4),
                            round(float(bmax[0] - 0.05), 4)],
                "z_margin": 0.35,
                "exit_x_min": round(float(bmax[0] - 0.05) + 0.10, 4),
                "exit_ground_tol": 0.08, **lane}
    if kind == "stairs":
        # The official stairs obstacle is a full staircase: 6 up steps + 0.8 m
        # landing + 6 down steps.  The judge needs the far end of the down flight:
        # "通过台阶" means climbing up AND descending back to the ground.
        return {"kind": "stairs", "top_z": float(bmax[2]),
                "goal_x_min": round(float(_rel("landing", geoms)[0] + 0.10), 4),
                "descend_end_x": round(float(bmax[0]), 4),
                "normal_z_min": 0.30, "normal_z_max": 0.80,
                "z_margin": 0.35, **lane}
    if kind == "slope":
        return {"kind": "slope", "rise_z": float(bmax[2]),
                "goal_x_min": round(float(_rel("landing", geoms)[0] + 0.10), 4),
                "exit_x_min": round(float(bmax[0]) - 0.10, 4),
                "exit_ground_tol": 0.08,
                "normal_z_min": 0.30, "normal_z_max": 0.80,
                "z_margin": 0.35, **lane}
    if kind == "wall":
        h = float(bmax[2]) if override_h is None else override_h
        return {"kind": "wall", "wall_h": h,
                "wall_x": round(float((bmin[0] + bmax[0]) / 2.0), 4),
                "goal_x_min": round(float(bmax[0] + 0.35), 4),
                "climb_z_min": round(BASE_REF_Z + 0.20 * h, 4), **lane}
    if kind == "gap":
        nb_max = None
        fb_min = None
        for g in geoms:
            n = _parse_attrs(g)["name"]
            if n.endswith("_near"):
                nb_max = _geom_bbox(g)[1]
            elif n.endswith("_far"):
                fb_min = _geom_bbox(g)[0]
        assert nb_max is not None and fb_min is not None
        plateau = float(nb_max[2])
        return {"kind": "gap",
                "gap_x": round(float((nb_max[0] + fb_min[0]) / 2.0), 4),
                "width": round(float(fb_min[0] - nb_max[0]), 4),
                "plateau_z": plateau,
                "near_x": round(float(nb_max[0]), 4),
                "far_x": round(float(fb_min[0]), 4),
                "z_cross_min": round(plateau + 0.28, 4),
                "gap_max_drop_m": round(min(0.5 * plateau, 0.103), 4),
                # v7.5 traverse semantics: the official trench continues with a
                # 1 m exit ramp down to normal ground; the terminal state requires
                # descending it (all four wheels back on the ground past ramp end).
                "exit_ramp_end": round(float(bmax[0]), 4),
                "exit_ground_tol": 0.08, **lane}
    raise ValueError(kind)


def _plum_judge() -> dict:
    """Judge contract for the owner-confirmed no-ramp 3-pile plum layout.

    ``exit_ramp_end`` keeps its old key name (judge compatibility): with no ramp
    it is simply the far edge of the 0.2 m exit platform (x=5.0 m); success
    requires stepping down to normal ground past that edge.
    """
    spec = gen_plum.plum_section()
    cube_half = spec["cube_half_m"]
    centers = spec["cube_centers_x_m"]
    first_left = float(centers[0]) - cube_half
    last_right = float(centers[-1]) + cube_half
    return {
        "kind": "plum",
        "top_z": float(spec["cube_top_m"]),
        # x span covering the 3 cubes (+/- 0.15 m tolerance, same convention as
        # the previous official plum level).
        "cube_x_range": [round(first_left - 0.15, 4), round(last_right + 0.15, 4)],
        # Exit platform near edge: reaching it is evidence of a full crossing.
        "goal_x_min": round(float(spec["exit_platform_x_range_m"][0]), 4),
        "z_margin": 0.35,
        "z_cross_min": 0.05,
        "lane_half_width": 0.7,
        "hold_s": 0.20,
        # No ramp: the exit platform far edge IS the step-down edge (old key name).
        "exit_ramp_end": round(float(spec["exit_platform_x_range_m"][1]), 4),
        "exit_ground_tol": 0.08,
    }


# ---------------------------------------------------------------------------
# Scene XML
# ---------------------------------------------------------------------------
def _scene_text(geoms: list[str], meshes: list[str], dx: float, dy: float) -> str:
    used = {_parse_attrs(g).get("mesh") for g in geoms}
    mesh_block = "\n".join("    " + m for m in meshes
                           if _parse_attrs(m)["name"] in used)
    obs_block = "\n".join("      " + g for g in geoms)
    return (
        '<mujoco model="part2_official_level">\n'
        f'  <include file="{ROBOT_INCLUDE}"/>\n'
        '  <statistic center="2.0 0 0.5" extent="3.0"/>\n'
        '  <visual><global azimuth="90" elevation="-20"/></visual>\n'
        '  <asset>\n'
        f'{mesh_block}\n'
        '  </asset>\n'
        '  <worldbody>\n'
        '    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="level_floor" type="plane" size="0 0 0.05"'
        ' rgba="0.20 0.20 0.20 1"/>\n'
        '    <body name="obstacle" pos="{} {} 0">\n'.format(round(dx, 6), round(dy, 6))
        + obs_block + "\n"
        '    </body>\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    )


def _ensure_asset_link() -> None:
    """Create the generated/assets -> robot assets symlink MuJoCo needs."""
    link = SCENE_DIR / "assets"
    if link.is_symlink() or link.exists():
        if not link.is_symlink():
            raise RuntimeError(
                f"{link} exists but is not a symlink; expected the blackW assets "
                f"symlink ({ROBOT_ASSETS_LINK})")
        return
    link.symlink_to(ROBOT_ASSETS_LINK)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _plum_geometry(dx: float, dy: float) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:
    geoms = gen_plum.mjcf_geoms(dx=dx, dy=dy, indent="")
    boxes = []
    for box in gen_plum.boxes(dx=dx, dy=dy):
        c = np.asarray(box["center_m"], dtype=float)
        h = np.asarray(box["half_size_m"], dtype=float)
        boxes.append((c - h, c + h))
    return geoms, boxes


def _shift_judge(judge: dict, dx: float, dy: float) -> dict:
    """Translate x/y-valued judge fields produced in the untranslated frame."""
    out = dict(judge)
    for k in ("goal_x_min", "wall_x", "near_x", "far_x", "gap_x", "descend_end_x",
              "exit_ramp_end", "exit_x_min"):
        if k in out:
            out[k] = round(float(out[k]) + dx, 4)
    if "x_range" in out:
        out["x_range"] = [round(float(out["x_range"][0]) + dx, 4),
                          round(float(out["x_range"][1]) + dx, 4)]
    if "cube_x_range" in out:
        out["cube_x_range"] = [round(float(out["cube_x_range"][0]) + dx, 4),
                               round(float(out["cube_x_range"][1]) + dx, 4)]
    return out


def build() -> dict:
    meshes, geom_lines = _parse_xml()
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_asset_link()

    levels = []
    produced_scenes: set[str] = set()
    for spec in _specs():
        if spec["kind"] == "plum":
            # First pass: place at dx=0 to compute the translation, then rebuild.
            _, boxes0 = _plum_geometry(0.0, 0.0)
            bmin = np.min([b[0] for b in boxes0], axis=0)
            bmax = np.max([b[1] for b in boxes0], axis=0)
            dx = RUNWAY_M - float(bmin[0])
            dy = -float(bmin[1] + bmax[1]) / 2.0
            geoms, boxes = _plum_geometry(dx, dy)
            bmin = np.min([b[0] for b in boxes], axis=0)
            bmax = np.max([b[1] for b in boxes], axis=0)
            judge = _plum_judge()
            source = ("owner-confirmed 3-pile plum layout (terrain/gen_plum.py): "
                      + ", ".join(_parse_attrs(g)["name"] for g in geoms))
        else:
            geoms = _select_geoms(geom_lines, spec["prefixes"])
            if not geoms:
                raise RuntimeError(f"no official geoms matched {spec['prefixes']}")
            if spec.get("override_h") is not None:
                geoms = _override_wall_height(geoms, spec["override_h"])
            boxes = [_geom_bbox(g) for g in geoms]
            bmin = np.min([b[0] for b in boxes], axis=0)
            bmax = np.max([b[1] for b in boxes], axis=0)
            dx = RUNWAY_M - float(bmin[0])
            dy = -float(bmin[1] + bmax[1]) / 2.0
            judge = _judge(spec["kind"], geoms, bmin, bmax, spec.get("override_h"))
            source = ("official terrain.zip map.xml: "
                      + ", ".join(_parse_attrs(g)["name"] for g in geoms[:3])
                      + ("..." if len(geoms) > 3 else ""))
        # All judge fields are computed in the untranslated frame, then shifted by
        # (dx, dy) into the generated scene's world frame.
        judge = _shift_judge(judge, dx, dy)
        key = (f"{spec['category']}_1" if spec["unit"] == "bool"
               else f"{spec['category']}_{spec['level']}{spec['unit']}")
        scene_name = f"part2_off_{key}.xml"
        (SCENE_DIR / scene_name).write_text(
            _scene_text(geoms, meshes, dx, dy), encoding="utf-8")
        produced_scenes.add(scene_name)
        levels.append({
            "key": key,
            "category": spec["category"],
            "level": spec["level"],
            "unit": spec["unit"],
            "scene": scene_name,
            "source": source,
            "obstacle_bbox": {"min": [round(float(v), 4) for v in bmin + [dx, dy, 0]],
                              "max": [round(float(v), 4) for v in bmax + [dx, dy, 0]]},
            "spawn": [0.0, 0.0, SPAWN_Z, 0.0],
            "spawn_raw": [0.0, 0.0, SPAWN_RAW_Z, 0.0],
            "spawn_default_pose": [0.0, 0.0, SPAWN_DEFAULT_Z, 0.0],
            "command": [0.5, 0.0, 0.0],
            # Runtime safety cap only; the judge, not the clock, decides success.
            "seconds": SECONDS,
            "judge": judge,
        })

    # Remove stale scene files from previous (non-official) manifests so the
    # generated directory only contains the current 20 levels + the assets link.
    for stale in SCENE_DIR.glob("part2_off_*.xml"):
        if stale.name not in produced_scenes:
            stale.unlink()
    # The pre-migration generator wrote the manifest inside generated/; the
    # runtime manifest now lives next to it (demo/scenes/part2/), so remove the
    # legacy copy if present to avoid two sources.
    legacy_manifest = SCENE_DIR / "part2_official_levels.json"
    if legacy_manifest.is_file():
        legacy_manifest.unlink()

    payload = {
        "schema_version": 2,
        "source_map": str(MAP_XML.relative_to(ROOT)),
        "levels": levels,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    expected = {"high_platform": 4, "stairs": 4, "slope": 4, "high_wall": 4,
                "trench": 3, "plum": 1}
    got: dict[str, int] = {}
    for level in levels:
        got[level["category"]] = got.get(level["category"], 0) + 1
    if got != expected or len(levels) != sum(expected.values()):
        raise RuntimeError(f"official level counts mismatch: got {got}, want {expected}")
    if any(level["seconds"] != SECONDS for level in levels):
        raise RuntimeError("every level must use the safety-cap seconds value")

    print(f"wrote {len(levels)} official levels -> {OUT_JSON}")
    print(f"scene dir: {SCENE_DIR}")
    for level in levels:
        print(f"  {level['key']:22s} {level['scene']:40s} seconds={level['seconds']:.1f}"
              f"  judge={json.dumps(level['judge'], ensure_ascii=False)}")
    return payload


if __name__ == "__main__":
    build()
