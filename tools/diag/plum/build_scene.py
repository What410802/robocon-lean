#!/usr/bin/env python3
"""Build the diagnostic *competition-plum* Part 2 level (scene + levels JSON).

Why this exists
---------------
The competition PDF plum field (梅花桩) is the 6-cube staggered two-row layout
that the team corrected inside the full-field map
`demo/scenes/part2/full_field/map_fixed.xml` (row y=12).  The eval-side
generated level `demo/scenes/part2/generated/part2_off_plum_1.xml` is STALE
(3-pile single-row, owner-confirmed 2026-11-02 geometry) and must not be used
as the competition-plum oracle.  This script derives an isolated diagnostic
level from the *corrected map geometry* so blind/deployable experts can be
benchmarked against it.  It is additive: it only writes under tools/diag/plum/.

Frame convention
----------------
Like every generated Part 2 level, the obstacle is translated so the lane is
centred on y=0 and the obstacle starts 1.8 m ahead of the spawn (RUNWAY_M).
`map_fixed.xml` already has the plum row at lane centre y=12 and the entry
platform at x=4.0..4.8, so the translation is (dx, dy) = (-2.2, -12.0):

    entry platform : x 1.8 .. 2.6,  y -0.4 .. 0.4, top z = 0.2
    row A (y=+0.2) : cubes x 2.8, 3.4, 4.0   (same as map y=12.2)
    row B (y=-0.2) : cubes x 3.1, 3.7, 4.3   (same as map y=11.8)
    exit platform  : x 4.5 .. 5.3,  y -0.4 .. 0.4, top z = 0.2

Relative geometry is identical to the old-repo regulation plum `plum_reg_1`
(entry 0..0.8, cubes row1 x=1.0/1.6/2.2 y=-0.2, row2 x=1.3/1.9/2.5 y=+0.2,
exit 2.7..3.5) shifted by +1.8 m; the map geoms are the source of truth.

Two level entries are written, sharing one scene and one judge block:

* `plum_comp_judge_1` -- judge-style spawn ON the entry platform centre
  (spawn_default_pose [2.2, 0, 0.75, 0]: 0.2 m platform + nominal 0.55 m base
  height, same convention as old-repo `plum_reg_1` spawn [0.4, 0, 0.875, 0]).
* `plum_comp_judge_mirror_1` -- the SAME competition geometry mirrored in y
  (row A on y=-0.2).  This is *not* the competition oracle: it is the old-repo
  `plum_reg_1` relative geometry (the regulation level is the y-mirror of the
  corrected map row) and exists only to check whether the old depth-CNN
  student, which mastered `plum_reg_1`, reproduces that success in the lean
  eval runner / on this diagnostic level.
* `plum_comp_field_1` -- full-field route-style spawn ON THE GROUND 1.8 m
  before the entry platform near face (spawn_default_pose [0, 0, 0.55, 0]);
  the policy must step up the 0.2 m entry platform first.

The levels manifest is selected in eval runs through
``ROBOCON_EVAL_LEVELS=<this dir>/plum_diag_levels.json`` (documented hook in
``eval/robocon_eval/part2/official.py``); eval/ and demo/ are not modified.

Usage (from the repo root):

    pixi run -e eval python tools/diag/plum/build_scene.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # repository root
MAP_XML = ROOT / "demo/scenes/part2/full_field/map_fixed.xml"
OUT_DIR = Path(__file__).resolve().parent
SCENE_NAME = "plum_comp_diag.xml"
LEVELS_NAME = "plum_diag_levels.json"
ROBOT_INCLUDE = "../../../demo/robots/blackW/blackW_description.xml"
ROBOT_ASSETS_LINK = "../../../demo/robots/blackW/assets"

RUNWAY_M = 1.8        # Part 2 protocol runway: obstacle starts 1.8 m ahead
PLUM_LANE_Y_M = 12.0  # map lane centre (plum row y=12)

# Expected post-translation geometry (self-check; derived from the PDF).
EXPECTED = {
    "entry": {"min": (1.8, -0.4, 0.0), "max": (2.6, 0.4, 0.2)},
    "cubes": [
        (2.8, 0.2, 0.2), (3.4, 0.2, 0.2), (4.0, 0.2, 0.2),
        (3.1, -0.2, 0.2), (3.7, -0.2, 0.2), (4.3, -0.2, 0.2),
    ],   # (x_centre, y_centre, top_z)
    "exit": {"min": (4.5, -0.4, 0.0), "max": (5.3, 0.4, 0.2)},
}
# Old-repo regulation level shifted by +1.8 m; kept as a regression reference.
OLD_REG_PLUM_JUDGE = {
    "cube_x_range": [0.75, 2.75],
    "goal_x_min": 2.7,
    "exit_ramp_end": 3.5,
}
JUDGE_SHIFT_M = 1.8

_GEOM_RE = re.compile(r"^\s*(<geom\b[^>]*/>)\s*$")


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf'{name}="([^"]*)"', tag)
    return m.group(1) if m else None


def _plum_geoms() -> list[str]:
    """Extract only the competition plum row geoms from the corrected map.

    The map also contains the 2/3/4/5-pile *incremental* plum sections
    (`plum_t2_*` ... `plum_t5_*`); those are NOT part of the competition field.
    """
    wanted = re.compile(r"plum_(entry|exit|r1_0[1-9]|r2_0[1-9])$")
    out = []
    for line in MAP_XML.read_text(encoding="utf-8").splitlines():
        m = _GEOM_RE.match(line)
        if not m:
            continue
        tag = m.group(1)
        name = _attr(tag, "name")
        if name and wanted.match(name):
            out.append(tag)
    return out


def _translate(tag: str, dx: float, dy: float) -> str:
    pos = _attr(tag, "pos")
    if pos is None:                       # default pos = 0 0 0
        return tag.replace("<geom", f'<geom pos="{dx:.6g} {dy:.6g} 0"', 1)
    vals = [float(v) for v in pos.split()]
    vals[0] += dx
    vals[1] += dy
    new = " ".join(f"{v:.6g}" for v in vals)
    return re.sub(r'pos="[^"]*"', f'pos="{new}"', tag, count=1)


def _box_aabb(tag: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([float(v) for v in (_attr(tag, "pos") or "0 0 0").split()])
    size = np.array([float(v) for v in _attr(tag, "size").split()])
    assert _attr(tag, "type") == "box", tag
    return pos - size, pos + size


SCENE_NAME_MIRROR = "plum_comp_diag_mirror.xml"


def _mirror_y(geom_lines: list[str]) -> list[str]:
    """Negate the y position of every obstacle geom (old plum_reg_1 orientation)."""
    out = []
    for tag in geom_lines:
        vals = [float(v) for v in (_attr(tag, "pos") or "0 0 0").split()]
        vals[1] = -vals[1]
        new = " ".join(f"{v:.6g}" for v in vals)
        out.append(re.sub(r'pos="[^"]*"', f'pos="{new}"', tag, count=1))
    return out


def _scene_text(geom_lines: list[str], model_name: str = "part2_diag_competition_plum_level",
                note: str = "") -> str:
    obs_block = "\n".join("      " + g for g in geom_lines)
    return (
        f'<mujoco model="{model_name}">\n'
        "  <!-- DIAGNOSTIC competition-plum level (梅花桩), derived from the\n"
        "       corrected full-field map demo/scenes/part2/full_field/map_fixed.xml\n"
        "       row y=12 (competition PDF p4): entry/exit platforms 800x800x200 mm,\n"
        "       6 x 200 mm cubes staggered in two rows.  Translation dx=-2.2 m\n"
        "       (RUNWAY 1.8 m) dy=-12 m (lane-centred y=0).  Generated by\n"
        "       tools/diag/plum/build_scene.py -- do not edit by hand.\n"
        f"       {note} -->\n"
        f'  <include file="{ROBOT_INCLUDE}"/>\n'
        '  <statistic center="2.6 0 0.3" extent="3.0"/>\n'
        '  <visual><global azimuth="90" elevation="-20"/></visual>\n'
        "  <worldbody>\n"
        '    <light pos="0 0 3" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="level_floor" type="plane" size="0 0 0.05"'
        ' rgba="0.20 0.20 0.20 1"/>\n'
        '    <body name="obstacle" pos="0 0 0">\n'
        f"{obs_block}\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


def _judge(entry_x_max: float, exit_x_min: float, exit_x_max: float) -> dict:
    """Plum judge block, same parameterisation as old-repo plum_reg_1.

    Historical key names are kept: ``exit_ramp_end`` is the exit-platform far
    edge (there is no ramp in the competition geometry) and success requires
    stepping down onto normal ground past it.  ``cube_x_range`` spans the pile
    field window: entry-platform far edge (-5 cm) .. exit-platform near edge
    (+5 cm), exactly the old plum_reg_1 convention shifted by +1.8 m.
    """
    return {
        "kind": "plum",
        "top_z": 0.2,
        "cube_x_range": [round(float(entry_x_max) - 0.05, 4),
                         round(float(exit_x_min) + 0.05, 4)],
        "goal_x_min": round(float(exit_x_min), 4),
        "z_margin": 0.35,
        "z_cross_min": 0.05,
        "lane_half_width": 0.7,
        "hold_s": 0.2,
        "exit_ramp_end": round(float(exit_x_max), 4),
        "exit_ground_tol": 0.08,
    }


def _level_entry(key: str, spawn_default: list[float], spawn_raw: list[float],
                 judge: dict, source: str) -> dict:
    return {
        "key": key,
        "category": "plum",
        "level": 1,
        "unit": "bool",
        "scene": SCENE_NAME,
        "source": source,
        "official": False,
        "obstacle_bbox": {"min": [1.8, -0.4, 0.0], "max": [5.3, 0.4, 0.2]},
        "spawn": spawn_raw,
        "spawn_raw": spawn_raw,
        "spawn_default_pose": spawn_default,
        "command": [0.5, 0.0, 0.0],
        "seconds": 60.0,
        "judge": judge,
    }


def _ensure_asset_link() -> None:
    link = OUT_DIR / "assets"
    if link.is_symlink() or link.exists():
        return
    link.symlink_to(ROBOT_ASSETS_LINK)


def build() -> dict:
    geom_tags = _plum_geoms()
    if len(geom_tags) != 8:
        raise RuntimeError(
            f"expected 8 competition plum geoms in {MAP_XML} (entry, exit, "
            f"6 cubes), found {len(geom_tags)}: "
            f"{[_attr(g, 'name') for g in geom_tags]}")
    names = [_attr(g, "name") for g in geom_tags]
    aabbs = {n: _box_aabb(g) for n, g in zip(names, geom_tags)}
    bmin = np.min([a[0] for a in aabbs.values()], axis=0)
    bmax = np.max([a[1] for a in aabbs.values()], axis=0)
    dx = RUNWAY_M - float(bmin[0])
    dy = -0.5 * float(bmin[1] + bmax[1])
    if (abs(dx + 2.2) > 1e-9) or (abs(dy + PLUM_LANE_Y_M) > 1e-9):
        raise RuntimeError(
            f"unexpected map translation dx={dx} dy={dy}; the corrected map "
            "geometry changed -- review the diagnostic level")
    translated = [_translate(g, dx, dy) for g in geom_tags]

    # -- self-check the translated boxes against the PDF-derived expectation --
    t = {n: _box_aabb(g) for n, g in zip(names, translated)}
    entry_lo, entry_hi = t["plum_entry"]
    exit_lo, exit_hi = t["plum_exit"]
    np.testing.assert_allclose(entry_lo, EXPECTED["entry"]["min"], atol=1e-9)
    np.testing.assert_allclose(entry_hi, EXPECTED["entry"]["max"], atol=1e-9)
    np.testing.assert_allclose(exit_lo, EXPECTED["exit"]["min"], atol=1e-9)
    np.testing.assert_allclose(exit_hi, EXPECTED["exit"]["max"], atol=1e-9)
    cubes = []
    for n in sorted(names):
        if "r1_" in n or "r2_" in n:
            lo, hi = t[n]
            cubes.append((float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2),
                          float(hi[2])))
    for got, want in zip(sorted(cubes), sorted(EXPECTED["cubes"])):
        np.testing.assert_allclose(got, want, atol=1e-9)

    judge = _judge(entry_hi[0], exit_lo[0], exit_hi[0])
    shifted_old = {
        k: ([round(x + JUDGE_SHIFT_M, 4) for x in v]
            if isinstance(v, (list, tuple)) else round(v + JUDGE_SHIFT_M, 4))
        for k, v in OLD_REG_PLUM_JUDGE.items()
    }
    expected_shifted = {
        "cube_x_range": judge["cube_x_range"],
        "goal_x_min": judge["goal_x_min"],
        "exit_ramp_end": judge["exit_ramp_end"],
    }
    if shifted_old != expected_shifted:
        raise RuntimeError(
            f"judge shift check failed: {shifted_old} != {expected_shifted}")

    source_common = (
        "competition PDF p4 plum (梅花桩): 6 x 200 mm cubes, two staggered "
        "rows (row A y=+0.2 x=2.8/3.4/4.0, row B y=-0.2 x=3.1/3.7/4.3), "
        "800x800x200 mm entry/exit platforms, no ramps; geometry derived from "
        "demo/scenes/part2/full_field/map_fixed.xml row y=12 by dx=-2.2 dy=-12 "
        "(tools/diag/plum/build_scene.py). EXTRA diagnostic level, not official.")
    levels = {
        "schema_version": 2,
        "source_map": "demo/scenes/part2/full_field/map_fixed.xml (plum row y=12)",
        "diagnostic": True,
        "levels": [
            _level_entry(
                "plum_comp_judge_1",
                spawn_default=[2.2, 0.0, 0.75, 0.0],
                spawn_raw=[2.2, 0.0, 0.875, 0.0],
                judge=judge,
                source=source_common + " Spawn ON the entry platform (judge-style)."),
            _level_entry(
                "plum_comp_field_1",
                spawn_default=[0.0, 0.0, 0.55, 0.0],
                spawn_raw=[0.0, 0.0, 0.675, 0.0],
                judge=judge,
                source=source_common + " Spawn on the ground 1.8 m before the "
                                      "entry platform (full-field route-style; "
                                      "must step up 0.2 m)."),
        ],
    }

    _ensure_asset_link()
    scene_path = OUT_DIR / SCENE_NAME
    scene_path.write_text(_scene_text(translated), encoding="utf-8")
    mirror_path = OUT_DIR / SCENE_NAME_MIRROR
    mirror_path.write_text(
        _scene_text(_mirror_y(translated),
                    model_name="part2_diag_competition_plum_mirror_level",
                    note="MIRROR of the competition row in y: this is the "
                         "old-repo plum_reg_1 relative geometry, used only as a "
                         "depth-student reproduction oracle."),
        encoding="utf-8")
    mirror_entry = _level_entry(
        "plum_comp_judge_mirror_1",
        spawn_default=[2.2, 0.0, 0.75, 0.0],
        spawn_raw=[2.2, 0.0, 0.875, 0.0],
        judge=judge,
        source=source_common + " MIRROR in y (row A y=-0.2): equals the old-repo "
                              "plum_reg_1 relative geometry. NOT the competition "
                              "oracle; depth-student reproduction probe only.")
    mirror_entry["scene"] = SCENE_NAME_MIRROR
    levels["levels"].append(mirror_entry)
    levels_path = OUT_DIR / LEVELS_NAME
    levels_path.write_text(
        json.dumps(levels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"scene": scene_path, "levels": levels_path, "judge": judge,
            "translation": [dx, dy], "geoms": names}


def verify() -> dict:
    """Compile the scene with MuJoCo and check the collision AABBs."""
    import mujoco

    out = build()
    model = mujoco.MjModel.from_xml_path(str(out["scene"]))
    got = {}
    for name in out["geoms"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"missing geom {name}"
        got[name] = (np.array(model.geom_pos[gid]), np.array(model.geom_size[gid]))
    entry = got["plum_entry"]
    np.testing.assert_allclose(entry[0], [2.2, 0.0, 0.1], atol=1e-5)
    np.testing.assert_allclose(entry[1], [0.4, 0.4, 0.1], atol=1e-5)
    exit_ = got["plum_exit"]
    np.testing.assert_allclose(exit_[0], [4.9, 0.0, 0.1], atol=1e-5)
    r1 = got["plum_r1_02"]
    np.testing.assert_allclose(r1[0], [3.4, 0.2, 0.1], atol=1e-5)
    r2 = got["plum_r2_03"]
    np.testing.assert_allclose(r2[0], [4.3, -0.2, 0.1], atol=1e-5)
    return {"model_nq": int(model.nq), "ngeom": int(model.ngeom), "ok": True}


if __name__ == "__main__":
    info = verify()
    out = build()
    print(f"wrote {out['scene'].relative_to(ROOT)}")
    print(f"wrote {out['levels'].relative_to(ROOT)} (3 entries: "
          "plum_comp_judge_1, plum_comp_field_1, plum_comp_judge_mirror_1)")
    print(f"translation dx={out['translation'][0]:.6g} dy={out['translation'][1]:.6g}")
    print(f"judge: {json.dumps(out['judge'])}")
    print(f"MuJoCo compile ok: nq={info['model_nq']} ngeom={info['ngeom']}")
