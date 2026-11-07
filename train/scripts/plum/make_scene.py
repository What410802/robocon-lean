#!/usr/bin/env python3
"""Write a replay scene matching a curriculum row of the plum training terrain.

``record_rollouts.py`` renders rollouts against a MuJoCo scene, and the rollout
itself ran on the training terrain for a given curriculum row.  Rendering an easy
row against the official (difficulty 1.0) scene would show barriers that were not
in the simulation, so the scene is rebuilt with the *same* ``function`` the
terrain generator calls -- the row is generated in a scratch spec, then its geoms
are copied into the official scene's ``obstacle`` body at the spawn-relative
offset.

    pixi run -e train python train/scripts/plum/make_scene.py --difficulty 0.4 \
        --out artifacts/plum/videos/plum_d0.4.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train" / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from alldog_mjlab.tasks.velocity.profiles.plum.terrain import (  # noqa: E402
    PlumCurriculumTerrainCfg,
)

OFFICIAL = ROOT / "demo/scenes/part2/generated/part2_off_plum_1.xml"
_COLOR = (0.25, 0.55, 0.9, 1.0)
_FRICTION = (1.0, 0.005, 0.0001)


def build_scene(difficulty: float, out: Path) -> Path:
    """Official scene with the plum obstacle replaced by one curriculum row."""
    t = PlumCurriculumTerrainCfg()

    # Generate the row exactly as the trainer does, in a scratch spec.
    scratch = mujoco.MjSpec()
    src = scratch.worldbody.add_body(name="terrain")
    t.function(difficulty, scratch, np.random.default_rng(0))
    x_off = t.size[0] / 2          # patch frame -> spawn-relative scene frame
    y_off = t.size[1] / 2

    spec = mujoco.MjSpec.from_file(str(OFFICIAL))
    obstacle = spec.body("obstacle")
    for geom in list(obstacle.geoms):
        spec.delete(geom)
    for i, g in enumerate(src.geoms):
        if g.name.startswith("floor"):
            continue               # the official scene has its own ground plane
        n = obstacle.add_geom()
        n.name = f"plum_row_{i:02d}"
        n.type = mujoco.mjtGeom.mjGEOM_BOX
        n.pos = [g.pos[0] - x_off, g.pos[1] - y_off, g.pos[2]]
        n.size = list(g.size)
        n.rgba = list(_COLOR)
        n.friction = list(_FRICTION)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml(), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--difficulty", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = build_scene(args.difficulty, args.out)
    print(f"wrote {p}")
    mujoco.MjModel.from_xml_path(str(p))  # fail loud if it does not compile
    print("compiles OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
