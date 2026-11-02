"""CPU geometry check for the competition-proxy terrain generator.

Builds the terrain spec via mjlab's TerrainGenerator (no GPU / no env), prints
geom counts, spawn origins and obstacle height envelope. Run from the training
repo root with `.venv/bin/python scripts/agent_scripts/check_competition_terrain.py`.
"""

from __future__ import annotations

import mujoco
import numpy as np

from alldog_mjlab.tasks.velocity.profiles.part1.terrain import (
    black_competition_terrain_generator_cfg,
)
from mjlab.terrains.terrain_generator import TerrainGenerator


def main() -> None:
    cfg = black_competition_terrain_generator_cfg()
    gen = TerrainGenerator(cfg)
    spec = mujoco.MjSpec()
    gen.compile(spec)
    model = spec.compile()

    names = [model.geom(i).name for i in range(model.ngeom)]
    bodies = [model.body(i).name for i in range(model.nbody)]
    print(f"sub_terrains={len(cfg.sub_terrains)} rows={cfg.num_rows} cols={gen._num_cols}")
    print(f"ngeom={model.ngeom} nbody={model.nbody} bodies={bodies[:3]}")
    print(f"geom names sample={names[:3]}")

    origins = gen.terrain_origins
    print(f"origins shape={origins.shape}")
    assert np.isfinite(origins).all(), "non-finite terrain origins"
    z_span = origins[..., 2].max() - origins[..., 2].min()
    print(f"origin z range={origins[..., 2].min():.3f}..{origins[..., 2].max():.3f}")

    # Surface envelope: top of all geoms except the very deep trench floor.
    tops = []
    for i in range(model.ngeom):
        pos = model.geom_pos[i]
        size = model.geom_size[i]
        gtype = model.geom_type[i]
        if gtype == mujoco.mjtGeom.mjGEOM_HFIELD:
            z = pos[2]
        elif gtype in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            z = pos[2] + size[2] if gtype == mujoco.mjtGeom.mjGEOM_BOX else pos[2] + size[1]
        else:
            continue
        tops.append(z)
    print(f"geom top z min={min(tops):.3f} max={max(tops):.3f} (expect up to ~0.55)")

    # Per-column top envelope tells whether every obstacle column has geometry.
    for col in range(gen._num_cols):
        col_top = []
        col_origins = origins[:, col]
        xmin = col_origins[:, 0].min() - 4.0
        xmax = col_origins[:, 0].max() + 4.0
        for i in range(model.ngeom):
            pos = model.geom_pos[i]
            if xmin <= pos[0] <= xmax:
                size = model.geom_size[i]
                gtype = model.geom_type[i]
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    col_top.append(pos[2] + size[2])
                elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    col_top.append(pos[2] + size[1])
        name = list(cfg.sub_terrains)[col]
        print(f"  col {col:2d} {name:16s} geoms_top={len(col_top):4d} max_z={max(col_top) if col_top else float('nan'):.2f}")

    print("OK: competition terrain geometry builds on CPU")


if __name__ == "__main__":
    main()
