#!/usr/bin/env python3
"""Self-test for PART2_WORLD_FRICTION (training-only terrain friction override).

Builds the exact slope/stairs terrain specs directly (no env, no GPU) and checks
that every terrain geom carries the requested friction.  MuJoCo combines contact
friction as the element-wise max of both geoms' friction, so the robot wheels keep
their 1.0 value while contacts become max(requested, 1.0).

Run inside the training repo:
  PYTHONPATH=$PWD/src .venv/bin/python scripts/agent_scripts/check_world_friction.py
  PART2_WORLD_FRICTION=1.2 PYTHONPATH=$PWD/src .venv/bin/python scripts/agent_scripts/check_world_friction.py
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from alldog_mjlab.tasks.velocity.profiles.part2.terrain import (
    ExactSlopeTerrainCfg,
    ExactStairsTerrainCfg,
)

MUJOCO_DEFAULT = (1.0, 0.005, 0.0001)


def _expected(raw: str) -> tuple[float, float, float]:
    mu = float(raw) if raw else 0.0
    return MUJOCO_DEFAULT if mu <= 0.0 else (mu, 0.005, 0.0001)


def _friction_sets(frictions) -> list[tuple[float, ...]]:
    return sorted({tuple(float(x) for x in f) for f in frictions})


def _matches(actual: list[tuple[float, ...]], expected: tuple[float, float, float]) -> bool:
    if len(actual) != 1:
        return False
    return bool(np.allclose(np.asarray(actual[0]), np.asarray(expected), atol=1e-6))


def build_terrain(cfg, difficulty: float = 1.0) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg.function(difficulty, spec, np.random.default_rng(0))
    return spec


def main() -> int:
    raw = os.environ.get("PART2_WORLD_FRICTION", "")
    expected = _expected(raw)
    print(f"PART2_WORLD_FRICTION={raw or '<unset>'} -> expected geom friction {expected}")
    failures = 0
    for name, cfg in (("slope", ExactSlopeTerrainCfg()), ("stairs", ExactStairsTerrainCfg())):
        spec = build_terrain(cfg)
        model = spec.compile()
        spec_fr = _friction_sets(g.friction for g in spec.geoms)
        model_fr = _friction_sets(row for row in model.geom_friction)
        ok = _matches(spec_fr, expected) and _matches(model_fr, expected)
        print(f"  {name:6s} geoms={len(spec.geoms):3d} spec={spec_fr} compiled={model_fr}"
              f" -> {'OK' if ok else 'FAIL'}")
        failures += int(not ok)
    if failures:
        print(f"SELFTEST FAIL ({failures})")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
