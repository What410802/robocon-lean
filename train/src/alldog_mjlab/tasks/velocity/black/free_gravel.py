"""Free-joint gravel bed for small-batch S1b fine-tuning / evaluation.

The batch path is ``black-gravel-flat``.  This module builds a *raised* stone bed
(3 cm base + 3 cm low curb; dog steps in from the surrounding plane) and covers
it with many small/irregular free-joint stones.  Every stone is a free joint in
every world, so this task is limited to 8-16 envs.

Compared with the first prototype:
- stones are 3-8 cm (not 15-19 cm slabs),
- shape is a random tri-axial ellipsoid (irregular, not a box),
- the bed is slightly raised and has a low curb; there is no tall fence,
- a static base under the stones prevents a foot from punching through to the
  exposed floor in gaps.
"""

from __future__ import annotations

import mujoco
import numpy as np

BED_HALF_WIDTH = 0.75       # 1.5 x 1.5 m bed
BED_BASE_HEIGHT = 0.06      # static pre-baked gravel base top (95% sink depth + margin)
CURB_HEIGHT = 0.06          # low curb; dog can step in; below 1/3 COM (0.14 m)
STONE_COUNT = 120
STATIC_STONE_COUNT = 700
SEED = 20261021


def _stone_geometry() -> list[tuple[float, float, float, float, float, float, float]]:
    """Return (x, y, z, rx, ry, rz, yaw) for every free stone."""
    rng = np.random.default_rng(SEED)
    stones: list[tuple[float, float, float, float, float, float, float]] = []
    attempts = 0
    while len(stones) < STONE_COUNT and attempts < 20000:
        attempts += 1
        x = rng.uniform(-BED_HALF_WIDTH + 0.02, BED_HALF_WIDTH - 0.02)
        y = rng.uniform(-BED_HALF_WIDTH + 0.02, BED_HALF_WIDTH - 0.02)
        # 3-8 cm full size => half axes 1.5-4 cm; slightly flatter in z.
        rx = rng.uniform(0.015, 0.040)
        ry = rng.uniform(0.015, 0.040)
        rz = rng.uniform(0.010, 0.035)
        # Avoid two stones starting in near-identical positions; we do not enable
        # stone-stone contacts, so this is only a placement/visual concern.
        if any((x - sx) ** 2 + (y - sy) ** 2 < (0.25 * (rx + srx)) ** 2 for sx, sy, _, srx, _, _, _ in stones):
            continue
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        z = BED_BASE_HEIGHT + rz + 0.002
        stones.append((x, y, z, rx, ry, rz, yaw))
    return stones


def _static_stone_geometry() -> list[tuple[float, float, float, float, float, float, float]]:
    """Settled static stones under the free layer (same visual size range)."""
    rng = np.random.default_rng(SEED + 17)
    stones: list[tuple[float, float, float, float, float, float, float]] = []
    for _ in range(STATIC_STONE_COUNT):
        x = rng.uniform(-BED_HALF_WIDTH + 0.01, BED_HALF_WIDTH - 0.01)
        y = rng.uniform(-BED_HALF_WIDTH + 0.01, BED_HALF_WIDTH - 0.01)
        rx = rng.uniform(0.020, 0.040)
        ry = rng.uniform(0.020, 0.040)
        rz = rng.uniform(0.010, 0.030)
        # Partially embedded in the base; dog sees/stands on the exposed cap.
        z = BED_BASE_HEIGHT + 0.5 * rz
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        stones.append((x, y, z, rx, ry, rz, yaw))
    return stones


def coverage_stats(samples: int = 1600) -> dict[str, float]:
    """Projected top-view coverage: free layer, static filler, and union."""
    free = _stone_geometry()
    static = _static_stone_geometry()
    bed_area = (2.0 * BED_HALF_WIDTH) ** 2
    xs = np.linspace(-BED_HALF_WIDTH, BED_HALF_WIDTH, samples)
    ys = np.linspace(-BED_HALF_WIDTH, BED_HALF_WIDTH, samples)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    def union_fraction(stones):
        mask = np.zeros_like(X, dtype=bool)
        for x, y, _, rx, ry, _, yaw in stones:
            c, s = np.cos(yaw), np.sin(yaw)
            dx, dy = X - x, Y - y
            ex = dx * c + dy * s
            ey = -dx * s + dy * c
            mask |= (ex / rx) ** 2 + (ey / ry) ** 2 <= 1.0
        return float(mask.mean())

    free_area = sum(np.pi * rx * ry for _, _, _, rx, ry, _, _ in free)
    static_area = sum(np.pi * rx * ry for _, _, _, rx, ry, _, _ in static)
    return {
        "free_stones": float(len(free)),
        "static_stones": float(len(static)),
        "bed_area_m2": bed_area,
        "free_sum_coverage_fraction": float(free_area / bed_area),
        "free_union_coverage_fraction": union_fraction(free),
        "static_union_coverage_fraction": union_fraction(static),
        "total_union_coverage_fraction": union_fraction(free + static),
        "base_top_m": BED_BASE_HEIGHT,
        "curb_height_m": CURB_HEIGHT,
    }


def add_free_gravel_bed(spec: mujoco.MjSpec) -> None:
    """Add a raised static gravel base + low curb + many free ellipsoid stones."""
    stone_color = [0.48, 0.43, 0.35, 1.0]
    base_color = [0.34, 0.31, 0.27, 1.0]

    # Static raised base: a stone-colored platform, not an exposed floor.
    base = spec.worldbody.add_body(name="free_gravel_base", pos=(0.0, 0.0, BED_BASE_HEIGHT * 0.5))
    base.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(BED_HALF_WIDTH, BED_HALF_WIDTH, BED_BASE_HEIGHT * 0.5),
        rgba=base_color,
        contype=1,
        conaffinity=1,
        friction=(1.0, 0.02, 0.001),
    )

    # Low curb, below the "flat" threshold (COM ~0.43 m -> 1/3 is 0.14 m).
    curb_thickness = 0.035
    for name, pos, size in (
        ("free_gravel_curb_xp", (BED_HALF_WIDTH, 0.0, CURB_HEIGHT * 0.5), (curb_thickness, BED_HALF_WIDTH + curb_thickness, CURB_HEIGHT * 0.5)),
        ("free_gravel_curb_xn", (-BED_HALF_WIDTH, 0.0, CURB_HEIGHT * 0.5), (curb_thickness, BED_HALF_WIDTH + curb_thickness, CURB_HEIGHT * 0.5)),
        ("free_gravel_curb_yp", (0.0, BED_HALF_WIDTH, CURB_HEIGHT * 0.5), (BED_HALF_WIDTH + curb_thickness, curb_thickness, CURB_HEIGHT * 0.5)),
        ("free_gravel_curb_yn", (0.0, -BED_HALF_WIDTH, CURB_HEIGHT * 0.5), (BED_HALF_WIDTH + curb_thickness, curb_thickness, CURB_HEIGHT * 0.5)),
    ):
        body = spec.worldbody.add_body(name=name, pos=pos)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            rgba=base_color,
            contype=1,
            conaffinity=1,
            friction=(1.0, 0.02, 0.001),
        )

    # Settled static stone filler: dog always lands on stone/bed, not exposed floor.
    static_body = spec.worldbody.add_body(name="free_gravel_static_stones")
    for index, (x, y, z, rx, ry, rz, yaw) in enumerate(_static_stone_geometry()):
        geom = static_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(x, y, z),
            size=(rx, ry, rz),
            rgba=stone_color,
            contype=1,
            conaffinity=1,
            friction=(1.0, 0.02, 0.001),
        )
        geom.quat = [np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)]
        del geom

    # Free top layer.
    for index, (x, y, z, rx, ry, rz, yaw) in enumerate(_stone_geometry()):
        body = spec.worldbody.add_body(name=f"free_gravel_{index:03d}", pos=(x, y, z))
        joint = body.add_freejoint()
        joint.damping = [0.15, 0.15, 0.15]
        geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=(rx, ry, rz),
            rgba=stone_color,
            # Stones collide with robot/base/terrain but not each other.
            contype=2,
            conaffinity=1,
            density=2200.0,
            friction=(1.0, 0.02, 0.001),
            solref=(0.02, 1.0),
            solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )
        # Random yaw + slight tilt gives irregular-looking stones.
        tilt = 0.0
        body.quat = [np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)]
        del joint, geom, tilt
