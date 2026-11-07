"""Plum training terrain: the official field, optionally with gap fill.

The deployment route climbs onto the entry platform with the ordinary (30 cm
capable) locomotion policy and *then* hands over to the plum policy, so the plum
teacher is born on the entry platform and only has to cross the cube field to
the exit platform.  That removes the entry step from the task entirely.

With the robot starting at platform height, a barrier-height curriculum is both
unnecessary and unsound.  The judge's drop rule is **absolute** (any wheel below
0.05 m while the base is over the cube field is a dropped attempt), so changing
the cube height changed the rule's physical meaning and every variant was
measured to fail:

* cube top 0.05 m -- the platform top IS the drop line, so a wheel resting on it
  counts as fallen: 128/128 episodes terminated at x ~ 2.57.
* cube top 0.08-0.14 m -- the riding surface clears the line, but the floor is
  still reachable through the 0.4 m gaps and the policy learned to exploit it:
  rolled traces at d=0.6 cross at wheel z ~ 0.02 with the wheels splayed to
  |y| = 0.60, outside the 0.2 m-wide cube rows.  That route does not exist on
  the official field.

So the terrain is the official field at every row, and the only curriculum axis
is the **intra-row gap**: ``gap_fill`` closes the 0.40 m longitudinal gaps at
difficulty 0 and opens them to the official size at 1.0.  That grades the actual
skill (place a wheel on a 0.2 m top and carry it across a 0.4 m gap) without
ever making the floor reachable, and it is the axis the deployment route keeps
when it hands over mid-field.

Difficulty 1.0 is exactly the rules-PDF field.  ``ExactPlumTerrainCfg`` (no
difficulty) is the same field with no gap fill and remains the single-terrain
default for judging/replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.terrains.terrain_generator import TerrainOutput

from alldog_mjlab.tasks.velocity.profiles.part2.terrain import (
    ExactPlumTerrainCfg, _OBSTACLE_RGBA, _box, _floor, _goal, _origin,
)
from .spec import SPEC

#: Official open gap between neighbours: 0.40 m in x, 0.40/0.10 m end holes.
OFFICIAL_GAP_M = 0.40
#: Residual gap at the first jumping rung: just over the 0.206 m wheel diameter,
#: so the rung above the closed one already requires a carry rather than a roll.
JUMP_MIN_M = 0.21
#: Curriculum rows (difficulty 0, 1/(N-1), ..., 1).
NUM_ROWS = 6


@dataclass(kw_only=True)
class PlumCurriculumTerrainCfg(ExactPlumTerrainCfg):
    """Official plum field with a gap/jump curriculum.

    Cube centres, cube size, row offsets and platform positions are the official
    values at every row.  Difficulty 0 closes every gap (lane riding: the robot
    learns to move along the field at height); above it the residual gap opens
    from just over the wheel diameter to the official 0.40 m.
    """

    official_gap: float = OFFICIAL_GAP_M
    jump_min: float = JUMP_MIN_M

    def residual_at(self, difficulty: float) -> float:
        """Open gap (m) between neighbours at this row, on BOTH axes.

        A linear fill from 0.40 to 0.00 spends more than half the ladder below
        the 0.206 m wheel diameter, where a wheel simply rolls over the dip --
        every rung there teaches the same thing, and the first rung that needs a
        jump lands on a cliff at exactly the diameter.  Measured: after 2000
        iterations the population sat at row 1.4-1.8 with 48 % success at
        d=0.3 and 0 % at d=0.6.

        So the ladder is split: difficulty 0 is fully closed (one rung of lane
        riding / speed), and above it the residual gap starts just over the
        wheel diameter and grows linearly to the official 0.40 m.  Every rung
        then asks for a real, slightly harder jump.
        """
        d = min(1.0, max(0.0, float(difficulty)))
        if d < 0.05:
            return 0.0
        # The ladder is sampled at d = 0, 1/(rows-1), ... 1; the rung just above
        # the closed one must be the SMALLEST jump, so the ramp spans from
        # d0 = 1/(rows-1) to 1 rather than from 0 (sampling it from 0 put the
        # first jumping rung at 0.248 m, nearly the official size).
        d0 = 1.0 / (NUM_ROWS - 1)
        return self.jump_min + (self.official_gap - self.jump_min) * (
            (d - d0) / max(1e-9, 1.0 - d0))

    def gap_fill_at(self, difficulty: float) -> float:
        """How much of each 0.40 m gap is filled at this row."""
        return max(0.0, self.official_gap - self.residual_at(difficulty))

    def bridge_at(self, difficulty: float) -> float:
        """End-hole bridging; the same residual schedule as the interior gaps."""
        return self.gap_fill_at(difficulty)

    def function(self, difficulty, spec, rng):
        del rng
        h = SPEC.platform_top          # official height at every row
        fill = self.gap_fill_at(difficulty)
        bridge = self.bridge_at(difficulty)
        body = spec.body("terrain")
        cy = self.size[1] / 2
        # Scene-frame -> terrain-patch frame (spawn at patch origin).
        x0 = self.size[0] / 2 + self.start_x
        geoms = [_floor(body, self.size)]
        half = SPEC.cube_half

        def slab(name, x_min, x_max, y_c, half_y, top):
            width = x_max - x_min
            if width <= 1e-9:
                return
            geoms.append(_box(
                body, (width / 2, half_y, top / 2),
                (x0 + (x_min + x_max) / 2 - self.start_x, cy + y_c, top / 2),
                _OBSTACLE_RGBA))

        # Entry and exit platforms at the official x ranges and height.
        for xr, name in ((SPEC.entry_x_range, "entry"), (SPEC.exit_x_range, "exit")):
            slab(name, xr[0], xr[1], 0.0, self.platform_half_y, h)

        # Official cubes, then the intra-row gap fill that closes them into a
        # 0.2 m-wide lane.  The lanes stay separate in y (0.4 m offset, 0.2 m
        # tiles), so this never degenerates into a slab.
        rows = {y: sorted(x for x, yy in SPEC.cube_centres if yy == y)
                for y in {y for _, y in SPEC.cube_centres}}
        for cyy, xs in rows.items():
            for cx in xs:
                geoms.append(_box(
                    body, (half, half, h / 2),
                    (x0 + (cx - self.start_x), cy + cyy, h / 2), _OBSTACLE_RGBA))
            if fill > 1e-9:
                for a, b in zip(xs, xs[1:]):
                    gap_a, gap_b = a + half, b - half
                    take = 0.5 * min(fill, gap_b - gap_a)
                    # Grow each neighbour by half the fill; they meet in the
                    # middle, so the lane is continuous at difficulty 0.
                    slab(f"fill_{cyy}_{a:.2f}", gap_a, gap_a + take, cyy, half, h)
                    slab(f"fill_{cyy}_{b:.2f}", gap_b - take, gap_b, cyy, half, h)
                # The two axes stay orthogonal: gap fill only closes the space
                # BETWEEN cubes, the end holes belong to ``bridge``.  Extending
                # the outer cubes as well would fill part of the entry hole at
                # mid difficulty and make the bridge rung non-monotone.

        if bridge > 1e-9:
            # Extend each platform toward the field, at platform width, to narrow
            # the end holes.  The wide hole is 0.40 m on ONE row at each end
            # (row 2 at the entry, row 1 at the exit -- the rectangles are not
            # point-symmetric); a full-width strip covers whichever row is wide.
            near = max(xs[0] for xs in rows.values()) - half
            far = min(xs[-1] for xs in rows.values()) + half
            take_in = min(bridge, near - SPEC.entry_x_range[1])
            take_out = min(bridge, SPEC.exit_x_range[0] - far)
            slab("entry_bridge", SPEC.entry_x_range[1],
                 SPEC.entry_x_range[1] + take_in, 0.0, self.platform_half_y, h)
            slab("exit_bridge", SPEC.exit_x_range[0] - take_out,
                 SPEC.exit_x_range[0], 0.0, self.platform_half_y, h)

        exit_end = SPEC.exit_x_range[1]
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            flat_patches=_goal(self, exit_end + 0.4),
        )
