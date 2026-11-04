"""Part 2 training terrains built from the single-source terrain spec.

Every dimension below comes from ``terrain/official/part2_terrain_spec.json``,
which is regenerated from the organiser's ``terrain.zip/map.xml`` by
``terrain/gen_part2_spec.py`` (repository root).  Do not hard-code obstacle sizes
here: hand-copied constants drifted from the judge geometry before (the slope
ramp was an older shape on one branch), so the numbers are data, the construction
is code.

The plum field is the rules-PDF layout (single source:
``terrain/gen_plum.py`` -> ``terrain/plum/plum_spec.json``: six cubes in two
staggered rows, platform-to-platform span 3.5 m); it is *not* parsed from the
official map, whose plum parameters the rules PDF itself says are wrong.

The difficulty rows interpolate between the official tier endpoints; the
protocol constants (1.8 m runway, 0.7 m lane) are also in the spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains import SubTerrainCfg

from alldog_mjlab.utils import knobs

# Repository root: train/src/alldog_mjlab/tasks/velocity/profiles/part2/terrain.py
# -> parents[7] == <repo root>.  The spec lives in the single-source terrain/ dir.
_SPEC_PATH = (
    Path(__file__).resolve().parents[7]
    / "terrain" / "official" / "part2_terrain_spec.json"
)
if not _SPEC_PATH.is_file():
    raise FileNotFoundError(
        f"Part 2 terrain spec not found: {_SPEC_PATH}; "
        "regenerate it with `python terrain/gen_part2_spec.py`")
_SPEC = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
_PROTOCOL = _SPEC["protocol"]
_CAT = _SPEC["categories"]
_PLATFORM = _CAT["high_platform"]
_SLOPE = _CAT["slope"]
_TRENCH = _CAT["trench"]
_WALL = _CAT["high_wall"]
_STAIRS = _CAT["stairs"]
_PLUM = _CAT["plum"]

for _required in ("high_platform", "slope", "trench", "high_wall", "stairs", "plum"):
    if _required not in _CAT:
        raise ValueError(
            f"part2_terrain_spec.json is missing category {_required!r}")

_FLOOR_RGBA = (0.45, 0.45, 0.45, 1.0)
_OBSTACLE_RGBA = (0.25, 0.55, 0.9, 1.0)
_RAMP_RGBA = (0.75, 0.72, 0.55, 1.0)
_TRENCH_RGBA = (0.3, 0.1, 0.1, 1.0)
_FLOOR_T = 0.05


def _world_friction() -> tuple[float, float, float] | None:
    """Optional training-only terrain friction override (diagnostic).

    MuJoCo combines contact friction as the element-wise max of both geoms'
    friction, so a value above the robot's 1.0 raises the effective contact
    friction without touching the robot model.  Unset (or 0) keeps the official
    judge value, i.e. MuJoCo's default 1.0 on the terrain geoms.

    Only used to probe physical limits (e.g. ``PART2_WORLD_FRICTION=1.2`` for
    the 45 deg slope); never set for deliverable training.
    """
    mu = knobs.get_float("PART2_WORLD_FRICTION", 0.0)
    if mu <= 0.0:
        return None
    return (mu, 0.005, 0.0001)


def _box(body, size, pos, rgba, quat=None):
    kwargs = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos)
    if quat is not None:
        kwargs["quat"] = quat
    friction = _world_friction()
    if friction is not None:
        kwargs["friction"] = friction
    geom = body.add_geom(**kwargs)
    return TerrainGeometry(geom=geom, color=rgba)


def _origin(size):
    return np.array([size[0] / 2, size[1] / 2, 0.0])


def _floor(body, size, rgba=_FLOOR_RGBA):
    return _box(
        body,
        (size[0] / 2, size[1] / 2, _FLOOR_T / 2),
        (size[0] / 2, size[1] / 2, -_FLOOR_T / 2),
        rgba,
    )




def _ramp(body, start_x, y_center, start_z, run, rise, half_width, rgba,
          thickness: float = 1.0):
    """Solid slab whose *top face* is the ramp surface (no leading-edge lip).

    The official slope/trench/plum ramps are solid wedges whose top surface goes
    exactly from (start_x, start_z) to (start_x + run, start_z + rise).  Earlier
    versions used a thin rotated plate centred on the mid-height, which lifted the
    surface by half the thickness (~2 cm) at the bottom edge: a small step that
    made steeper slopes systematically harder in training than in the judge
    scenes (BC policy cleared the 35 deg judge scene but never the training
    terrain).  This helper aligns the top face with the wedge surface and lets the
    rest of the slab sit below ground.
    """
    angle = float(np.arctan2(rise, run))
    half_t = 0.5 * float(thickness)
    half_len = float(np.hypot(run, rise)) / 2.0
    normal = np.array([-np.sin(angle), 0.0, np.cos(angle)])
    mid = np.array([start_x + run / 2.0, y_center, start_z + rise / 2.0])
    center = mid - half_t * normal
    return _box(
        body,
        (half_len, half_width, half_t),
        (float(center[0]), float(center[1]), float(center[2])),
        rgba,
        quat=(float(np.cos(angle / 2)), 0.0, float(-np.sin(angle / 2)), 0.0),
    )


def _goal(cfg, x):
    return {"goal": np.array([[x, cfg.size[1] / 2, 0.0]])}


@dataclass(kw_only=True)
class ExactHighPlatformTerrainCfg(SubTerrainCfg):
    """Official Part2 high platform: one 1.5 x 1.6 m box, no helper step.

    Geometry follows ``terrain.zip/map.xml`` (platform_30cm/40cm geoms:
    ``size=(0.75, 0.8, h/2)``); the previous local level XML's 1.2 x 1.2 m
    platform was too small. ``side_walls`` keeps the robot in the official
    0.7 m scoring lane so it cannot learn the proxy bypass.
    """

    platform_height_range: tuple[float, float] = tuple(_PLATFORM["height_range_m"])
    platform_half_x: float = _PLATFORM["half_x_m"]
    platform_half_y: float = _PLATFORM["half_y_m"]
    platform_x: float = _PLATFORM["center_x_m"]
    # Official ability_field has no physical side walls; the judge only enforces
    # a lateral lane. Keep the option for debugging, but do not train with walls.
    side_walls: bool = False
    lane_half_width: float = _PROTOCOL["lane_half_width_m"]
    wall_height: float = 0.60
    wall_thickness: float = 0.05

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        h = self.platform_height_range[0] + difficulty * (
            self.platform_height_range[1] - self.platform_height_range[0]
        )
        cx = self.size[0] / 2 + self.platform_x
        geom = _box(
            body,
            (self.platform_half_x, self.platform_half_y, h / 2),
            (cx, self.size[1] / 2, h / 2),
            _OBSTACLE_RGBA,
        )
        geometries = [_floor(body, self.size), geom]
        if self.side_walls:
            half_y = self.size[1] / 2
            for sign in (-1.0, 1.0):
                geometries.append(
                    _box(
                        body,
                        (self.size[0] / 2, self.wall_thickness / 2, self.wall_height / 2),
                        (
                            self.size[0] / 2,
                            half_y + sign * self.lane_half_width,
                            self.wall_height / 2,
                        ),
                        _OBSTACLE_RGBA,
                    )
                )
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geometries,
            flat_patches=_goal(self, self.size[0] / 2 + self.platform_x + 0.4),
        )


@dataclass(kw_only=True)
class ExactSlopeTerrainCfg(SubTerrainCfg):
    """Official Part2 slope: up ramp + landing + down ramp, 1.4 m wide, 20..45 deg.

    From ``map.xml`` the real slope is ``slope_XXdeg_up`` (2 m horizontal run,
    rise ``2*tan(angle)``), a 0.8 m landing, and a mirrored ``slope_XXdeg_down``
    ramp back to the floor.  All three parts are required for the judge's
    traversal success; the earlier training-only version omitted the down ramp.
    """

    slope_deg_range: tuple[float, float] = tuple(_SLOPE["deg_range"])
    ramp_horizontal: float = _SLOPE["ramp_horizontal_m"]
    ramp_width: float = _SLOPE["ramp_width_m"]
    landing_half_x: float = _SLOPE["landing_half_x_m"]
    # Official slope ramp starts 1.8 m ahead of the spawn (judge goal_x_min=3.9).
    ramp_start_x: float = _SLOPE["ramp_start_x_m"]

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        deg = self.slope_deg_range[0] + difficulty * (
            self.slope_deg_range[1] - self.slope_deg_range[0]
        )
        angle = float(np.deg2rad(deg))
        rise = float(np.tan(angle) * self.ramp_horizontal)
        ramp = _ramp(
            body,
            self.size[0] / 2 + self.ramp_start_x,
            self.size[1] / 2,
            0.0,
            self.ramp_horizontal,
            rise,
            self.ramp_width / 2,
            _RAMP_RGBA,
        )
        landing_cx = (self.size[0] / 2 + self.ramp_start_x
                      + self.ramp_horizontal + self.landing_half_x)
        top = _box(
            body,
            (self.landing_half_x, self.ramp_width / 2, rise / 2),
            (landing_cx, self.size[1] / 2, rise / 2),
            _OBSTACLE_RGBA,
        )
        # Official ``slope_XXdeg_down`` mesh: a mirror of the up ramp from the
        # far landing edge back to the floor (same 2 m run).  The training patch
        # used to stop at the landing block, i.e. it ended in a ``rise``-high
        # cliff.  The slope success predicate requires all wheels back on the
        # ground past ``PART2_SLOPE_EXIT_X``, so without the down ramp the
        # training success was *impossible* (success≈0 even for a policy that
        # passes the judge scene, and slope45 could never be learned).
        down_start = landing_cx + self.landing_half_x
        down = _ramp(
            body, down_start, self.size[1] / 2, rise,
            self.ramp_horizontal, -rise, self.ramp_width / 2, _RAMP_RGBA,
        )
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=[_floor(body, self.size), ramp, top, down],
            flat_patches=_goal(self, down_start + self.ramp_horizontal + 0.4),
        )


@dataclass(kw_only=True)
class ExactTrenchTerrainCfg(SubTerrainCfg):
    """Official Part2 trench: 0.3 m raised plateau, 1 m entry/exit ramps, 0.1..0.3 m gap.

    From ``terrain.zip/map.xml`` the official trench is not a pit: the crossing
    sits on a 0.30 m high plateau (1.4 m channel) entered/exited by 1 m wedge
    ramps, and the "gap" is a 0.10/0.20/0.30 m slot through that plateau.
    """

    gap_width_range: tuple[float, float] = tuple(_TRENCH["gap_range_m"])
    plateau_height: float = _TRENCH["plateau_m"]
    ramp_run: float = _TRENCH["ramp_run_m"]
    near_half_x: float = _TRENCH["near_half_x_m"]
    channel_half_y: float = _TRENCH["channel_half_y_m"]
    # Official trench near plateau edge sits at x=3.8 (judge near_x/far_x).
    start_x: float = _TRENCH["start_x_m"]

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        width = self.gap_width_range[0] + difficulty * (
            self.gap_width_range[1] - self.gap_width_range[0]
        )
        cy = self.size[1] / 2
        x0 = self.size[0] / 2 + self.start_x
        angle = float(np.arctan2(self.plateau_height, self.ramp_run))
        ramp_len = float(np.hypot(self.ramp_run, self.plateau_height))
        half_y = self.channel_half_y
        geoms = [_floor(body, self.size)]
        # Entry ramp: ground -> plateau top (solid wedge, top face exact).
        entry_cx = x0 + self.ramp_run / 2
        geoms.append(_ramp(
            body, x0, cy, 0.0, self.ramp_run, self.plateau_height,
            half_y, _RAMP_RGBA))
        # Near plateau block, then the gap, then the far block.
        near_cx = x0 + self.ramp_run + self.near_half_x
        geoms.append(_box(body, (self.near_half_x, half_y, self.plateau_height / 2),
                          (near_cx, cy, self.plateau_height / 2), _OBSTACLE_RGBA))
        far_start = near_cx + self.near_half_x + width
        far_cx = far_start + self.near_half_x
        geoms.append(_box(body, (self.near_half_x, half_y, self.plateau_height / 2),
                          (far_cx, cy, self.plateau_height / 2), _OBSTACLE_RGBA))
        # Exit ramp: plateau top -> ground (solid wedge, top face exact).
        exit_start = far_cx + self.near_half_x
        exit_cx = exit_start + self.ramp_run / 2
        geoms.append(_ramp(
            body, exit_start, cy, self.plateau_height, self.ramp_run,
            -self.plateau_height, half_y, _RAMP_RGBA))
        exit_end = exit_start + self.ramp_run
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            # v7.5-aligned: progress/goal reward must keep paying through the 1 m
            # exit ramp (judge success = back on normal ground past the ramp), so
            # the goal patch sits 0.4 m after the ramp end (flat floor).
            flat_patches=_goal(self, exit_end + 0.4),
        )


@dataclass(kw_only=True)
class ExactHighWallTerrainCfg(SubTerrainCfg):
    """Official Part2 high wall: 1.4 m wide x 0.12 m thick, 10..40 cm high."""

    wall_height_range: tuple[float, float] = tuple(_WALL["height_range_m"])
    wall_half_y: float = _WALL["half_y_m"]
    wall_thickness: float = _WALL["thickness_m"]
    # Official cross_bar position in the isolated Part2 scene (judge wall_x=1.86).
    wall_x: float = _WALL["center_x_m"]

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        h = self.wall_height_range[0] + difficulty * (
            self.wall_height_range[1] - self.wall_height_range[0]
        )
        cx = self.size[0] / 2 + self.wall_x
        wall = _box(
            body,
            (self.wall_thickness / 2, self.wall_half_y, h / 2),
            (cx, self.size[1] / 2, h / 2),
            _OBSTACLE_RGBA,
        )
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=[_floor(body, self.size), wall],
            flat_patches=_goal(self, self.size[0] / 2 + self.wall_x + 0.5),
        )


@dataclass(kw_only=True)
class ExactStairsTerrainCfg(SubTerrainCfg):
    """Official Part2 stairs: 6 up steps + 0.8 m landing + 6 down steps.

    ``map.xml`` names the whole object ``stairs_XXcm_{up,landing,down}`` and the
    official ``stairs_XXcm.png`` label covers both flights, so "通过台阶" means
    climbing up, crossing the landing and descending back to the ground on the
    far side.  Official levels are single-step heights 10/15/20/30 cm and total
    heights 0.60/0.90/1.20/1.80 m; the difficulty interpolates 0.6..1.8 m.
    """

    total_height_range: tuple[float, float] = tuple(_STAIRS["total_height_range_m"])
    num_steps: int = _STAIRS["num_steps"]
    tread: float = _STAIRS["tread_m"]
    half_width: float = _STAIRS["half_width_m"]
    # Official staircase runs 1.8 .. 6.2 m in the isolated scene (judge
    # descend_end_x=6.2).
    start_x: float = _STAIRS["start_x_m"]
    landing_half_x: float = _STAIRS["landing_half_x_m"]

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        total = self.total_height_range[0] + difficulty * (
            self.total_height_range[1] - self.total_height_range[0]
        )
        geoms = [_floor(body, self.size)]
        cy = self.size[1] / 2
        x = self.size[0] / 2 + self.start_x
        # Up flight: each step is a full column of the level height.
        for i in range(self.num_steps):
            top = total * (i + 1) / self.num_steps
            geoms.append(_box(body, (self.tread / 2, self.half_width, top / 2),
                              (x, cy, top / 2), _OBSTACLE_RGBA))
            x += self.tread
        # Landing.
        landing_cx = x + self.landing_half_x
        geoms.append(_box(body, (self.landing_half_x, self.half_width, total / 2),
                          (landing_cx, cy, total / 2), _OBSTACLE_RGBA))
        x += 2.0 * self.landing_half_x
        # Down flight: the official scene has 6 full-height columns with tops
        # 6h,5h,4h,3h,2h,h (the first down column is flush with the landing, so
        # the descent starts at 4.4 m and ends at 6.2 m, matching
        # ``descend_end_x=6.2``).  The previous loop dropped the flush column and
        # started the descent one tread early (5 columns 5h..h, ending at 5.9 m).
        for i in range(self.num_steps):
            top = total * (self.num_steps - i) / self.num_steps
            geoms.append(_box(body, (self.tread / 2, self.half_width, top / 2),
                              (x + self.tread / 2, cy, top / 2), _OBSTACLE_RGBA))
            x += self.tread
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            flat_patches=_goal(self, x + 0.4),
        )


@dataclass(kw_only=True)
class ExactPlumTerrainCfg(SubTerrainCfg):
    """Part 2 plum: rules-PDF layout (six cubes, two staggered rows, no ramps).

    Single source of truth: ``terrain/gen_plum.py`` ->
    ``terrain/plum/plum_spec.json``.  Geometry: 0.8 x 0.8 x 0.2 m entry platform
    (near face 1.8 m ahead of the spawn), six 0.2 m cubes in two staggered rows,
    0.8 x 0.8 x 0.2 m exit platform, then the 0.2 m step down to normal ground.

    The field is fixed, so the difficulty parameter does not change its shape.
    """

    start_x: float = _PLUM["start_x_m"]
    platform_half_x: float = _PLUM["platform_half_x_m"]
    platform_half_y: float = _PLUM["platform_half_y_m"]
    platform_height: float = _PLUM["platform_height_m"]
    cube_half: float = _PLUM["cube_half_m"]
    cube_top: float = _PLUM["cube_top_m"]
    cube_centres: tuple[tuple[float, float], ...] = tuple(
        (float(x), float(y)) for x, y in _PLUM["cube_centres"])
    exit_platform_x_range: tuple[float, float] = tuple(
        float(v) for v in _PLUM["exit_platform_x_range_m"])

    def function(self, difficulty, spec, rng):
        del difficulty, rng  # fixed official field: no difficulty interpolation
        body = spec.body("terrain")
        cy = self.size[1] / 2
        # Scene-frame -> terrain-patch frame (spawn at patch origin).
        x0 = self.size[0] / 2 + self.start_x
        geoms = [_floor(body, self.size)]
        # Entry platform: near face starts at x0 (1.8 m runway).
        geoms.append(_box(
            body,
            (self.platform_half_x, self.platform_half_y, self.platform_height / 2),
            (x0 + self.platform_half_x, cy, self.platform_height / 2),
            _OBSTACLE_RGBA,
        ))
        # Six cubes in two staggered rows (spec centres are scene-frame x/y).
        for center_x, center_y in self.cube_centres:
            geoms.append(_box(
                body,
                (self.cube_half, self.cube_half, self.cube_top / 2),
                (x0 + (center_x - self.start_x), cy + center_y, self.cube_top / 2),
                _OBSTACLE_RGBA,
            ))
        # Exit platform, then normal ground: the 0.2 m drop is the step-down the
        # judge requires beyond ``exit_ramp_end`` (training success also requires
        # getting back down to the ground; see predicates.py).
        exit_near = self.exit_platform_x_range[0]
        exit_cx = x0 + (exit_near - self.start_x) + self.platform_half_x
        geoms.append(_box(
            body,
            (self.platform_half_x, self.platform_half_y, self.platform_height / 2),
            (exit_cx, cy, self.platform_height / 2),
            _OBSTACLE_RGBA,
        ))
        exit_end = exit_cx + self.platform_half_x
        return TerrainOutput(
            origin=_origin(self.size),
            geometries=geoms,
            flat_patches=_goal(self, exit_end + 0.4),
        )
