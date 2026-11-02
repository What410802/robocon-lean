"""Competition-course terrain proxies for the Part 1 / Part 2 obstacles.

The training env uses mjlab's terrain generator, not the Part 1/2 MJCF scenes:
each patch below approximates one competition obstacle type with randomized
layout scope and a difficulty sweep, so the PPO curriculum sees the obstacle
distribution instead of a single fixed course.  Real Part 1/2 scenes remain the
final judge; these patches only shape the training distribution.

Obstacle dimensions follow ``comp/CONTRACT.md`` and the Part 1/2 scenes
(2026-10-24 calibrated): slalom poles, bridge blocks (0.4 x 1.0 m, 0.15 m gaps,
0.20 m top), bridge bands (0.2 x 1.5 m, 0.10 m gaps), height bar (pass-under),
high wall (0.10-0.30 m), trench (0.15-0.35 m gap), platform (0.10-0.40 m),
stepping stones (plum, 0.20-0.30 m cubes at 0.20 m height), plus the existing
flat / slope / stairs / gravel proxies for basic skills.

Courses run along +x from the patch spawn origin (patch centre): the robot
spawns on the flat floor at the centre and meets the obstacle ~1 m ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG

_FLOOR_THICKNESS = 0.05
_FLOOR_RGBA = (0.45, 0.45, 0.45, 1.0)
_OBSTACLE_RGBA = (0.85, 0.55, 0.20, 1.0)
_WALL_RGBA = (0.55, 0.55, 0.62, 1.0)
_STONE_RGBA = (0.30, 0.60, 0.85, 1.0)

# Patch layout constants (patch size is 8 x 8 m from BLACK_CONFIG.terrain).
_COURSE_OFFSET_X = 0.7  # first obstacle centre starts ~0.7 m ahead of spawn


def _lerp(lo: float, hi: float, difficulty: float) -> float:
    return lo + float(np.clip(difficulty, 0.0, 1.0)) * (hi - lo)


def _box(
    body: mujoco.MjsBody,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> TerrainGeometry:
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
    )
    return TerrainGeometry(geom=geom, color=rgba)


def _cylinder(
    body: mujoco.MjsBody,
    radius: float,
    half_height: float,
    pos: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> TerrainGeometry:
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(radius, half_height, 0.0),
        pos=pos,
    )
    return TerrainGeometry(geom=geom, color=rgba)


def _patch_floor(
    body: mujoco.MjsBody,
    size: tuple[float, float],
    rgba: tuple[float, float, float, float] = _FLOOR_RGBA,
) -> TerrainGeometry:
    """Solid floor whose top surface is z = 0 (the patch ground plane)."""
    return _box(
        body,
        (size[0] / 2, size[1] / 2, _FLOOR_THICKNESS / 2),
        (size[0] / 2, size[1] / 2, -_FLOOR_THICKNESS / 2),
        rgba,
    )


def _spawn_origin(size: tuple[float, float]) -> np.ndarray:
    return np.array([size[0] / 2, size[1] / 2, 0.0])


def _course_start(size: tuple[float, float], offset: float = _COURSE_OFFSET_X) -> float:
    return size[0] / 2 + offset


def _goal_patches(cfg: "SubTerrainCfg") -> dict[str, np.ndarray] | None:
    """Return the far-end goal flat patch when the generator asks for it."""
    sampling = cfg.flat_patch_sampling
    if not sampling or "goal" not in sampling:
        return None
    goal_x = cfg.size[0] / 2 + cfg.goal_dist  # type: ignore[attr-defined]
    return {"goal": np.array([[goal_x, cfg.size[1] / 2, 0.0]])}


@dataclass(kw_only=True)
class BridgeBlocksTerrainCfg(SubTerrainCfg):
    """Part 1 bridge_blocks: five 0.4 x 1.0 m blocks, 0.15 m gaps, 0.20 m top."""

    num_blocks: int = 5
    block_length: float = 0.4
    block_width: float = 1.0
    block_height: float = 0.2
    gap_range: tuple[float, float] = (0.05, 0.15)

    goal_dist: float = 3.4

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        gap = _lerp(*self.gap_range, difficulty)
        start = _course_start(self.size)
        half_h = self.block_height / 2
        for i in range(self.num_blocks):
            cx = start + i * (self.block_length + gap) + self.block_length / 2
            geometries.append(
                _box(
                    body,
                    (self.block_length / 2, self.block_width / 2, half_h),
                    (cx, self.size[1] / 2, half_h),
                    _OBSTACLE_RGBA,
                )
            )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class BridgeBandsTerrainCfg(SubTerrainCfg):
    """Part 1 bridge_bands: three 0.2 x 1.5 m bands, 0.10 m gaps, 0.20 m top."""

    num_bands: int = 3
    band_length: float = 0.2
    band_width: float = 1.5
    band_height: float = 0.2
    gap_range: tuple[float, float] = (0.05, 0.10)

    goal_dist: float = 1.8

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        gap = _lerp(*self.gap_range, difficulty)
        start = _course_start(self.size)
        half_h = self.band_height / 2
        for i in range(self.num_bands):
            cx = start + i * (self.band_length + gap) + self.band_length / 2
            geometries.append(
                _box(
                    body,
                    (self.band_length / 2, self.band_width / 2, half_h),
                    (cx, self.size[1] / 2, half_h),
                    _OBSTACLE_RGBA,
                )
            )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class HeightBarTerrainCfg(SubTerrainCfg):
    """Part 1 height_bar: pass under the beam without dropping it.

    The bar is static in training (the drop rule is checked by the judge).  Two
    side walls force the robot through the doorway, so the curriculum learns the
    crouch/low-posture skill rather than walking around the bar.
    """

    door_width: float = 1.2
    post_radius: float = 0.03
    beam_half_y: float = 0.6
    clearance_range: tuple[float, float] = (0.60, 0.55)
    side_wall_height: float = 0.9

    goal_dist: float = 2.4

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        center_y = self.size[1] / 2
        bar_x = self.size[0] / 2 + 1.5
        clearance = _lerp(*self.clearance_range, difficulty)
        post_h = clearance + 0.1
        half_y = self.door_width / 2
        for sign in (-1.0, 1.0):
            geometries.append(
                _cylinder(
                    body,
                    self.post_radius,
                    post_h / 2,
                    (bar_x, center_y + sign * half_y, post_h / 2),
                    _WALL_RGBA,
                )
            )
        geometries.append(
            _box(
                body,
                (0.015, self.beam_half_y, 0.015),
                (bar_x, center_y, clearance + 0.015),
                _OBSTACLE_RGBA,
            )
        )
        side_wall_len = (self.size[1] / 2 - half_y) / 2
        for sign in (-1.0, 1.0):
            geometries.append(
                _box(
                    body,
                    (0.05, side_wall_len, self.side_wall_height / 2),
                    (bar_x, center_y + sign * (half_y + side_wall_len), self.side_wall_height / 2),
                    _WALL_RGBA,
                )
            )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class HighWallTerrainCfg(SubTerrainCfg):
    """Part 1/2 high wall: a full-width wall the robot has to climb over."""

    wall_height_range: tuple[float, float] = (0.10, 0.30)
    wall_thickness: float = 0.06

    goal_dist: float = 2.2

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        wall_x = self.size[0] / 2 + 1.2
        height = _lerp(*self.wall_height_range, difficulty)
        geometries.append(
            _box(
                body,
                (self.wall_thickness / 2, self.size[1] / 2, height / 2),
                (wall_x, self.size[1] / 2, height / 2),
                _WALL_RGBA,
            )
        )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class SlalomTerrainCfg(SubTerrainCfg):
    """Part 1 slalom: alternating poles forcing an S-shaped path."""

    num_poles: int = 3
    pole_radius: float = 0.04
    pole_height: float = 0.7
    spacing_x: float = 0.8
    offset_y_range: tuple[float, float] = (0.55, 0.85)

    goal_dist: float = 2.8

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        center_y = self.size[1] / 2
        start = _course_start(self.size)
        offset_y = _lerp(*self.offset_y_range, difficulty)
        for i in range(self.num_poles):
            y = center_y + (offset_y if i % 2 == 0 else -offset_y)
            geometries.append(
                _cylinder(
                    body,
                    self.pole_radius,
                    self.pole_height / 2,
                    (start + i * self.spacing_x, y, self.pole_height / 2),
                    _OBSTACLE_RGBA,
                )
            )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class TrenchTerrainCfg(SubTerrainCfg):
    """Part 2 trench: a floor gap 0.15-0.35 m wide to step over/through."""

    gap_width_range: tuple[float, float] = (0.15, 0.35)
    gap_depth: float = 0.30

    goal_dist: float = 2.4

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        gap = _lerp(*self.gap_width_range, difficulty)
        gap_start = self.size[0] / 2 + 1.4 - gap / 2
        gap_end = gap_start + gap
        half_t = _FLOOR_THICKNESS / 2
        geometries = [
            _box(
                body,
                (gap_start / 2, self.size[1] / 2, half_t),
                (gap_start / 2, self.size[1] / 2, -half_t),
                _FLOOR_RGBA,
            ),
            _box(
                body,
                ((self.size[0] - gap_end) / 2, self.size[1] / 2, half_t),
                ((self.size[0] + gap_end) / 2, self.size[1] / 2, -half_t),
                _FLOOR_RGBA,
            ),
            _box(
                body,
                (gap / 2, self.size[1] / 2, 0.01),
                (gap_start + gap / 2, self.size[1] / 2, -self.gap_depth),
                _WALL_RGBA,
            ),
        ]
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class PlatformTerrainCfg(SubTerrainCfg):
    """Part 2 high platform: climb a 0.10-0.40 m platform (half-height step first)."""

    platform_height_range: tuple[float, float] = (0.10, 0.40)
    platform_size: float = 0.7

    goal_dist: float = 2.2

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        center_y = self.size[1] / 2
        height = _lerp(*self.platform_height_range, difficulty)
        step_x = self.size[0] / 2 + 1.0
        top_x = self.size[0] / 2 + 1.6
        geometries.append(
            _box(
                body,
                (0.35, 0.7, height / 4),
                (step_x, center_y, height / 4),
                _STONE_RGBA,
            )
        )
        geometries.append(
            _box(
                body,
                (self.platform_size / 2, self.platform_size / 2, height / 2),
                (top_x, center_y, height / 2),
                _OBSTACLE_RGBA,
            )
        )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


@dataclass(kw_only=True)
class SteppingStonesTerrainCfg(SubTerrainCfg):
    """Part 2 plum (stepping stones): 0.20-0.30 m cubes at 0.20 m height."""

    num_stones: int = 4
    stone_size_range: tuple[float, float] = (0.30, 0.20)
    stone_height: float = 0.2
    spacing_x: float = 0.55
    offset_y: float = 0.28

    goal_dist: float = 3.4

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geometries = [_patch_floor(body, self.size)]
        center_y = self.size[1] / 2
        stone = _lerp(*self.stone_size_range, difficulty)
        half_h = self.stone_height / 2
        start_x = self.size[0] / 2 + 0.7
        geometries.append(
            _box(
                body,
                (0.4, 0.4, half_h),
                (start_x, center_y, half_h),
                _STONE_RGBA,
            )
        )
        for i in range(self.num_stones):
            sign = 1.0 if i % 2 == 0 else -1.0
            geometries.append(
                _box(
                    body,
                    (stone / 2, stone / 2, half_h),
                    (
                        start_x + 0.55 + i * self.spacing_x,
                        center_y + sign * self.offset_y,
                        half_h,
                    ),
                    _OBSTACLE_RGBA,
                )
            )
        geometries.append(
            _box(
                body,
                (0.4, 0.4, half_h),
                (start_x + 0.55 + self.num_stones * self.spacing_x, center_y, half_h),
                _STONE_RGBA,
            )
        )
        return TerrainOutput(
            origin=_spawn_origin(self.size),
            geometries=geometries,
            flat_patches=_goal_patches(self),
        )


def black_competition_terrain_generator_cfg() -> TerrainGeneratorCfg:
    """Mixed competition-proxy terrain with a 0..1 difficulty sweep.

    One column per obstacle type (mjlab curriculum mode); proportions weight the
    robot spawn distribution.  Easy rows are flat/slopes/stairs, later rows add
    gravel, bridges, bars, walls, slalom, trenches, platforms and stones.
    """
    terrain_params = BLACK_CONFIG.terrain
    slope_range = (0.0, 1.0)  # 45 deg at difficulty 1.0
    stairs_step = (
        terrain_params.stair_step_height_base,
        terrain_params.stair_step_height_base + terrain_params.stair_step_height_gain,
    )
    return TerrainGeneratorCfg(
        curriculum=True,
        size=terrain_params.size,
        border_width=terrain_params.border_width,
        num_rows=6,
        difficulty_range=(0.0, 1.0),
        add_lights=True,
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=0.06),
            "smooth_slope_up": HfPyramidSlopedTerrainCfg(
                proportion=0.05,
                slope_range=slope_range,
                platform_width=terrain_params.platform_width,
                horizontal_scale=terrain_params.horizontal_scale,
                vertical_scale=terrain_params.vertical_scale,
                inverted=False,
            ),
            "smooth_slope_down": HfPyramidSlopedTerrainCfg(
                proportion=0.05,
                slope_range=slope_range,
                platform_width=terrain_params.platform_width,
                horizontal_scale=terrain_params.horizontal_scale,
                vertical_scale=terrain_params.vertical_scale,
                inverted=True,
            ),
            "stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=0.06,
                step_height_range=stairs_step,
                step_width=terrain_params.stair_step_width,
                platform_width=terrain_params.platform_width,
            ),
            "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.06,
                step_height_range=stairs_step,
                step_width=terrain_params.stair_step_width,
                platform_width=terrain_params.platform_width,
            ),
            "gravel": HfDiscreteObstaclesTerrainCfg(
                proportion=0.08,
                obstacle_width_range=(0.05, 0.20),
                obstacle_height_range=(0.02, 0.10),
                num_obstacles=120,
                platform_width=1.5,
                horizontal_scale=0.05,
                vertical_scale=0.005,
            ),
            "bridge_blocks": BridgeBlocksTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.12),
            "bridge_bands": BridgeBandsTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.12),
            "height_bar": HeightBarTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.12),
            "high_wall": HighWallTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.12),
            "slalom": SlalomTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.08),
            "trench": TrenchTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.10),
            "platform": PlatformTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.10),
            "stepping_stones": SteppingStonesTerrainCfg(flat_patch_sampling={"goal": FlatPatchSamplingCfg(num_patches=1, patch_radius=0.4)}, proportion=0.08),
        },
    )
