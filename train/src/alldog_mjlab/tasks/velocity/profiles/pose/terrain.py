"""Platform-edge sub-terrain for the pose-primitive flat task (blackW).

Why this exists
---------------
The plan-A Part1 route bridges obstacles with a flat velocity-tracking policy
(``vt_pose_*``).  Its measured failure modes near obstacles are edge-related:
after an expert crossing the primitive turns on top of a ~0.2 m platform (e.g.
``bridge_platform_r`` at the bridge-blocks exit) and a wheel can drop off the
edge; the harness also records the flat policy tripping over step edges it
cannot see.  The 2026-11-01 direction: train the primitive itself to be
edge-robust — under a command that points back onto the platform, the dropped
wheel must come back up.

Terrain design (one sub-terrain, ``pose_edge``):

* an 8x8 flat patch (same size as the rough-task grid) with one platform box
  and, with probability ``block_prob``, one small block (bridge-block
  footprint);
* the platform height follows the difficulty row, ``h = h_min + d (h_max -
  h_min)``; the default 0.05..0.20 m sweep covers the Part1 edge heights
  (gravel-pit lip ~0.08 m, t-stairs steps 0.10 m, bridge blocks / far platform
  0.20 m);
* two placement modes per cell (sampled by the generator rng):
  - ON-TOP: the platform covers the patch-centre spawn origin and the output
    origin z equals the platform height, so the robot starts on top.  Random
    envelope commands then drive it across edges (drop-off / straddle /
    re-climb) — the northeast-platform turn scenario.  The spawn-randomization
    xy range used with this terrain (±0.12 m, see ``assembly``) stays inside
    the platform footprint (half extents 0.6/0.9 m minus the ≤0.25 m placement
    offset leaves ≥0.35 m margin), so an on-top spawn never intersects the
    platform side.
  - BESIDE: the platform's nearest edge sits 0.38..0.55 m from the spawn box
    (orientation-aware placement: centre distance = clearance + box support
    extent, ~0.98..1.45 m), so episodes practise approaching and mounting an
    edge from the floor at envelope speeds — close enough that one 5 s command
    segment reaches the edge, which the round-1 fixed 1.3..1.7 m placement
    rarely allowed.

Training-only code: the deployment/eval contract (57x6 obs, 16 act) is
unchanged.  Heights and proportions are knobs (``RC_POSE_EDGE_*``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mjlab.terrains import BoxFlatTerrainCfg, SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainOutput

from alldog_mjlab.tasks.velocity.profiles.part2.terrain import (
	_OBSTACLE_RGBA,
	_box,
	_floor,
)
from alldog_mjlab.utils import knobs


def _clamp(v: float, lo: float, hi: float) -> float:
	return float(min(max(v, lo), hi))


@dataclass(kw_only=True)
class PoseEdgePlatformTerrainCfg(SubTerrainCfg):
	"""Flat patch + one platform (and optionally one block) with climbable edges."""

	height_range: tuple[float, float] = (0.05, 0.20)
	"""Platform/block top height at difficulty 0 / 1 [m]."""
	platform_half: tuple[float, float] = (0.60, 0.90)
	"""Platform half extents [m] (1.2 x 1.8 m footprint, cf. bridge_platform_r)."""
	block_half: tuple[float, float] = (0.20, 0.50)
	"""Small block half extents [m] (0.4 x 1.0 m footprint, cf. bridge blocks)."""
	on_top_prob: float = 0.45
	"""Probability that the platform covers the spawn origin (on-top mode)."""
	on_top_offset_max: float = 0.25
	"""Max platform-centre offset from the patch centre in on-top mode [m]."""
	beside_clearance_range: tuple[float, float] = (0.38, 0.55)
	"""Nearest-edge clearance from the spawn box in beside mode [m]; the centre
	distance becomes clearance + orientation-dependent box support extent.
	Lower bound 0.38 keeps the worst-case spawn footprint (xy randomization
	+ front-wheel reach, ~0.35 m) clear of the platform side."""
	block_prob: float = 0.5
	"""Probability of adding the small block on the opposite side."""
	border_margin: float = 0.45
	"""Keep every box this far inside the patch border [m]."""

	def function(
		self, difficulty: float, spec, rng: np.random.Generator
	) -> TerrainOutput:
		h_min, h_max = self.height_range
		height = h_min + difficulty * (h_max - h_min)
		body = spec.body("terrain")
		geoms = [_floor(body, self.size)]
		cx, cy = self.size[0] / 2.0, self.size[1] / 2.0

		def support_extent(half: tuple[float, float], theta: float) -> float:
			"""Axis-aligned box boundary distance along (cos t, sin t)."""
			ct, st = abs(math.cos(theta)), abs(math.sin(theta))
			return min(half[0] / max(ct, 1e-9), half[1] / max(st, 1e-9))

		def place(half: tuple[float, float], radius: float, theta: float) -> tuple[float, float]:
			px = _clamp(
				cx + radius * math.cos(theta),
				half[0] + self.border_margin,
				self.size[0] - half[0] - self.border_margin,
			)
			py = _clamp(
				cy + radius * math.sin(theta),
				half[1] + self.border_margin,
				self.size[1] - half[1] - self.border_margin,
			)
			return px, py

		on_top = rng.random() < self.on_top_prob
		half = tuple(float(v) for v in self.platform_half)
		if rng.random() < 0.5:
			half = (half[1], half[0])
		theta = rng.uniform(0.0, 2.0 * math.pi)
		if on_top:
			radius = rng.uniform(0.0, self.on_top_offset_max)
		else:
			# Orientation-aware near placement: hold the platform's nearest
			# edge `clearance` from the spawn box (+-xy spawn randomization) so
			# a spawn never penetrates the platform, while keeping the edge
			# close enough (0.95..1.35 m) that one 5 s command segment at
			# envelope speed reaches it — round 1's fixed 1.3..1.7 m gave too
			# few floor->edge approach samples for the mount skill to emerge.
			clearance = rng.uniform(*self.beside_clearance_range)
			radius = clearance + support_extent(half, theta)
		px, py = place(half, radius, theta)
		geoms.append(
			_box(
				body,
				(half[0], half[1], height / 2.0),
				(px, py, height / 2.0),
				_OBSTACLE_RGBA,
			)
		)

		if rng.random() < self.block_prob:
			bh = tuple(float(v) for v in self.block_half)
			if rng.random() < 0.5:
				bh = (bh[1], bh[0])
			theta2 = theta + math.pi + rng.uniform(-0.7, 0.7)
			clearance2 = rng.uniform(*self.beside_clearance_range)
			r2 = clearance2 + support_extent(bh, theta2)
			bx, by = place(bh, r2, theta2)
			geoms.append(
				_box(
					body,
					(bh[0], bh[1], height / 2.0),
					(bx, by, height / 2.0),
					_OBSTACLE_RGBA,
				)
			)

		origin_z = height if on_top else 0.0
		return TerrainOutput(
			origin=np.array([cx, cy, origin_z]), geometries=geoms
		)


def pose_edge_terrain_generator_cfg() -> TerrainGeneratorCfg:
	"""Two-column generator: pure flat + platform edges (knob-configured).

	``curriculum=True`` gives exactly two columns (one per sub-terrain);
	``proportion`` is the env spawn weight.  Rows are the height sweep; the
	terrain-levels curriculum term stays popped by the flat task, so envs keep a
	static random row in ``[0, num_rows - 1]`` (uniform height coverage, no
	curriculum dynamics to tune).
	"""
	num_rows = knobs.get_int("RC_POSE_EDGE_ROWS", 6)
	if num_rows < 2:
		raise ValueError(f"RC_POSE_EDGE_ROWS={num_rows} must be >= 2")
	h_min = knobs.get_float("RC_POSE_EDGE_H_MIN", 0.05)
	h_max = knobs.get_float("RC_POSE_EDGE_H_MAX", 0.20)
	if not (0.0 < h_min < h_max <= 0.40):
		raise ValueError(
			f"RC_POSE_EDGE_H_MIN/MAX=({h_min}, {h_max}) must satisfy 0 < min < max <= 0.40"
		)
	flat_prop = knobs.get_float("RC_POSE_EDGE_FLAT_PROP", 1.0)
	edge_prop = knobs.get_float("RC_POSE_EDGE_PROP", 2.0)
	if flat_prop <= 0.0 or edge_prop <= 0.0:
		raise ValueError("RC_POSE_EDGE_FLAT_PROP / RC_POSE_EDGE_PROP must be > 0")
	return TerrainGeneratorCfg(
		curriculum=True,
		size=(8.0, 8.0),
		border_width=20.0,
		num_rows=num_rows,
		difficulty_range=(0.0, 1.0),
		add_lights=True,
		sub_terrains={
			"flat": BoxFlatTerrainCfg(proportion=flat_prop),
			"pose_edge": PoseEdgePlatformTerrainCfg(
				proportion=edge_prop,
				height_range=(h_min, h_max),
				on_top_prob=knobs.get_float("RC_POSE_EDGE_ON_TOP_P", 0.45),
			),
		},
	)
