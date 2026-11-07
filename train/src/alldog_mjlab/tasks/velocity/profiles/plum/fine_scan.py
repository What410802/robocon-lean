"""Privileged fine terrain scan for the plum teacher.

Training-time ONLY.  Four downward ray grids, one attached to each wheel body,
report the wheel-centre clearance above the terrain at 1 cm resolution.  A
0.2 m cube is ~20 cells wide here, so the teacher can resolve exactly where a
cube rim is -- the coarse 187-ray trunk scan cannot.

At deployment the student predicts this vector from a depth image instead
(see ``student.py``); the scan never enters a deployable policy.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg

FINE_SCAN_GROUP = "fine_scan"
FINE_SCAN_SENSOR = "fine_terrain_scan"
WHEEL_BODIES = tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR"))

# Old repo's successful plum teacher used 0.01 m / 0.20 m / 4 frames = 1764 rays.
RESOLUTION = 0.01
PATCH_SIZE = 0.20
Z_OFFSET = 0.25          # ray origins must sit above the 0.2 m cube tops
MAX_DISTANCE = 2.0
OBS_SCALE = 1.0
RAW_CLIP = (-0.5, 0.6)   # metres of wheel-centre clearance


def _cells(resolution: float, size: float) -> int:
    """Odd ray count per axis, so the grid is symmetric about the wheel centre."""
    n = max(3, int(round(size / resolution)))
    return n + 1 if n % 2 == 0 else n


@dataclass(frozen=True)
class RaisedGridPatternCfg:
    """Square grid of downward rays lifted by ``z_offset`` in the attached frame.

    ``GridPatternCfg`` hard-codes a zero local z origin; a wheel-centre-height
    origin cannot see a 0.2 m cube standing above it.  This is the minimal
    extension used by the old repo (kept verbatim in behaviour).
    """

    size: float = PATCH_SIZE
    resolution: float = RESOLUTION
    z_offset: float = Z_OFFSET

    def generate_rays(self, mj_model, device):
        n = _cells(self.resolution, self.size)
        centre = (n - 1) / 2.0
        coords = (torch.arange(n, device=device, dtype=torch.float32) - centre) * self.resolution
        gx, gy = torch.meshgrid(coords, coords, indexing="xy")
        offsets = torch.zeros((n * n, 3), device=device, dtype=torch.float32)
        offsets[:, 0], offsets[:, 1] = gx.flatten(), gy.flatten()
        offsets[:, 2] = self.z_offset
        directions = torch.tensor((0.0, 0.0, -1.0), device=device).expand(n * n, 3).clone()
        return offsets, directions

    @property
    def rays_per_frame(self) -> int:
        return _cells(self.resolution, self.size) ** 2


def fine_scan_dim() -> int:
    return len(WHEEL_BODIES) * RaisedGridPatternCfg().rays_per_frame


def add_fine_scan(cfg: ManagerBasedRlEnvCfg) -> int:
    """Attach the sensor + a single-frame privileged observation group."""
    assert cfg.scene is not None
    names = {s.name for s in (cfg.scene.sensors or ())}
    if FINE_SCAN_SENSOR in names:
        raise ValueError(f"sensor {FINE_SCAN_SENSOR!r} already present")
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        RayCastSensorCfg(
            name=FINE_SCAN_SENSOR,
            frame=tuple(ObjRef(type="body", name=b, entity="robot") for b in WHEEL_BODIES),
            pattern=RaisedGridPatternCfg(),
            ray_alignment="yaw",
            max_distance=MAX_DISTANCE,
            exclude_parent_body=True,
            include_geom_groups=(0,),   # terrain collision geoms only
        ),
    )
    cfg.observations[FINE_SCAN_GROUP] = ObservationGroupCfg(
        terms={
            "height_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": FINE_SCAN_SENSOR, "offset": 0.0, "miss_value": 0.0},
                scale=OBS_SCALE,
                clip=RAW_CLIP,
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=None,
        flatten_history_dim=False,
    )
    dim = fine_scan_dim()
    print(f"[plum/fine_scan] {len(WHEEL_BODIES)} x {RaisedGridPatternCfg().rays_per_frame} "
          f"rays = {dim}-D privileged actor group {FINE_SCAN_GROUP!r}", flush=True)
    return dim
