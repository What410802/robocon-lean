"""Simulated front depth camera for the plum student.

This module is the single source for the camera contract: ``DEPTH_CONTRACT`` is
written verbatim into the exported ``depth.json`` and is asserted equal to the
eval runner's ``DEFAULT_DEPTH_CONTRACT`` by ``scripts/plum/smoke.py``.  Both
sides render ideal simulated depth, so -- per the recipe decision -- NO camera
domain randomisation (noise / dropout / latency / extrinsic jitter) is applied.
Training and judging therefore see the same observation distribution.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg

CAMERA_GROUP = "camera"
DEPTH_TERM = "depth"
SENSOR_NAME = "depth_camera"

# Must match eval/robocon_eval/policy_runner.py::DEFAULT_DEPTH_CONTRACT exactly.
DEPTH_CONTRACT = {
    "camera_name": "depth_cam",
    "parent_body": "d435_link",
    "pos": [0.01, 0.0, 0.0],
    "quat": [0.5610, 0.4304, -0.4304, -0.5610],
    "fovy": 70.0,
    "width": 64,
    "height": 48,
    "min_depth": 0.10,
    "cutoff": 3.0,
    "input_layout": "1,1,H,W",
    "visible_geom_groups": [0],
    "hide_robot_geoms": True,
}

# rsl_rl CNNModel encoder: 3 conv layers -> global average pool -> 64-D latent.
DEPTH_CNN_CFG = {
    "output_channels": (16, 32, 64),
    "kernel_size": (5, 3, 3),
    "stride": (2, 2, 2),
    "padding": "zeros",
    "global_pool": "avg",
    "flatten": True,
}
CNN_LATENT_DIM = 64


def camera_depth(
    env: ManagerBasedRlEnv,
    sensor_name: str = SENSOR_NAME,
    cutoff_distance: float | None = None,
    min_depth: float | None = None,
) -> torch.Tensor:
    """Normalized depth ``(B, 1, H, W)`` in [0, 1].

    Mirrors ``policy_runner.render_depth()``: no-hit/sky -> cutoff, near clamp
    ``min_depth``, then divide by cutoff.
    """
    cutoff = DEPTH_CONTRACT["cutoff"] if cutoff_distance is None else float(cutoff_distance)
    near = DEPTH_CONTRACT["min_depth"] if min_depth is None else float(min_depth)
    sensor: CameraSensor = env.scene[sensor_name]
    raw = sensor.data.depth
    assert raw is not None, f"camera {sensor_name!r} produced no depth"
    depth = raw.permute(0, 3, 1, 2)  # (B, H, W, 1) -> (B, 1, H, W)
    far = torch.full_like(depth, cutoff)
    depth = torch.where(torch.isfinite(depth) & (depth > 0.0), depth, far)
    return torch.clamp(depth, min=near, max=cutoff) / cutoff


def add_depth_camera(cfg: ManagerBasedRlEnvCfg) -> None:
    """Attach the depth sensor + a corruption-free 2-D observation group."""
    assert cfg.scene is not None
    names = {s.name for s in (cfg.scene.sensors or ())}
    if SENSOR_NAME in names:
        raise ValueError(f"sensor {SENSOR_NAME!r} already present")
    # NOTE: ``camera_name`` in this mjlab version means "wrap an EXISTING scene
    # camera"; leaving it None makes mjlab CREATE a camera named ``SENSOR_NAME``
    # at pos/quat/fovy.  The eval judge injects its own camera named
    # DEPTH_CONTRACT["camera_name"] ("depth_cam") into the CPU scene.  The two
    # names differ by design; what must match is the pose / FOV / resolution /
    # geom-group contract (asserted in scripts/plum/smoke.py).
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        CameraSensorCfg(
            name=SENSOR_NAME,
            parent_body=f"robot/{DEPTH_CONTRACT['parent_body']}",
            pos=tuple(DEPTH_CONTRACT["pos"]),
            quat=tuple(DEPTH_CONTRACT["quat"]),
            fovy=DEPTH_CONTRACT["fovy"],
            width=DEPTH_CONTRACT["width"],
            height=DEPTH_CONTRACT["height"],
            data_types=("depth",),
            use_textures=False,
            use_shadows=False,
            enabled_geom_groups=(0,),
        ),
    )
    cfg.observations[CAMERA_GROUP] = ObservationGroupCfg(
        terms={
            DEPTH_TERM: ObservationTermCfg(func=camera_depth, params={"sensor_name": SENSOR_NAME})
        },
        concatenate_terms=True,
        enable_corruption=False,   # no camera DR: simulation-only judging
    )
    print(f"[plum/depth] {DEPTH_CONTRACT['width']}x{DEPTH_CONTRACT['height']} "
          f"fovy={DEPTH_CONTRACT['fovy']} cutoff={DEPTH_CONTRACT['cutoff']} m, "
          "no camera DR", flush=True)
