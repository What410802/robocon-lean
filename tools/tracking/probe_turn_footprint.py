#!/usr/bin/env python3
"""Turn-footprint probe: the flat-ground area a model needs to turn in.

Motivation (user, 2026-11-01): the bridge-A-west choreography turns the robot
90 deg on a 1.0 x 1.0 m platform, and the runs failed with the robot drifting
0.5-0.95 m off the platform — i.e. the turn needs more room than the platform
provides.  The support-rectangle probe measures centroid drift per 90 deg, but
NOT the swept footprint of the four wheels, which is what determines the
required flat area (platform size / clearance to edges).

Measurements per (policy, wz):
* drive a pure in-place turn from rest until |dyaw| >= target (90/180 deg);
* record every wheel position over the turn, in the frame where the robot's
  initial heading is +x and the initial rectangle centroid is the origin;
* report:
  - footprint bbox (min/max x/y of all wheel samples): the free rectangle
    the turn sweeps;
  - footprint radius: max distance from the origin to any wheel sample — the
    radius of the free disc needed if orientation is arbitrary;
  - margin over the static rectangle: footprint extent minus the resting
    rectangle extent (~0.53 x 0.42 for blackW) — the extra room the turn
    costs beyond standing still;
  - centroid drift and turn time.

Usage:
    MUJOCO_GL=egl .pixi/envs/eval/bin/python tools/tracking/probe_turn_footprint.py \
        --policies vt_pose_v1,artifacts/velocity-tracking/pose/vt_pose_v2_r7 \
        --out artifacts/velocity-tracking/pose/turn_footprint.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from robocon_eval.policy_runner import PolicyRunner  # noqa: E402

SCENE = ROOT / "demo/robots/blackW/smoke_flat.xml"
WHEEL_RADIUS = 0.103


def _label(policy: str) -> str:
	p = Path(policy)
	return p.name if (p / "policy.pt").is_file() else policy


class FootProbe:
	def __init__(self, policy: str):
		self.r = PolicyRunner("blackW", 1, policy, scene=SCENE,
		                      command=(0.0, 0.0, 0.0), settle_s=1.0, seed=0,
		                      profile="diagnostic")
		self.r.reset()
		self.r.settle(1.0)

	def wheel_pts(self) -> np.ndarray:
		"""(4, 2) world XY of the wheel centres."""
		return np.stack([
			np.asarray(self.r.data.geom_xpos[g])[:2]
			for g in self.r.wheel_geoms.values()
		])

	def yaw(self) -> float:
		q = np.asarray(self.r.data.qpos, dtype=np.float64)
		w, x, y, z = (float(v) for v in q[3:7])
		return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

	def turn(self, wz: float, target_rad: float, timeout_s: float = 15.0) -> dict:
		pts0 = self.wheel_pts()
		c0 = pts0.mean(axis=0)
		yaw0 = self.yaw()
		# initial heading frame: rotate world deltas into the start heading.
		c, s = math.cos(-yaw0), math.sin(-yaw0)
		R = np.array([[c, -s], [s, c]])
		rest = (R @ (pts0 - c0).T).T
		samples = []
		t0 = float(self.r.data.time)
		fall = False
		dyaw = 0.0
		while float(self.r.data.time) - t0 < timeout_s:
			self.r.step(np.asarray((0.0, 0.0, wz), dtype=np.float64))
			dyaw = self.yaw() - yaw0
			rel = (R @ (self.wheel_pts() - c0).T).T
			samples.append(rel)
			fr = self.r.frame()
			if float(fr["tilt_deg"]) > 70.0 or float(fr["base"][2]) < 0.2:
				fall = True
				break
			if abs(dyaw) >= target_rad:
				break
		allpts = np.stack(samples) if samples else rest[None]
		extreme = allpts.reshape(-1, 2)
		bbox = [float(extreme[:, 0].min()), float(extreme[:, 0].max()),
		        float(extreme[:, 1].min()), float(extreme[:, 1].max())]
		radius = float(np.linalg.norm(extreme, axis=1).max())
		rest_extent = [float(rest[:, 0].min()), float(rest[:, 0].max()),
		               float(rest[:, 1].min()), float(rest[:, 1].max())]
		# extra room beyond standing still, per axis (>= 0)
		margin = [max(0.0, lo - rlo) if False else abs(lo - rlo)
		          for lo, rlo in zip(bbox[:2], rest_extent[:2])]
		margin = [abs(bbox[0] - rest_extent[0]), abs(bbox[1] - rest_extent[1]),
		          abs(bbox[2] - rest_extent[2]), abs(bbox[3] - rest_extent[3])]
		return {
			"wz": wz, "target_deg": round(math.degrees(target_rad), 1),
			"dyaw_deg": round(math.degrees(dyaw), 1),
			"footprint_bbox_m": [round(v, 3) for v in bbox],
			"footprint_radius_m": round(radius, 3),
			"rest_rectangle_m": [round(v, 3) for v in rest_extent],
			"extra_room_m": [round(v, 3) for v in margin],
			"turn_time_s": round(float(self.r.data.time) - t0, 2),
			"fall": fall,
		}


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--policies", default="vt_pose_v1")
	ap.add_argument("--targets-deg", default="90,180")
	ap.add_argument("--out", type=Path,
	                default=ROOT / "artifacts/velocity-tracking/pose/turn_footprint.json")
	args = ap.parse_args()
	targets = [math.radians(float(v)) for v in args.targets_deg.split(",")]
	result = {"scene": str(SCENE), "policies": {}}
	for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
		name = _label(policy)
		print(f"\n=== {name}", flush=True)
		rows = []
		for target in targets:
			for wz in (0.3, -0.3):
				row = FootProbe(policy).turn(wz, target)
				rows.append(row)
				print(f"  {row['target_deg']:5.0f}deg wz={wz:+.2f}: dyaw={row['dyaw_deg']:+6.1f} "
				      f"bbox=[{row['footprint_bbox_m'][0]:+.2f},{row['footprint_bbox_m'][1]:+.2f}]x"
				      f"[{row['footprint_bbox_m'][2]:+.2f},{row['footprint_bbox_m'][3]:+.2f}] "
				      f"radius={row['footprint_radius_m']:.2f} extra={row['extra_room_m']} "
				      f"t={row['turn_time_s']:.1f}s fall={row['fall']}", flush=True)
		r90 = [r_["footprint_radius_m"] for r_ in rows
		       if r_["target_deg"] == 90.0 and not r_["fall"]]
		result["policies"][name] = {"rows": rows,
		                            "radius_90deg_max": max(r90) if r90 else None}
		print(f"--- {name}: 90deg footprint radius max = "
		      f"{result['policies'][name]['radius_90deg_max']}", flush=True)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(json.dumps(result, indent=2) + "\n")
	print(f"\nsaved {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
