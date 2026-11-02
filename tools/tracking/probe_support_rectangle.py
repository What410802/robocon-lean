#!/usr/bin/env python3
"""Support-rectangle tracking probe for the pose-primitive policy (blackW).

Criterion (user, 2026-11-01): the velocity-tracking model must make the
*wheel-contact rectangle* follow the command, not just the body — in-place
rotation should rotate the rectangle about its own centroid (small drift),
and translation should move the centroid at the commanded speed.  The
position-loop primitive needs this so every obstacle expert that follows
starts from a well-placed stance.

Measurements (flat smoke scene, settle 1.0 s, hold ~3 s, tail steady state):

* rotate (0,0,+-wz): rectangle-centroid drift |Δc|, net yaw turned Δψ,
  drift ratio |Δc| / |Δψ| [m/rad] (target <= ~0.05 m/rad, i.e. ~0.08 m per
  90 deg) and drift speed |Δc|/T [m/s];
* translate (vx,0,0): centroid speed vs commanded vx (target within 20%).

The rectangle uses the four wheel-centre XY positions (ground projection of
the contact points for an upright wheel).

Usage:
    MUJOCO_GL=egl .pixi/envs/eval/bin/python tools/tracking/probe_support_rectangle.py \
        --policies vt_pose_v1,artifacts/velocity-tracking/pose/vt_pose_v2_r3 \
        --out artifacts/velocity-tracking/pose/support_rect.json
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


def _label(policy: str) -> str:
	p = Path(policy)
	return p.name if (p / "policy.pt").is_file() else policy


class RectProbe:
	def __init__(self, policy: str, cmd=(0.0, 0.0, 0.0)):
		self.r = PolicyRunner("blackW", 1, policy, scene=SCENE,
		                      command=cmd, settle_s=1.0, seed=0,
		                      profile="diagnostic")
		self.r.reset()
		self.r.settle(1.0)

	def rect(self) -> tuple[np.ndarray, float]:
		"""Centroid (x, y) and yaw of the wheel-contact rectangle."""
		pts = []
		for leg in ("FL", "FR", "RL", "RR"):
			g = self.r.wheel_geoms[leg]
			pos = np.asarray(self.r.data.geom_xpos[g], dtype=np.float64)
			pts.append(pos[:2])
		pts = np.stack(pts)
		centroid = pts.mean(axis=0)
		front = 0.5 * (pts[0] + pts[1])  # FL, FR midpoint
		rear = 0.5 * (pts[2] + pts[3])  # RL, RR midpoint
		yaw = math.atan2(front[1] - rear[1], front[0] - rear[0])
		return centroid, yaw

	def hold(self, cmd, seconds: float) -> dict:
		n = int(round(seconds / self.r.policy_dt))
		tilts = []
		for _ in range(n):
			self.r.step(np.asarray(cmd, dtype=np.float64))
			fr = self.r.frame()
			tilts.append(float(fr["tilt_deg"]))
			if float(fr["tilt_deg"]) > 70.0 or float(fr["base"][2]) < 0.2:
				break
		return {"steps": len(tilts), "tilt": max(tilts) if tilts else 0.0,
		        "fall": bool(tilts and (tilts[-1] > 70.0))}

	def rotate(self, wz: float, seconds: float = 3.0) -> dict:
		c0, y0 = self.rect()
		info = self.hold((0.0, 0.0, wz), seconds)
		c1, y1 = self.rect()
		dyaw = (y1 - y0 + math.pi) % (2.0 * math.pi) - math.pi
		drift = float(np.linalg.norm(c1 - c0))
		return {"cmd": [0.0, 0.0, wz], "dyaw_rad": round(dyaw, 3),
		        "centroid_drift_m": round(drift, 4),
		        "drift_m_per_rad": round(drift / max(abs(dyaw), 1e-6), 4),
		        "drift_m_per_90deg": round(drift / max(abs(dyaw), 1e-6)
		                                  * (math.pi / 2.0), 4),
		        "drift_speed_mps": round(drift / max(seconds, 1e-6), 4),
		        **info}

	def translate(self, vx: float, seconds: float = 2.4) -> dict:
		c0, _ = self.rect()
		info = self.hold((vx, 0.0, 0.0), seconds)
		c1, _ = self.rect()
		disp = c1 - c0
		speed = float(np.linalg.norm(disp)) / seconds
		return {"cmd": [vx, 0.0, 0.0],
		        "centroid_speed_mps": round(speed, 4),
		        "speed_ratio": round(speed / max(abs(vx), 1e-6), 3),
		        "lateral_drift_m": round(abs(float(disp[1])), 4)
		        if abs(vx) > 0.05 else 0.0,
		        **info}


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--policies", default="vt_pose_v1")
	ap.add_argument("--out", type=Path,
	                default=ROOT / "artifacts/velocity-tracking/pose/support_rect.json")
	args = ap.parse_args()
	result = {"scene": str(SCENE), "policies": {}}
	for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
		name = _label(policy)
		print(f"\n=== {name}", flush=True)
		rows = {"rotate": [], "translate": []}
		for wz in (0.2, -0.2, 0.3, -0.3, 0.45, -0.45):
			row = RectProbe(policy).rotate(wz)
			rows["rotate"].append(row)
			print(f"  rotate wz={wz:+.2f} -> dyaw={row['dyaw_rad']:+.2f} rad "
			      f"drift={row['centroid_drift_m']:.3f} m "
			      f"({row['drift_m_per_90deg']:.3f} m/90deg) fall={row['fall']}",
			      flush=True)
		for vx in (0.15, 0.30):
			row = RectProbe(policy).translate(vx)
			rows["translate"].append(row)
			print(f"  translate vx={vx:+.2f} -> centroid speed "
			      f"{row['centroid_speed_mps']:.3f} m/s "
			      f"(ratio {row['speed_ratio']:.2f}) fall={row['fall']}", flush=True)
		rot_ok = [r_["drift_m_per_90deg"] for r_ in rows["rotate"] if not r_["fall"]]
		tr_ok = [r_["speed_ratio"] for r_ in rows["translate"] if not r_["fall"]]
		summary = {
			"rotate_drift_m_per_90deg_max": round(max(rot_ok), 4) if rot_ok else None,
			"translate_speed_ratio_min": round(min(tr_ok), 3) if tr_ok else None,
		}
		result["policies"][name] = {"rows": rows, "summary": summary}
		print(f"--- {name} summary: {summary}", flush=True)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(json.dumps(result, indent=2) + "\n")
	print(f"\nsaved {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
