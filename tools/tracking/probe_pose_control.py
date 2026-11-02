#!/usr/bin/env python3
"""Low-speed command-response probe for closed-loop pose primitives (blackW).

Criterion (user, 2026-11-01): the pose-alignment policy must have monotonic vx/vy/wz
response inside the operating range and no command dead zone, so a position loop can
use it.  Steady-state response is measured on the flat smoke scene for:

* vx   : 0.05 .. 0.40 m/s  (+ reverse -0.05/-0.10)
* vy   : +-0.03 .. +-0.15 m/s at vx=0.10
* wz   : +-0.05 .. +-0.40 rad/s at vx=0.0 and vx=0.10

Outputs a JSON summary with per-command measured response, monotonicity/dead-zone
flags, and a recommendation.  Run with the eval env:

    MUJOCO_GL=egl .pixi/envs/eval/bin/python tools/tracking/probe_pose_control.py \
        --policies flat,vt_flat_v2,vt_flat_v1 --out artifacts/velocity-tracking/pose/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from robocon_eval.policy_runner import PolicyRunner  # noqa: E402

SCENE = ROOT / "demo/robots/blackW/smoke_flat.xml"


def run_one(policy: str, cmd: tuple[float, float, float], hold_s: float = 2.2) -> dict:
    """Fresh runner per command; return steady tail mean body velocities."""
    r = PolicyRunner("blackW", 1, policy, scene=SCENE,
                     command=cmd, settle_s=1.0, seed=0, profile="diagnostic")
    r.reset()
    r.settle(1.0)
    n = int(round(hold_s / r.policy_dt))
    vx, vy, wz, tilts, zs = [], [], [], [], []
    fall = False
    for _ in range(n):
        r.step(np.asarray(cmd, dtype=np.float64))
        qvel = np.asarray(r.data.qvel, dtype=np.float64)
        vx.append(float(qvel[0]))
        vy.append(float(qvel[1]))
        wz.append(float(qvel[5]) if qvel.size > 5 else 0.0)
        fr = r.frame(cmd)
        tilts.append(float(fr["tilt_deg"]))
        zs.append(float(fr["base"][2]))
        if float(fr["tilt_deg"]) > 70.0 or float(fr["base"][2]) < 0.2:
            fall = True
            break
    tail = max(1, int(0.8 / r.policy_dt))
    return {
        "cmd": [float(v) for v in cmd],
        "vx": float(np.mean(vx[-tail:])),
        "vy": float(np.mean(vy[-tail:])),
        "wz": float(np.mean(wz[-tail:])),
        "tilt": float(max(tilts)) if tilts else 0.0,
        "z": float(np.mean(zs[-tail:])) if zs else 0.0,
        "fall": fall,
        "steps": len(vx),
    }


def sweep(policy: str, axis: str, values: list[float], base=(0.0, 0.0, 0.0),
          hold_s: float = 2.2) -> list[dict]:
    out = []
    for v in values:
        cmd = list(base)
        cmd[{"vx": 0, "vy": 1, "wz": 2}[axis]] = v
        row = run_one(policy, tuple(cmd), hold_s=hold_s)
        row["cmd_axis"] = axis
        row["cmd_value"] = float(v)
        out.append(row)
        print(f"{policy:14s} {axis}={v:+.2f} -> meas={row[axis]:+.4f} "
              f"(vx={row['vx']:+.3f} vy={row['vy']:+.3f} wz={row['wz']:+.3f}) "
              f"tilt={row['tilt']:.1f} fall={row['fall']}", flush=True)
    return out


def analyse(rows: list[dict], axis: str, min_gain: float = 0.35) -> dict:
    rows = [r for r in rows if not r["fall"]]
    xs = np.array([r["cmd_value"] for r in rows])
    ys = np.array([r[axis] for r in rows])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    sign_ok = bool(np.all(np.sign(ys[xs > 0]) >= 0) and np.all(np.sign(ys[xs < 0]) <= 0))
    if len(xs) >= 2:
        mono = np.diff(ys) >= -0.02
        mono_ok = bool(np.all(mono))
        # dead zone: gain vs command near zero; require response >= 30% of command or 0.02
        gain = np.abs(ys) / np.maximum(np.abs(xs), 1e-6)
        dead = bool(np.any((np.abs(ys) < 0.02) & (np.abs(xs) >= 0.05)))
        sat = bool(np.any(gain < min_gain))
    else:
        mono_ok = dead = sat = False
    return {"axis": axis, "rows": rows, "sign_ok": sign_ok,
            "monotonic": mono_ok, "dead_zone": dead, "low_gain": sat,
            "ok": bool(sign_ok and mono_ok and not dead and not sat)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="flat,vt_flat_v2,vt_flat_v1")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/velocity-tracking/pose/probe.json")
    args = ap.parse_args()
    result = {"scene": str(SCENE), "policies": {}}
    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        print(f"\n=== {policy}", flush=True)
        axes = {}
        axes["vx"] = sweep(policy, "vx", [0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                                          -0.05, -0.10])
        axes["vy"] = sweep(policy, "vy", [0.03, 0.06, 0.10, 0.15, -0.03, -0.06, -0.10],
                           base=(0.10, 0.0, 0.0))
        axes["wz_vx0"] = sweep(policy, "wz", [0.05, 0.10, 0.20, 0.30, 0.40,
                                              -0.05, -0.10, -0.20, -0.30, -0.40],
                               base=(0.0, 0.0, 0.0))
        # in-place turn is critical for alignment; a short forward creep can help
        axes["wz_vx010"] = sweep(policy, "wz", [0.10, 0.20, 0.30, -0.10, -0.20, -0.30],
                                 base=(0.10, 0.0, 0.0))
        summary = {name: analyse(rows, axis) for name, (rows, axis) in {
            "vx": (axes["vx"], "vx"),
            "vy": (axes["vy"], "vy"),
            "wz_vx0": (axes["wz_vx0"], "wz"),
            "wz_vx010": (axes["wz_vx010"], "wz"),
        }.items()}
        result["policies"][policy] = {"raw": axes, "summary": summary}
        print(f"--- {policy} summary:", {k: v["ok"] for k, v in summary.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
