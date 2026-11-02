#!/usr/bin/env python3
"""RecoveryLoco-style robustness probe for the velocity-tracking primitive.

The training recipe samples random initial states (yaw/roll/pitch, root
velocity, joint offsets), pushes the trunk periodically and randomizes wheel
friction.  This probe checks that the exported policy actually tolerates those
disturbances on the deployment contract (CPU MuJoCo, 50 Hz policy, 5 ms PD).

Cases per policy (each starts from a fresh reset + settle):
  1. yaw offset  +-30 deg, command 0.3 m/s
  2. roll offset +-15 deg, command 0.3 m/s
  3. pitch offset +-15 deg, command 0.3 m/s
  4. root velocity impulse x=+0.6 m/s, command 0.3 m/s
  5. root velocity impulse y=+0.5 m/s, command 0.3 m/s
  6. yaw-rate impulse +0.5 rad/s, command 0 m/s
  7. joint position offset +-0.2 rad on leg joints, command 0.3 m/s
  8. mid-run push: +0.5 m/s x at t=1.2 s while holding 0.3 m/s

Pass = no fall (tilt<45 deg, z>0.25) and the command is re-acquired
(|vx_measured - cmd_x| <= 0.20 m/s sustained 0.5 s) within 2.0 s.

Usage::
    PYTHONPATH=eval MUJOCO_GL=egl .pixi/envs/eval/bin/python \
        tools/tracking/robustness_probe.py --policies vt_v2_800 --out ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from robocon_eval.policy_runner import PolicyRunner  # noqa: E402

SCENE = ROOT / "demo/robots/blackW/smoke_flat.xml"
CMD = np.array([0.30, 0.0, 0.0])
RECOVERY_ERR_MPS = 0.20
RECOVERY_HOLD_S = 0.5
RECOVERY_WINDOW_S = 2.0
TILT_FAIL_DEG = 45.0
BASE_Z_MIN = 0.25


def _tilt_deg(r: PolicyRunner) -> float:
    q = np.asarray(r.data.qpos[3:7], dtype=np.float64)
    w, x, y, z = (float(v) for v in q)
    rzz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, rzz))))


def _vx_body(r: PolicyRunner) -> float:
    q = np.asarray(r.data.qpos[3:7], dtype=np.float64)
    w, x, y, z = (float(v) for v in q)
    # rotate world velocity into body frame
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return float((rot.T @ np.asarray(r.data.qvel[:3], dtype=np.float64))[0])


def _apply_yaw_offset(r: PolicyRunner, deg: float) -> None:
    half = math.radians(deg) * 0.5
    qz = np.array((math.cos(half), 0.0, 0.0, math.sin(half)))
    q = np.asarray(r.data.qpos[3:7], dtype=np.float64)
    # quaternion multiply qz * q
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = q
    out = np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )
    r.data.qpos[3:7] = out / np.linalg.norm(out)


def _apply_roll_pitch_offset(r: PolicyRunner, roll_deg: float, pitch_deg: float) -> None:
    half_r = math.radians(roll_deg) * 0.5
    half_p = math.radians(pitch_deg) * 0.5
    qx = np.array((math.cos(half_r), math.sin(half_r), 0.0, 0.0))
    qy = np.array((math.cos(half_p), 0.0, math.sin(half_p), 0.0))
    q = np.asarray(r.data.qpos[3:7], dtype=np.float64)

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )

    out = mul(qx, mul(qy, q))
    r.data.qpos[3:7] = out / np.linalg.norm(out)


def _run_case(policy: str, name: str, *, perturb, cmd: np.ndarray,
              push_at: float | None = None, seconds: float = 4.0) -> dict:
    r = PolicyRunner("blackW", 1, policy, scene=SCENE, command=(0.0, 0.0, 0.0),
                     settle_s=1.0, seed=0, profile="diagnostic")
    try:
        r.reset()
        r.settle(1.0)
        perturb(r)
        mujoco.mj_forward(r.model, r.data)
        t = 0.0
        ok_since = None
        recovery_s = None
        max_tilt = 0.0
        min_z = 99.0
        fallen = False
        pushed = False
        while t < seconds:
            if push_at is not None and not pushed and t >= push_at:
                r.data.qvel[0] += 0.5
                r.data.qvel[1] += 0.0
                mujoco.mj_forward(r.model, r.data)
                pushed = True
            r.step(cmd)
            t += float(r.policy_dt)
            tilt = _tilt_deg(r)
            max_tilt = max(max_tilt, tilt)
            min_z = min(min_z, float(r.data.qpos[2]))
            if tilt > TILT_FAIL_DEG or float(r.data.qpos[2]) < BASE_Z_MIN:
                fallen = True
                break
            err = abs(_vx_body(r) - float(cmd[0]))
            if err <= RECOVERY_ERR_MPS:
                if ok_since is None:
                    ok_since = t
                elif recovery_s is None and t - ok_since >= RECOVERY_HOLD_S:
                    recovery_s = t - RECOVERY_HOLD_S
            else:
                ok_since = None
        recovered = recovery_s is not None and recovery_s <= RECOVERY_WINDOW_S
        return {
            "case": name,
            "cmd": cmd.tolist(),
            "fallen": fallen,
            "recovered": bool(recovered),
            "recovery_s": None if recovery_s is None else round(float(recovery_s), 3),
            "max_tilt_deg": round(max_tilt, 2),
            "min_base_z_m": round(min_z, 4),
            "passed": bool(not fallen and recovered),
        }
    finally:
        r.close()


def run_policy(policy: str) -> dict:
    cases: list[dict] = []
    cases.append(_run_case(
        policy, "yaw_offset_+30", perturb=lambda r: _apply_yaw_offset(r, 30.0),
        cmd=CMD))
    cases.append(_run_case(
        policy, "yaw_offset_-30", perturb=lambda r: _apply_yaw_offset(r, -30.0),
        cmd=CMD))
    cases.append(_run_case(
        policy, "roll_offset_+15", perturb=lambda r: _apply_roll_pitch_offset(r, 15.0, 0.0),
        cmd=CMD))
    cases.append(_run_case(
        policy, "roll_offset_-15", perturb=lambda r: _apply_roll_pitch_offset(r, -15.0, 0.0),
        cmd=CMD))
    cases.append(_run_case(
        policy, "pitch_offset_+15", perturb=lambda r: _apply_roll_pitch_offset(r, 0.0, 15.0),
        cmd=CMD))
    cases.append(_run_case(
        policy, "vel_impulse_x_+0.6",
        perturb=lambda r: r.data.qvel.__setitem__(0, 0.6), cmd=CMD))
    cases.append(_run_case(
        policy, "vel_impulse_y_+0.5",
        perturb=lambda r: r.data.qvel.__setitem__(1, 0.5), cmd=CMD))
    cases.append(_run_case(
        policy, "yawrate_impulse_+0.5",
        perturb=lambda r: r.data.qvel.__setitem__(5, 0.5),
        cmd=np.zeros(3), seconds=3.0))
    cases.append(_run_case(
        policy, "joint_offset_+0.2",
        perturb=lambda r: r.data.qpos.__setitem__(
            r.qpos_adr, r.data.qpos[r.qpos_adr] + 0.2), cmd=CMD))
    cases.append(_run_case(
        policy, "joint_offset_-0.2",
        perturb=lambda r: r.data.qpos.__setitem__(
            r.qpos_adr, r.data.qpos[r.qpos_adr] - 0.2), cmd=CMD))
    cases.append(_run_case(
        policy, "push_while_tracking_+0.5",
        perturb=lambda r: None, cmd=CMD, push_at=1.2))
    passed = sum(1 for c in cases if c["passed"])
    return {
        "policy": policy,
        "cases": cases,
        "summary": {"passed": passed, "total": len(cases),
                    "all_pass": passed == len(cases)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True, help="comma-separated policy dirs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out: dict[str, object] = {"policies": {}}
    all_ok = True
    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        print(f"=== {policy} ===")
        result = run_policy(policy)
        out["policies"][policy] = result
        for c in result["cases"]:
            print(f"  {c['case']:26s} passed={c['passed']} "
                  f"tilt={c['max_tilt_deg']:5.1f} recovery={c['recovery_s']}")
        s = result["summary"]
        print(f"  --> {s['passed']}/{s['total']} passed")
        all_ok = all_ok and s["all_pass"]
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"saved {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
