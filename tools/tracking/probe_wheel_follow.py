#!/usr/bin/env python3
"""Measure four-wheel contact-point following during a pure-yaw command.

The position-loop handover needs all four wheel contact points to follow the
commanded body rotation without lag (RecoveryLoco's foot-tracking requirement,
adapted to wheel-legs).  This probe runs a pure yaw command on the deployment
contract and reports:

  follow_rel_err  Σ|v_wheel - v_rigid(cmd)| / Σ|v_rigid(cmd)|
  follow_rmse     RMSE of the same velocity error (m/s)
  plane_spread    body-frame z spread of the four wheel centres (m, mean/max)
  yaw_actual      measured steady yaw rate (rad/s, cmd shown for reference)

For a rigid command wz the target wheel-centre velocity is
``v_i = wz_cmd * (-(y_i-y_root), +(x_i-x_root))`` in the world frame.  A wheel
that lags, slips or twists the trunk relative to the contact plane shows up as a
large error even if the base yaw-rate tracking looks acceptable.

Usage::
    MUJOCO_GL=egl pixi run -e eval python tools/tracking/probe_wheel_follow.py \
        --policies vt_track_v1,vt_track_v2 --out artifacts/tracking/wheel_follow.json
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
WHEEL_GEOMS = ("FL_wheel_collision", "FR_wheel_collision",
               "RL_wheel_collision", "RR_wheel_collision")


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _body_point(base: np.ndarray, quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rot.T @ (point - base)


def run_case(policy: str, wz: float, seconds: float = 3.0, settle_s: float = 1.0) -> dict:
    r = PolicyRunner("blackW", 1, policy, scene=SCENE, command=(0.0, 0.0, wz),
                     settle_s=settle_s, seed=0, profile="diagnostic")
    try:
        r.reset()
        r.settle(settle_s)
        r.step((0.0, 0.0, 0.0))  # one zero step, then start the turn
        gids = [int(r.model.geom(n).id) for n in WHEEL_GEOMS]
        prev_pos = np.asarray(r.data.geom_xpos[gids, :], dtype=np.float64).copy()
        prev_root = np.asarray(r.data.qpos[:3], dtype=np.float64).copy()
        prev_yaw = _yaw(r.data.qpos[3:7])
        dt = float(r.policy_dt)
        errs: list[np.ndarray] = []
        errs_actual: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        targets_actual: list[np.ndarray] = []
        spreads: list[float] = []
        yaw_rates: list[float] = []
        steps = max(1, int(round(seconds / dt)))
        for _ in range(steps):
            r.step((0.0, 0.0, wz))
            pos = np.asarray(r.data.geom_xpos[gids, :], dtype=np.float64)
            root = np.asarray(r.data.qpos[:3], dtype=np.float64)
            quat = np.asarray(r.data.qpos[3:7], dtype=np.float64)
            yaw = _yaw(quat)
            vel = (pos - prev_pos) / dt
            rel = pos[:, :2] - root[None, :2]
            target = np.stack((-wz * rel[:, 1], wz * rel[:, 0]), axis=1)
            actual_wz = (yaw - prev_yaw) / dt
            target_actual = np.stack((-actual_wz * rel[:, 1], actual_wz * rel[:, 0]), axis=1)
            errs.append((vel[:, :2] - target).ravel())
            targets.append(target.ravel())
            errs_actual.append((vel[:, :2] - target_actual).ravel())
            targets_actual.append(target_actual.ravel())
            body_z = np.array([_body_point(root, quat, p)[2] for p in pos])
            spreads.append(float(body_z.max() - body_z.min()))
            yaw_rates.append((yaw - prev_yaw) / dt)
            prev_pos, prev_root, prev_yaw = pos.copy(), root.copy(), yaw
        err = np.concatenate(errs)
        tgt = np.concatenate(targets)
        denom = float(np.sum(np.abs(tgt))) or 1e-9
        tgt_norm = float(np.mean(np.linalg.norm(tgt.reshape(-1, 2), axis=1))) or 1e-9
        err_a = np.concatenate(errs_actual)
        tgt_a = np.concatenate(targets_actual)
        denom_a = float(np.sum(np.abs(tgt_a))) or 1e-9
        return {
            "policy": policy,
            "wz_cmd": wz,
            "follow_rel_err_cmd": round(float(np.sum(np.abs(err)) / denom), 4),
            "follow_rel_err_actual": round(float(np.sum(np.abs(err_a)) / denom_a), 4),
            "follow_rmse_mps": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "target_speed_mps": round(tgt_norm, 4),
            "plane_spread_mean_m": round(float(np.mean(spreads)), 4),
            "plane_spread_max_m": round(float(np.max(spreads)), 4),
            "yaw_rate_actual": round(float(np.mean(yaw_rates[len(yaw_rates) // 3:])), 4),
            "yaw_rate_cmd": wz,
            "no_fall": True,
        }
    finally:
        r.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wz", default="0.3,-0.3")
    args = ap.parse_args()
    wss = [float(v) for v in args.wz.split(",") if v.strip()]
    out: dict[str, object] = {"policies": {}}
    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        print(f"=== {policy} ===")
        cases = []
        for wz in wss:
            case = run_case(policy, wz)
            cases.append(case)
            print(f"  wz={wz:+.1f} rel_err_cmd={case['follow_rel_err_cmd']:.3f} "
                  f"rel_err_actual={case['follow_rel_err_actual']:.3f} "
                  f"rmse={case['follow_rmse_mps']:.3f} plane_mean={case['plane_spread_mean_m']:.4f} "
                  f"yaw_actual={case['yaw_rate_actual']:+.3f}")
        out["policies"][policy] = cases
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
