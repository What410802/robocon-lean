#!/usr/bin/env python3
"""Record a locked-wheel policy gait and dump per-step diagnostic data.

This is the missing "look at the gait" tool: it drives a policy through a fixed
command script (stand -> forward -> stop -> in-place yaw -> combined), records
base/joint/wheel/contact data at every 50 Hz step, renders an MP4 with foot and
joint overlays, and prints per-phase summaries (tracking, foot lift, stance
rectangle, slip, joint range).

Usage:
  MUJOCO_GL=egl pixi run -e eval python tools/tracking/record_gait_diagnostic.py \
    --policy artifacts/gait_diag/policy --out artifacts/gait_diag \
    --scene demo/robots/blackW/scene_part1.xml

The policy directory must contain ``policy.pt`` + a deployment ``config.yaml``
with ``lock_wheels: true`` (the policy's wheel outputs are ignored and the wheel
joints are held at q=0 with the wheel rl_kp/rl_kd).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from robocon_eval.policy_runner import PolicyRunner  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
NOMINAL = {"FL": (0.2553, 0.2231), "FR": (0.2570, -0.2234),
           "RL": (-0.1525, 0.2382), "RR": (-0.1513, -0.2367)}
CENTER = (0.0521, 0.0003)
HALF = (0.2050, 0.2380)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W, H = 640, 480
OUT_W, OUT_H = 900, 720

# (label, duration_s, (vx, vy, wz))
SCRIPT = (
    ("stand", 2.0, (0.0, 0.0, 0.0)),
    ("fwd_0.15", 3.0, (0.15, 0.0, 0.0)),
    ("stop", 1.5, (0.0, 0.0, 0.0)),
    ("lat_+0.10", 2.5, (0.0, 0.10, 0.0)),
    ("yaw_+0.3", 3.0, (0.0, 0.0, 0.30)),
    ("yaw_vy", 2.5, (0.0, 0.10, 0.30)),
    ("yaw_-0.3", 3.0, (0.0, 0.0, -0.30)),
    ("combined", 3.0, (0.10, 0.10, 0.20)),
    ("stop2", 1.5, (0.0, 0.0, 0.0)),
)


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def yaw_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def body_xy(base: np.ndarray, yaw: float, p: np.ndarray) -> tuple[float, float]:
    dx, dy = p[0] - base[0], p[1] - base[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def contact_forces(model, data, wheel_ids):
    force = {leg: 0.0 for leg in LEGS}
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        for leg, gid in wheel_ids.items():
            if gid in (g1, g2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f6)
                force[leg] += float(np.linalg.norm(f6[:3]))
    return force


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="policy dir with policy.pt/config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default=str(ROOT / "demo/robots/blackW/scene_part1.xml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth 50 Hz step")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = str(Path(args.policy).resolve())

    runner = PolicyRunner("blackW", 1, policy,
                          scene=Path(args.scene), settle_s=args.settle,
                          seed=args.seed)
    runner.reset()
    runner.settle(args.settle)
    model, data = runner.model, runner.data
    wheel_ids = {leg: int(g) for leg, g in runner.wheel_geoms.items()}
    qadr = np.asarray(runner.qpos_adr)
    madr = np.asarray(runner.qvel_adr)
    names = [str(n) for n in runner.joint_names]

    # Map joint names -> action-scale policy order (runner.joint_names order).
    leg_joint_idx = {leg: [i for i, n in enumerate(names)
                           if n.startswith(leg) and "_wheel_" not in n] for leg in LEGS}
    wheel_joint_idx = [i for i, n in enumerate(names) if "_wheel_" in n]

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.3, 135.0, -24.0
    f13, f11, f16 = font(13), font(11), font(16)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", "25", "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         str(out / "gait_diagnostic.mp4")],
        stdin=subprocess.PIPE,
    )
    rows: list[dict] = []
    phase_rows: dict[str, list[dict]] = {}
    frame_i = 0
    max_frames = 1250
    for label, duration, cmd in SCRIPT:
        n_steps = int(round(duration / runner.policy_dt))
        phase_rows[label] = []
        for k in range(n_steps):
            runner.step(np.asarray(cmd))
            d = runner.data
            base = np.array(d.qpos[0:3], dtype=np.float64)
            quat = np.array(d.qpos[3:7], dtype=np.float64)
            yaw = yaw_quat(quat)
            q = np.array(d.qpos[qadr], dtype=np.float64)
            dq = np.array(d.qvel[madr], dtype=np.float64)
            wheels = {}
            for leg, gid in wheel_ids.items():
                wheels[leg] = np.array(d.geom_xpos[gid], dtype=np.float64)
            forces = contact_forces(model, d, wheel_ids)
            feet = {leg: body_xy(base, yaw, wheels[leg]) for leg in LEGS}
            # approximate foot horizontal velocity from previous row (same leg)
            if rows:
                prev = rows[-1]
                prev_feet = {leg: np.array(prev["wheels_w"][leg]) for leg in LEGS}
                foot_speed = {leg: float(np.linalg.norm(wheels[leg][:2] - prev_feet[leg][:2])
                                         / runner.policy_dt) for leg in LEGS}
            else:
                foot_speed = {leg: 0.0 for leg in LEGS}
            body_v = np.array([d.qvel[0], d.qvel[1], 0.0])
            c, s = math.cos(yaw), math.sin(yaw)
            vx_b = c * body_v[0] + s * body_v[1]
            vy_b = -s * body_v[0] + c * body_v[1]
            # gyro/body angular velocity from free joint qvel (body frame)
            wz_b = float(d.qvel[5]) if d.qvel.size >= 6 else 0.0
            # joint ranges of interest
            row = {
                "t": float(d.time - runner.t0), "phase": label, "cmd": list(cmd),
                "base": base.tolist(), "yaw": yaw, "quat": quat.tolist(),
                "vx_b": float(vx_b), "vy_b": float(vy_b), "wz_b": wz_b,
                "q": q.tolist(), "dq": dq.tolist(),
                "names": names,
                "wheels_w": {leg: wheels[leg].tolist() for leg in LEGS},
                "feet_body": {leg: list(feet[leg]) for leg in LEGS},
                "contact_force": forces,
                "foot_speed": foot_speed,
            }
            rows.append(row)
            phase_rows[label].append(row)
            if k % args.stride == 0:
                cam.lookat = base
                renderer.update_scene(d, camera=cam)
                im = Image.fromarray(renderer.render())
                canvas = Image.new("RGB", (OUT_W, OUT_H), (16, 18, 22))
                canvas.paste(im, (0, 92))
                dr = ImageDraw.Draw(canvas)
                dr.rectangle((0, 0, OUT_W, 90), fill=(0, 0, 0))
                dr.text((8, 2), f"locked gait diagnostic  t={row['t']:5.2f}s  phase={label}",
                        font=f16, fill=(245, 245, 245))
                dr.text((8, 26), f"cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}  actual vx={vx_b:+.2f} vy={vy_b:+.2f} wz={wz_b:+.2f} m/s",
                        font=f13, fill=(200, 220, 255))
                dr.text((8, 48), f"base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) yaw={math.degrees(yaw):+6.1f}deg  slip-ish L/R",
                        font=f11, fill=(180, 200, 220))
                dr.text((8, 66), f"wheel z lift: " + "  ".join(
                    f"{leg}={wheels[leg][2]-0.103:+.3f}" for leg in LEGS), font=f11, fill=(255, 220, 120))
                # bottom data panel
                top = 92 + H
                dr.rectangle((0, top, OUT_W, OUT_H), fill=(0, 0, 0))
                for i, leg in enumerate(LEGS):
                    x, y = feet[leg]
                    qv = [q[j] for j in leg_joint_idx[leg]]
                    dr.text((8, top + 4 + i * 18),
                            f"{leg}: body=({x:+.3f},{y:+.3f})  contact={forces[leg]:5.1f}N  "
                            f"joints(hip/thigh/calf)=({qv[0]:+.2f},{qv[1]:+.2f},{qv[2]:+.2f})",
                            font=f11, fill=(230, 230, 230))
                # top-down inset
                ix, iy, iw, ih = OUT_W - 230, 98, 220, 220
                dr.rectangle((ix, iy, ix + iw, iy + ih), fill=(245, 245, 240), outline=(30, 30, 30))
                sc = 300.0
                cx, cy = ix + iw / 2 - CENTER[0] * sc, iy + ih / 2
                def sx(x): return cx + (x) * sc
                def sy(y): return cy - (y) * sc
                # nominal support rectangle
                nx0, nx1 = -HALF[0], HALF[0]
                ny0, ny1 = -HALF[1], HALF[1]
                dr.rectangle((sx(nx0), sy(ny1), sx(nx1), sy(ny0)), outline=(150, 150, 150), width=1)
                for leg in LEGS:
                    x, y = feet[leg]
                    px, py = sx(x), sy(y)
                    col = (30, 150, 60) if forces[leg] > 0.5 else (210, 120, 40)
                    dr.ellipse((px - 4, py - 4, px + 4, py + 4), fill=col)
                    dr.line((px - 7, py, px + 7, py), fill=col, width=1)
                    dr.line((px, py - 7, px, py + 7), fill=col, width=1)
                    dr.text((px + 6, py - 6), leg, font=f11, fill=(20, 20, 20))
                dr.ellipse((sx(0) - 5, sy(0) - 5, sx(0) + 5, sy(0) + 5), fill=(20, 80, 220))
                dr.text((ix + 4, iy + 2), "top: body-frame feet (green=contact)", font=f11,
                        fill=(30, 30, 30))
                dr.text((ix + 4, iy + ih - 16), f"rect x±{HALF[0]:.2f} y±{HALF[1]:.2f} m", font=f11,
                        fill=(60, 60, 60))
                if ff.stdin is not None:
                    ff.stdin.write(np.asarray(canvas).tobytes())
                    frame_i += 1
            if len(rows) >= max_frames:
                break
        if len(rows) >= max_frames:
            break
    if ff.stdin is not None:
        ff.stdin.close()
    ff.wait(timeout=180)
    runner.close()

    (out / "gait_steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    summary = {}
    for label, rs in phase_rows.items():
        if not rs:
            continue
        feet = {leg: np.array([r["feet_body"][leg] for r in rs]) for leg in LEGS}
        lifts = {leg: max(0.0, float(np.max(feet[leg][:, 0] * 0 + 0))) for leg in LEGS}
        # lift = world wheel z - 0.103 (use wheels_w)
        lifts = {leg: float(np.mean([r["wheels_w"][leg][2] - 0.103 for r in rs])) for leg in LEGS}
        max_lift = {leg: float(np.max([r["wheels_w"][leg][2] - 0.103 for r in rs])) for leg in LEGS}
        x_ext = float(np.max(np.abs(feet["FL"][:, 0] - CENTER[0])) if False else max(
            np.max(np.abs(feet[leg][:, 0] - CENTER[0])) for leg in LEGS))
        y_ext = max(np.max(np.abs(feet[leg][:, 1] - CENTER[1])) for leg in LEGS)
        contact_duty = {leg: float(np.mean([r["contact_force"][leg] > 0.5 for r in rs])) for leg in LEGS}
        slip = {leg: float(np.mean([r["foot_speed"][leg] for r in rs
                                    if r["contact_force"][leg] > 0.5])) for leg in LEGS}
        summary[label] = {
            "cmd": rs[0]["cmd"],
            "vx_mean": avg(rs, "vx_b"),
            "vy_mean": avg(rs, "vy_b"),
            "wz_mean": avg(rs, "wz_b"),
            "track_err_xy": float(np.mean([abs(r["vx_b"] - r["cmd"][0]) + abs(r["vy_b"] - r["cmd"][1]) for r in rs])),
            "track_err_wz": float(np.mean([abs(r["wz_b"] - r["cmd"][2]) for r in rs])),
            "foot_lift_mean": {leg: round(lifts[leg], 4) for leg in LEGS},
            "foot_lift_max": {leg: round(max_lift[leg], 4) for leg in LEGS},
            "contact_duty": {leg: round(contact_duty[leg], 3) for leg in LEGS},
            "contact_speed_mean": {leg: round(slip[leg], 3) for leg in LEGS},
            "support_x_extent": round(float(x_ext), 4),
            "support_y_extent": round(float(y_ext), 4),
            "base_z_mean": avg(rs, "base") if False else round(float(np.mean([r["base"][2] for r in rs])), 4),
            "x_drift": round(float(rs[-1]["base"][0] - rs[0]["base"][0]), 4),
            "y_drift": round(float(rs[-1]["base"][1] - rs[0]["base"][1]), 4),
            "yaw_drift": round(float(wrap(rs[-1]["yaw"] - rs[0]["yaw"])), 4),
        }
    (out / "gait_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"rendered {frame_i} frames -> {out / 'gait_diagnostic.mp4'}")
    print(f"raw steps -> {out / 'gait_steps.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
