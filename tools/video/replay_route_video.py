#!/usr/bin/env python3
"""Replay recorded /eval_state JSONL with expected-route / wheel overlay.

Usage:
  MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route/states.jsonl artifacts/route/route.mp4 artifacts/route/plan.jsonl

The overlay draws the current expected leg/target (recorded by the route node),
actual base pose, and the four wheel contact points in the base frame, plus a
top-down inset with the expected route and actual trace.  The goal is review, not
demo quality.
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
SCENE = ROOT / "demo/robots/blackW/scene_part1.xml"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W, H = 640, 480
OUT_W, OUT_H = 800, 620


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def yaw_quat(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_plan(path: Path):
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    events.sort(key=lambda e: e.get("t", 0.0))
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("plan", type=Path, nargs="?")
    ap.add_argument("--scene", type=Path, default=SCENE,
                    help="MuJoCo scene used for replay (Part2 route field etc.)")
    ap.add_argument("--operator-only", action="store_true",
                    help="show only the scene and the auto-operator CLI command "
                         "(no telemetry/wheel/top-down debug overlay)")
    args = ap.parse_args()
    src = args.states
    dst = args.video
    plan_path = args.plan
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    plan = load_plan(plan_path) if plan_path and plan_path.exists() else []
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    qadr = [int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])]) for i in range(model.nu)]
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -25.0
    try:
        font = ImageFont.truetype(FONT, 13)
        font_small = ImageFont.truetype(FONT, 11)
        font_big = ImageFont.truetype(FONT, 16)
    except Exception:  # noqa: BLE001
        font = font_small = font_big = ImageFont.load_default()
    out_w, out_h = (W, H + 64) if args.operator_only else (OUT_W, OUT_H)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", "25", "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(dst)],
        stdin=subprocess.PIPE,
    )
    trace = []
    cur = {"leg": "start", "cmd": "start", "target": None}
    n = 0
    for fi, fr in enumerate(frames):
        t = float(fr.get("t", fi * 0.02))
        while plan and plan[0].get("t", 0.0) <= t:
            cur = plan.pop(0)
        leg = cur.get("leg", "?")
        target = cur.get("target")
        base = np.asarray(fr["base"], dtype=float)
        quat = np.asarray(fr["quat"], dtype=float)
        yaw = yaw_quat(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        feet = {}
        for k, p in fr.get("feet", {}).items():
            dx, dy = float(p[0]) - base[0], float(p[1]) - base[1]
            feet[k] = (c * dx + s * dy, -s * dx + c * dy)
        trace.append((base[0], base[1]))
        if fi % 2 == 0:
            data.qpos[0:3] = base
            data.qpos[3:7] = quat
            for k, adr in enumerate(qadr):
                data.qpos[adr] = float(fr["q"][k])
            mujoco.mj_forward(model, data)
            cam.lookat = base
            renderer.update_scene(data, camera=cam)
            im = Image.fromarray(renderer.render())
            cli = cur.get("cli", "") or "(none)"
            if args.operator_only:
                canvas = Image.new("RGB", (W, H + 64), (0, 0, 0))
                canvas.paste(im, (0, 64))
                dr = ImageDraw.Draw(canvas)
                dr.text((8, 6), "Part1 auto-operator", font=font_small,
                        fill=(150, 150, 150))
                # Wrap long CLI strings over two lines instead of clipping.
                line = f"> CLI: {cli}"
                if len(line) > 76:
                    dr.text((8, 24), line[:76], font=font, fill=(255, 220, 120))
                    dr.text((8, 44), "       " + line[76:], font=font,
                            fill=(255, 220, 120))
                else:
                    dr.text((8, 28), line, font=font, fill=(255, 220, 120))
                ff.stdin.write(np.asarray(canvas).tobytes())
                n += 1
                continue
            canvas = Image.new("RGB", (OUT_W, OUT_H), (16, 18, 22))
            canvas.paste(im, (0, 100))
            dr = ImageDraw.Draw(canvas)
            # top header + internal CLI/REPL command line
            dr.rectangle((0, 0, OUT_W, 98), fill=(0, 0, 0))
            dr.text((8, 2), f"Part1 auto-route  t={t:6.2f}s  leg={leg}", font=font_big, fill=(245, 245, 245))
            tgt = f"target=({target[0]:.2f},{target[1]:.2f})" if target else "target=—"
            dr.text((8, 26), f"base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) yaw={math.degrees(yaw):+6.1f}deg  {tgt}",
                    font=font, fill=(200, 220, 255))
            cli = cur.get("cli", "")
            corr = cur.get("corr", {})
            cmd = cur.get("cmd", "")
            dr.text((8, 52), f"> CLI: {cli}", font=font, fill=(255, 220, 120))
            if corr:
                ctxt = (f"corr: dx={corr.get('dx',0):+.3f}m "
                        f"dy={corr.get('dy',0):+.3f}m "
                        f"dyaw={math.degrees(corr.get('dyaw',0)):+.1f}deg")
            else:
                ctxt = "corr: — (no pose_move_to target)"

            dr.text((8, 70), f"> {ctxt}   cmd={cmd}", font=font_small, fill=(120, 255, 160))
            # bottom wheel/rectangle bar
            dr.rectangle((0, 100 + H, OUT_W, OUT_H), fill=(0, 0, 0))
            if len(feet) == 4:
                order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                labels = ["FL", "FR", "RL", "RR"]
                vals = []
                for lab, key in zip(labels, order):
                    x, y = feet[key]
                    vals.append(f"{lab}({x:+.2f},{y:+.2f})")
                dr.text((8, 100 + H + 4), "wheel contact (base frame): " + "  ".join(vals),
                        font=font, fill=(230, 230, 230))
                fl, fr, rl, rr = [feet[k] for k in order]
                front_track = abs(fl[1] - fr[1])
                rear_track = abs(rl[1] - rr[1])
                left_wb = abs(fl[0] - rl[0])
                right_wb = abs(fr[0] - rr[0])
                rear_asym = abs(rl[0] - rr[0])
                bad = rear_asym > 0.20 or abs(front_track - rear_track) > 0.12
                col = (255, 120, 100) if bad else (120, 230, 150)
                dr.text((8, 100 + H + 20),
                        f"tracks F/R={front_track:.2f}/{rear_track:.2f}m  wheelbase L/R={left_wb:.2f}/{right_wb:.2f}m  rear-x diff={rear_asym:.2f}m"
                        + ("  RECT-BROKEN" if bad else "  rect-ok"),
                        font=font_small, fill=col)
                # tiny wheel diagram
                cx, cy, sc = 710, 600, 55
                dr.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), outline=(100, 100, 100))
                for (x, y), lab in zip([fl, fr, rl, rr], labels):
                    px = cx + x * sc
                    py = cy - y * sc
                    dr.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(120, 220, 255))
                    dr.text((px + 5, py - 6), lab, font=font_small, fill=(220, 220, 220))
            # top-down inset (right, over scene)
            ix, iy, iw, ih = OUT_W - 230, 106, 222, 180
            dr.rectangle((ix, iy, ix + iw, iy + ih), fill=(245, 245, 240), outline=(30, 30, 30))
            sx = lambda x: ix + (x / 7.2) * iw
            sy = lambda y: iy + ih - ((y + 14.0) / 15.0) * ih
            # expected route from plan targets
            pts = []
            for e in plan:
                if e.get("target"):
                    pts.append(tuple(e["target"]))
            if target:
                pts.append(tuple(target))
            for a, b in zip(pts, pts[1:]):
                dr.line((sx(a[0]), sy(a[1]), sx(b[0]), sy(b[1])), fill=(80, 150, 240), width=2)
            for p in pts:
                dr.ellipse((sx(p[0]) - 2, sy(p[1]) - 2, sx(p[0]) + 2, sy(p[1]) + 2), fill=(40, 100, 200))
            if len(trace) > 1:
                dr.line([(sx(x), sy(y)) for x, y in trace[-3000:]], fill=(30, 30, 30), width=1)
            if target:
                dr.ellipse((sx(target[0]) - 5, sy(target[1]) - 5, sx(target[0]) + 5, sy(target[1]) + 5),
                           outline=(220, 40, 40), width=2)
            dr.ellipse((sx(base[0]) - 4, sy(base[1]) - 4, sx(base[0]) + 4, sy(base[1]) + 4),
                       fill=(20, 160, 80))
            dr.text((ix + 4, iy + 2), "expected route (blue) / actual (black)", font=font_small, fill=(30, 30, 30))
            canvas = canvas.resize((OUT_W, OUT_H))
            ff.stdin.write(np.asarray(canvas).tobytes())
            n += 1
    ff.stdin.close()
    ff.wait(timeout=120)
    print(f"rendered {n} frames -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
