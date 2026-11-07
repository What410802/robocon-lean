#!/usr/bin/env python3
"""Render recorded /eval_state JSONL without running robot physics.

Usage:
  MUJOCO_GL=egl pixi run -e eval python tools/video/replay_route_video.py \
    artifacts/route/states.jsonl artifacts/route/route.mp4 artifacts/route/plan.jsonl

Use --cli-only for a demo video: the scene plus a black strip containing only
"> CLI: ...". The default review layout includes route and wheel telemetry.
Frames are sampled by simulation timestamp, so both 2 ms and 20 ms traces play
at 1x speed. Only mj_forward is called; policies and physics are not rerun.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import textwrap
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


def sample_frames(path: Path, fps: int, start: float | None = None,
                  end: float | None = None):
    """Stream the latest recorded state at each output timestamp, at 1x speed."""
    previous = None
    origin = None
    index = 0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            current = json.loads(line)
            t = float(current["t"])
            if not math.isfinite(t):
                raise ValueError("trace has a non-finite timestamp")
            if previous is None:
                origin = max(t, start) if start is not None else t
            elif t < float(previous["t"]) - 1e-9:
                raise ValueError("trace timestamps must be nondecreasing")
            # Hold the previous state until the next recorded timestamp. At an
            # exact timestamp use the new state, including the last duplicate.
            while previous is not None:
                output_t = origin + index / fps
                if output_t >= t - 1e-9 or (end is not None and output_t > end + 1e-9):
                    break
                yield output_t, previous
                index += 1
            if end is not None and t > end + 1e-9:
                return
            previous = current
    if previous is not None:
        while origin + index / fps <= float(previous["t"]) + 1e-9:
            output_t = origin + index / fps
            if end is not None and output_t > end + 1e-9:
                break
            yield output_t, previous
            index += 1


def cli_canvas(scene: Image.Image, cli: str, font) -> Image.Image:
    """Keep annotations outside the scene, with no title or telemetry."""
    canvas = Image.new("RGB", (W, H + 64), (0, 0, 0))
    canvas.paste(scene, (0, 64))
    dr = ImageDraw.Draw(canvas)
    line = f"> CLI: {cli}"
    char_width = max(1.0, dr.textlength("M", font=font))
    lines = textwrap.wrap(line, width=int((W - 16) / char_width),
                          subsequent_indent="       ") or ["> CLI:"]
    if len(lines) > 3:
        raise ValueError("CLI annotation exceeds the three-line demo strip")
    top = max(4, (64 - 18 * len(lines)) // 2)
    for number, text in enumerate(lines):
        dr.text((8, top + 18 * number), text, font=font, fill=(255, 220, 120))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("plan", type=Path, nargs="?")
    ap.add_argument("--scene", type=Path, default=SCENE,
                    help="MuJoCo scene used for replay (Part2 route field etc.)")
    layout = ap.add_mutually_exclusive_group()
    layout.add_argument("--operator-only", action="store_true",
                    help="show only the scene and the auto-operator CLI command "
                         "(no telemetry/wheel/top-down debug overlay)")
    layout.add_argument("--cli-only", action="store_true",
                        help="minimal demo: only '> CLI: ...' outside the scene")
    ap.add_argument("--fps", type=int, default=25, help="output frame rate (1x simulation time)")
    ap.add_argument("--start-time", type=float, help="first simulation timestamp to render")
    ap.add_argument("--end-time", type=float, help="last simulation timestamp to render")
    ap.add_argument("--preview", type=Path, help="also save the first rendered frame as an image")
    args = ap.parse_args()
    if args.fps <= 0:
        ap.error("--fps must be positive")
    if any(value is not None and not math.isfinite(value)
           for value in (args.start_time, args.end_time)):
        ap.error("time bounds must be finite")
    if args.start_time is not None and args.end_time is not None and args.end_time < args.start_time:
        ap.error("--end-time must be at or after --start-time")
    if args.cli_only and (args.plan is None or not args.plan.is_file()):
        ap.error("--cli-only requires the recorded plan JSONL for its CLI annotations")
    src = args.states
    dst = args.video
    plan_path = args.plan
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames = sample_frames(src, args.fps, args.start_time, args.end_time)
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
    out_w, out_h = (W, H + 64) if args.operator_only or args.cli_only else (OUT_W, OUT_H)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{out_w}x{out_h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(dst)],
        stdin=subprocess.PIPE,
    )
    trace = []
    cur = {"leg": "start", "cmd": "start", "target": None}
    n = 0

    def write_frame(canvas):
        nonlocal n
        if args.preview is not None and n == 0:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(args.preview)
        ff.stdin.write(np.asarray(canvas).tobytes())
        n += 1

    for t, fr in frames:
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
        data.qpos[0:3] = base
        data.qpos[3:7] = quat
        for k, adr in enumerate(qadr):
            data.qpos[adr] = float(fr["q"][k])
        mujoco.mj_forward(model, data)
        cam.lookat = base
        renderer.update_scene(data, camera=cam)
        im = Image.fromarray(renderer.render())
        cli = cur.get("cli", "") or "(none)"
        if args.cli_only:
            write_frame(cli_canvas(im, cli, font))
            continue
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
            write_frame(canvas)
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
        write_frame(canvas)
    ff.stdin.close()
    code = ff.wait(timeout=120)
    renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited with status {code}")
    if n == 0:
        raise ValueError("no recorded states inside the requested time window")
    print(f"rendered {n} frames -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
