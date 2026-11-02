#!/usr/bin/env python3
"""Render one campaign trace as an annotated review video.

The trace format intentionally stores base/support/contact evidence rather than
the full qpos.  This script replays the *same deterministic policy* with the
same command/duration/settle, renders the actual robot, and overlays the
quantities needed to judge a trial:

* the commanded body-frame velocity ``vx, vy, wz``;
* the measured body-frame velocity (from the free-joint qvel and the IMU gyro);
* time, policy/level, judge status/reason, and a progress bar.

Usage:
    pixi run -e eval python scripts/render_review.py \
        --trace artifacts/eval-campaign-20261027/traces/comp_high600/high_platform_30cm/trial-01.jsonl \
        --out /tmp/review_platform30.mp4

The default 640x360 / 50 fps is deliberately review quality, not demo quality.
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

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "eval"))

from robocon_eval.part2.official import canonicalize_level  # noqa: E402
from robocon_eval.part2.v5_judge import Part2OfficialJudge  # noqa: E402
from robocon_eval import rules  # noqa: E402
from robocon_eval.policy_runner import PolicyRunner  # noqa: E402
from robocon_eval.scoring import score_for_level  # noqa: E402

SMALL_W, SMALL_H = 640, 420
W, H = 2 * SMALL_W, SMALL_H
FPS = 50
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Per-category review camera.  Side views for height/geometry, a bit more
# oblique for platform/slope so forward motion is visible.
CAMERAS = {
    "high_platform": {"azimuth": 120.0, "elevation": -18.0, "distance": 2.6},
    "stairs": {"azimuth": 95.0, "elevation": -15.0, "distance": 2.8},
    "slope": {"azimuth": 105.0, "elevation": -15.0, "distance": 2.8},
    "high_wall": {"azimuth": 95.0, "elevation": -12.0, "distance": 2.2},
    "trench": {"azimuth": 95.0, "elevation": -18.0, "distance": 2.4},
    "plum": {"azimuth": 90.0, "elevation": -35.0, "distance": 3.0},
}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _fmt(v: float) -> str:
    return f"{v:+6.2f}"


def _read_meta(trace: Path) -> dict:
    with trace.open() as fh:
        first = fh.readline()
    meta = json.loads(first)
    if meta.get("kind") != "meta":
        raise SystemExit(f"{trace}: first line is not a trace meta")
    return meta


def _read_frames(trace: Path) -> list[dict]:
    frames: list[dict] = []
    with trace.open() as fh:
        next(fh)
        for line in fh:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def _render_models(runner: PolicyRunner) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """Side view gets fill lights; top view keeps the scene lighting.

    The fill lights make the side view readable but over-expose large, flat
    tops (platform/trench/plum) from directly above.  Two visual-only models
    avoid that trade-off; physics is unchanged.
    """
    spec_side = mujoco.MjSpec.from_file(str(runner.scene))
    for pos, diffuse, ambient in (
        ([2.0, 0.0, 3.0], [0.55, 0.55, 0.55], [0.06, 0.06, 0.06]),
        ([1.0, -2.0, 2.0], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]),
    ):
        light = spec_side.worldbody.add_light()
        light.pos = pos
        light.dir = [0.0, 0.0, -1.0]
        light.diffuse = diffuse
        light.ambient = ambient
        light.castshadow = False
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    spec_top = mujoco.MjSpec.from_file(str(runner.scene))
    return spec_side.compile(), spec_top.compile()


def _velocities(runner: PolicyRunner) -> tuple[np.ndarray, np.ndarray]:
    d = runner.data
    quat = np.array(d.qpos[3:7], dtype=np.float64)
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    # Free joint qvel[0:3] is world-frame linear velocity; policy commands are
    # body-frame.  The IMU gyro is already body-frame.
    body_v = R.T @ np.asarray(d.qvel[0:3], dtype=np.float64)
    try:
        gyro = np.array(d.sensor("imu_gyro").data, dtype=np.float64)[0:3]
    except Exception:  # noqa: BLE001 - all current scenes have the sensor
        gyro = np.asarray(d.qvel[3:6], dtype=np.float64)
    return body_v, gyro


def _draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: float,
              cmd: float, scale: float, color: tuple[int, int, int]) -> None:
    width = 150
    mid = x + width // 2
    draw.text((x, y - 14), label, font=_font(FONT_MONO, 13), fill=(210, 210, 210))
    draw.line((x, y, x + width, y), fill=(120, 120, 120), width=1)
    draw.line((mid, y - 5, mid, y + 5), fill=(180, 180, 180), width=1)
    # command as a hollow marker
    cx = int(mid + max(-1.0, min(1.0, cmd / scale)) * width / 2)
    draw.rectangle((cx - 3, y - 7, cx + 3, y + 7), outline=(245, 245, 245), width=1)
    # measured as a filled bar
    vx = int(mid + max(-1.0, min(1.0, value / scale)) * width / 2)
    draw.line((mid, y, vx, y), fill=color, width=5)


def _overlay(img: np.ndarray, meta: dict, frame: dict, cmd: np.ndarray,
             body_v: np.ndarray, gyro: np.ndarray, judge_status: str,
             judge_reason: str, duration: float, label: str) -> Image.Image:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, "RGBA")
    # top and bottom panels
    draw.rectangle((0, 0, W, 58), fill=(0, 0, 0, 155))
    draw.rectangle((0, H - 92, W, H), fill=(0, 0, 0, 170))
    level = (meta.get("level") or {}).get("key", "?")
    title = f"{meta.get('robot', '?')} | {meta.get('policy', '?')} | {level} | {label}"
    draw.text((10, 5), title, font=_font(FONT_BOLD, 15), fill=(245, 245, 245))
    t = float(frame["t"])
    draw.text((W - 190, 5), f"t={t:5.2f}/{duration:4.1f}s", font=_font(FONT_MONO, 15),
              fill=(230, 230, 230))
    status_col = {"success": (80, 220, 120), "fail": (245, 110, 100)}.get(
        judge_status, (245, 210, 90))
    draw.text((W - 190, 28), f"judge: {judge_status}", font=_font(FONT_MONO, 14),
              fill=status_col)
    if judge_reason:
        draw.text((10, 44), f"reason: {judge_reason}", font=_font(FONT_MONO, 13),
                  fill=(240, 160, 150))
    base_yaw = math.degrees(float(frame.get("yaw", 0.0)))
    sup = frame.get("support") or {}
    sup_yaw = None
    try:
        front = np.array([(sup["FL"][0] + sup["FR"][0]) / 2,
                          (sup["FL"][1] + sup["FR"][1]) / 2])
        rear = np.array([(sup["RL"][0] + sup["RR"][0]) / 2,
                         (sup["RL"][1] + sup["RR"][1]) / 2])
        d = front - rear
        sup_yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    except Exception:
        pass
    yaw_txt = f"yaw base {base_yaw:+5.1f}deg"
    if sup_yaw is not None:
        yaw_txt += f"  sup {sup_yaw:+5.1f}deg"
    draw.text((W - 360, 47), yaw_txt, font=_font(FONT_MONO, 11),
              fill=(220, 220, 220))
    draw.text((10, 30), f"cmd  vx {_fmt(cmd[0])}  vy {_fmt(cmd[1])}  wz {_fmt(cmd[2])}",
              font=_font(FONT_MONO, 14), fill=(150, 210, 255))
    draw.line((SMALL_W, 58, SMALL_W, H - 92), fill=(150, 150, 150, 180), width=1)
    draw.text((10, 60), "SIDE", font=_font(FONT_BOLD, 12), fill=(180, 220, 255))
    draw.text((SMALL_W + 10, 60), "TOP", font=_font(FONT_BOLD, 12), fill=(180, 255, 220))
    # measured velocities and bars
    speed = float(np.linalg.norm(body_v))
    act = (f"act  vx {_fmt(body_v[0])}  vy {_fmt(body_v[1])}  "
           f"wz {_fmt(gyro[2])}  |v| {speed:4.2f}")
    draw.text((10, H - 88), act, font=_font(FONT_MONO, 14), fill=(255, 220, 140))
    _draw_bar(draw, 10, H - 52, "vx", float(body_v[0]), float(cmd[0]), 1.0, (120, 220, 255))
    _draw_bar(draw, 180, H - 52, "vy", float(body_v[1]), float(cmd[1]), 0.5, (120, 220, 255))
    _draw_bar(draw, 350, H - 52, "wz", float(gyro[2]), float(cmd[2]), 1.0, (255, 180, 120))
    # progress
    frac = 0.0 if duration <= 0 else min(1.0, max(0.0, t / duration))
    draw.rectangle((W - 120, H - 34, W - 10, H - 24), outline=(150, 150, 150), width=1)
    draw.rectangle((W - 119, H - 33, W - 119 + int(108 * frac), H - 25), fill=(120, 220, 255))
    draw.text((W - 120, H - 54), f"{100 * frac:3.0f}%", font=_font(FONT_MONO, 12),
              fill=(200, 200, 200))
    return im


def render(trace: Path, out: Path, settle_s: float, preview: Path | None,
           preview_frac: float, label: str | None = None,
           stop_at_success: bool = True) -> dict:
    meta = _read_meta(trace)
    trace_frames = _read_frames(trace)
    if label is None:
        label = trace.stem.replace("trial-", "trial ")
    if meta.get("part") != 2:
        raise SystemExit("this renderer currently supports Part2 traces")
    level = canonicalize_level(meta["level"], require_key=True)
    command = np.asarray(meta.get("command") or level.get("command") or (0.5, 0, 0),
                         dtype=np.float64)
    duration = float(meta.get("duration_s") or level.get("seconds") or 15.0)
    runner = PolicyRunner(
        meta["robot"], 2, meta["policy"], level["key"],
        steps=None, duration_s=duration, command=tuple(command), settle_s=settle_s,
    )
    runner.settle(settle_s)
    spawn_offset = meta.get("spawn_offset") or {}
    if spawn_offset.get("dy") or spawn_offset.get("dyaw_deg"):
        runner.offset_spawn(float(spawn_offset.get("dy", 0.0)),
                            float(spawn_offset.get("dyaw_deg", 0.0)))
    render_model_side, render_model_top = _render_models(runner)
    render_data_side = mujoco.MjData(render_model_side)
    render_data_top = mujoco.MjData(render_model_top)
    renderer_side = mujoco.Renderer(render_model_side, SMALL_H, SMALL_W)
    renderer_top = mujoco.Renderer(render_model_top, SMALL_H, SMALL_W)
    cam = mujoco.MjvCamera()
    preset = CAMERAS.get(level["category"], CAMERAS["high_platform"])
    cam.azimuth = preset["azimuth"]
    cam.elevation = preset["elevation"]
    cam.distance = preset["distance"]
    cam_top = mujoco.MjvCamera()
    cam_top.azimuth = 90.0
    cam_top.elevation = -90.0
    cam_top.distance = 2.4
    judge = Part2OfficialJudge(level)
    frames = max(1, int(round(duration / runner.policy_dt)))
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    preview_at = max(1, int(round(frames * preview_frac)))
    last = None
    last_annotated = None
    preview_saved = False
    encoded = 0
    try:
        for i in range(frames):
            # Replay the actual per-step command from the trace.  Fixed-command
            # traces have cmd == nominal; adaptive traces have a nonzero wz.
            step_cmd = command
            if i < len(trace_frames) and "cmd" in trace_frames[i]:
                step_cmd = np.asarray(trace_frames[i]["cmd"], dtype=np.float64)
            last = runner.step(step_cmd)
            base = np.asarray(last["base"], dtype=np.float64)
            if i < len(trace_frames):
                trace_base = np.asarray(trace_frames[i]["base"], dtype=np.float64)
                err = float(np.max(np.abs(base - trace_base)))
                if err > 1e-3:
                    raise SystemExit(
                        f"replay diverged at t={last['t']}: max|base-trace|={err:.4g}; "
                        "refusing to render a video that is not the judged rollout")
            support = last["support"]
            wx = np.array([support[k][0] for k in ("FL", "FR", "RL", "RR")])
            wy = np.array([support[k][1] for k in ("FL", "FR", "RL", "RR")])
            wz = np.array([support[k][2] for k in ("FL", "FR", "RL", "RR")])
            if runner.support_frame != "wheel_lowest":
                wz = wz - runner.wheel_radius
            status = judge.update(float(last["t"]), base, float(last["tilt_deg"]),
                                  wheel_x=wx, wheel_y=wy, wheel_z=wz)
            body_v, gyro = _velocities(runner)
            render_data_side.qpos[:] = runner.data.qpos
            render_data_side.qvel[:] = runner.data.qvel
            render_data_top.qpos[:] = runner.data.qpos
            render_data_top.qvel[:] = runner.data.qvel
            mujoco.mj_forward(render_model_side, render_data_side)
            mujoco.mj_forward(render_model_top, render_data_top)
            cam.lookat[:] = [float(base[0]) + 0.35, float(base[1]), 0.52]
            renderer_side.update_scene(render_data_side, cam)
            side = renderer_side.render()
            cam_top.lookat[:] = [float(base[0]), float(base[1]), 0.0]
            renderer_top.update_scene(render_data_top, cam_top)
            top = renderer_top.render()
            img = np.concatenate([side, top], axis=1)
            annotated = _overlay(img, meta, last, step_cmd, body_v, gyro,
                                 status, judge.reason, duration, label)
            last_annotated = annotated
            if i + 1 == preview_at and preview is not None:
                preview.parent.mkdir(parents=True, exist_ok=True)
                annotated.save(preview)
                preview_saved = True
            proc.stdin.write(annotated.tobytes())
            encoded += 1
            if stop_at_success and status == "success":
                break
    finally:
        if preview is not None and not preview_saved and last_annotated is not None:
            preview.parent.mkdir(parents=True, exist_ok=True)
            last_annotated.save(preview)
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=120)
        renderer_side.close()
        renderer_top.close()
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed with {proc.returncode}")
    result_status = judge.status
    report = judge.finalize("trace_end")
    report["score"] = (score_for_level(rules.part2(),
                                       level["category"], level["level"])
                       if report["status"] == "success" else 0)
    result = {
        "trace": str(trace), "video": str(out), "level": level["key"],
        "policy": meta["policy"], "command": command.tolist(),
        "command_replay": "trace_per_frame",
        "duration_s": duration, "frames": frames,
        "final_base": last["base"] if last else None,
        "judge": report,
    }
    (out.with_suffix(".json")).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("video", "level", "judge")}, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--preview", type=Path, default=None)
    ap.add_argument("--preview-frac", type=float, default=0.5)
    ap.add_argument("--label", default=None)
    ap.add_argument("--full", action="store_true",
                    help="keep the whole level duration instead of ending at success")
    a = ap.parse_args()
    render(a.trace, a.out, a.settle, a.preview, a.preview_frac, a.label,
           stop_at_success=not a.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
