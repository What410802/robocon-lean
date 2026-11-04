#!/usr/bin/env python3
"""Replay one full-field route run through the eval Part 2 judge, leg by leg.

The full-field route is an integration demo, not an official scoring unit: the
route stops each leg with its own geometric gate.  To check that those gates are
consistent with the *judge* (and to stop treating the route gates as a second
criterion), this tool projects each leg's field-frame trajectory back into the
isolated official level frame and feeds the real ``Part2OfficialJudge``.

Projection (from the full-field map geometry and the generated manifest bbox):

* platform: route crosses S->N, official level crosses W->E -> rotate 90 deg
  (field +y -> level +x, field +x -> level -y), bbox-fitted;
* stairs/slope/trench: pure translation (same +x crossing, same y lane centre);
* high_wall: route crosses E->W, official level crosses +x -> 180 deg rotation
  (mirror both axes), bbox-fitted.

Inputs are the route recorder files (``*_states.jsonl`` + ``*_plan.jsonl``).
Wheel support points in the route states are wheel *centres*; the judge wants
wheel bottoms, so the tool subtracts ``WHEEL_RADIUS_M`` (0.103 m), the same
convention as ``record_eval_trace.py``.

Usage (repository root)::

    pixi run -e eval python tools/route/replay_field_part2.py \
        --tag real_v22            # artifacts/part2_route3/<tag>_{states,plan}.jsonl
    pixi run -e eval python tools/route/replay_field_part2.py \
        --states artifacts/part2_route3/real_v22_states.jsonl \
        --plan artifacts/part2_route3/real_v22_plan.jsonl --out /tmp/eval.json

The exit code is 0; use ``--require-all`` to make a non-success verdict on a
crossed leg (or a missing verdict) fail the command (CI/integration use).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from robocon_eval import rules, scoring  # noqa: E402
from robocon_eval.part2 import official  # noqa: E402
from robocon_eval.part2.v5_judge import (  # noqa: E402
    WHEEL_RADIUS_M, Part2OfficialJudge)
from robocon_eval.trace_part2_v5 import frame_tilt_deg  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "part2_route3"
FIELD_MAP = ROOT / "demo/scenes/part2/full_field/part2_fixed_field.xml"
SUPPORT_LEGS = ("FL", "FR", "RL", "RR")
# The field rows use short geom prefixes; the wall is named cross_bar_N while
# its level keys are high_wall_*cm (same 20/30/40 cm mapping as the route).
WALL_GEOM_PREFIX = {
    "high_wall_20cm": "cross_bar_5",
    "high_wall_30cm": "cross_bar_6",
    "high_wall_40cm": "cross_bar_7",
}
CATEGORY_OBSTACLE = {
    "high_platform": "high_platform",
    "stairs": "stairs",
    "slope": "slope",
    "high_wall": "high_wall",
    "trench": "trench",
    "plum": "plum",
}


def _quat_rotate(q, v: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(a) for a in q)
    r = np.array([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dtype=float).reshape(3, 3)
    return r @ v


def geom_aabb(model, gid: int):
    """World AABB of one MuJoCo geom (boxes and meshes; route field uses both)."""
    import mujoco

    gtype = int(model.geom_type[gid])
    pos = np.asarray(model.geom_pos[gid], dtype=float)
    quat = np.asarray(model.geom_quat[gid], dtype=float)
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = np.asarray(model.geom_size[gid], dtype=float)
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                    corners.append(pos + _quat_rotate(quat, local))
        pts = np.asarray(corners)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        adr = int(model.mesh_vertadr[mid])
        num = int(model.mesh_vertnum[mid])
        verts = np.asarray(model.mesh_vert[adr:adr + num], dtype=float)
        pts = np.asarray([pos + _quat_rotate(quat, v) for v in verts])
    else:
        raise ValueError(f"unsupported geom type {gtype} for AABB")
    return pts.min(axis=0), pts.max(axis=0)


def field_bbox(model, level: dict):
    """AABB of the field row instance consumed by this level."""
    import mujoco

    category = level["category"]
    key = level["key"]
    if category == "high_wall":
        prefix = WALL_GEOM_PREFIX[key]
    elif category == "high_platform":
        prefix = key.replace("high_", "", 1)      # field geom: platform_20cm
    else:
        prefix = key
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name or not name.startswith(prefix):
            continue
        gmin, gmax = geom_aabb(model, gid)
        lo = np.minimum(lo, gmin)
        hi = np.maximum(hi, gmax)
    if not np.all(np.isfinite(lo)):
        raise RuntimeError(f"no field geoms matched {prefix!r} for {key}")
    return lo, hi


def make_map(category: str, field_lo, field_hi, level_lo, level_hi):
    """Affine field->level map; see module docstring for the axis conventions."""
    fx0, fy0 = float(field_lo[0]), float(field_lo[1])
    fx1, fy1 = float(field_hi[0]), float(field_hi[1])
    lx0, ly0 = float(level_lo[0]), float(level_lo[1])
    lx1, ly1 = float(level_hi[0]), float(level_hi[1])

    def mapped(x: float, y: float) -> tuple[float, float]:
        if category == "high_platform":
            sx = (lx1 - lx0) / (fy1 - fy0)
            sy = (ly1 - ly0) / (fx1 - fx0)
            return lx0 + (y - fy0) * sx, ly1 - (x - fx0) * sy
        if category == "high_wall":
            sx = (lx1 - lx0) / (fx1 - fx0)
            sy = (ly1 - ly0) / (fy1 - fy0)
            return lx1 - (x - fx0) * sx, ly1 - (y - fy0) * sy
        sx = (lx1 - lx0) / (fx1 - fx0)
        sy = (ly1 - ly0) / (fy1 - fy0)
        return (lx0 + (x - fx0) * sx,
                (ly0 + ly1) / 2.0 + (y - (fy0 + fy1) / 2.0) * sy)

    return mapped


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def leg_windows(plan: list[dict]) -> list[dict]:
    """Route plan -> per-leg time windows and bypass flags."""
    windows: list[dict] = []
    current: dict | None = None
    for entry in plan:
        leg = str(entry.get("leg") or "")
        cmd = str(entry.get("cmd") or "")
        t = float(entry.get("t", 0.0))
        if not leg or leg == "start":
            continue
        if cmd.startswith(leg + "(") and "policy=" in cmd:
            current = {"leg": leg, "start": t, "end": None, "bypass": False}
            continue
        if current is not None and current["leg"] == leg and cmd == "ensure_transfer":
            current["end"] = t
            windows.append(current)
            current = None
            continue
        if cmd.startswith(leg + "(") and "bypass" in cmd:
            windows.append({"leg": leg, "start": t, "end": t, "bypass": True})
    if current is not None:                       # run stopped inside a leg
        current["end"] = math.inf
        windows.append(current)
    return windows


def judge_leg(frames: list[dict], window: dict, level: dict, mapped) -> dict:
    judge = Part2OfficialJudge(level)
    time_to_frames = {}
    for frame in frames:
        time_to_frames[round(float(frame["t"]), 6)] = frame
    times = sorted(time_to_frames)
    lo = bisect_left(times, window["start"])
    hi = bisect_right(times, window["end"])
    updated = 0
    for t in times[lo:hi]:
        frame = time_to_frames[t]
        base = np.asarray(frame["base"], dtype=float)
        bx, by = mapped(float(base[0]), float(base[1]))
        feet = frame.get("feet") or {}
        if not all(f"{leg}_foot" in feet for leg in SUPPORT_LEGS):
            continue
        wx, wy, wz = [], [], []
        for leg in SUPPORT_LEGS:
            p = feet[f"{leg}_foot"]
            x, y = mapped(float(p[0]), float(p[1]))
            wx.append(x)
            wy.append(y)
            wz.append(float(p[2]) - WHEEL_RADIUS_M)
        judge.update(t, np.array([bx, by, float(base[2])]), frame_tilt_deg(frame),
                     wheel_x=np.array(wx), wheel_y=np.array(wy),
                     wheel_z=np.array(wz))
        updated += 1
    report = judge.finalize("window_end")
    report["leg"] = window["leg"]
    report["level"] = level["key"]
    report["category"] = level["category"]
    report["frames"] = updated
    report["window"] = [window["start"],
                        None if math.isinf(window["end"]) else window["end"]]
    if report["status"] == "success":
        report["score"] = scoring.score_for_level(
            rules.part2(), CATEGORY_OBSTACLE[level["category"]], level["level"])
    else:
        report["score"] = 0
    return report


def run(states: Path, plan: Path, map_path: Path, out: Path | None,
        require_all: bool) -> int:
    import mujoco

    manifest = official.load_levels()
    by_key = {entry["key"]: entry for entry in manifest}
    model = mujoco.MjModel.from_xml_path(str(map_path))
    frames = load_jsonl(states)
    plan_entries = load_jsonl(plan)
    windows = leg_windows(plan_entries)
    rows = []
    total = 0
    for window in windows:
        leg = window["leg"]
        if window.get("bypass") or leg not in by_key:
            if "plum" in leg:
                rows.append({"leg": leg, "status": "bypass",
                             "reason": "no deployable expert; transit only",
                             "score": 0})
                continue
            raise SystemExit(f"leg {leg!r}: no matching official level / bypass flag")
        level = by_key[leg]
        field_lo, field_hi = field_bbox(model, level)
        level_lo = np.asarray(level["obstacle_bbox"]["min"], dtype=float)
        level_hi = np.asarray(level["obstacle_bbox"]["max"], dtype=float)
        mapped = make_map(level["category"], field_lo, field_hi, level_lo, level_hi)
        report = judge_leg(frames, window, level, mapped)
        report["field_bbox"] = [field_lo.tolist(), field_hi.tolist()]
        rows.append(report)
        total += int(report["score"])

    print(f"# field -> eval judge: {states.name} ({len(frames)} frames, "
          f"{len(windows)} legs)")
    for row in rows:
        status = row["status"]
        line = f"{row['leg']:22s} {status:8s} score={row['score']:3d}"
        if status == "success":
            line += f"  detail={{'frames': {row['frames']}, 'max_x': {row.get('detail', {}).get('max_x')}}}"
        elif status == "fail":
            line += f"  reason={row.get('reason')}"
        else:
            line += f"  ({row.get('reason')})"
        print(line)
    print(f"# eval-judged total (best level per leg): {total} / 1300")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"states": str(states), "plan": str(plan),
                                   "legs": rows, "total": total,
                                   "judge_version": part2_judge_version()},
                                  indent=2, ensure_ascii=False) + "\n")
        print(f"# wrote {out}")
    if require_all:
        bad = [row for row in rows if row["status"] not in ("success", "bypass")]
        if bad:
            print(f"# --require-all: {len(bad)} leg(s) not success: "
                  + ", ".join(row["leg"] for row in bad), file=sys.stderr)
            return 1
    return 0


def part2_judge_version() -> str:
    from robocon_eval.part2.v5_judge import JUDGE_VERSION
    return JUDGE_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=None,
                        help="shortcut for artifacts/part2_route3/<tag>_{states,plan}.jsonl")
    parser.add_argument("--states", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--map", type=Path, default=FIELD_MAP)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-all", action="store_true",
                        help="exit 1 unless every crossed leg is eval-success")
    args = parser.parse_args()
    if args.tag:
        states = ARTIFACTS / f"{args.tag}_states.jsonl"
        plan = ARTIFACTS / f"{args.tag}_plan.jsonl"
    else:
        if not args.states or not args.plan:
            parser.error("provide --tag or both --states and --plan")
        states, plan = args.states, args.plan
    return run(states, plan, args.map, args.out, args.require_all)


if __name__ == "__main__":
    raise SystemExit(main())
