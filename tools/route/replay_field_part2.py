#!/usr/bin/env python3
"""Replay the canonical Part 2 field plan with support-point judges.

Exact target geoms and crossing directions come from part2_route_plan.py.
Tiers present in the official manifest are projected into that level's frame.
Intermediate field-only tiers reuse the terrain generator's judge profile,
derived from the actual compiled target geometry; they are explicitly unscored.
Scores are the best successful official tier per category, never summed across
all tiers. This remains a field/demo comparison, not an official standalone run.
Continuous same-policy wall gates may observe at most 0.5 s beyond the original
window solely to finish an active terminal hold, before next-wall contact.

--require-all requires completed, successful windows for every planned leg.
For deliberate debug subsets repeat --expected-leg <canonical name or index>.
Empty/partial recordings and legacy bypasses cannot pass that check.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT / "demo/rl_sar/scripts"))
sys.path.insert(0, str(ROOT / "terrain"))

from robocon_eval import rules, scoring  # noqa: E402
from robocon_eval.part2 import official  # noqa: E402
from robocon_eval.part2.v5_judge import (  # noqa: E402
    WHEEL_RADIUS_M, Part2OfficialJudge)
from robocon_eval.trace_part2_v5 import frame_tilt_deg  # noqa: E402
from gen_part2_levels import _judge as terrain_judge  # noqa: E402
from part2_route_plan import LEGS as PLAN_LEGS, SCENE_RELATIVE, SKIPPED_TIERS  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "part2_route3"
FIELD_MAP = ROOT / SCENE_RELATIVE
SUPPORT_LEGS = ("FL", "FR", "RL", "RR")
WALL_OBSERVATION_TAIL_MAX_S = 0.5
WALL_CONTINUATION_GAP_MAX_S = 0.05
WALL_OBSERVATION_SAMPLE_GAP_MAX_S = 0.04
PLAN_BY_NAME = {leg["leg"]: leg for leg in PLAN_LEGS}
CATEGORY_NAMES = {"platform": "high_platform", "wall": "high_wall"}
CATEGORY_OBSTACLE = {
    "high_platform": "high_platform",
    "stairs": "stairs",
    "slope": "slope",
    "high_wall": "high_wall",
    "trench": "trench",
    "plum": "plum",
}


def official_level_key(leg: dict) -> str:
    category = CATEGORY_NAMES.get(leg["category"], leg["category"])
    if category == "plum":
        return "plum_1"
    unit = "deg" if category == "slope" else "cm"
    return f"{category}_{int(leg['difficulty'])}{unit}"


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


def target_boxes(model, leg: dict):
    """Resolve every exact target from the plan, rejecting missing geometry."""
    import mujoco
    boxes = []
    for name in leg["target_geoms"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f"leg {leg['leg']}: target geom missing: {name}")
        boxes.append((name, *geom_aabb(model, gid)))
    if not boxes:
        raise RuntimeError(f"leg {leg['leg']}: no target geoms")
    return boxes


def field_bbox(model, leg: dict):
    boxes = target_boxes(model, leg)
    return (np.min([lo for _, lo, _ in boxes], axis=0),
            np.max([hi for _, _, hi in boxes], axis=0))


def make_map(category: str, field_lo, field_hi, level_lo, level_hi, *, yaw=None):
    """Project along the planned heading, then normalise to the official bbox.

    Official profiles advance along +x. Rotate the field bbox and every sample
    into that lane frame before scaling its two axes. The omitted-yaw default
    preserves old standalone callers; production always supplies the leg yaw.
    Vertical coordinates and support/hold predicates are never transformed.
    """
    if yaw is None:
        yaw = math.pi / 2 if category == "high_platform" else 0.0
    if not math.isfinite(yaw):
        raise ValueError("projection heading must be finite")
    c, s = math.cos(yaw), math.sin(yaw)
    c = 0.0 if abs(c) < 1e-12 else c
    s = 0.0 if abs(s) < 1e-12 else s

    def local(x, y):
        return x * c + y * s, -x * s + y * c

    corners = np.asarray([local(x, y)
                          for x in (field_lo[0], field_hi[0])
                          for y in (field_lo[1], field_hi[1])])
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    level_lower = np.asarray(level_lo[:2], dtype=float)
    level_upper = np.asarray(level_hi[:2], dtype=float)
    if (not np.isfinite([*lower, *upper, *level_lower, *level_upper]).all()
            or np.any(upper <= lower) or np.any(level_upper <= level_lower)):
        raise ValueError("projection bounding boxes must have finite positive horizontal spans")
    scale = (level_upper - level_lower) / (upper - lower)

    def mapped(x: float, y: float) -> tuple[float, float]:
        point = level_lower + (np.asarray(local(x, y)) - lower) * scale
        return float(point[0]), float(point[1])

    return mapped


def field_profile(model, leg: dict):
    """Use the generator's schema on exact field geometry, with no score claim."""
    origin_x, origin_y = leg["stand"]
    c, s = math.cos(leg["lane_yaw"]), math.sin(leg["lane_yaw"])

    def mapped(x, y):
        dx, dy = x - origin_x, y - origin_y
        return dx * c + dy * s, -dx * s + dy * c

    tags, mins, maxs = [], [], []
    for name, lo, hi in target_boxes(model, leg):
        corners = np.asarray([(*mapped(x, y), z)
                              for x in (lo[0], hi[0])
                              for y in (lo[1], hi[1])
                              for z in (lo[2], hi[2])])
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        centre, half = (lower + upper) / 2, (upper - lower) / 2
        pos = " ".join(str(v) for v in centre)
        size = " ".join(str(v) for v in half)
        tags.append(f'<geom name="{name}" type="box" pos="{pos}" size="{size}"/>')
        mins.append(lower)
        maxs.append(upper)
    lo, hi = np.min(mins, axis=0), np.max(maxs, axis=0)
    kind = {"trench": "gap"}.get(leg["category"], leg["category"])
    category = CATEGORY_NAMES.get(leg["category"], leg["category"])
    level = {"key": "field_" + leg["leg"], "category": category,
             "level": leg["difficulty"],
             "unit": "deg" if category == "slope" else "cm",
             "judge": terrain_judge(kind, tags, lo, hi, None),
             "obstacle_bbox": {"min": lo.tolist(), "max": hi.tolist()}}
    return level, mapped


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def leg_windows(plan: list[dict]) -> list[dict]:
    """Recorded expert attempts, including incomplete attempts and legacy bypasses."""
    windows: list[dict] = []
    current: dict | None = None
    current_start: tuple[str, str, float] | None = None
    interrupted_after_close = False
    for entry in plan:
        leg = str(entry.get("leg") or "")
        cmd = str(entry.get("cmd") or "")
        t = float(entry.get("t", 0.0))
        if not leg or leg == "start":
            continue
        starts_attempt = cmd.startswith(leg + "(") and "policy=" in cmd
        if current is None and windows and cmd != "cross_complete" and not starts_attempt:
            interrupted_after_close = True
        if starts_attempt:
            event = (leg, cmd, t)
            if current is not None and current_start == event:
                # set_cmd/emit_plan opens the attempt; switch_topic then emits
                # the same command at the same sim instant with updated CLI.
                # That status update is not a second, zero-length attempt.
                continue
            if current is not None:
                current["end"] = t
                windows.append(current)
            policy_match = re.search(r"\bpolicy=([^,\s)]+)", cmd)
            current = {"leg": leg, "start": t, "end": None,
                       "bypass": False, "closed": False,
                       "policy": policy_match.group(1) if policy_match else "",
                       "end_cmd": None,
                       "continuation_from_previous": bool(windows and windows[-1].get("closed")
                                                          and not interrupted_after_close)}
            current_start = event
            continue
        if (current is not None and current["leg"] == leg
                and cmd in ("cross_complete", "ensure_transfer")):
            current["end"] = t
            current["closed"] = True
            current["end_cmd"] = cmd
            windows.append(current)
            current = None
            current_start = None
            interrupted_after_close = False
            continue
        if cmd.startswith(leg + "(") and "bypass" in cmd:
            windows.append({"leg": leg, "start": t, "end": t,
                            "bypass": True, "closed": False})
    if current is not None:                       # run stopped inside a leg
        current["end"] = math.inf
        windows.append(current)
    return windows


def wall_observation_tail(windows: list[dict], index: int) -> dict | None:
    """Only adjacent, uninterrupted same-policy wall windows may share a hold.

    Crossing evidence must already exist at the first window's gate. The judge
    below can use this bounded observation interval only to finish that existing
    hold, before any contact with the next wall. It cannot restart a broken hold.
    """
    if index + 1 >= len(windows):
        return None
    current, following = windows[index:index + 2]
    leg, next_leg = PLAN_BY_NAME.get(current["leg"]), PLAN_BY_NAME.get(following["leg"])
    if not leg or not next_leg or leg["category"] != "wall" or next_leg["category"] != "wall":
        return None
    names = list(PLAN_BY_NAME)
    if names.index(next_leg["leg"]) != names.index(leg["leg"]) + 1:
        return None
    gap = following["start"] - current["end"]
    if (not all(row.get("closed") and not row.get("bypass") for row in (current, following))
            or current.get("end_cmd") != "cross_complete"
            or not following.get("continuation_from_previous")
            or not leg.get("continuous_lane") or not next_leg.get("continuous_lane")
            or not current.get("policy") or current["policy"] != following.get("policy")
            or not -1e-6 <= gap <= WALL_CONTINUATION_GAP_MAX_S + 1e-6):
        return None
    return {"end": round(min(current["end"] + WALL_OBSERVATION_TAIL_MAX_S, following["end"]), 6),
            "next_target_geoms": tuple(next_leg["target_geoms"])}


def _contacts_before_next_target(frame: dict, next_targets: tuple[str, ...]) -> str:
    """A missing/malformed contact stream cannot establish an uncontaminated tail."""
    contacts = frame.get("contacts")
    if (not isinstance(contacts, list)
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                   or any(not isinstance(name, str) or not name for name in pair)
                   for pair in contacts)):
        return "invalid_contacts"
    if any(name in next_targets for pair in contacts for name in pair):
        return "next_target_contact"
    return ""


def judge_leg(frames: list[dict], window: dict, level: dict, mapped,
              *, scored: bool = True) -> dict:
    judge = Part2OfficialJudge(level)
    time_to_frames = {}
    for frame in frames:
        time_to_frames[round(float(frame["t"]), 6)] = frame
    times = sorted(time_to_frames)
    lo = bisect_left(times, window["start"])
    hi = bisect_right(times, window["end"])
    updated = 0
    complete_supports = True

    def update(frame):
        try:
            base = np.asarray(frame["base"], dtype=float)
            feet = frame.get("feet") or {}
            support = np.asarray([feet[f"{leg}_foot"] for leg in SUPPORT_LEGS], dtype=float)
            if (base.shape != (3,) or support.shape != (4, 3)
                    or not np.isfinite(base).all() or not np.isfinite(support).all()):
                return False
            bx, by = mapped(float(base[0]), float(base[1]))
            xy = np.asarray([mapped(float(p[0]), float(p[1])) for p in support])
            judge.update(float(frame["t"]), np.array([bx, by, float(base[2])]), frame_tilt_deg(frame),
                         wheel_x=xy[:, 0], wheel_y=xy[:, 1],
                         wheel_z=support[:, 2] - WHEEL_RADIUS_M)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return True

    for t in times[lo:hi]:
        if not update(time_to_frames[t]):
            complete_supports = False
            continue
        updated += 1

    observation_end = window["end"]
    tail = window.get("observation_tail")
    tail_stop = None
    if tail is not None:
        # No additional wall-crossing evidence may be earned in the tail.
        # Only a terminal predicate already holding at cross_complete qualifies.
        last_t = times[hi - 1] if hi > lo else None
        if (level["judge"]["kind"] != "wall" or judge.status != "running"
                or not judge.crossed or judge.held <= 0.0):
            tail_stop = "no_pending_terminal_hold"
        elif (not complete_supports or last_t is None
              or window["end"] - last_t > WALL_OBSERVATION_SAMPLE_GAP_MAX_S + 1e-6):
            tail_stop = "incomplete_support_observation"
        else:
            tail_stop = _contacts_before_next_target(time_to_frames[last_t], tail["next_target_geoms"])
            if not tail_stop:
                tail_stop = "tail_limit"
                for t in times[hi:bisect_right(times, tail["end"])]:
                    frame = time_to_frames[t]
                    contact_fault = _contacts_before_next_target(frame, tail["next_target_geoms"])
                    if contact_fault:
                        tail_stop = contact_fault
                        break
                    if t - last_t > WALL_OBSERVATION_SAMPLE_GAP_MAX_S + 1e-6:
                        tail_stop = "incomplete_support_observation"
                        break
                    if not update(frame):
                        tail_stop = "incomplete_support_observation"
                        break
                    observation_end = t
                    updated += 1
                    last_t = t
                    if judge.status == "success":
                        tail_stop = "terminal_hold_complete"
                        break
                    if judge.status != "running" or judge.held <= 0.0:
                        tail_stop = "terminal_hold_broken"
                        break
    report = judge.finalize("window_end")
    report["leg"] = window["leg"]
    report["level"] = level["key"]
    report["category"] = level["category"]
    report["frames"] = updated
    report["window"] = [window["start"],
                        None if math.isinf(window["end"]) else window["end"]]
    report["observation_end"] = None if math.isinf(observation_end) else observation_end
    report["tail_s"] = round(max(0.0, observation_end - window["end"]), 6) if tail else 0.0
    if tail is not None:
        report["tail_stop"] = tail_stop
    if report["status"] == "success" and scored:
        report["score"] = scoring.score_for_level(
            rules.part2(), CATEGORY_OBSTACLE[level["category"]], level["level"])
    else:
        report["score"] = 0
    return report


def expected_plan_names(selectors: list[str] | None) -> list[str]:
    if selectors is None:
        return list(PLAN_BY_NAME)
    selected = []
    for selector in selectors:
        name = next((name for name in PLAN_BY_NAME if name.lower() == selector.lower()), None)
        if name is None:
            try:
                index = int(selector) - 1
            except ValueError:
                index = -1
            name = PLAN_LEGS[index]["leg"] if 0 <= index < len(PLAN_LEGS) else None
        if name is None:
            raise ValueError(f"unknown expected leg {selector!r}; use a canonical name or 1..{len(PLAN_LEGS)}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("expected leg list must not be empty")
    return selected


def run(states: Path, plan: Path, map_path: Path, out: Path | None,
        require_all: bool, expected_legs: list[str] | None = None) -> int:
    import mujoco

    manifest = official.load_levels()
    by_key = {entry["key"]: entry for entry in manifest}
    model = mujoco.MjModel.from_xml_path(str(map_path))
    frames = load_jsonl(states)
    plan_entries = load_jsonl(plan)
    windows = leg_windows(plan_entries)
    expected = expected_plan_names(expected_legs)
    rows = []
    best_by_category = {}
    for index, window in enumerate(windows):
        name = window["leg"]
        if name not in PLAN_BY_NAME:
            raise ValueError(f"leg {name!r} is absent from the canonical plan; legacy routes require their original revision")
        leg = PLAN_BY_NAME[name]
        if window.get("bypass"):
            rows.append({"leg": name, "status": "fail",
                         "reason": "legacy bypass is not a completed canonical crossing",
                         "judge_scope": "no_crossing", "score": 0, "frames": 0,
                         "closed": False})
            continue
        field_lo, field_hi = field_bbox(model, leg)
        level_key = official_level_key(leg)
        scored = level_key in by_key
        if scored:
            level = by_key[level_key]
            level_lo = np.asarray(level["obstacle_bbox"]["min"], dtype=float)
            level_hi = np.asarray(level["obstacle_bbox"]["max"], dtype=float)
            # A matching label cannot turn a different physical height into an
            # official tier. In that case retain an unscored geometry verdict.
            scored = bool(np.isclose(field_hi[2], level_hi[2], atol=1e-4))
        if scored:
            mapped = make_map(level["category"], field_lo, field_hi, level_lo, level_hi,
                              yaw=leg["lane_yaw"])
            scope = "official_profile_projection"
        else:
            level, mapped = field_profile(model, leg)
            scope = "field_geometry"
        tail = wall_observation_tail(windows, index)
        observed_window = {**window, "observation_tail": tail} if tail else window
        report = judge_leg(frames, observed_window, level, mapped, scored=scored)
        report["judge_scope"] = scope
        report["closed"] = bool(window["closed"])
        report["target_geoms"] = list(leg["target_geoms"])
        report["field_bbox"] = [field_lo.tolist(), field_hi.tolist()]
        rows.append(report)
        category = report["category"]
        best_by_category[category] = max(best_by_category.get(category, 0), int(report["score"]))

    for name in expected:
        if not any(row["leg"] == name for row in rows):
            rows.append({"leg": name, "status": "missing", "reason": "no recorded expert window",
                         "judge_scope": "no_trace", "score": 0, "frames": 0, "closed": False})
    total = sum(best_by_category.values())
    incomplete = [name for name in expected if not any(
        row["leg"] == name and row["status"] == "success"
        and row.get("closed") and row.get("frames", 0) > 0 for row in rows)]

    print(f"# field -> eval judge: {states.name} ({len(frames)} frames, "
          f"{len(windows)} legs)")
    for row in rows:
        status = row["status"]
        line = (f"{row['leg']:22s} {status:8s} score={row['score']:3d} "
                f"scope={row['judge_scope']}")
        if status == "success":
            line += f"  detail={{'frames': {row['frames']}, 'max_x': {row.get('detail', {}).get('max_x')}}}"
            if row.get("tail_s", 0.0) > 0.0:
                line += f"  observation_tail={row['tail_s']:.3f}s"
        elif status == "fail":
            line += f"  reason={row.get('reason')}"
        else:
            line += f"  ({row.get('reason')})"
        print(line)
    print(f"# field/demo projected score (best official level per category): {total} / 1300")
    print(f"# completed successful legs: {len(expected) - len(incomplete)}/{len(expected)} expected; "
          "field-only tiers are unscored")
    if SKIPPED_TIERS:
        print(f"# explicitly omitted tiers (not crossed): {SKIPPED_TIERS}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"states": str(states), "plan": str(plan),
                                   "legs": rows, "total": total,
                                   "best_by_category": best_by_category,
                                   "expected_legs": expected, "incomplete_legs": incomplete,
                                   "skipped_tiers": SKIPPED_TIERS,
                                   "scope": "field_demo_projection",
                                   "judge_version": part2_judge_version()},
                                  indent=2, ensure_ascii=False) + "\n")
        print(f"# wrote {out}")
    if require_all:
        if incomplete:
            print(f"# --require-all: {len(incomplete)} expected leg(s) incomplete or unsuccessful: "
                  + ", ".join(incomplete), file=sys.stderr)
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
                        help="exit 1 unless every canonical leg has a completed successful support-point window")
    parser.add_argument("--expected-leg", action="append", default=None,
                        help="explicit debug subset: canonical name or 1-based index; repeat for several legs")
    args = parser.parse_args()
    if args.tag:
        states = ARTIFACTS / f"{args.tag}_states.jsonl"
        plan = ARTIFACTS / f"{args.tag}_plan.jsonl"
    else:
        if not args.states or not args.plan:
            parser.error("provide --tag or both --states and --plan")
        states, plan = args.states, args.plan
    return run(states, plan, args.map, args.out, args.require_all, args.expected_leg)


if __name__ == "__main__":
    raise SystemExit(main())
