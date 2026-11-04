#!/usr/bin/env python3
"""Benchmark policies on the diagnostic competition-plum level.

Runs the eval-side in-process runner (``eval/robocon_eval/policy_runner.py``)
against ``tools/diag/plum/plum_comp_diag.xml`` for a grid of blind (proprio-only)
experts and, optionally, the old-repo depth-CNN student.  The diagnostic level
manifest is selected with ``ROBOCON_EVAL_LEVELS``, so ``eval/`` is untouched.

Grid (defaults, from task-2):
    policies : obstacle, row2p_p40w40_v1, vt_band_v1,
               s1d_trench_reward_v75_m100, st30_ppo_st30_s100_600
    spawn    : plum_comp_judge_1 (ON entry platform) / plum_comp_field_1 (ground)
    command  : fixed 0.5 / fixed 0.8 / adaptive 0.5 / adaptive 0.8
    dy offset: 0, +0.05, -0.05, +0.10, -0.10 m (applied after settle)

Adaptive command = the standard eval AdaptiveCommand (vx hold, y->0, yaw->0;
k_y=0.6, k_yaw=1.2, as in ``python -m eval.robocon_eval.policy_runner --adaptive``).

Usage (from the repo root; eval env):
    pixi run -e eval python tools/diag/plum/bench_plum.py --tag blind_grid
    pixi run -e eval python tools/diag/plum/bench_plum.py --tag depth_probe \
        --policies route2_v3_depth --variants judge --modes fixed05 fixed08 \
        adaptive05 adaptive08 --offsets 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repo root
DIAG = Path(__file__).resolve().parent
LOCAL_POLICIES = ROOT / "artifacts/part2_experts/plum/policies"
RESULTS = ROOT / "artifacts/part2_experts/plum/results"

# Must be set before robocon_eval.part2.official resolves the manifest (both the
# runner's level lookup and the judge replay's canonicalize_level read it).
os.environ["ROBOCON_EVAL_LEVELS"] = str(DIAG / "plum_diag_levels.json")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robocon_eval.policy_runner import AdaptiveCommand, run  # noqa: E402

SCENE = "tools/diag/plum/plum_comp_diag.xml"

POLICIES = {
    # named policy dirs under demo/rl_sar/policy/blackW/
    "obstacle": "obstacle",
    "row2p_p40w40_v1": "row2p_p40w40_v1",
    "s1d_trench_reward_v75_m100": "s1d_trench_reward_v75_m100",
    "st30_ppo_st30_s100_600": "st30_ppo_st30_s100_600",
    # local copies (old repo / deployment artifacts, not in the lean policy root)
    "vt_band_v1": str(LOCAL_POLICIES / "vt_band_v1"),
    "route2_v3_depth": str(LOCAL_POLICIES / "route2_v3"),
}
VARIANTS = {
    "judge": "plum_comp_judge_1",   # spawn ON the entry platform (judge-style)
    "field": "plum_comp_field_1",   # spawn on ground 1.8 m before the entry platform
}
MODES = {
    "fixed05": (0.5, False),
    "fixed08": (0.8, False),
    "adaptive05": (0.5, True),
    "adaptive08": (0.8, True),
}
OFFSETS = [0.0, 0.05, -0.05, 0.10, -0.10]


# ---------------------------------------------------------------------------
# trace diagnostics (milestones judged from the recorded frames)
# ---------------------------------------------------------------------------
def _wheels(frame: dict):
    support = frame.get("support") or {}
    legs = ("FL", "FR", "RL", "RR")
    if not all(leg in support for leg in legs):
        return None
    return [[float(v) for v in support[leg]] for leg in legs]


def milestones(trace_path: Path) -> dict:
    """Extract qualitative milestones from a trace (wheel-bottom support frame)."""
    with trace_path.open() as fh:
        meta = json.loads(fh.readline())
        frames = [json.loads(line) for line in fh if line.strip()]
    out = {
        "trace_frames": len(frames),
        "support_frame": meta.get("support_frame"),
        "max_base_x": None,
        "min_wheel_z": None,
        "max_base_y": None,
        "entry_platform_standing": False,
        "field_entered": False,
        "field_max_x_no_drop": None,
        "exit_platform_standing": False,
        "ground_beyond_exit": False,
        "first_drop": None,
        "max_base_x_before_failure": None,
        "first_lane_violation": None,
        "final_base": None,
    }
    if not frames:
        return out
    window = (2.55, 4.55)
    lane_limit = 0.72
    import numpy as np
    failure_t = None
    for frame in frames:
        base = frame.get("base") or [0, 0, 0]
        wheels = _wheels(frame)
        if wheels is None:
            continue
        w = np.asarray(wheels, dtype=float)
        bx, by = float(base[0]), float(base[1])
        out["max_base_x"] = bx if out["max_base_x"] is None else max(out["max_base_x"], bx)
        out["max_base_y"] = (abs(by) if out["max_base_y"] is None
                             else max(out["max_base_y"], abs(by)))
        out["min_wheel_z"] = (float(w[:, 2].min()) if out["min_wheel_z"] is None
                              else min(out["min_wheel_z"], float(w[:, 2].min())))
        top = (w[:, 2] >= 0.14) & (w[:, 2] <= 0.25)
        if 1.75 <= bx <= 2.65 and top.all():
            out["entry_platform_standing"] = True
        if bx >= 2.6:
            out["field_entered"] = True
        if 2.55 <= bx <= 4.55 and bool((w[:, 2] >= 0.05).all()):
            out["field_max_x_no_drop"] = (bx if out["field_max_x_no_drop"] is None
                                          else max(out["field_max_x_no_drop"], bx))
        if bx >= 4.5 and top.all():
            out["exit_platform_standing"] = True
        if (bx >= 5.40 and (w[:, 0] >= 5.32).all()
                and (w[:, 2] <= 0.08).all()):
            out["ground_beyond_exit"] = True
        min_z = float(w[:, 2].min())
        if (out["first_drop"] is None and window[0] <= bx <= window[1]
                and min_z < 0.05):
            legs = ("FL", "FR", "RL", "RR")
            out["first_drop"] = {
                "t": frame.get("t"), "base_x": round(bx, 4),
                "min_wheel_z": min_z,
                "leg": legs[int(np.argmin(w[:, 2]))]}
            failure_t = frame.get("t")
        if (out["first_lane_violation"] is None
                and (abs(by) > lane_limit
                     or bool(np.any(np.abs(w[:, 1]) > lane_limit)))):
            out["first_lane_violation"] = {"t": frame.get("t"),
                                           "base_y": round(by, 4)}
            failure_t = failure_t if failure_t is not None else frame.get("t")
        if failure_t is None or float(frame.get("t", 0.0)) <= float(failure_t):
            out["max_base_x_before_failure"] = (
                bx if out["max_base_x_before_failure"] is None
                else max(out["max_base_x_before_failure"], bx))
        out["final_base"] = [round(float(v), 4) for v in base]
    return out


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def run_one(policy_spec: str, policy_name: str, variant: str, mode: str,
            offset: float, outdir: Path, duration_judge: float,
            duration_field: float) -> dict:
    vx, adaptive = MODES[mode]
    level = VARIANTS[variant]
    duration = duration_judge if variant == "judge" else duration_field
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{policy_name}__{variant}__{mode}__dy{offset:+.2f}"
    trace = outdir / f"{tag}.jsonl"
    result_path = outdir / f"{tag}.json"
    command_fn = (AdaptiveCommand(vx=vx, k_y=0.6, k_yaw=1.2) if adaptive else None)
    started = time.time()
    try:
        result = run(
            "blackW", 2, policy_spec, level, trace,
            scene=SCENE,
            duration_s=duration,
            command=(vx, 0.0, 0.0),
            settle_s=1.0,
            warmup_s=0.0,
            seed=0,
            stop_on_fall=True,
            profile="diagnostic",
            command_fn=command_fn,
            # always apply the offset path (even dy=0) so every grid cell gets
            # the same post-settle state refresh; None is kept as a reference
            # in the depth/knife-edge run.
            spawn_offset=(offset, 0.0),
        )
        result["milestones"] = milestones(Path(trace))
        result["grid"] = {"policy_name": policy_name, "variant": variant,
                          "mode": mode, "dy": offset,
                          "vx": vx, "adaptive": adaptive,
                          "level": level, "duration_s": duration}
    except Exception as exc:  # noqa: BLE001 - keep the grid running
        result = {"error": f"{type(exc).__name__}: {exc}", "trace": str(trace),
                  "grid": {"policy_name": policy_name, "variant": variant,
                           "mode": mode, "dy": offset, "vx": vx,
                           "adaptive": adaptive, "level": level,
                           "duration_s": duration}}
    result["wall_time_s"] = round(time.time() - started, 3)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return {
        "policy": policy_name,
        "variant": variant,
        "mode": mode,
        "dy": offset,
        "status": result.get("status", "error"),
        "score": result.get("score"),
        "reason": result.get("reason"),
        "sim_time_s": result.get("sim_time_s"),
        "stopped": result.get("stopped"),
        "warnings": result.get("warnings"),
        "error": result.get("error"),
        "milestones": result.get("milestones"),
        "wall_time_s": result["wall_time_s"],
        "result_path": str(result_path),
    }


def aggregate(rows: list[dict]) -> dict:
    cells: dict = {}
    for row in rows:
        key = (row["policy"], row["variant"], row["mode"])
        cell = cells.setdefault(key, {"attempts": 0, "successes": 0,
                                      "best_max_x": None, "reasons": {}})
        cell["attempts"] += 1
        if row["status"] == "success":
            cell["successes"] += 1
        ms = row.get("milestones") or {}
        mx = ms.get("max_base_x")
        if mx is not None:
            cell["best_max_x"] = mx if cell["best_max_x"] is None else max(cell["best_max_x"], mx)
        reason = (row.get("reason") or row.get("error") or "?").split(":")[0]
        cell["reasons"][reason] = cell["reasons"].get(reason, 0) + 1
    table = [
        {"policy": k[0], "variant": k[1], "mode": k[2], **v}
        for k, v in sorted(cells.items())
    ]
    per_policy: dict = {}
    for row in rows:
        p = per_policy.setdefault(row["policy"], {"attempts": 0, "successes": 0})
        p["attempts"] += 1
        if row["status"] == "success":
            p["successes"] += 1
    successes = [r for r in rows if r["status"] == "success"]
    return {
        "attempts": len(rows),
        "successes": len(successes),
        "success_cells": [
            {k: r[k] for k in ("policy", "variant", "mode", "dy", "status",
                               "reason", "result_path")}
            for r in successes
        ],
        "per_policy": per_policy,
        "cells": table,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="results subdir/tag")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES),
                    choices=sorted(POLICIES))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=sorted(VARIANTS))
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=sorted(MODES))
    ap.add_argument("--offsets", nargs="+", type=float, default=OFFSETS)
    ap.add_argument("--duration-judge", type=float, default=12.0)
    ap.add_argument("--duration-field", type=float, default=16.0)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args(argv)
    outdir = args.outdir or (RESULTS / args.tag)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.policies, args.variants, args.modes,
                                  args.offsets))
    rows = []
    started = time.time()
    for i, (policy_name, variant, mode, offset) in enumerate(grid, start=1):
        row = run_one(POLICIES[policy_name], policy_name, variant, mode, offset,
                      outdir, args.duration_judge, args.duration_field)
        rows.append(row)
        ms = row.get("milestones") or {}
        print(f"[{i}/{len(grid)}] {policy_name:28s} {variant:5s} {mode:10s} "
              f"dy={offset:+.2f}  {row['status']:7s} "
              f"max_x={ms.get('max_base_x')} "
              f"reason={row['reason'] or row.get('error') or ''} "
              f"({row['wall_time_s']:.2f}s)", flush=True)
    summary = {
        "tag": args.tag,
        "scene": SCENE,
        "levels_manifest": str(DIAG / "plum_diag_levels.json"),
        "policies": {p: POLICIES[p] for p in args.policies},
        "grid": {"variants": args.variants, "modes": args.modes,
                 "offsets": args.offsets,
                 "duration_judge_s": args.duration_judge,
                 "duration_field_s": args.duration_field},
        "wall_time_s": round(time.time() - started, 3),
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    summary_path = RESULTS / f"{args.tag}__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    agg = summary["aggregate"]
    print(f"\n{agg['successes']}/{agg['attempts']} success; "
          f"per_policy={json.dumps(agg['per_policy'])}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
