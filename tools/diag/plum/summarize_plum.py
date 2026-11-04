#!/usr/bin/env python3
"""Aggregate the competition-plum benchmark results into report tables.

Reads the per-run JSONs written by ``bench_plum.py`` / ``probe_depth_student.py``
and writes ``artifacts/part2_experts/plum/results/analysis.json`` plus markdown
tables on stdout (copy-pasted into REPORT.md).

Usage (plain python, no deps):
    python tools/diag/plum/summarize_plum.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "artifacts/part2_experts/plum/results"
GRID = RESULTS / "blind_grid"
DEPTH = RESULTS / "depth_student__summary.json"

POLICIES = ["obstacle", "row2p_p40w40_v1", "vt_band_v1",
            "s1d_trench_reward_v75_m100", "st30_ppo_st30_s100_600",
            "route2_v3_depth"]
VARIANTS = ["judge", "field"]
MODES = ["fixed05", "fixed08", "adaptive05", "adaptive08"]


def _load_runs() -> list[dict]:
    rows = []
    for path in sorted(GRID.glob("*.json")):
        d = json.loads(path.read_text())
        g = d.get("grid") or {}
        if not g:
            continue
        rows.append({
            "policy": g["policy_name"], "variant": g["variant"],
            "mode": g["mode"], "dy": g["dy"],
            "status": d.get("status"), "score": d.get("score"),
            "reason": d.get("reason"), "stopped": d.get("stopped"),
            "warnings": d.get("warnings") or [],
            "milestones": d.get("milestones") or {},
            "judge_detail": (d.get("judge") or {}).get("detail") or {},
            "result_path": str(path),
        })
    return rows


def _cell(rows: list[dict]) -> dict:
    cell = {
        "attempts": len(rows),
        "successes": sum(1 for r in rows if r["status"] == "success"),
        "offset_api_used": bool(rows),
        "deepest_base_x_before_failure": None,
        "entry_platform_mounts": 0,
        "exit_platform_standing": 0,
        "first_drop_legs": collections.Counter(),
        "first_drop_x": [],
        "fell": 0,
        "warnings": sorted({w for r in rows for w in r["warnings"]}),
    }
    for r in rows:
        ms = r["milestones"]
        mx = ms.get("max_base_x_before_failure")
        if mx is not None:
            cell["deepest_base_x_before_failure"] = (
                mx if cell["deepest_base_x_before_failure"] is None
                else max(cell["deepest_base_x_before_failure"], mx))
        if ms.get("entry_platform_standing"):
            cell["entry_platform_mounts"] += 1
        if ms.get("exit_platform_standing"):
            cell["exit_platform_standing"] += 1
        fd = ms.get("first_drop")
        if fd:
            cell["first_drop_legs"][fd.get("leg", "?")] += 1
            cell["first_drop_x"].append(fd.get("base_x"))
        if r["stopped"] == "fall":
            cell["fell"] += 1
    if cell["first_drop_x"]:
        xs = sorted(cell["first_drop_x"])
        cell["first_drop_x_mean"] = round(sum(xs) / len(xs), 3)
    cell["first_drop_legs"] = dict(cell["first_drop_legs"])
    return cell


def main() -> int:
    rows = _load_runs()
    analysis: dict = {
        "grid_runs": len(rows),
        "grid_successes": sum(1 for r in rows if r["status"] == "success"),
        "cells": {},
        "per_policy": {},
        "depth": {},
    }
    md = []
    for variant in VARIANTS:
        sub = [r for r in rows if r["variant"] == variant]
        md.append(f"\n### variant `{variant}` — all policies (dy offsets "
                  f"-0.10..+0.10, 5 per cell)\n")
        md.append("| policy | mode | succ/att | deepest base x | mounts onto entry "
                  "platform | first-drop leg | first-drop x |")
        md.append("|---|---|---|---|---|---|---|")
        for policy in POLICIES:
            for mode in MODES:
                cell_rows = [r for r in sub if r["policy"] == policy
                             and r["mode"] == mode]
                if not cell_rows:
                    continue
                c = _cell(cell_rows)
                analysis["cells"][f"{policy}|{variant}|{mode}"] = c
                deepest = c["deepest_base_x_before_failure"]
                legs = ", ".join(f"{k}x{v}" for k, v in
                                 sorted(c["first_drop_legs"].items())) or "-"
                md.append(
                    f"| {policy} | {mode} | {c['successes']}/{c['attempts']} | "
                    f"{'-' if deepest is None else round(deepest, 3)} | "
                    f"{c['entry_platform_mounts']}/{c['attempts']} | {legs} | "
                    f"{c.get('first_drop_x_mean', '-')} |")
    # per policy totals
    md.append("\n### per-policy totals (all variants/modes/offsets)\n")
    md.append("| policy | attempts | successes | deepest base x |")
    md.append("|---|---|---|---|")
    for policy in POLICIES:
        sub = [r for r in rows if r["policy"] == policy]
        if not sub:
            continue
        c = _cell(sub)
        analysis["per_policy"][policy] = c
        deepest = c["deepest_base_x_before_failure"]
        md.append(f"| {policy} | {c['attempts']} | {c['successes']} | "
                  f"{'-' if deepest is None else round(deepest, 3)} |")

    # depth student summary (including the y-mirror reproduction probe)
    if DEPTH.is_file():
        depth = json.loads(DEPTH.read_text())
        by_orientation = collections.defaultdict(lambda: {"attempts": 0,
                                                          "successes": 0})
        for r in depth["rows"]:
            key = (r["orientation"], "field" if "__field__" in r["tag"] else "judge")
            by_orientation[key]["attempts"] += 1
            if r["status"] == "success":
                by_orientation[key]["successes"] += 1
        analysis["depth"] = {
            "policy": depth["policy"],
            "successes": depth["successes"],
            "attempts": depth["attempts"],
            "by_orientation_variant": {
                f"{k[0]}|{k[1]}": v for k, v in by_orientation.items()},
            "rows": [{k: r[k] for k in ("tag", "orientation", "mode", "dy",
                                        "status", "reason", "judge_detail")}
                     for r in depth["rows"]],
        }
        md.append("\n### depth student route2_v3 (auto-detected depth.json)\n")
        md.append("| setup | attempts | successes |")
        md.append("|---|---|---|")
        for key, val in sorted(by_orientation.items()):
            md.append(f"| {key[0]} / {key[1]} | {val['attempts']} | "
                      f"{val['successes']} |")

    out = RESULTS / "analysis.json"
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
