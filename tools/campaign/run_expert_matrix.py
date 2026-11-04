#!/usr/bin/env python3
"""Part 2 per-obstacle expert matrix campaign (blackW, 16-DoF wheel-leg).

Runs a policy x official-level x command-mode x spawn-offset matrix through the
in-process eval runner (``robocon_eval.policy_runner.run``), replays the v5 Part 2
judge from each trace, and writes a machine-readable + human-readable report:

    artifacts/part2_experts/summary.json
    artifacts/part2_experts/summary.md
    artifacts/part2_experts/runs/<tag>.json      (per-run cache/evidence)
    artifacts/part2_experts/policy_sources.json  (written by the copy step)

Protocol (matches the authoritative ``~/RoboCon/artifacts/vtaudit-20261031``
benchmark unless noted):

* ``profile='official'``, ``settle_s=1.0``, ``stop_on_fall=True``; the MuJoCo
  sim is deterministic, so one run per (policy, level, mode, offset) is one
  trial -- robustness comes from spawn offsets and command modes, not seeds.
* primary duration: 28 s (stairs: 30 s, matching the old benchmark's 30 s; the
  official safety cap is 120 s and success is a terminal state resolved much
  earlier).  A ``duration`` phase reruns the winners at the old benchmark
  durations (15/18/30 s) and at 24/32 s to show the margin.
* command modes: adaptive vx=0.5 with the exact training gains
  (k_y=0.8,k_yaw=2.0,k_vy=0.6,k_vheading=0.0,vy_gain=0.5,max_vy=0.2,max_wz=0.7,
  center_y=0.0); fixed vx=0.5 and speed variants 0.3/0.35/0.8/1.0; adaptive 0.8.
* phases (default order): matrix -> modes -> align -> duration -> repro.
  Results are cached per run, so rerunning with the same command is cheap and
  ``--aggregate-only`` rebuilds the report from cache.

Usage
-----
    pixi run -e eval python tools/campaign/run_expert_matrix.py
    pixi run -e eval python tools/campaign/run_expert_matrix.py --phases matrix,modes
    pixi run -e eval python tools/campaign/run_expert_matrix.py --aggregate-only
    pixi run -e eval python tools/campaign/run_expert_matrix.py \
        --only vt_band_v1:stairs_30cm:adaptive05

Only writes under ``tools/campaign/`` and ``artifacts/part2_experts/``; trace
files are written to ``/dev/shm`` and deleted after judging.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
ART = BASE / "artifacts/part2_experts"
RUNS_DIR = ART / "runs"
TRACE_DIR = Path("/dev/shm/part2_experts_traces")
POLICY_ROOT = BASE / "demo/rl_sar/policy/blackW"

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
ADAPTIVE_GAINS = dict(
    k_y=0.8, k_yaw=2.0, k_vy=0.6, k_vheading=0.0,
    vy_gain=0.5, max_vy=0.2, max_wz=0.7, center_y=0.0,
)

# ``kind`` is adaptive (feedback command, vx held) or fixed (constant command).
MODES: dict[str, dict[str, Any]] = {
    "adaptive05": {"kind": "adaptive", "vx": 0.5},
    "adaptive08": {"kind": "adaptive", "vx": 0.8},
    "fixed03": {"kind": "fixed", "command": (0.3, 0.0, 0.0)},
    "fixed035": {"kind": "fixed", "command": (0.35, 0.0, 0.0)},
    "fixed05": {"kind": "fixed", "command": (0.5, 0.0, 0.0)},
    "fixed08": {"kind": "fixed", "command": (0.8, 0.0, 0.0)},
    "fixed10": {"kind": "fixed", "command": (1.0, 0.0, 0.0)},
    # Probe-only speed variants (single-level mode-window characterisation).
    "adaptive040": {"kind": "adaptive", "vx": 0.40},
    "adaptive045": {"kind": "adaptive", "vx": 0.45},
    "adaptive055": {"kind": "adaptive", "vx": 0.55},
    "adaptive060": {"kind": "adaptive", "vx": 0.60},
    "adaptive070": {"kind": "adaptive", "vx": 0.70},
    "fixed045": {"kind": "fixed", "command": (0.45, 0.0, 0.0)},
    "fixed060": {"kind": "fixed", "command": (0.60, 0.0, 0.0)},
    "fixed070": {"kind": "fixed", "command": (0.70, 0.0, 0.0)},
    "fixed090": {"kind": "fixed", "command": (0.90, 0.0, 0.0)},
}
# Preference order when several modes solve the same level (lower = preferred):
# nominal adaptive first, then nominal fixed, then the speed variants.
MODE_ORDER = ["adaptive05", "fixed05", "fixed08", "fixed10", "adaptive08",
              "fixed035", "fixed03"]

PRIMARY_DURATION = {"stairs": 30.0}
DEFAULT_DURATION = 28.0
OLD_BENCH_DURATION = {  # durations of the authoritative vtaudit-20261031 runs
    "high_platform": 15.0, "stairs": 30.0, "slope": 18.0,
    "trench": 15.0, "high_wall": 15.0,
}
# Fine alignment grid for the stairs_30cm bottleneck (route pre-expert gate).
BASIN_DY = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
BASIN_DYAW = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0,
              1.0, 2.0, 3.0, 4.0, 5.0]
BASIN_FULL_GRID_MODES = ["adaptive05", "adaptive055", "adaptive070"]
BASIN_VX_MODES = ["adaptive045", "adaptive055"]
BASIN_VX_OFFSETS = [(0.0, 0.0), (0.0, 5.0), (0.05, 0.0), (0.0, -5.0)]
# Fine neighborhood around the promising vx=0.70 row (dy=-0.05..-0.03 all pass
# in the coarse grid) -- tests whether the patch is locally stable or alias noise.
BASIN2_MODE = "adaptive070"
BASIN2_CENTER = (-0.04, 0.0)
BASIN2_DY = [round(-0.06 + 0.005 * i, 3) for i in range(9)]      # -0.060..-0.020
BASIN2_DYAW = [round(-2.0 + 0.5 * i, 1) for i in range(9)]       # -2.0..+2.0

# Required alignment-sensitivity offsets: (dy m, dyaw deg).
ALIGN_OFFSETS = [
    (0.05, 0.0), (-0.05, 0.0), (0.10, 0.0), (-0.10, 0.0),
    (0.0, 2.0), (0.0, -2.0), (0.0, 5.0), (0.0, -5.0),
]

# ---------------------------------------------------------------------------
# Level catalog (targets = the max official difficulty needed by the field)
# ---------------------------------------------------------------------------
LEVEL_GROUPS = {
    "high_platform": ["high_platform_40cm", "high_platform_30cm", "high_platform_20cm"],
    "stairs": ["stairs_30cm", "stairs_20cm", "stairs_15cm"],
    "slope": ["slope_35deg", "slope_30deg", "slope_20deg"],
    "trench": ["trench_30cm", "trench_20cm"],
    "high_wall": ["high_wall_40cm", "high_wall_30cm"],
}
TARGET_LEVEL = {
    "high_platform": "high_platform_40cm",
    "stairs": "stairs_30cm",
    "slope": "slope_35deg",
    "trench": "trench_30cm",
    "high_wall": "high_wall_40cm",
}
# Context-only levels (beyond the route's max difficulty; measured, not required).
CONTEXT_LEVELS = ["slope_45deg"]
MATRIX_LEVELS = [lv for group in LEVEL_GROUPS.values() for lv in group]

# ---------------------------------------------------------------------------
# Policy roster
# ---------------------------------------------------------------------------
# The task-1 candidate roster.
CORE_POLICIES = [
    "row2p_p40w40_v1", "st30_ppo_st30_s100_600", "s1d_trench_reward_v75_m100",
    "obstacle", "vt_band_v1", "vt_mode_v2", "vt_freezehid_1349", "vtg09_1100",
    "vt_polish_1175", "fixed-stairs30-final", "spec_stairs_v2_800", "stairs30-long",
]
# Additional in-repo policies worth measuring for "best deployable expert".
EXTRA_POLICIES = [
    "vt_track_v1", "vt_track_v2", "vt_pose_v1", "vt_edge_v1",
    "vtg095_1100", "vt_mixture_v1", "flat", "stair",
]
# Policies whose advertised speciality is stairs (all also measured on stairs).
STAIRS_SPECIALISTS = [
    "st30_ppo_st30_s100_600", "fixed-stairs30-final", "spec_stairs_v2_800",
    "stairs30-long", "exp-stairs30-curric", "exp-climbstairs-final",
    "exp-r3-stairs-399", "exp-stairs27", "guard3_stairs_199",
    "guard4_stairs_450", "ts_trial_guard4_stairs_m799",
]
# Reference generalists used in the targeted mode sweeps.
GENERALISTS_40 = ["row2p_p40w40_v1", "vt_mode_v2", "vt_freezehid_1349",
                  "vt_mixture_v1", "vt_band_v1", "vtg09_1100"]
TRENCH_REFERENCE = ["s1d_trench_reward_v75_m100", "row2p_p40w40_v1",
                    "vt_mode_v2", "vt_freezehid_1349", "vt_band_v1"]
STAIRS_REFERENCE = ["vt_band_v1", "vt_track_v2", "vt_pose_v1", "vt_edge_v1",
                    "row2p_p40w40_v1", "vt_mode_v2"]

# Route-level preference: equal-tolerance winners may still be re-ordered when a
# category has a deployment reason (here: trench adaptive08 keeps a symmetric
# yaw tolerance; the s1d specialist tolerates dy+-0.10 and yaw +-2 only).
RECOMMENDED_MODE_OVERRIDE = {"trench": "adaptive08"}

MODE_SWEEP_MODES = {
    "high_platform": ["adaptive05", "fixed05", "fixed035", "fixed03", "fixed08"],
    "stairs": ["adaptive05", "fixed05", "fixed03", "fixed08", "fixed10"],
    "slope": ["adaptive05", "fixed05", "fixed035", "fixed03", "fixed08"],
    "trench": ["fixed05", "fixed08", "fixed10", "adaptive05", "adaptive08"],
    "high_wall": ["adaptive05", "fixed05", "fixed035", "fixed03", "fixed08"],
}

# Authoritative old-benchmark evidence (vtaudit-20261031, judge v7.6) that the
# lean port must reproduce; ("old" duration = OLD_BENCH_DURATION).
# Authoritative baseline evidence from campaigns/*/summary.json (matrix.log run,
# judge v7.6, legacy scenes; the current scenes differ only by the blackW include
# path, so the MuJoCo model is identical).
REPRO_EXPECTED = [
    # policy, level, mode, expected status
    ("vt_band_v1", "stairs_30cm", "adaptive05", "success"),        # only stairs30 solver
    ("vt_band_v1", "high_wall_40cm", "adaptive05", "success"),
    ("vt_band_v1", "slope_35deg", "adaptive05", "success"),
    ("row2p_p40w40_v1", "high_platform_40cm", "adaptive05", "success"),
    ("row2p_p40w40_v1", "high_wall_40cm", "adaptive05", "success"),
    ("row2p_p40w40_v1", "slope_35deg", "adaptive05", "success"),
    ("row2p_p40w40_v1", "stairs_30cm", "adaptive05", "fail"),
    ("row2p_p40w40_v1", "trench_30cm", "adaptive05", "fail"),
    ("vt_mode_v2", "high_platform_40cm", "adaptive05", "success"),
    ("vt_mode_v2", "high_wall_40cm", "adaptive05", "success"),
    ("vt_mode_v2", "trench_30cm", "adaptive05", "fail"),
    ("s1d_trench_reward_v75_m100", "trench_30cm", "fixed08", "success"),
    ("s1d_trench_reward_v75_m100", "trench_30cm", "fixed10", "success"),
    ("s1d_trench_reward_v75_m100", "trench_30cm", "fixed05", "fail"),
    ("fixed-stairs30-final", "stairs_30cm", "adaptive05", "fail"),
    ("st30_ppo_st30_s100_600", "stairs_30cm", "adaptive05", "fail"),
]


# ---------------------------------------------------------------------------
# Environment / level helpers (no mujoco import at module level so that
# --aggregate-only stays light; workers import the runner themselves).
# ---------------------------------------------------------------------------
def _judge_version() -> str:
    try:
        from robocon_eval.part2.v5_judge import JUDGE_VERSION
        return str(JUDGE_VERSION)
    except Exception:  # noqa: BLE001 - report scripts must never die for this
        return "unknown"


def _level_catalog() -> dict[str, dict]:
    from robocon_eval.part2.official import load_levels
    return {e["key"]: e for e in load_levels()}


def _duration_for(category: str) -> float:
    return float(PRIMARY_DURATION.get(category, DEFAULT_DURATION))


def _old_duration_for(category: str) -> float:
    return float(OLD_BENCH_DURATION[category])


def _git_rev() -> dict:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(args, cwd=BASE, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    rev = _run("git", "rev-parse", "HEAD")
    dirty = _run("git", "status", "--porcelain")
    return {"rev": rev, "dirty": bool(dirty) if dirty is not None else None}


def _policy_meta(name: str) -> dict:
    d = POLICY_ROOT / name
    meta: dict[str, Any] = {"dir": str(d.relative_to(BASE)) if d.is_dir() else None}
    for field, fname in (("policy_sha256", "policy.pt"), ("config_sha256", "config.yaml")):
        p = d / fname
        if p.is_file():
            import hashlib
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            meta[field] = h.hexdigest()
    manifest = d / "manifest.json"
    if manifest.is_file():
        try:
            meta["manifest"] = json.loads(manifest.read_text())
        except Exception:  # noqa: BLE001
            pass
    return meta


def _role(name: str) -> str:
    if name in CORE_POLICIES:
        return "core"
    if name in EXTRA_POLICIES:
        return "extra"
    return "stairs_specialist"


ROLE_ORDER = {"core": 0, "extra": 1, "stairs_specialist": 2}


# ---------------------------------------------------------------------------
# Job specification / execution
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{v:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")


def make_tag(phase: str, policy: str, level: str, mode: str,
             offset: tuple[float, float], duration: float) -> str:
    dy, dyaw = offset
    return (f"{phase}__{policy}__{level}__{mode}"
            f"__dy{_fmt(dy)}__dyaw{_fmt(dyaw)}__d{duration:g}")


def make_spec(phase: str, policy: str, level: str, mode: str,
              offset: tuple[float, float], duration: float,
              catalog: dict[str, dict]) -> dict:
    entry = catalog[level]
    return {
        "tag": make_tag(phase, policy, level, mode, offset, duration),
        "phase": phase,
        "policy": policy,
        "level": level,
        "category": entry["category"],
        "level_value": int(entry["level"]),
        "mode": mode,
        "offset": [float(offset[0]), float(offset[1])],
        "duration_s": float(duration),
    }


def _worker(spec: dict) -> dict:
    """Execute one run inside a worker process (must be importable)."""
    import hashlib

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass

    from robocon_eval.policy_runner import AdaptiveCommand, run

    mode = MODES[spec["mode"]]
    command_fn = None
    command = None
    if mode["kind"] == "adaptive":
        command_fn = AdaptiveCommand(vx=mode["vx"], **ADAPTIVE_GAINS)
    else:
        command = tuple(mode["command"])
    offset = tuple(spec["offset"])
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace = TRACE_DIR / f"{abs(hash(spec['tag'])):x}_{os.getpid()}.jsonl"
    started = time.time()
    try:
        result = run(
            "blackW", 2, spec["policy"], spec["level"], trace,
            duration_s=spec["duration_s"], command=command,
            settle_s=1.0, warmup_s=0.0, seed=0, profile="official",
            command_fn=command_fn,
            spawn_offset=offset if offset != (0.0, 0.0) else None,
        )
        status = result.get("status") or "fail"
        reason = result.get("reason") or ""
        score = int(result.get("score") or 0)
        detail = {k: v for k, v in (result.get("judge") or {}).get("detail", {}).items()
                  if isinstance(v, (int, float, bool, str))}
        rec = {
            "tag": spec["tag"], "phase": spec["phase"], "policy": spec["policy"],
            "level": spec["level"], "category": spec["category"],
            "level_value": spec["level_value"], "mode": spec["mode"],
            "offset": list(offset), "duration_s": spec["duration_s"],
            "status": status, "score": score, "reason": reason,
            "sim_time_s": result.get("sim_time_s"), "frames": result.get("frames"),
            "stopped": result.get("stopped"), "finite": bool(result.get("finite")),
            "judge_detail": detail,
            "policy_sha256": result.get("policy_sha256"),
            "config_sha256": result.get("config_sha256"),
            "scene_sha256": result.get("scene_sha256"),
            "warnings": list(result.get("warnings") or []),
            "wall_time_s": round(float(result.get("wall_time_s") or 0.0), 3),
        }
    except Exception as exc:  # noqa: BLE001 - a broken run is evidence, not a crash
        rec = {
            "tag": spec["tag"], "phase": spec["phase"], "policy": spec["policy"],
            "level": spec["level"], "category": spec["category"],
            "level_value": spec["level_value"], "mode": spec["mode"],
            "offset": list(offset), "duration_s": spec["duration_s"],
            "status": "error", "score": 0,
            "reason": f"runner_error:{type(exc).__name__}:{exc}",
            "sim_time_s": None, "frames": None, "stopped": "error",
            "finite": False, "judge_detail": {}, "policy_sha256": None,
            "config_sha256": None, "scene_sha256": None, "warnings": [],
            "wall_time_s": round(time.time() - started, 3),
        }
    finally:
        try:
            trace.unlink()
        except OSError:
            pass
    rec["campaign_wall_time_s"] = round(time.time() - started, 3)
    return rec


def _record_path(tag: str) -> Path:
    return RUNS_DIR / f"{tag}.json"


def config_key(rec: dict) -> tuple:
    """Identity of a measured configuration (a rerun is bit-identical)."""
    offset = rec.get("offset") or [0.0, 0.0]
    return (rec.get("policy"), rec.get("level"), rec.get("mode"),
            round(float(offset[0]), 6), round(float(offset[1]), 6),
            round(float(rec.get("duration_s") or 0.0), 6))


def execute_specs(specs: list[dict], jobs: int, force: bool = False,
                  known: dict[tuple, dict] | None = None) -> dict[str, dict]:
    """Run specs with a process pool, using per-run JSON files as cache.

    ``known`` maps config_key -> already-measured record from another phase; a
    spec whose configuration is already measured is not rerun (deterministic
    sim), it simply reuses that evidence.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    known = known or {}
    records: dict[str, dict] = {}
    todo: list[dict] = []
    for spec in specs:
        if not force and config_key(spec) in known:
            continue
        path = _record_path(spec["tag"])
        if path.is_file() and not force:
            try:
                records[spec["tag"]] = json.loads(path.read_text())
                continue
            except Exception:  # noqa: BLE001 - corrupt cache -> rerun
                pass
        todo.append(spec)
    n_cached = len(records)
    if not todo:
        print(f"  all {n_cached} run(s) cached", flush=True)
        return records
    print(f"  {n_cached} cached, running {len(todo)} with {jobs} workers", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_worker, spec): spec for spec in todo}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 - worker bootstrap failure
                rec = {"tag": spec["tag"], **{k: spec[k] for k in
                                              ("phase", "policy", "level", "category",
                                               "level_value", "mode", "duration_s")},
                       "offset": spec["offset"], "status": "error",
                       "score": 0, "reason": f"pool_error:{type(exc).__name__}:{exc}",
                       "finite": False, "judge_detail": {}, "wall_time_s": None,
                       "sim_time_s": None, "frames": None, "stopped": "error",
                       "policy_sha256": None, "config_sha256": None,
                       "scene_sha256": None, "warnings": []}
            records[spec["tag"]] = rec
            _record_path(spec["tag"]).write_text(json.dumps(rec, indent=1) + "\n")
            done += 1
            if done % 25 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"    {done}/{len(todo)} runs  ({rate:.2f} runs/s)", flush=True)
    print(f"  {len(todo)} run(s) in {time.time() - t0:.1f}s", flush=True)
    return records


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------
def _dedupe(specs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for spec in specs:
        if spec["tag"] in seen:
            continue
        seen.add(spec["tag"])
        out.append(spec)
    return out


def phase_matrix(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    specs = []
    broad = CORE_POLICIES + EXTRA_POLICIES
    for policy in broad:
        for level in MATRIX_LEVELS:
            category = catalog[level]["category"]
            for mode in ("adaptive05", "fixed05"):
                specs.append(make_spec("matrix", policy, level, mode, (0.0, 0.0),
                                       _duration_for(category), catalog))
    # Stairs specialists are also measured on the stairs levels in the matrix.
    for policy in STAIRS_SPECIALISTS:
        if policy in broad:
            continue
        for level in LEVEL_GROUPS["stairs"]:
            for mode in ("adaptive05", "fixed05"):
                specs.append(make_spec("matrix", policy, level, mode, (0.0, 0.0),
                                       _duration_for("stairs"), catalog))
    return _dedupe(specs)


def _successful_offset0_policies(results: dict[str, dict], levels: list[str]) -> list[str]:
    """Policies with >=1 offset-0 success on any of ``levels``, best first."""
    ranked: dict[str, tuple] = {}
    for rec in results.values():
        if rec.get("status") != "success" or rec.get("level") not in levels:
            continue
        if tuple(rec.get("offset") or (0.0, 0.0)) != (0.0, 0.0):
            continue
        mode = rec.get("mode", "")
        key = (int(rec.get("level_value") or 0),
               -(MODE_ORDER.index(mode) if mode in MODE_ORDER else 99))
        ranked[rec["policy"]] = max(ranked.get(rec["policy"], (-1, -999)), key)
    return [p for p, _ in sorted(ranked.items(),
                                 key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))]


def phase_modes(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    specs = []
    # -- high_platform ------------------------------------------------------
    cands = GENERALISTS_40 + _successful_offset0_policies(
        results, LEVEL_GROUPS["high_platform"])[:4]
    for policy in _dedupe_names(cands):
        for level in ["high_platform_40cm", "high_platform_30cm"]:
            for mode in MODE_SWEEP_MODES["high_platform"]:
                specs.append(make_spec("modes", policy, level, mode, (0.0, 0.0),
                                       _duration_for("high_platform"), catalog))
    # -- high_wall ----------------------------------------------------------
    cands = GENERALISTS_40 + _successful_offset0_policies(
        results, LEVEL_GROUPS["high_wall"])[:4]
    for policy in _dedupe_names(cands):
        for level in ["high_wall_40cm", "high_wall_30cm"]:
            for mode in MODE_SWEEP_MODES["high_wall"]:
                specs.append(make_spec("modes", policy, level, mode, (0.0, 0.0),
                                       _duration_for("high_wall"), catalog))
    # -- slope --------------------------------------------------------------
    cands = GENERALISTS_40 + _successful_offset0_policies(
        results, LEVEL_GROUPS["slope"])[:4]
    for policy in _dedupe_names(cands):
        for level in LEVEL_GROUPS["slope"] + CONTEXT_LEVELS:
            for mode in MODE_SWEEP_MODES["slope"]:
                specs.append(make_spec("modes", policy, level, mode, (0.0, 0.0),
                                       _duration_for("slope"), catalog))
    # -- trench -------------------------------------------------------------
    cands = TRENCH_REFERENCE + _successful_offset0_policies(
        results, LEVEL_GROUPS["trench"])[:4]
    for policy in _dedupe_names(cands):
        for level in LEVEL_GROUPS["trench"]:
            for mode in MODE_SWEEP_MODES["trench"]:
                specs.append(make_spec("modes", policy, level, mode, (0.0, 0.0),
                                       _duration_for("trench"), catalog))
    # -- stairs -------------------------------------------------------------
    cands = STAIRS_SPECIALISTS + STAIRS_REFERENCE + _successful_offset0_policies(
        results, LEVEL_GROUPS["stairs"])[:4]
    for policy in _dedupe_names(cands):
        for level in ["stairs_30cm", "stairs_20cm"]:
            for mode in MODE_SWEEP_MODES["stairs"]:
                specs.append(make_spec("modes", policy, level, mode, (0.0, 0.0),
                                       _duration_for("stairs"), catalog))
    return _dedupe(specs)


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen and (POLICY_ROOT / n / "config.yaml").is_file():
            seen.add(n)
            out.append(n)
    return out


def _offset0_results(results: dict[str, dict]) -> list[dict]:
    return [r for r in results.values()
            if tuple(r.get("offset") or (0.0, 0.0)) == (0.0, 0.0)]


def _record_rank(rec: dict, target_duration: float) -> tuple:
    mode = rec.get("mode", "")
    return (
        0 if rec.get("status") == "success" else 1,
        abs(float(rec.get("duration_s") or 0.0) - target_duration),
        MODE_ORDER.index(mode) if mode in MODE_ORDER else 99,
        ROLE_ORDER.get(_role(rec.get("policy", "")), 9),
        rec.get("policy", ""),
    )


def best_at_level(results: dict[str, dict], level: str) -> dict | None:
    """Best offset-0 record at ``level`` under the canonical protocol first."""
    from robocon_eval.part2.official import get_level
    category = get_level(level)["category"]
    target_duration = _duration_for(category)
    offset0 = [r for r in _offset0_results(results) if r.get("level") == level]
    canonical = [r for r in offset0
                 if abs(float(r.get("duration_s") or 0.0) - target_duration) < 1e-9
                 and r.get("phase") in ("matrix", "modes", "repro")]
    candidates = canonical or offset0
    if not candidates:
        return None
    return min(candidates, key=lambda r: _record_rank(r, target_duration))


def best_by_category(results: dict[str, dict]) -> dict[str, dict]:
    """Highest level reached per category + the primary expert/mode there.

    ``success_configs`` / ``n_*`` describe the TARGET level (the deployment
    need); ``reached_success_configs`` describes the highest level actually
    reached when it differs (e.g. stairs_20cm when stairs_30cm is unsolved).
    """
    out: dict[str, dict] = {}
    for category, target in TARGET_LEVEL.items():
        levels = LEVEL_GROUPS[category]
        reached_level = None
        reached_rec = None
        for level in levels:  # hardest first
            rec = best_at_level(results, level)
            if rec is not None and rec.get("status") == "success":
                reached_level, reached_rec = level, rec
                break

        def _at(level: str) -> list[dict]:
            return [r for r in _offset0_results(results)
                    if r.get("level") == level
                    and r.get("phase") in ("matrix", "modes", "repro")
                    and abs(float(r.get("duration_s") or 0.0)
                            - _duration_for(category)) < 1e-9]

        target_tested = _at(target)
        target_cfg = sorted({(r["policy"], r["mode"]) for r in target_tested
                             if r.get("status") == "success"})
        reached_tested = _at(reached_level) if reached_level else []
        reached_cfg = sorted({(r["policy"], r["mode"]) for r in reached_tested
                              if r.get("status") == "success"})
        fail_reasons: dict[str, int] = {}
        for r in target_tested:
            if r.get("status") == "success":
                continue
            reason = str(r.get("reason") or r.get("status") or "fail")
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        out[category] = {
            "target_level": target,
            "reached_level": reached_level,
            "reached_level_value": int(reached_rec["level_value"]) if reached_rec else 0,
            "policy": reached_rec["policy"] if reached_rec else None,
            "mode": reached_rec["mode"] if reached_rec else None,
            "reached": reached_level == target,
            "n_success_configs": len(target_cfg),
            "n_configs_tested": len({(r["policy"], r["mode"]) for r in target_tested}),
            "success_configs": [{"policy": p, "mode": m} for p, m in target_cfg],
            "reached_success_configs": [{"policy": p, "mode": m} for p, m in reached_cfg],
            "fail_reasons": dict(sorted(fail_reasons.items(), key=lambda kv: -kv[1])),
        }
    return out


def phase_align(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    """Measure the required spawn-offset tolerance for the top candidates.

    Runs the best offset-0 success plus up to 3 runner-up (policy, mode)
    configurations per category at the highest reached level.
    """
    specs = []
    base = best_by_category(results)
    for category, info in base.items():
        level = info["reached_level"]
        if not level:
            continue
        for policy, mode in _align_candidates(results, category, level):
            for dy, dyaw in ALIGN_OFFSETS:
                specs.append(make_spec("align", policy, level, mode, (dy, dyaw),
                                       _duration_for(category), catalog))
    return _dedupe(specs)


def phase_duration(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    specs = []
    best = recommend_by_category(results)
    for category, info in best.items():
        if not info["reached_level"]:
            continue
        level = info["reached_level"]
        primary = _duration_for(category)
        durations = {_old_duration_for(category), 24.0, 32.0}
        for duration in sorted(d for d in durations if abs(d - primary) > 1e-9):
            specs.append(make_spec("duration", info["policy"], level, info["mode"],
                                   (0.0, 0.0), duration, catalog))
    return _dedupe(specs)


def phase_probe(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    """Single-level mode-window probes for the two hardest categories.

    * stairs_30cm (only vt_band_v1 solves it): how wide is the speed window?
    * trench_30cm (only s1d solves it): which speeds stay inside the slot rule?
    """
    specs = []
    stairs_modes = ["adaptive040", "adaptive045", "adaptive055", "adaptive060",
                    "adaptive070", "fixed045", "fixed060", "fixed070"]
    for mode in stairs_modes:
        specs.append(make_spec("probe", "vt_band_v1", "stairs_30cm", mode,
                               (0.0, 0.0), _duration_for("stairs"), catalog))
    trench_modes = ["fixed060", "fixed070", "fixed090", "adaptive060", "adaptive070"]
    for mode in trench_modes:
        specs.append(make_spec("probe", "s1d_trench_reward_v75_m100", "trench_30cm",
                               mode, (0.0, 0.0), _duration_for("trench"), catalog))
    return _dedupe(specs)


def phase_basin(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    """Fine (dy, dyaw) pass map for the stairs_30cm bottleneck + vx widening probes.

    143 adaptive-vx=0.5 configs on a 13x11 grid plus adaptive 0.45/0.55 at four
    representative offsets (does a different vx widen the basin?).
    """
    specs = []
    duration = _duration_for("stairs")
    for mode in BASIN_FULL_GRID_MODES:
        for dy in BASIN_DY:
            for dyaw in BASIN_DYAW:
                specs.append(make_spec("basin", "vt_band_v1", "stairs_30cm", mode,
                                       (dy, dyaw), duration, catalog))
    for mode in BASIN_VX_MODES:
        for dy, dyaw in BASIN_VX_OFFSETS:
            specs.append(make_spec("basin", "vt_band_v1", "stairs_30cm", mode,
                                   (dy, dyaw), duration, catalog))
    for dy in BASIN2_DY:
        for dyaw in BASIN2_DYAW:
            specs.append(make_spec("basin", "vt_band_v1", "stairs_30cm", BASIN2_MODE,
                                   (dy, dyaw), duration, catalog))
    return _dedupe(specs)


def phase_repro(catalog: dict[str, dict], results: dict[str, dict]) -> list[dict]:
    specs = []
    for policy, level, mode, _expected in REPRO_EXPECTED:
        category = catalog[level]["category"]
        specs.append(make_spec("repro", policy, level, mode, (0.0, 0.0),
                               _old_duration_for(category), catalog))
    return _dedupe(specs)


# ---------------------------------------------------------------------------
# Aggregation / report
# ---------------------------------------------------------------------------
def _short_reason(reason: str, limit: int = 42) -> str:
    reason = str(reason or "")
    return reason if len(reason) <= limit else reason[: limit - 1] + "…"


LEVEL_CATEGORY = {lv: cat for cat, lvs in LEVEL_GROUPS.items() for lv in lvs}


def _best_cell(recs: list[dict], category: str) -> str:
    if not recs:
        return "·"
    successes = [r for r in recs if r.get("status") == "success"]
    if successes:
        modes = sorted({r["mode"] for r in successes},
                       key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else 99)
        return "✓ " + ",".join(modes)
    best = min(recs, key=lambda r: _record_rank(r, _duration_for(category)))
    return "✗ " + _short_reason(best.get("reason"))


def _policy_strength(results: dict[str, dict]) -> dict[str, int]:
    """Number of distinct official levels a policy has a canonical success on."""
    levels: dict[str, set] = {}
    for r in _offset0_results(results):
        if r.get("status") != "success":
            continue
        if abs(float(r.get("duration_s") or 0.0)
               - _duration_for(r.get("category", ""))) > 1e-9:
            continue
        levels.setdefault(r.get("policy", ""), set()).add(r.get("level"))
    return {p: len(v) for p, v in levels.items()}


def _align_candidates(results: dict[str, dict], category: str, level: str,
                      k: int = 6) -> list[tuple[str, str]]:
    """Top offset-0 successes at ``level`` (unique policy+mode), canonical first."""
    duration = _duration_for(category)
    uniq: dict[tuple[str, str], dict] = {}
    for r in _offset0_results(results):
        if r.get("level") != level or r.get("status") != "success":
            continue
        if abs(float(r.get("duration_s") or 0.0) - duration) > 1e-9:
            continue
        if r.get("phase") not in ("matrix", "modes", "repro"):
            continue
        uniq.setdefault((r["policy"], r["mode"]), r)
    strength = _policy_strength(results)

    def rank(r: dict) -> tuple:
        mode = r["mode"]
        return (MODE_ORDER.index(mode) if mode in MODE_ORDER else 99,
                -strength.get(r["policy"], 0),
                ROLE_ORDER.get(_role(r["policy"]), 9), r["policy"])

    ranked = sorted(uniq.values(), key=rank)
    return [(r["policy"], r["mode"]) for r in ranked[:k]]


_PHASE_PREFERENCE = {"matrix": 0, "modes": 1, "align": 2, "repro": 3, "duration": 4}


def _alignment_rows(results: dict[str, dict], level: str, policy: str, mode: str,
                    duration: float) -> list[dict]:
    rows = []
    for dy, dyaw in [(0.0, 0.0)] + ALIGN_OFFSETS:
        matches = [r for r in results.values()
                   if r.get("level") == level and r.get("policy") == policy
                   and r.get("mode") == mode
                   and tuple(r.get("offset") or (0.0, 0.0)) == (dy, dyaw)
                   and abs(float(r.get("duration_s") or 0.0) - duration) < 1e-9]
        if not matches:
            continue
        rec = min(matches, key=lambda r: _PHASE_PREFERENCE.get(r.get("phase"), 9))
        rows.append({
            "dy": dy, "dyaw_deg": dyaw,
            "status": rec.get("status"), "reason": rec.get("reason") or "",
            "score": rec.get("score"), "sim_time_s": rec.get("sim_time_s"),
        })
    return rows


def _candidate_alignment(results: dict[str, dict], category: str, level: str,
                         policy: str, mode: str) -> dict:
    rows = _alignment_rows(results, level, policy, mode, _duration_for(category))
    n_pass = sum(1 for r in rows if r["status"] == "success")
    base = next((r for r in rows if (r["dy"], r["dyaw_deg"]) == (0.0, 0.0)), None)
    return {
        "policy": policy, "mode": mode, "reached_level": level,
        "offset0_status": base["status"] if base else None,
        "n_pass": n_pass, "n_tested": len(rows),
        "pass_offsets": [{"dy": r["dy"], "dyaw_deg": r["dyaw_deg"]}
                         for r in rows
                         if r["status"] == "success"
                         and (r["dy"], r["dyaw_deg"]) != (0.0, 0.0)],
        "offsets": rows,
    }


def recommend_by_category(results: dict[str, dict]) -> dict[str, dict]:
    """Primary expert per category, upgraded to the most alignment-robust one.

    Before the ``align`` phase has run this is simply the best offset-0 success.
    Afterwards the candidate set (primary winner + runner-ups, all measured at
    the required spawn offsets) is re-ranked by spawn-offset tolerance, so the
    recommended expert is the most deployment-robust measured choice.
    """
    best = best_by_category(results)
    out: dict[str, dict] = {}
    for category, info in best.items():
        info = dict(info)
        level = info["reached_level"]
        candidates = _align_candidates(results, category, level) if level else []
        alignments = [_candidate_alignment(results, category, level, policy, mode)
                      for policy, mode in candidates]
        measured = [a for a in alignments
                    if a["n_tested"] > 0 and a["offset0_status"] == "success"]
        strength = _policy_strength(results)
        if measured:
            preferred_mode = RECOMMENDED_MODE_OVERRIDE.get(category)
            pool = measured
            if preferred_mode:
                preferred = [a for a in measured if a["mode"] == preferred_mode]
                if preferred:
                    pool = preferred
            chosen = min(pool, key=lambda a: (
                -a["n_pass"],
                MODE_ORDER.index(a["mode"]) if a["mode"] in MODE_ORDER else 99,
                -strength.get(a["policy"], 0),
                ROLE_ORDER.get(_role(a["policy"]), 9), a["policy"]))
            info["policy"], info["mode"] = chosen["policy"], chosen["mode"]
            info["individual_alignment"] = chosen
            info["alternatives"] = [a for a in alignments
                                    if (a["policy"], a["mode"])
                                    != (chosen["policy"], chosen["mode"])]
            basis = ("most alignment-robust measured candidate: "
                     f"{chosen['n_pass']}/{chosen['n_tested']} spawn offsets pass")
            if preferred_mode and chosen["mode"] == preferred_mode:
                basis = (f"route-preferred mode `{preferred_mode}` "
                         f"({chosen['n_pass']}/{chosen['n_tested']} spawn offsets pass); "
                         "chosen for the symmetric yaw tolerance")
            info["selection_basis"] = basis
        else:
            info["individual_alignment"] = None
            info["alternatives"] = alignments
            info["selection_basis"] = "best offset-0 success (alignment not measured yet)"
        out[category] = info
    return out


def _alignment_view(best: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for category, info in best.items():
        chosen = info.get("individual_alignment")
        if not chosen:
            out[category] = {"reached_level": info["reached_level"],
                             "policy": info["policy"], "mode": info["mode"],
                             "base": None, "n_pass": 0, "n_tested": 0, "offsets": []}
            continue
        out[category] = {
            "reached_level": chosen["reached_level"],
            "policy": chosen["policy"], "mode": chosen["mode"],
            "base": chosen["offset0_status"],
            "n_pass": chosen["n_pass"], "n_tested": chosen["n_tested"],
            "offsets": chosen["offsets"],
        }
    return out


def _special_comparisons(results: dict[str, dict]) -> dict:
    offset0 = _offset0_results(results)

    def cell(policy: str, level: str, mode: str | None = None,
             duration: float | None = None) -> dict | None:
        recs = [r for r in offset0 if r.get("policy") == policy and r.get("level") == level
                and (mode is None or r.get("mode") == mode)
                and (duration is None or abs(float(r.get("duration_s") or 0.0) - duration) < 1e-9)]
        if not recs:
            return None
        succ = [r for r in recs if r.get("status") == "success"]
        pick = min(succ, key=lambda r: _record_rank(r, _duration_for(r["category"]))) if succ \
            else min(recs, key=lambda r: _record_rank(r, _duration_for(r["category"])))
        return {"policy": policy, "level": level, "mode": pick["mode"],
                "status": pick.get("status"), "reason": pick.get("reason") or "",
                "duration_s": pick.get("duration_s")}

    stairs_policies = _dedupe_names(STAIRS_SPECIALISTS + ["vt_band_v1", "row2p_p40w40_v1",
                                                          "vt_mode_v2", "vtg09_1100"])
    stairs = {}
    for policy in stairs_policies:
        stairs[policy] = {level: cell(policy, level) for level in LEVEL_GROUPS["stairs"]}

    generalists = _dedupe_names(["row2p_p40w40_v1", "vt_mode_v2", "vt_freezehid_1349",
                                 "vt_mixture_v1", "vt_band_v1", "vtg09_1100",
                                 "st30_ppo_st30_s100_600"])
    gen_levels = ["high_platform_40cm", "high_wall_40cm", "slope_35deg", "slope_45deg"]
    generalist = {policy: {level: cell(policy, level) for level in gen_levels}
                  for policy in generalists}

    trench = {}
    for policy in _dedupe_names(TRENCH_REFERENCE):
        trench[policy] = {}
        for level in LEVEL_GROUPS["trench"]:
            trench[policy][level] = {
                mode: cell(policy, level, mode=mode,
                           duration=_duration_for("trench"))
                for mode in MODE_SWEEP_MODES["trench"]
            }
    return {"stairs_specialists": stairs, "generalists_40cm": generalist,
            "trench_modes": trench}


def _duration_sensitivity(results: dict[str, dict], best: dict[str, dict]) -> dict:
    out: dict[str, dict] = {}
    for category, info in best.items():
        if not info["reached_level"]:
            out[category] = {"reached_level": None, "runs": []}
            continue
        level = info["reached_level"]
        runs: list[dict] = []
        seen: set[float] = set()
        for r in results.values():
            if (r.get("level") != level or r.get("policy") != info["policy"]
                    or r.get("mode") != info["mode"]
                    or tuple(r.get("offset") or (0.0, 0.0)) != (0.0, 0.0)):
                continue
            duration = float(r.get("duration_s") or 0.0)
            if duration in seen:
                continue
            seen.add(duration)
            runs.append({"duration_s": duration, "status": r.get("status"),
                         "reason": r.get("reason") or "", "phase": r.get("phase")})
        runs.sort(key=lambda r: r["duration_s"])
        out[category] = {"reached_level": level, "policy": info["policy"],
                         "mode": info["mode"], "runs": runs}
    return out


def _reproducibility(results: dict[str, dict]) -> dict:
    rows = []
    n_match = 0
    for policy, level, mode, expected in REPRO_EXPECTED:
        old_duration = _old_duration_for(_level_catalog()[level]["category"])
        matches = [r for r in results.values()
                   if r.get("policy") == policy and r.get("level") == level
                   and r.get("mode") == mode
                   and tuple(r.get("offset") or (0.0, 0.0)) == (0.0, 0.0)
                   and abs(float(r.get("duration_s") or 0.0) - old_duration) < 1e-9]
        matches.sort(key=lambda r: _PHASE_PREFERENCE.get(r.get("phase"), 9))
        observed = matches[0].get("status") if matches else "missing"
        match = observed == expected
        n_match += int(match)
        rows.append({
            "policy": policy, "level": level, "mode": mode,
            "duration_s": matches[0].get("duration_s") if matches else None,
            "expected": expected, "observed": observed,
            "reason": (matches[0].get("reason") or "") if matches else "",
            "match": match,
        })
    return {"n_match": n_match, "n": len(rows), "all_match": n_match == len(rows),
            "rows": rows}


def _failure_reasons(results: dict[str, dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in _offset0_results(results):
        if r.get("status") == "success":
            continue
        reason = str(r.get("reason") or r.get("status") or "fail")
        per = out.setdefault(r["level"], {})
        per[reason] = per.get(reason, 0) + 1
    return {level: dict(sorted(per.items(), key=lambda kv: -kv[1]))
            for level, per in sorted(out.items())}


def _repro_block(info: dict) -> dict:
    """Exact command lines to reproduce the chosen expert through the eval runner."""
    if not info.get("policy"):
        return {}
    policy, level, mode = info["policy"], info["reached_level"], info["mode"]
    from robocon_eval.part2.official import get_level
    duration = _duration_for(get_level(level)["category"])
    m = MODES[mode]
    if m["kind"] == "adaptive":
        fn = f"AdaptiveCommand(vx={m['vx']}, **{json.dumps(ADAPTIVE_GAINS, sort_keys=True)})"
        command = "None"
    else:
        fn = "None"
        command = repr(tuple(m["command"]))
    script = (
        "from pathlib import Path\n"
        "from robocon_eval.policy_runner import run, AdaptiveCommand\n"
        f"result = run('blackW', 2, '{policy}', '{level}', Path('/tmp/{policy}_{level}.jsonl'),\n"
        f"            duration_s={duration}, command={command}, profile='official',\n"
        f"            command_fn={fn}, spawn_offset=None)\n"
        "print(result['status'], result['score'], result['reason'])\n"
    )
    cmd = (f"pixi run -e eval python tools/campaign/run_expert_matrix.py "
           f"--only {policy}:{level}:{mode} --force")
    return {
        "policy_dir": f"demo/rl_sar/policy/blackW/{policy}",
        "campaign_cli": cmd,
        "python_snippet": script,
    }


def _canonical_rows(results: dict[str, dict]) -> list[dict]:
    """Offset-0 records at each category's primary duration (any canonical phase).

    Cross-phase dedupe may store a measurement under another phase tag, so the
    per-level matrix view unions matrix/modes/repro at the protocol duration.
    """
    out = []
    for r in results.values():
        if tuple(r.get("offset") or (0.0, 0.0)) != (0.0, 0.0):
            continue
        if r.get("phase") not in ("matrix", "modes", "repro"):
            continue
        if abs(float(r.get("duration_s") or 0.0)
               - _duration_for(r.get("category", ""))) > 1e-9:
            continue
        out.append(r)
    return out


def _probe_view(results: dict[str, dict]) -> dict[str, dict]:
    """Mode-window probes grouped by level."""
    out: dict[str, dict] = {}
    for r in results.values():
        if r.get("phase") != "probe":
            continue
        entry = out.setdefault(r["level"], {"policy": r["policy"], "rows": []})
        mode_spec = MODES.get(r["mode"], {})
        entry["rows"].append({
            "mode": r["mode"],
            "kind": mode_spec.get("kind"),
            "vx": mode_spec.get("vx"),
            "command": list(mode_spec["command"]) if "command" in mode_spec else None,
            "status": r.get("status"), "reason": r.get("reason") or "",
            "score": r.get("score"), "duration_s": r.get("duration_s"),
        })
    for entry in out.values():
        entry["rows"].sort(key=lambda row: (row["kind"] or "", row["vx"] or 0.0))
    return out


def _stairs_basin(results: dict[str, dict]) -> dict | None:
    """Fine (dy, dyaw) pass maps for the stairs_30cm bottleneck.

    The primary map is vt_band_v1 / adaptive vx=0.5 (the only matrix solver);
    adaptive vx=0.55 and 0.70 grids test whether a different speed widens the
    entry basin.  Records may live under another phase tag because of cross-phase
    dedupe (a deterministic config is measured once), so all phases are searched.
    """
    duration = _duration_for("stairs")
    grids: dict[str, dict] = {}
    mode_order = ["basin", "align", "matrix", "modes", "probe", "repro", "duration"]
    for mode in BASIN_FULL_GRID_MODES:
        cells: dict[tuple[float, float], dict] = {}
        for r in results.values():
            if r.get("policy") != "vt_band_v1" or r.get("level") != "stairs_30cm" \
                    or r.get("mode") != mode:
                continue
            if abs(float(r.get("duration_s") or 0.0) - duration) > 1e-9:
                continue
            offset = r.get("offset") or [0.0, 0.0]
            key = (round(float(offset[0]), 3), round(float(offset[1]), 3))
            if key[0] not in BASIN_DY or key[1] not in BASIN_DYAW:
                continue
            prev = cells.get(key)
            if prev is None or mode_order.index(r.get("phase")) < mode_order.index(
                    prev.get("phase")):
                cells[key] = r
        if not cells:
            continue
        rows = [[cells.get((dy, dyaw), {"status": "missing", "reason": ""})
                 for dyaw in BASIN_DYAW] for dy in BASIN_DY]
        iy0 = BASIN_DY.index(0.0)
        jyaw0 = BASIN_DYAW.index(0.0)
        n_tested = sum(1 for row in rows for c in row if c["status"] != "missing")
        n_pass = sum(1 for row in rows for c in row if c["status"] == "success")

        # Largest all-pass rectangle containing (dy=0, dyaw=0).
        best = None
        ny, nja = len(BASIN_DY), len(BASIN_DYAW)
        for i0 in range(iy0, -1, -1):
            for i1 in range(iy0, ny):
                for j0 in range(jyaw0, -1, -1):
                    for j1 in range(jyaw0, nja):
                        ok = True
                        for i in range(i0, i1 + 1):
                            for j in range(j0, j1 + 1):
                                if rows[i][j]["status"] != "success":
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        area = (i1 - i0 + 1) * (j1 - j0 + 1)
                        cand = {"dy_min": BASIN_DY[i0], "dy_max": BASIN_DY[i1],
                                "dyaw_min": BASIN_DYAW[j0], "dyaw_max": BASIN_DYAW[j1],
                                "area_cells": area,
                                "n_dy": i1 - i0 + 1, "n_dyaw": j1 - j0 + 1}
                        if best is None or (area, cand["n_dy"], cand["n_dyaw"]) > (
                                best["area_cells"], best["n_dy"], best["n_dyaw"]):
                            best = cand
        dy_line = [i for i in range(ny)
                   if rows[i][jyaw0]["status"] == "success"]
        dyaw_line = [j for j in range(nja)
                     if rows[iy0][j]["status"] == "success"]

        def _contiguous(line: list[dict], idx0: int) -> tuple[int, int]:
            neg = 0
            i = idx0 - 1
            while i >= 0 and line[i]["status"] == "success":
                neg += 1
                i -= 1
            pos = 0
            i = idx0 + 1
            while i < len(line) and line[i]["status"] == "success":
                pos += 1
                i += 1
            return neg, pos

        dy_neg, dy_pos = _contiguous([rows[i][jyaw0] for i in range(ny)], iy0)
        dyaw_neg, dyaw_pos = _contiguous([rows[iy0][j] for j in range(nja)], jyaw0)

        def _rate(cell_list):
            cell_list = [c for c in cell_list if c["status"] != "missing"]
            return (sum(1 for c in cell_list if c["status"] == "success")
                    / len(cell_list)) if cell_list else 0.0

        pos_dy = [rows[i][j] for i in range(iy0 + 1, ny) for j in range(nja)]
        neg_dy = [rows[i][j] for i in range(0, iy0) for j in range(nja)]
        pos_yaw = [rows[i][j] for i in range(ny) for j in range(jyaw0 + 1, nja)]
        neg_yaw = [rows[i][j] for i in range(ny) for j in range(0, jyaw0)]
        reasons: dict[str, int] = {}
        for row in rows:
            for c in row:
                if c["status"] not in ("success", "missing"):
                    reasons[c["reason"] or "fail"] = reasons.get(c["reason"] or "fail", 0) + 1

        def _cell(dy, dyaw):
            c = cells.get((round(dy, 3), round(dyaw, 3)))
            return {"dy": dy, "dyaw_deg": dyaw,
                    "status": c["status"] if c else "missing",
                    "reason": (c.get("reason") or "") if c else ""}

        grids[mode] = {
            "mode": mode,
            "vx": MODES[mode].get("vx"),
            "n_tested": n_tested, "n_pass": n_pass,
            "pass_rate": round(n_pass / n_tested, 4) if n_tested else 0.0,
            "grid": [[_cell(dy, dyaw) for dyaw in BASIN_DYAW] for dy in BASIN_DY],
            "largest_all_pass_rectangle_around_origin": best,
            "origin_axis_pass": {
                "max_abs_dy_on_dyaw0": max((abs(BASIN_DY[i]) for i in dy_line), default=0.0),
                "max_abs_dyaw_deg_on_dy0": max((abs(BASIN_DYAW[j]) for j in dyaw_line),
                                               default=0.0),
                "contiguous_dy_min": BASIN_DY[iy0 - dy_neg],
                "contiguous_dy_max": BASIN_DY[iy0 + dy_pos],
                "contiguous_dyaw_min": BASIN_DYAW[jyaw0 - dyaw_neg],
                "contiguous_dyaw_max": BASIN_DYAW[jyaw0 + dyaw_pos],
            },
            "best_row_dy": max(BASIN_DY, key=lambda dy: sum(
                1 for cell in rows[BASIN_DY.index(dy)] if cell["status"] == "success")),
            "best_col_dyaw_deg": max(BASIN_DYAW, key=lambda yaw: sum(
                1 for i in range(ny) if rows[i][BASIN_DYAW.index(yaw)]["status"] == "success")),
            "pattern": {
                "pass_rate_dy_positive": round(_rate(pos_dy), 4),
                "pass_rate_dy_negative": round(_rate(neg_dy), 4),
                "pass_rate_dyaw_positive": round(_rate(pos_yaw), 4),
                "pass_rate_dyaw_negative": round(_rate(neg_yaw), 4),
            },
            "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }
    if not grids:
        return None

    # vx offset probes (do 0.45 / 0.55 widen the basin at representative offsets?).
    vx_probe: dict[tuple, dict] = {}
    for mode in BASIN_VX_MODES:
        for dy, dyaw in BASIN_VX_OFFSETS:
            key = (mode, round(dy, 3), round(dyaw, 3))
            for r in results.values():
                if (r.get("policy") != "vt_band_v1"
                        or r.get("level") != "stairs_30cm"
                        or r.get("mode") != mode):
                    continue
                if abs(float(r.get("duration_s") or 0.0) - duration) > 1e-9:
                    continue
                offset = r.get("offset") or [0.0, 0.0]
                if (round(float(offset[0]), 3), round(float(offset[1]), 3)) \
                        != (round(dy, 3), round(dyaw, 3)):
                    continue
                prev = vx_probe.get(key)
                if prev is None or mode_order.index(r.get("phase")) < mode_order.index(
                        prev.get("phase")):
                    vx_probe[key] = r
    probe_rows = []
    for (mode, dy, dyaw), r in sorted(vx_probe.items(),
                                      key=lambda kv: (MODES[kv[0][0]]["vx"],
                                                      kv[0][1], kv[0][2])):
        probe_rows.append({"mode": mode, "vx": MODES[mode].get("vx"),
                           "dy": dy, "dyaw_deg": dyaw,
                           "status": r.get("status"),
                           "reason": r.get("reason") or ""})

    primary = grids.get("adaptive05") or next(iter(grids.values()))
    out = {
        "protocol": {"policy": "vt_band_v1", "level": "stairs_30cm",
                     "primary_mode": "adaptive05",
                     "grid_modes": list(grids.keys()),
                     "duration_s": duration,
                     "dy_grid": BASIN_DY, "dyaw_deg_grid": BASIN_DYAW},
        "primary_mode": primary["mode"],
        "grids": grids,
        "vx_probe": probe_rows,
        # Convenience mirror of the primary (adaptive vx=0.5) grid.
        "n_tested": primary["n_tested"], "n_pass": primary["n_pass"],
        "pass_rate": primary["pass_rate"], "grid": primary["grid"],
        "largest_all_pass_rectangle_around_origin":
            primary["largest_all_pass_rectangle_around_origin"],
        "origin_axis_pass": primary["origin_axis_pass"],
        "pattern": primary["pattern"],
        "failure_reasons": primary["failure_reasons"],
    }
    return out


def _stairs_fine_neighborhood(results: dict[str, dict]) -> dict | None:
    """Fine 9x9 (dy, dyaw) neighborhood at vx=0.70 around dy=-0.04 m."""
    duration = _duration_for("stairs")
    phase_order = ["basin", "align", "matrix", "modes", "probe", "repro", "duration"]
    cells: dict[tuple[float, float], dict] = {}
    for r in results.values():
        if (r.get("policy") != "vt_band_v1" or r.get("level") != "stairs_30cm"
                or r.get("mode") != BASIN2_MODE):
            continue
        if abs(float(r.get("duration_s") or 0.0) - duration) > 1e-9:
            continue
        offset = r.get("offset") or [0.0, 0.0]
        key = (round(float(offset[0]), 3), round(float(offset[1]), 1))
        if key[0] not in BASIN2_DY or key[1] not in BASIN2_DYAW:
            continue
        prev = cells.get(key)
        if prev is None or phase_order.index(r.get("phase")) < phase_order.index(
                prev.get("phase")):
            cells[key] = r
    if not cells:
        return None
    rows = [[cells.get((dy, dyaw), {"status": "missing", "reason": ""})
             for dyaw in BASIN2_DYAW] for dy in BASIN2_DY]
    n_tested = sum(1 for row in rows for c in row if c["status"] != "missing")
    n_pass = sum(1 for row in rows for c in row if c["status"] == "success")
    cy, cz = BASIN2_CENTER
    iy0 = BASIN2_DY.index(cy)
    jyaw0 = BASIN2_DYAW.index(cz)
    # All-pass rectangle around the probe centre.
    best = None
    ny, nja = len(BASIN2_DY), len(BASIN2_DYAW)
    for i0 in range(iy0, -1, -1):
        for i1 in range(iy0, ny):
            for j0 in range(jyaw0, -1, -1):
                for j1 in range(jyaw0, nja):
                    ok = all(rows[i][j]["status"] == "success"
                             for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))
                    if not ok:
                        continue
                    area = (i1 - i0 + 1) * (j1 - j0 + 1)
                    cand = {"dy_min": BASIN2_DY[i0], "dy_max": BASIN2_DY[i1],
                            "dyaw_min": BASIN2_DYAW[j0], "dyaw_max": BASIN2_DYAW[j1],
                            "area_cells": area, "n_dy": i1 - i0 + 1,
                            "n_dyaw": j1 - j0 + 1}
                    if best is None or (area, cand["n_dy"], cand["n_dyaw"]) > (
                            best["area_cells"], best["n_dy"], best["n_dyaw"]):
                        best = cand

    def _cell(dy, dyaw):
        c = cells.get((round(dy, 3), round(dyaw, 1)))
        return {"dy": dy, "dyaw_deg": dyaw,
                "status": c["status"] if c else "missing",
                "reason": (c.get("reason") or "") if c else ""}

    row_pass = {dy: sum(1 for c in rows[i] if c["status"] == "success")
                for i, dy in enumerate(BASIN2_DY)}
    return {
        "mode": BASIN2_MODE, "vx": MODES[BASIN2_MODE].get("vx"),
        "center": {"dy": cy, "dyaw_deg": cz},
        "dy_grid": BASIN2_DY, "dyaw_deg_grid": BASIN2_DYAW,
        "n_tested": n_tested, "n_pass": n_pass,
        "pass_rate": round(n_pass / n_tested, 4) if n_tested else 0.0,
        "grid": [[_cell(dy, dyaw) for dyaw in BASIN2_DYAW] for dy in BASIN2_DY],
        "largest_all_pass_rectangle_around_center": best,
        "row_pass_counts": {f"{dy:+.3f}": cnt for dy, cnt in row_pass.items()},
        "best_row_dy": max(row_pass, key=row_pass.get),
        "best_row_pass": max(row_pass.values(), default=0),
    }


def _route_usage(best: dict[str, dict], alignment: dict[str, dict],
                 basin: dict | None) -> list[dict]:
    """Compact deployment table for the full-field route."""
    rows: list[dict] = []
    for category, info in best.items():
        level = info.get("reached_level")
        if not level:
            rows.append({"category": category, "level": None, "policy": None,
                         "mode": None, "baseline": None, "alignment_gate": "no success"})
            continue
        mode = MODES.get(info["mode"], {})
        if mode.get("kind") == "adaptive":
            vx = float(mode["vx"])
            command = {"kind": "adaptive", "vx": vx, "gains": ADAPTIVE_GAINS}
        else:
            vx = float(mode["command"][0])
            command = {"kind": "fixed", "command": list(mode["command"])}
        al = alignment.get(category, {})
        pass_offsets = [{"dy": o["dy"], "dyaw_deg": o["dyaw_deg"]}
                        for o in al.get("offsets", [])
                        if o["status"] == "success"
                        and (o["dy"], o["dyaw_deg"]) != (0.0, 0.0)]
        if category == "stairs" and basin and basin.get(
                "largest_all_pass_rectangle_around_origin"):
            rect = basin["largest_all_pass_rectangle_around_origin"]
            gate = (f"stairs basin all-pass rectangle (vx=0.5): dy in "
                    f"[{rect['dy_min']:+.2f}, {rect['dy_max']:+.2f}] m, dyaw in "
                    f"[{rect['dyaw_min']:+.0f}, {rect['dyaw_max']:+.0f}] deg "
                    f"({rect['area_cells']} cells of the measured 13x11 grid)")
            other = [(m, g) for m, g in (basin.get("grids") or {}).items()
                     if m != basin.get("primary_mode")
                     and g.get("largest_all_pass_rectangle_around_origin")]
            if other:
                best_alt = max(other, key=lambda kv: kv[1][
                    "largest_all_pass_rectangle_around_origin"]["area_cells"])
                alt_rect = best_alt[1]["largest_all_pass_rectangle_around_origin"]
                if (alt_rect["area_cells"], alt_rect["n_dy"], alt_rect["n_dyaw"]) > (
                        rect["area_cells"], rect["n_dy"], rect["n_dyaw"]):
                    gate += (f"; WIDER basin at vx={best_alt[1]['vx']:g}: dy in "
                             f"[{alt_rect['dy_min']:+.2f}, {alt_rect['dy_max']:+.2f}] m, "
                             f"dyaw in [{alt_rect['dyaw_min']:+.0f}, "
                             f"{alt_rect['dyaw_max']:+.0f}] deg "
                             f"({alt_rect['area_cells']} cells)")
            fine = basin.get("fine_neighborhood_vx070")
            if fine and fine.get("largest_all_pass_rectangle_around_center"):
                fr = fine["largest_all_pass_rectangle_around_center"]
                gate += (f"; best contiguous pocket: vx={fine['vx']:g} around "
                         f"(dy={fine['center']['dy']:+.2f}, dyaw=0) -> dy in "
                         f"[{fr['dy_min']:+.3f}, {fr['dy_max']:+.3f}] m, dyaw in "
                         f"[{fr['dyaw_min']:+.1f}, {fr['dyaw_max']:+.1f}] deg "
                         f"({fr['area_cells']} all-pass cells)")
        else:
            dys = sorted({round(o["dy"], 3) for o in pass_offsets})
            yaws = sorted({round(o["dyaw_deg"], 1) for o in pass_offsets})
            gate = (f"measured pass offsets: dy {dys}, dyaw {yaws} deg "
                    "(non-monotone: do not interpolate)")
        rows.append({
            "category": category,
            "level": level,
            "policy": info["policy"],
            "policy_dir": f"demo/rl_sar/policy/blackW/{info['policy']}",
            "mode": info["mode"],
            "command_kind": mode.get("kind"),
            "vx": vx,
            "command": command,
            "baseline": al.get("base"),
            "alignment_tolerance": f"{al.get('n_pass', 0)}/{al.get('n_tested', 0)}",
            "alignment_gate": gate,
            "pass_offsets": pass_offsets,
            "alternatives": [
                {"policy": a["policy"], "mode": a["mode"],
                 "tolerance": f"{a['n_pass']}/{a['n_tested']}"}
                for a in (info.get("alternatives") or [])
            ],
        })
    return rows


def build_summary(results: dict[str, dict], catalog: dict[str, dict],
                  phase_wall: dict[str, float], campaign_wall: float) -> dict:
    best = recommend_by_category(results)
    for info in best.values():
        info["repro"] = _repro_block(info)
    alignment = _alignment_view(best)
    special = _special_comparisons(results)
    duration = _duration_sensitivity(results, best)
    repro = _reproducibility(results)
    probe_view = _probe_view(results)
    basin = _stairs_basin(results)
    fine = _stairs_fine_neighborhood(results)
    if basin is not None and fine is not None:
        basin["fine_neighborhood_vx070"] = fine
    by_level = {}
    for category, levels in LEVEL_GROUPS.items():
        for level in levels:
            rec = best_at_level(results, level)
            succ = [r for r in _offset0_results(results)
                    if r.get("level") == level and r.get("status") == "success"
                    and r.get("phase") in ("matrix", "modes", "repro")
                    and abs(float(r.get("duration_s") or 0.0)
                            - _duration_for(category)) < 1e-9]
            by_level[level] = {
                "category": category,
                "level": int(get_level_value(catalog, level)),
                "best": ({k: rec.get(k) for k in ("policy", "mode", "status", "reason",
                                                  "score", "duration_s")}
                         if rec else None),
                "success_configs": [{"policy": r["policy"], "mode": r["mode"]}
                                    for r in sorted(succ, key=lambda r: _record_rank(
                                        r, _duration_for(category)))],
                "n_success_configs": len({(r["policy"], r["mode"]) for r in succ}),
            }
    # The category recommendation wins at its reached level.
    for category, info in best.items():
        level = info.get("reached_level")
        if not level:
            continue
        matches = [r for r in _offset0_results(results)
                   if r.get("level") == level and r.get("policy") == info["policy"]
                   and r.get("mode") == info["mode"]
                   and abs(float(r.get("duration_s") or 0.0)
                           - _duration_for(category)) < 1e-9]
        if matches:
            rec = min(matches, key=lambda r: _PHASE_PREFERENCE.get(r.get("phase"), 9))
            by_level[level]["best"] = {k: rec.get(k) for k in
                                       ("policy", "mode", "status", "reason", "score",
                                        "duration_s")}
            by_level[level]["recommended"] = True
    policies: dict[str, dict] = {}
    for rec in results.values():
        name = rec.get("policy")
        if name and name not in policies:
            meta = _policy_meta(name)
            meta["role"] = _role(name)
            policies[name] = meta
    modes_desc = {}
    for name, spec in MODES.items():
        if spec["kind"] == "adaptive":
            modes_desc[name] = {"kind": "adaptive", "vx": spec["vx"],
                                "gains": ADAPTIVE_GAINS}
        else:
            modes_desc[name] = {"kind": "fixed", "command": list(spec["command"])}
    summary = {
        "schema_version": 1,
        "campaign": "part2_expert_matrix",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign_command": ("pixi run -e eval python tools/campaign/run_expert_matrix.py "
                             "--jobs 6   # phases: " + ",".join(PHASE_ORDER)),
        "last_invocation_command": " ".join([sys.executable] + sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "mujoco": _pkg_version("mujoco"),
            "torch": _pkg_version("torch"),
            "numpy": _pkg_version("numpy"),
            "judge_version": _judge_version(),
            "git": _git_rev(),
        },
        "protocol": {
            "profile": "official", "settle_s": 1.0, "warmup_s": 0.0,
            "stop_on_fall": True, "seed": 0,
            "sim_deterministic": True,
            "primary_duration_s": {**{"default": DEFAULT_DURATION},
                                   **PRIMARY_DURATION},
            "old_benchmark_duration_s": OLD_BENCH_DURATION,
            "duration_phase_s": [24.0, 32.0],
            "spawn_offsets": [{"dy": dy, "dyaw_deg": dyaw}
                              for dy, dyaw in ALIGN_OFFSETS],
            "command_modes": modes_desc,
            "note": "sim is deterministic: varying seed alone does nothing; "
                    "robustness is measured with spawn_offset and command modes",
        },
        "levels": {k: {"category": v["category"], "level": v["level"],
                       "seconds": v.get("seconds"), "judge_kind": v["judge"]["kind"]}
                   for k, v in catalog.items()},
        "policies": policies,
        "totals": {
            "runs": len(results),
            "errors": sum(1 for r in results.values() if r.get("status") == "error"),
            "successes": sum(1 for r in results.values() if r.get("status") == "success"),
            "runner_wall_time_s": round(sum(float(r.get("wall_time_s") or 0.0)
                                            for r in results.values()), 2),
            "session_wall_time_s": round(campaign_wall, 2),
            "phase_wall_time_s": {k: round(v, 2) for k, v in phase_wall.items()},
        },
        "best_by_level": by_level,
        "best_by_category": best,
        "alignment": alignment,
        "duration_sensitivity": duration,
        "reproducibility": repro,
        "stairs_specialists": special["stairs_specialists"],
        "generalist_comparison": special["generalists_40cm"],
        "trench_modes": special["trench_modes"],
        "failure_reasons": _failure_reasons(results),
        "runs": list(results.values()),
        "matrix": _canonical_rows(results),
        "probe": probe_view,
        "stairs30_basin": basin,
        "route_usage": _route_usage(best, alignment, basin),
        "notes": _notes(best, alignment, repro, special, duration, probe_view, basin),
    }
    return summary


def get_level_value(catalog: dict[str, dict], level: str) -> int:
    return int(catalog[level]["level"])


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


def _notes(best: dict, alignment: dict, repro: dict, special: dict,
           duration: dict, probe: dict | None = None,
           basin: dict | None = None) -> list[str]:
    notes: list[str] = []
    for category, info in best.items():
        target = info["target_level"]
        align = alignment.get(category, {})
        basis = info.get("selection_basis", "")
        passing = f"{info['n_success_configs']}/{info['n_configs_tested']} configs pass at {target}"
        if info["reached"]:
            notes.append(
                f"{category}: {target} reached — recommended {info['policy']} "
                f"[{info['mode']}]; alignment "
                f"{align.get('n_pass', 0)}/{align.get('n_tested', 0)} required offsets pass; "
                f"{passing} ({basis}).")
        elif info["reached_level"]:
            notes.append(
                f"{category}: {target} NOT reached; best is {info['reached_level']} "
                f"({info['policy']} [{info['mode']}], alignment "
                f"{align.get('n_pass', 0)}/{align.get('n_tested', 0)}); {passing}.")
        else:
            notes.append(f"{category}: no successful config measured.")
    stairs = special["stairs_specialists"]
    vt_band_ok = (stairs.get("vt_band_v1", {}) or {}).get("stairs_30cm")
    specialist_ok = [p for p, cells in stairs.items()
                     if p not in ("vt_band_v1", "row2p_p40w40_v1", "vt_mode_v2", "vtg09_1100")
                     and (cells.get("stairs_30cm") or {}).get("status") == "success"]
    if vt_band_ok and vt_band_ok.get("status") == "success" and not specialist_ok:
        notes.append("stairs_30cm: only vt_band_v1 passes; none of the "
                     f"{len(STAIRS_SPECIALISTS)} stairs specialists beats it in the measured modes.")
    elif specialist_ok:
        notes.append(f"stairs_30cm: stairs specialist(s) passing: {specialist_ok}.")
    trench = special["trench_modes"]
    s1d = trench.get("s1d_trench_reward_v75_m100", {})
    trench30 = s1d.get("trench_30cm", {})
    trench30_ok = [m for m, c in trench30.items() if c and c.get("status") == "success"]
    if trench30_ok:
        notes.append(f"trench_30cm: s1d_trench_reward_v75_m100 passes with modes "
                     f"{trench30_ok}; adaptive/fixed 0.5 does not.")
    for level, info in (probe or {}).items():
        passing = [r["mode"] for r in info["rows"] if r["status"] == "success"]
        notes.append(f"{level} mode window: {info['policy']} passes "
                     f"{passing if passing else 'no additional speed probe mode'}; "
                     f"failures: " + ", ".join(sorted({str(r['reason']) for r in info['rows']
                                                       if r['status'] != 'success'})) + ".")
    wall = best.get("high_wall") or {}
    wall_alts = {(a["policy"], a["mode"]): a for a in (wall.get("alternatives") or [])}
    vt_wall = alignment.get("high_wall", {})
    obstacle = wall_alts.get(("obstacle", "adaptive05"))
    row2p = wall_alts.get(("row2p_p40w40_v1", "adaptive05"))
    if wall.get("policy"):
        parts = [f"high_wall: recommended {wall.get('policy')} [{wall.get('mode')}] "
                 f"({vt_wall.get('n_pass', 0)}/{vt_wall.get('n_tested', 0)} offsets pass)"]
        if obstacle:
            parts.append(f"obstacle [adaptive05] equal-tolerance alternative "
                         f"({obstacle['n_pass']}/{obstacle['n_tested']})")
        if row2p:
            parts.append(f"row2p_p40w40_v1 only {row2p['n_pass']}/{row2p['n_tested']} "
                         "— poor wall choice at nonzero entry error")
        notes.append("; ".join(parts) + ".")
    if basin:
        rect = basin.get("largest_all_pass_rectangle_around_origin")
        notes.append(
            f"stairs_30cm basin: {basin['n_pass']}/{basin['n_tested']} grid points pass "
            f"({basin['pass_rate']:.0%}); "
            + (f"largest all-pass rectangle around (0,0): dy in "
               f"[{rect['dy_min']:+.2f},{rect['dy_max']:+.2f}] m, dyaw in "
               f"[{rect['dyaw_min']:+.0f},{rect['dyaw_max']:+.0f}] deg."
               if rect else "no all-pass rectangle contains (0,0).")
            + f" Pattern: dy>0 pass {basin['pattern']['pass_rate_dy_positive']:.0%} vs "
              f"dy<0 {basin['pattern']['pass_rate_dy_negative']:.0%}; dyaw>0 "
              f"{basin['pattern']['pass_rate_dyaw_positive']:.0%} vs dyaw<0 "
              f"{basin['pattern']['pass_rate_dyaw_negative']:.0%}.")
        best_mode, best_grid = max(
            (basin.get("grids") or {}).items(),
            key=lambda kv: (kv[1]["largest_all_pass_rectangle_around_origin"] or
                            {"area_cells": 0})["area_cells"]) \
            if basin.get("grids") else (None, None)
        fine = basin.get("fine_neighborhood_vx070")
        if fine:
            rect2 = fine.get("largest_all_pass_rectangle_around_center")
            notes.append(
                f"stairs_30cm fine neighborhood at vx={fine['vx']:g} around "
                f"(dy=-0.04, dyaw=0): {fine['n_pass']}/{fine['n_tested']} pass "
                f"({fine['pass_rate']:.0%}); best row dy={fine['best_row_dy']:+.3f} "
                f"({fine['best_row_pass']}/{len(fine['dyaw_deg_grid'])} yaw values pass)"
                + (f"; all-pass rectangle around the probe centre dy "
                   f"[{rect2['dy_min']:+.3f},{rect2['dy_max']:+.3f}] m, dyaw "
                   f"[{rect2['dyaw_min']:+.1f},{rect2['dyaw_max']:+.1f}] deg."
                   if rect2 else "."))
        if basin.get("grids"):
            max_rate_mode, max_rate_grid = max(
                basin["grids"].items(), key=lambda kv: kv[1]["pass_rate"])
            notes.append(
                f"stairs_30cm basin by speed: vx=0.50 {basin['pass_rate']:.0%} pass, "
                f"vx=0.55 {basin['grids'].get('adaptive055', {}).get('pass_rate', 0):.0%}, "
                f"vx=0.70 {max_rate_grid['pass_rate']:.0%} "
                f"(highest; {max_rate_grid['n_pass']}/{max_rate_grid['n_tested']}); "
                "the pass map is patchy on a 1 cm / 1 deg scale, so exact-offset "
                "passes are not a tolerance envelope.")
        better = [p for p in basin["vx_probe"]
                  if p["status"] == "success" and p["mode"] != "adaptive05"]
        if better:
            notes.append("stairs_30cm vx probe: " + "; ".join(
                f"vx={p['vx']:g} at (dy={p['dy']:+.2f},dyaw={p['dyaw_deg']:+.1f}) passes"
                for p in better) + ".")
        else:
            notes.append("stairs_30cm vx probe: no 0.45/0.55 probe point passed where "
                         "adaptive vx=0.5 was measured.")
    mism = [row for row in repro["rows"] if not row["match"]]
    if mism:
        notes.append("reproducibility mismatches vs vtaudit-20261031: " +
                     "; ".join(f"{r['policy']}/{r['level']}/{r['mode']} "
                               f"expected {r['expected']} got {r['observed']}"
                               for r in mism))
    else:
        notes.append(f"reproducibility: {repro['n_match']}/{repro['n']} authoritative "
                     "old-benchmark outcomes reproduced exactly.")
    for category, info in duration.items():
        runs = info.get("runs") or []
        if not runs:
            continue
        compact = ", ".join(f"{r['duration_s']:g}s={r['status']}" for r in runs)
        notes.append(f"{category} duration sensitivity: {compact}.")
    return notes


def write_summary(summary: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (ART / "summary.md").write_text(render_markdown(summary))
    basin = summary.get("stairs30_basin")
    if basin:
        (ART / "stairs30_basin.json").write_text(
            json.dumps(basin, indent=1, ensure_ascii=False) + "\n")


def render_markdown(s: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Part 2 per-obstacle expert matrix — blackW (16-DoF wheel-leg, vt-family)")
    add("")
    add(f"* campaign: `{s['campaign']}`  ")
    add(f"* created (UTC): {s['created_at']}  ")
    add(f"* judge: `{s['environment']['judge_version']}`  ")
    add(f"* runner: eval-side in-process MuJoCo (`robocon_eval.policy_runner.run`), "
        f"profile `official`, settle 1 s, stop-on-fall  ")
    add(f"* command: `{s['campaign_command']}`  ")
    add(f"* totals: {s['totals']['runs']} runs, {s['totals']['successes']} successes, "
        f"{s['totals']['errors']} runner errors; "
        f"runner wall {s['totals']['runner_wall_time_s']} s "
        f"(session {s['totals']['session_wall_time_s']} s)" +
        (f", last phases: {', '.join(f'{k} {v:.0f}s' for k, v in s['totals']['phase_wall_time_s'].items())}"
         if s["totals"].get("phase_wall_time_s") else ""))
    add("")
    add("## Protocol")
    add("")
    add("| item | value |")
    add("|---|---|")
    primary = s["protocol"]["primary_duration_s"]
    add(f"| primary duration | {primary.get('default')} s (stairs {primary.get('stairs')} s); "
        f"safety cap 120 s, success is a terminal state |")
    add("| adaptive command | `AdaptiveCommand(vx=0.5, k_y=0.8, k_yaw=2.0, k_vy=0.6, "
        "k_vheading=0.0, vy_gain=0.5, max_vy=0.2, max_wz=0.7, center_y=0.0)` |")
    add("| fixed command | constant body-frame `(vx, 0, 0)` from the mode table |")
    add("| sim determinism | identical reruns for a fixed (policy, level, mode, offset) — "
        "robustness measured with spawn offsets, not seeds |")
    offs = ", ".join("(dy={:+.2f}, dyaw={:+.1f}\u00b0)".format(o["dy"], o["dyaw_deg"])
                     for o in s["protocol"]["spawn_offsets"])
    add(f"| alignment offsets | {offs} |")
    add("")
    add("## Best deployable expert per category")
    add("")
    add("| category | target level | reached | recommended expert | mode | passing configs at target "
        "| alignment tolerance | selection basis | fail reasons (top) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for category, info in s["best_by_category"].items():
        align = s["alignment"].get(category, {})
        reasons = ", ".join(f"`{k}`×{v}" for k, v in list(info["fail_reasons"].items())[:2])
        add(f"| {category} | {info['target_level']} | "
            f"{'✓ ' + info['reached_level'] if info['reached_level'] else '✗'} | "
            f"{info['policy'] or '—'} | {info['mode'] or '—'} | "
            f"{info['n_success_configs']}/{info['n_configs_tested']} | "
            f"{align.get('n_pass', 0)}/{align.get('n_tested', 0)} offsets | "
            f"{info.get('selection_basis', '—')} | {reasons} |")
    add("")
    add("## Route usage (recommended deployment set)")
    add("")
    add("| category | level | expert | policy dir | mode | command | baseline | "
        "offset tolerance | alignment gate for the route |")
    add("|---|---|---|---|---|---|---|---|---|")
    for row in s.get("route_usage") or []:
        if not row.get("policy"):
            add(f"| {row['category']} | — | — | — | — | — | — | — | no success |")
            continue
        cmd = (f"adaptive vx={row['vx']:g}" if row["command_kind"] == "adaptive"
               else f"fixed vx={row['vx']:g}")
        add(f"| {row['category']} | {row['level']} | {row['policy']} | "
            f"`{row['policy_dir']}` | {row['mode']} | {cmd} | {row['baseline']} | "
            f"{row['alignment_tolerance']} | {row['alignment_gate']} |")
    add("")
    add("## Alignment tolerance (winner per category)")
    add("")
    add("Baseline `(0,0)` plus the required ±dy/±dyaw offsets; ✓ = judge success. "
        "Tolerance is *measured*, not a monotone box: the trajectory is chaotic in "
        "contact, so e.g. the stairs winner passes `dy=+0.05` and `dyaw=+5` but not "
        "`dyaw=-2`; judge which offsets are safe from the table, not by interpolation.")
    add("")
    for category, info in s["alignment"].items():
        add(f"**{category}** — {info['policy']} [{info['mode']}] on "
            f"`{info['reached_level']}`: {info['n_pass']}/{info['n_tested']} pass")
        add("")
        add("| dy (m) | dyaw (deg) | status | reason |")
        add("|---|---|---|---|")
        for row in info["offsets"]:
            mark = "✓ success" if row["status"] == "success" else f"✗ {row['status']}"
            add(f"| {row['dy']:+.2f} | {row['dyaw_deg']:+.1f} | {mark} | "
                f"{'—' if not row['reason'] else row['reason']} |")
        add("")
        alts = s["best_by_category"].get(category, {}).get("alternatives") or []
        if alts:
            add("Runner-up tolerance (same level):")
            add("")
            add("| candidate | mode | tolerance | passing offsets |")
            add("|---|---|---|---|")
            for alt in alts:
                offs = ", ".join(f"dy={o['dy']:+.2f}/dyaw={o['dyaw_deg']:+.1f}"
                                 for o in alt.get("pass_offsets") or []) or "—"
                add(f"| {alt['policy']} | {alt['mode']} | "
                    f"{alt['n_pass']}/{alt['n_tested']} | {offs} |")
            add("")
    basin = s.get("stairs30_basin")
    if basin:
        add("## stairs_30cm alignment basin (vt_band_v1, adaptive vx=0.5)")
        add("")
        add(f"Fine grid: {len(basin['protocol']['dy_grid'])} dy x "
            f"{len(basin['protocol']['dyaw_deg_grid'])} dyaw = "
            f"{basin['n_tested']} measured points, {basin['n_pass']} pass "
            f"({basin['pass_rate']:.0%}). `✓` pass, `✗` fail, `·` not measured.")
        add("")
        dyaw_grid = basin["protocol"]["dyaw_deg_grid"]
        add("| dy (m) \\ dyaw (deg) | " + " | ".join(f"{y:+.0f}" for y in dyaw_grid) + " |")
        add("|---" * (len(dyaw_grid) + 1) + "|")
        for i, dy in enumerate(basin["protocol"]["dy_grid"]):
            cells = []
            for cell in basin["grid"][i]:
                if cell["status"] == "success":
                    cells.append("✓")
                elif cell["status"] == "missing":
                    cells.append("·")
                else:
                    cells.append("✗")
            add(f"| {dy:+.2f} | " + " | ".join(cells) + " |")
        add("")
        rect = basin.get("largest_all_pass_rectangle_around_origin")
        if rect:
            add(f"* Largest all-pass rectangle around (0,0): dy in "
                f"[{rect['dy_min']:+.2f}, {rect['dy_max']:+.2f}] m, dyaw in "
                f"[{rect['dyaw_min']:+.0f}, {rect['dyaw_max']:+.0f}] deg "
                f"({rect['n_dy']}x{rect['n_dyaw']} = {rect['area_cells']} cells).")
        axis = basin["origin_axis_pass"]
        add(f"* Contiguous axis cut through origin: dy in "
            f"[{axis['contiguous_dy_min']:+.2f}, {axis['contiguous_dy_max']:+.2f}] m "
            f"at dyaw=0, dyaw in [{axis['contiguous_dyaw_min']:+.0f}, "
            f"{axis['contiguous_dyaw_max']:+.0f}] deg at dy=0 (grid steps "
            f"{BASIN_DY[1] - BASIN_DY[0]:g} m / "
            f"{BASIN_DYAW[1] - BASIN_DYAW[0]:g} deg).")
        add(f"* Non-contiguous extent (max |passing| value on the axis): |dy| up to "
            f"{axis['max_abs_dy_on_dyaw0']:+.2f} m, |dyaw| up to "
            f"{axis['max_abs_dyaw_deg_on_dy0']:+.0f} deg.")
        alt_modes = [(m, g) for m, g in (basin.get("grids") or {}).items()
                     if m != basin.get("primary_mode")]
        if alt_modes:
            add("")
            add("Alternative-speed basins (full grids in `stairs30_basin.json`):")
            add("")
            add("| mode | vx | pass rate | all-pass rectangle (dy / dyaw) | axis |dy| / |dyaw| |")
            add("|---|---|---|---|---|")
            for mode, grid in alt_modes:
                rect = grid.get("largest_all_pass_rectangle_around_origin")
                rect_txt = (f"dy [{rect['dy_min']:+.2f},{rect['dy_max']:+.2f}], "
                            f"dyaw [{rect['dyaw_min']:+.0f},{rect['dyaw_max']:+.0f}]"
                            if rect else "none around origin")
                ax = grid["origin_axis_pass"]
                add(f"| {mode} | {grid['vx']:g} | {grid['n_pass']}/{grid['n_tested']} "
                    f"({grid['pass_rate']:.0%}) | {rect_txt} | "
                    f"{ax['max_abs_dy_on_dyaw0']:+.2f} m / "
                    f"{ax['max_abs_dyaw_deg_on_dy0']:+.0f} deg |")
        pat = basin["pattern"]
        add(f"* Asymmetry: dy>0 pass {pat['pass_rate_dy_positive']:.0%} vs dy<0 "
            f"{pat['pass_rate_dy_negative']:.0%}; dyaw>0 "
            f"{pat['pass_rate_dyaw_positive']:.0%} vs dyaw<0 "
            f"{pat['pass_rate_dyaw_negative']:.0%}.")
        add(f"* Failure reasons: " + ", ".join(
            f"`{k}`×{v}" for k, v in basin["failure_reasons"].items()) + ".")
        add("")
        fine = basin.get("fine_neighborhood_vx070")
        if fine:
            rect2 = fine.get("largest_all_pass_rectangle_around_center")
            add("")
            add(f"Fine neighborhood at vx={fine['vx']:g} around "
                f"(dy={fine['center']['dy']:+.2f}, dyaw={fine['center']['dyaw_deg']:+.1f}): "
                f"{fine['n_pass']}/{fine['n_tested']} pass ({fine['pass_rate']:.0%}), "
                f"best row dy={fine['best_row_dy']:+.3f} m "
                f"({fine['best_row_pass']}/{len(fine['dyaw_deg_grid'])} pass).")
            add("")
            dyaw2 = fine["dyaw_deg_grid"]
            add("| dy (m) \\ dyaw (deg) | " + " | ".join(f"{y:+.1f}" for y in dyaw2) + " |")
            add("|---" * (len(dyaw2) + 1) + "|")
            for i, dy in enumerate(fine["dy_grid"]):
                cells = []
                for cell in fine["grid"][i]:
                    cells.append("✓" if cell["status"] == "success"
                                 else ("·" if cell["status"] == "missing" else "✗"))
                add(f"| {dy:+.3f} | " + " | ".join(cells) + " |")
            add("")
            if rect2:
                add(f"* All-pass rectangle around the probe centre: dy in "
                    f"[{rect2['dy_min']:+.3f}, {rect2['dy_max']:+.3f}] m, dyaw in "
                    f"[{rect2['dyaw_min']:+.1f}, {rect2['dyaw_max']:+.1f}] deg "
                    f"({rect2['n_dy']}x{rect2['n_dyaw']} = {rect2['area_cells']} cells).")
                add("")
        if basin.get("vx_probe"):
            add("vx widening probe (does another adaptive vx widen the basin?):")
            add("")
            add("| mode | vx | dy (m) | dyaw (deg) | status | reason |")
            add("|---|---|---|---|---|---|")
            for row in basin["vx_probe"]:
                add(f"| {row['mode']} | {row['vx']:g} | {row['dy']:+.2f} | "
                    f"{row['dyaw_deg']:+.1f} | {row['status']} | "
                    f"{'—' if not row['reason'] else row['reason']} |")
            add("")
    add("## Per-level matrix (offset 0, primary duration)")
    add("")
    add("Cell = passing command modes (✓) or the representative failure reason (✗). "
        "Policy rows, official-level columns; `·` = not run.")
    add("")
    # Build matrix table: rows = policies, columns = levels.
    levels = [lv for group in LEVEL_GROUPS.values() for lv in group]
    policies = sorted({r["policy"] for r in s["matrix"]})
    matrix = s["matrix"]
    add("| policy | " + " | ".join(levels) + " |")
    add("|---" * (len(levels) + 1) + "|")
    for policy in policies:
        cells = []
        for level in levels:
            recs = [r for r in matrix if r["policy"] == policy and r["level"] == level]
            cells.append(_best_cell(recs, LEVEL_CATEGORY[level]))
        add(f"| {policy} | " + " | ".join(cells) + " |")
    add("")
    add("## Stairs specialists vs vt_band_v1 (stairs_30cm is the hard wall)")
    add("")
    add("| policy | stairs_30cm (best mode) | stairs_20cm (best mode) |")
    add("|---|---|---|")
    for policy, cells in s["stairs_specialists"].items():
        def _c(c):
            if not c:
                return "·"
            if c["status"] == "success":
                return f"✓ {c['mode']}"
            return f"✗ {c['mode']} ({c['status']}: {_short_reason(c['reason'])})"
        add(f"| {policy} | {_c(cells.get('stairs_30cm'))} | {_c(cells.get('stairs_20cm'))} |")
    add("")
    add("## Generalists on the hard levels (row2p vs vt_mode_v2 vs vt_freezehid ...)")
    add("")
    levels40 = ["high_platform_40cm", "high_wall_40cm", "slope_35deg", "slope_45deg"]
    add("| policy | " + " | ".join(levels40) + " |")
    add("|---" * (len(levels40) + 1) + "|")
    for policy, cells in s["generalist_comparison"].items():
        row = []
        for level in levels40:
            c = cells.get(level)
            if not c:
                row.append("·")
            elif c["status"] == "success":
                row.append(f"✓ {c['mode']}")
            else:
                row.append(f"✗ {c['mode']} ({_short_reason(c['reason'], 24)})")
        add(f"| {policy} | " + " | ".join(row) + " |")
    add("")
    add("## Trench mode sweep (trench_30cm needs speed)")
    add("")
    modes = list(MODE_SWEEP_MODES["trench"])
    add("| policy | level | " + " | ".join(modes) + " |")
    add("|---" * (len(modes) + 2) + "|")
    for policy, per_level in s["trench_modes"].items():
        for level, cells in per_level.items():
            row = []
            for mode in modes:
                c = cells.get(mode)
                if not c:
                    row.append("·")
                else:
                    row.append("✓" if c["status"] == "success"
                               else f"✗{_short_reason(c['reason'], 18)}")
            add(f"| {policy} | {level} | " + " | ".join(row) + " |")
    add("")
    probe = s.get("probe") or {}
    if probe:
        add("## Mode-window probes (hard levels)")
        add("")
        add("How wide is the command window for the single-mode hard solvers? "
            "(These probe modes are characterisation only, not part of the matrix.)")
        add("")
        add("| level | policy | mode | command | status | reason |")
        add("|---|---|---|---|---|---|")
        for level, info in probe.items():
            for row in info["rows"]:
                if row.get("kind") == "adaptive":
                    cmd = f"adaptive vx={row['vx']:g}"
                else:
                    cmd = f"fixed vx={row['command'][0]:g}" if row.get("command") else "—"
                mark = "✓" if row["status"] == "success" else f"✗ {row['status']}"
                add(f"| {level} | {info['policy']} | {row['mode']} | {cmd} | {mark} | "
                    f"{'—' if not row['reason'] else row['reason']} |")
        add("")
    add("## Duration sensitivity of the winners")
    add("")
    add("| category | level | policy [mode] | durations → judge status |")
    add("|---|---|---|---|")
    for category, info in s["duration_sensitivity"].items():
        runs = info.get("runs") or []
        add(f"| {category} | {info.get('reached_level') or '—'} | "
            f"{info.get('policy') or '—'} [{info.get('mode') or '—'}] | "
            + ", ".join(f"{r['duration_s']:g}s={r['status']}" for r in runs) + " |")
    add("")
    add("## Reproducibility vs the authoritative old benchmark (vtaudit-20261031)")
    add("")
    add(f"{s['reproducibility']['n_match']}/{s['reproducibility']['n']} historical "
        "outcomes reproduced with the old benchmark durations.")
    add("")
    add("| policy | level | mode | duration | expected | observed | match |")
    add("|---|---|---|---|---|---|---|")
    for row in s["reproducibility"]["rows"]:
        dur = (f"{row['duration_s']:g}s" if row["duration_s"] is not None else "?")
        add(f"| {row['policy']} | {row['level']} | {row['mode']} | "
            f"{dur} | {row['expected']} | {row['observed']} | "
            f"{'✓' if row['match'] else '✗'} |")
    add("")
    add("## Failure-reason census (offset 0, all modes)")
    add("")
    add("| level | failure reasons (count) |")
    add("|---|---|")
    for level, reasons in s["failure_reasons"].items():
        add(f"| {level} | " + ", ".join(f"`{k}`×{v}" for k, v in reasons.items()) + " |")
    add("")
    add("## Exact commands to reproduce the chosen experts")
    add("")
    add("```bash")
    add("# campaign entry point (all phases; per-run cache makes reruns cheap)")
    add("pixi run -e eval python tools/campaign/run_expert_matrix.py")
    add("```")
    add("")
    for category, info in s["best_by_category"].items():
        repro = info.get("repro") or {}
        if not repro:
            continue
        add(f"### {category}: {info['policy']} [{info['mode']}] on "
            f"`{info['reached_level']}`")
        add("")
        add("```bash")
        add(repro["campaign_cli"])
        add("```")
        add("")
        add("```python")
        add(repro["python_snippet"].rstrip())
        add("```")
        add("")
    add("## Notes / surprises")
    add("")
    for note in s["notes"]:
        add(f"* {note}")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
PHASE_ORDER = ["matrix", "modes", "align", "duration", "probe", "basin", "repro"]
PHASE_BUILDERS = {
    "matrix": phase_matrix,
    "modes": phase_modes,
    "align": phase_align,
    "duration": phase_duration,
    "probe": phase_probe,
    "basin": phase_basin,
    "repro": phase_repro,
}


def _parse_only(text: str) -> tuple[str, str, str, tuple[float, float]]:
    """``policy:level:mode[:dy:dyaw]``"""
    parts = text.split(":")
    if len(parts) not in (3, 5):
        raise SystemExit("--only expects policy:level:mode[:dy:dyaw]")
    policy, level, mode = parts[:3]
    offset = (float(parts[3]), float(parts[4])) if len(parts) == 5 else (0.0, 0.0)
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; known: {sorted(MODES)}")
    return policy, level, mode, offset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Part 2 expert matrix campaign (writes artifacts/part2_experts/)")
    ap.add_argument("--phases", default=",".join(PHASE_ORDER),
                    help=f"comma list from {PHASE_ORDER} (default all)")
    ap.add_argument("--jobs", type=int, default=6,
                    help="parallel worker processes (default 6, keep <= CPU budget/2)")
    ap.add_argument("--force", action="store_true",
                    help="ignore the per-run cache and rerun")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild summary.json/summary.md from the run cache only")
    ap.add_argument("--only", default=None,
                    help="run one config and print the result: policy:level:mode[:dy:dyaw]")
    ap.add_argument("--duration", type=float, default=None,
                    help="duration override for --only (default: protocol duration)")
    args = ap.parse_args(argv)
    if args.jobs < 1 or args.jobs > 6:
        raise SystemExit("--jobs must be in 1..6")

    catalog = _level_catalog()
    started = time.time()

    if args.only:
        policy, level, mode, offset = _parse_only(args.only)
        duration = args.duration if args.duration is not None else _duration_for(
            catalog[level]["category"])
        spec = make_spec("only", policy, level, mode, offset, duration, catalog)
        print(json.dumps(_worker(spec), indent=2))
        return 0

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in PHASE_BUILDERS]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; known: {PHASE_ORDER}")

    # Load full cache first so later phases can select candidates from it.
    results: dict[str, dict] = {}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for path in RUNS_DIR.glob("*.json"):
        try:
            rec = json.loads(path.read_text())
            results[rec["tag"]] = rec
        except Exception:  # noqa: BLE001
            pass
    print(f"cache: {len(results)} run(s) under {RUNS_DIR}", flush=True)

    phase_wall: dict[str, float] = {}
    if not args.aggregate_only:
        for phase in phases:
            print(f"[phase {phase}]", flush=True)
            t0 = time.time()
            specs = PHASE_BUILDERS[phase](catalog, results)
            known: dict[tuple, dict] = {}
            for rec in results.values():
                known.setdefault(config_key(rec), rec)
            new = execute_specs(specs, args.jobs, force=args.force, known=known)
            results.update(new)
            phase_wall[phase] = time.time() - t0
            n_reused = len(specs) - len(new)
            print(f"[phase {phase}] {len(specs)} spec(s), {n_reused} already measured, "
                  f"{len(new)} new, in {phase_wall[phase]:.1f}s", flush=True)

    summary = build_summary(results, catalog, phase_wall, time.time() - started)
    write_summary(summary)
    print(json.dumps({
        "summary_json": str(ART / "summary.json"),
        "summary_md": str(ART / "summary.md"),
        "runs": summary["totals"]["runs"],
        "successes": summary["totals"]["successes"],
        "best_by_category": {k: {kk: v[kk] for kk in
                                 ("target_level", "reached_level", "policy", "mode",
                                  "n_success_configs", "n_configs_tested")}
                             for k, v in summary["best_by_category"].items()},
        "reproducibility": {"n_match": summary["reproducibility"]["n_match"],
                            "n": summary["reproducibility"]["n"]},
        "runner_wall_time_s": summary["totals"]["runner_wall_time_s"],
        "session_wall_time_s": summary["totals"]["session_wall_time_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
