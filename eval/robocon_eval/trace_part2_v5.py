"""Part 2 (blackW wheel) trace replay.

The trace ``meta`` line carries the official level entry, so a replay is
self-describing; the entry is still verified against the canonical generated
manifest (``demo/scenes/part2/part2_official_levels.json``) so a forged trace
cannot downgrade a 40 cm wall to 10 cm thresholds.

Frames are judged independently of the recording cadence: the judge advances its
hold timers from consecutive ``t`` timestamps, so both the judged-run cadence
(2 ms, ``eval_state_every:=1``) and the legacy 20 ms recording replay correctly.
The optional ``support_frame`` meta field selects how the recorded support points
are interpreted (``wheel_lowest`` = geometric wheel bottom; anything else, the
default, is the legacy wheel-body centre which gets the radius subtracted).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import rules
from .part2.official import canonicalize_level
from .part2.v5_judge import JUDGE_VERSION, WHEEL_RADIUS_M, Part2OfficialJudge
from .scoring import score_for_level
from .trace import LEGS, read_trace


def frame_tilt_deg(frame: dict) -> float:
    """Tilt from the frame, preferring the recorded tilt, else the base quaternion."""
    for key in ("tilt_deg", "tilt"):
        value = frame.get(key)
        if value is not None:
            return float(value)
    quat = frame.get("quat")
    if quat is not None and len(quat) == 4:
        w, x, y, z = (float(v) for v in quat)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm > 1e-9:
            cos_theta = (1.0 - 2.0 * (x * x + y * y)) / (norm * norm)
            return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_theta)))))
    return 0.0


def replay_part2_v5(path: str | Path, *, levels_path: str | Path | None = None) -> dict:
    meta, frames = read_trace(path)
    if meta.get("part") != 2:
        raise ValueError(f"not a Part 2 trace: {meta.get('part')!r}")
    declared = meta.get("level") or meta.get("level_entry")
    if not isinstance(declared, dict):
        raise ValueError("Part 2 trace meta needs level (official level entry)")
    # The trace copy is self-describing but untrusted: use the canonical entry
    # for thresholds and reject a declared judge subset that differs from it.
    level = canonicalize_level(declared, require_key=True, path=levels_path)
    support_frame = meta.get("support_frame", "wheel_center")
    judge = Part2OfficialJudge(level)
    for frame in frames:
        support = frame.get("support") or {}
        have_support = all(leg in support for leg in LEGS)
        wx = np.array([float(support[leg][0]) for leg in LEGS]) if have_support else None
        wy = np.array([float(support[leg][1]) for leg in LEGS]) if have_support else None
        wz = np.array([float(support[leg][2]) for leg in LEGS]) if have_support else None
        if wz is not None and support_frame != "wheel_lowest":
            # Legacy traces recorded the wheel body centre; subtract the radius.
            wz = wz - WHEEL_RADIUS_M
        judge.update(float(frame["t"]), np.asarray(frame["base"], dtype=float),
                     frame_tilt_deg(frame), wheel_x=wx, wheel_z=wz, wheel_y=wy)
    report = judge.finalize("trace_end")
    if report["status"] == "success":
        report["score"] = score_for_level(rules.part2(), level["category"], level["level"])
    else:
        report["score"] = 0
    report["judge_version"] = JUDGE_VERSION
    report["level_key"] = level.get("key")
    report["category"] = level.get("category")
    report["level"] = level.get("level")
    report["trace"] = {"path": str(path), "frames": len(frames), "meta": meta,
                       "frame_dt_s": (float(frames[-1]["t"]) - float(frames[0]["t"]))
                       / max(1, len(frames) - 1) if len(frames) > 1 else None}
    return report


__all__ = ["replay_part2_v5", "frame_tilt_deg"]
