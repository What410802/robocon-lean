"""Scoring semantics (SPEC §3.5 / §4.3).

All rounding is floor; Part 2 is a best-success-level curve lookup; Part 1 is
per-obstacle points plus the return bonus, with the attempt/session aggregation
used by the judge.  The numbers come from :mod:`robocon_eval.rules` -- never
hard-code a curve or attempt count here.
"""

from __future__ import annotations

import math

from . import rules


def curve_points_floor(obstacle: dict, value: float) -> int:
    """Curve score for one Part 2 level value (floor, endpoint and interpolation rules)."""
    points = [(float(pt["value"]), int(pt["points"])) for pt in obstacle["score_curve"]]
    if obstacle.get("unit") == "bool":
        return points[0][1] if float(value) == 1.0 else 0
    first_v, first_p = points[0]
    last_v, last_p = points[-1]
    v = float(value)
    if v == first_v:
        return first_p
    if v < first_v:
        return 0
    if v >= last_v:
        return last_p
    for (v0, p0), (v1, p1) in zip(points, points[1:]):
        if v0 <= v <= v1:
            return int(math.floor(p0 + (p1 - p0) * (v - v0) / (v1 - v0)))
    raise AssertionError("unreachable")


def score_for_level(part2: dict, obstacle_id: str, level: float | int | None) -> int:
    """Part 2 category score at the best successful level (None = no success)."""
    if level is None:
        return 0
    for obstacle in part2["obstacles"]:
        if obstacle["id"] == obstacle_id:
            return curve_points_floor(obstacle, level)
    raise KeyError(f"unknown part2 obstacle {obstacle_id!r}")


def part2_score(part2: dict, best_levels: dict[str, float | int | None]) -> dict:
    """Sum of category scores; also returns the per-category breakdown."""
    breakdown = {oid: score_for_level(part2, oid, best_levels.get(oid))
                 for oid in rules.PART2_OBSTACLES}
    return {"total": sum(breakdown.values()), "by_obstacle": breakdown}


def part1_attempt_score(part1: dict, tasks: dict[str, dict], returned_to_start: bool) -> dict:
    """Score one Part 1 attempt.

    ``tasks[id] = {"success": bool, "score": int, "skipped": bool}``; skipped
    tasks score 0 and block the return bonus (the +100 requires all 8
    obstacles at full score *and* a return to the start area).
    """
    by_obs: dict[str, int] = {}
    all_full = True
    for obstacle in part1["obstacles"]:
        oid = obstacle["id"]
        state = tasks.get(oid, {})
        pts = int(state.get("score", 0)) if state.get("success") else 0
        pts = max(0, min(pts, int(obstacle["per_obstacle_score"])))
        by_obs[oid] = pts
        if (state.get("skipped") or not state.get("success")
                or pts < int(obstacle["per_obstacle_score"])):
            all_full = False
    bonus = int(part1["return_bonus"]["score"]) if (returned_to_start and all_full) else 0
    obstacle_score = sum(by_obs.values())
    return {
        "obstacle_score": obstacle_score,
        "return_bonus": bonus,
        "total": obstacle_score + bonus,
        "by_obstacle": by_obs,
        "all_full": all_full,
    }


def part1_session_summary(attempts: list[dict]) -> dict:
    """Best score attempt, fastest time among attempts with that score (P1-SCORE-03)."""
    valid = [a for a in attempts if not a.get("skipped", False)]
    if not valid:
        return {"final_score": 0, "final_time_s": None, "attempts_used": len(attempts)}
    best = max(int(a["score"]) for a in valid)
    times = [float(a["duration_s"]) for a in valid if int(a["score"]) == best]
    return {
        "final_score": best,
        "final_time_s": min(times) if times else None,
        "attempts_used": len(attempts),
    }


__all__ = [
    "curve_points_floor", "score_for_level", "part2_score",
    "part1_attempt_score", "part1_session_summary",
]
