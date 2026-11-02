"""Formal Part 1 whole-course attempt protocol (attempts + timer + return bonus).

The course itself is driven externally (scripted diagnostic route or a human operator
through the control stack); this module owns the *rule bookkeeping*:

- one attempt = one continuous run over the Part 1 field; judge states are fed in
  continuously through ``Part1JudgeV2``;
- 3 attempts per run; a fall with pause+recovery does NOT consume an attempt and the
  paused wall time is excluded from the attempt duration;
- starting/restarting and operator skip consume an attempt (current organizer reading);
- obstacle-level skip (the course offers skip for unpassable obstacles) marks that
  obstacle failed/skipped but the attempt continues;
- attempt score = sum of obstacle scores + 100 return bonus iff all 8 obstacles
  succeeded AND the robot re-entered the start area after the last success;
- final score = best attempt score; final time = fastest attempt duration.

This module is pure Python and covered by ``eval/tests/test_course_events.py``;
the MuJoCo runner only has to stream ``update`` calls and operator events.  All
rule constants (attempts, return bonus, recovery/stop tolerances) come from
:mod:`robocon_eval.rules`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .judge import ZONES, AttemptManager, Part1JudgeV2
from .. import rules

MAX_ATTEMPTS = rules.PART1_ATTEMPTS
RETURN_BONUS = float(rules.PART1_RETURN_BONUS)


class StableStopTracker:
    """Timer stop rule (confirmed 2026-10-22): stop the clock once the robot is stable.

    A robot that has returned to the start area still needs to settle; the clock stops
    after ``hold_s`` seconds with speed < ``speed_max`` and tilt < ``tilt_max``.
    ``dt`` must be the real physics timestep (2 ms judged / 5 ms control), not the
    policy period.
    """

    def __init__(self, speed_max: float = rules.PART1_STOP_SPEED_MAX_MPS,
                 tilt_max: float = rules.PART1_STOP_TILT_MAX_DEG,
                 hold_s: float = rules.PART1_STOP_HOLD_S) -> None:
        self.speed_max = speed_max
        self.tilt_max = tilt_max
        self.hold_s = hold_s
        self.held = 0.0

    def update(self, speed: float, tilt_deg: float, dt: float) -> bool:
        if speed < self.speed_max and tilt_deg < self.tilt_max:
            self.held += float(dt)
        else:
            self.held = 0.0
        return self.held >= self.hold_s


@dataclass
class AttemptResult:
    attempt: int
    score: float = 0.0
    obstacle_score: float = 0.0
    return_bonus: float = 0.0
    duration_s: float = 0.0
    skipped: bool = False
    reason: str = ""
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"attempt": self.attempt, "score": self.score,
                "obstacle_score": self.obstacle_score, "return_bonus": self.return_bonus,
                "duration_s": self.duration_s, "skipped": self.skipped,
                "reason": self.reason, "events": self.events}


class Part1Course:
    """One whole-run session (up to 3 attempts) with pause/recovery/skip events."""

    def __init__(self, max_attempts: int = MAX_ATTEMPTS,
                 partial_up: float = 50.0, partial_down: float = 50.0) -> None:
        self.manager = AttemptManager(max_attempts=max_attempts)
        self.judge = Part1JudgeV2(partial_up=partial_up, partial_down=partial_down)
        self.partial_up = partial_up
        self.partial_down = partial_down
        self.active = False
        self.attempt_no = 0
        self.active_time = 0.0
        self.paused = False
        self.prev_t: float | None = None
        self.pause_pose: tuple[tuple[float, float], float] | None = None
        self.recovery_violation = False
        self.results: list[AttemptResult] = []
        self._current_events: list[dict] = []
        self._start_area_hits = 0
        self._prev_in_start = False

    # ---------------------------------------------------------------- lifecycle
    def begin_attempt(self, t: float) -> bool:
        if self.active:
            raise RuntimeError("attempt already running")
        if not self.manager.begin(t):
            return False
        self.judge = Part1JudgeV2(partial_up=self.partial_up, partial_down=self.partial_down)
        self.active = True
        self.attempt_no += 1
        self.active_time = 0.0
        self.attempt_start_t = float(t)
        self.last_sim_t = float(t)
        self.recovery_violation = False
        self.pause_pose = None
        self.paused = False
        self.prev_t = float(t)
        self._start_area_hits = 0
        self._prev_in_start = False
        self._current_events = []
        return True

    def update(self, t: float, base_xyz, yaw: float = 0.0, **kwargs) -> dict:
        if not self.active:
            raise RuntimeError("no active attempt; call begin_attempt() first")
        t = float(t)
        # Duration is derived from the simulator clock at end_attempt: while paused the
        # physics clock is frozen, so no manual pause subtraction is needed (and the
        # operator fake clock can never leak into the ledger).
        self.last_sim_t = t
        report = self.judge.update(t, base_xyz, yaw, **kwargs)
        in_start = (ZONES["start_area"][0] <= base_xyz[0] <= ZONES["start_area"][1]
                    and ZONES["start_area"][2] <= base_xyz[1] <= ZONES["start_area"][3])
        if in_start and not self._prev_in_start:
            self._start_area_hits += 1
        self._prev_in_start = in_start
        return report

    def pause(self, t: float, base_xyz=None, yaw: float | None = None) -> None:
        """Fall detected -> timer paused (does not consume an attempt by itself)."""
        if not self.active or self.paused:
            return
        self.paused = True
        if base_xyz is not None and yaw is not None:
            self.pause_pose = ((float(base_xyz[0]), float(base_xyz[1])), float(yaw))
        sim_t = float(self.last_sim_t)
        self._current_events.append({"t": sim_t, "event": "pause", "operator_t": float(t),
                                     "pose": list(base_xyz) if base_xyz is not None else None})
        self.manager.note_recovery(self.last_sim_t)

    def resume(self, t: float, base_xyz=None, yaw: float | None = None) -> None:
        if not self.active or not self.paused:
            return
        # Rules: recovery must resume from the same position and heading (no teleport).
        if self.pause_pose is not None and base_xyz is not None and yaw is not None:
            (px, py), pyaw = self.pause_pose
            moved = ((float(base_xyz[0]) - px) ** 2 + (float(base_xyz[1]) - py) ** 2) ** 0.5
            dyaw = abs((float(yaw) - pyaw + 3.14159265) % (2 * 3.14159265) - 3.14159265)
            if (moved > rules.PART1_RECOVERY_POS_TOL_M
                    or dyaw > rules.PART1_RECOVERY_YAW_TOL_RAD):
                self.recovery_violation = True
                self._current_events.append({"t": float(t), "event": "recovery_pose_violation",
                                             "moved_m": round(moved, 3),
                                             "dyaw_rad": round(dyaw, 3)})
        self.paused = False
        if self.manager._current and self.manager._current["pauses"]:
            self.manager._current["pauses"][-1][1] = float(self.last_sim_t)  # sim clock
        sim_t = float(self.last_sim_t)
        self._current_events.append({"t": sim_t, "event": "resume", "operator_t": float(t),
                                     "pose": list(base_xyz) if base_xyz is not None else None})
        self.manager.events.append({"t": sim_t, "event": "resume", "operator_t": float(t),
                                    "consumes_attempt": False,
                                    "attempt": self.attempt_no})

    def skip_obstacle(self, t: float, name: str) -> None:
        if not self.active:
            raise RuntimeError("no active attempt")
        obs = self.judge.obstacles.get(name)
        if obs is None:
            raise KeyError(name)
        if obs.result.status in ("success", "fail"):
            return
        obs.result.status = "skipped"
        obs.result.reason = "operator_skip"
        self._current_events.append({"t": float(self.last_sim_t), "event": "skip_obstacle",
                                     "operator_t": float(t), "obstacle": name})

    def end_attempt(self, t: float, reason: str = "finished") -> AttemptResult:
        if not self.active:
            raise RuntimeError("no active attempt")
        self.judge.finalize(reason)
        report = self.judge.report()
        obstacle_score = float(report["obstacle_score"])
        return_bonus = float(report["return_bonus"])  # judge enforces "after all obstacles full"
        if self.recovery_violation:
            obstacle_score = 0.0
            return_bonus = 0.0
            reason = "recovery_pose_violation"
        score = obstacle_score + return_bonus
        duration = max(0.0, float(t) - float(self.attempt_start_t))
        result = AttemptResult(attempt=self.attempt_no, score=score, obstacle_score=obstacle_score,
                               return_bonus=return_bonus, duration_s=duration,
                               skipped=False, reason=reason, events=list(self._current_events))
        self.recovery_violation = False
        attempt_report = {"score": score, "obstacle_score": obstacle_score,
                          "return_bonus": return_bonus}
        self.results.append(result)
        self.manager.end(t, attempt_report, reason=reason)
        # attempt duration in the ledger must be the *active* time (pause excluded)
        self.manager.attempts[-1]["duration_s"] = result.duration_s
        self.manager.attempts[-1]["duration_note"] = "active_time_pause_excluded"
        self.active = False
        self.paused = False
        return result

    def abort_attempt(self, t: float, reason: str = "operator_abort") -> AttemptResult:
        """Operator abort/restart: consumes the attempt, score 0."""
        if not self.active:
            raise RuntimeError("no active attempt")
        duration = max(0.0, float(self.last_sim_t) - float(self.attempt_start_t))
        result = AttemptResult(attempt=self.attempt_no, score=0.0, duration_s=duration,
                               skipped=True, reason=reason, events=list(self._current_events))
        sim_t = float(self.last_sim_t)
        self.manager.skip(sim_t, reason=reason)
        self.manager.events[-1]["operator_t"] = float(t)
        self.recovery_violation = False
        self.pause_pose = None
        self.manager.attempts[-1]["duration_s"] = result.duration_s
        self.manager.attempts[-1]["duration_note"] = "active_time_pause_excluded"
        self.results.append(result)
        self.active = False
        self.paused = False
        return result

    # ---------------------------------------------------------------- results
    def summary(self) -> dict:
        summary = self.manager.summary()
        # snapshot live manager lists so an earlier attempt's exported report cannot be
        # mutated by later attempts (audit/reproducibility)
        summary["attempts"] = json.loads(json.dumps(summary.get("attempts") or []))
        summary["events"] = json.loads(json.dumps(summary.get("events") or []))
        summary["attempt_results"] = json.loads(json.dumps([r.to_dict() for r in self.results]))
        scored = [r for r in self.results if not r.skipped]
        best_score = max((r.score for r in scored), default=0.0)
        # confirmed rule: highest score first; on a tie take the fastest attempt
        best_candidates = [r for r in scored if r.score == best_score]
        summary["final_score"] = best_score
        summary["final_time_s"] = min((r.duration_s for r in best_candidates), default=0.0)
        summary["final_time_scope"] = "min_duration_among_best_score_attempts"
        summary["attempts_allowed"] = self.manager.max_attempts
        return summary

    def export_report(self) -> dict:
        return {"summary": self.summary(),
                "rules": {"return_bonus": RETURN_BONUS,
                          "recovery_consumes_attempt": False,
                          "skip_obstacle_consumes_attempt": False,
                          "final_score": "best_attempt",
                          "final_time": "min_duration_among_best_score_attempts"}}


# ---------------------------------------------------------------- self-test
def _mk_update(course: Part1Course, t: float, x: float, y: float, z: float = 0.45,
               tags: dict | None = None, ground: set | None = None, bar_z: float = 0.325) -> dict:
    return course.update(t, (x, y, z), 0.0, foot_tags=tags or {}, ground_feet=ground or set(),
                         bar_z=bar_z)


def run_self_test() -> None:
    # Attempt 1: all obstacles full + return to start after all full -> 900.
    c = Part1Course()
    assert c.begin_attempt(0.0)
    for obs in c.judge.obstacles.values():
        obs.result.status = "success"
        obs.result.score = 100.0
    c.update(0.0, (3.7, -9.0, 0.45), 0.0)     # first start-area entry
    c.update(0.1, (2.8, -9.0, 0.45), 0.0)     # leave
    c.update(0.2, (3.7, -9.0, 0.45), 0.0)     # re-enter after all full -> bonus
    result = c.end_attempt(0.3)
    assert result.score == 900.0, result.to_dict()
    assert result.return_bonus == 100.0, result.to_dict()
    summary = c.summary()
    assert summary["final_score"] == 900.0
    assert summary["attempts_used"] == 1
    assert abs(c.manager.attempts[0]["duration_s"] - result.duration_s) < 1e-9

    # Attempt 2: pause/resume does not consume; paused wall time is excluded.
    assert c.begin_attempt(10.0)
    c.update(10.1, (3.7, -9.0, 0.45), 0.0)
    c.pause(10.2, (3.7, -9.0, 0.45), 0.0)
    c.resume(15.0, (3.7, -9.0, 0.45), 0.0)    # operator fake clock (physics frozen)
    c.update(10.3, (3.7, -9.0, 0.45), 0.0)    # simulator clock continues after pause
    r2 = c.abort_attempt(10.4, "restart")
    assert r2.skipped and r2.score == 0.0
    assert r2.duration_s < 1.0, f"paused time must be excluded: {r2.duration_s}"
    summary = c.summary()
    assert summary["attempts_used"] == 2 and summary["recoveries"] == 1

    # Attempt 3: partial T-stairs (one arm) must not trigger the return bonus.
    assert c.begin_attempt(20.0)
    for obs in c.judge.obstacles.values():
        obs.result.status = "success"
        obs.result.score = 50.0 if obs.name == "t_stairs" else 100.0
    c.update(20.1, (3.7, -9.0, 0.45), 0.0)
    c.update(20.2, (2.8, -9.0, 0.45), 0.0)
    c.update(20.3, (3.7, -9.0, 0.45), 0.0)
    r3 = c.end_attempt(20.4)
    assert r3.return_bonus == 0.0, r3.to_dict()
    assert r3.score == 750.0, r3.to_dict()
    summary = c.summary()
    assert summary["attempts_used"] == 3
    assert summary["final_score"] == 900.0, summary
    assert not c.begin_attempt(30.0), "4th attempt must be rejected"

    # Recovery pose violation (teleport) invalidates the attempt score.
    c4 = Part1Course()
    assert c4.begin_attempt(0.0)
    for obs in c4.judge.obstacles.values():
        obs.result.status = "success"
        obs.result.score = 100.0
    c4.update(0.1, (3.7, -9.0, 0.45), 0.0)
    c4.pause(0.2, (3.7, -9.0, 0.45), 0.0)
    c4.resume(1.0, (2.0, -9.0, 0.45), 0.0)    # moved 1.7 m -> violation
    c4.update(1.1, (3.7, -9.0, 0.45), 0.0)
    r4 = c4.end_attempt(1.2)
    assert r4.score == 0.0 and r4.reason == "recovery_pose_violation", r4.to_dict()

    # stable-stop timer: real physics dt, must hold for 0.5 s of stable motion
    tracker = StableStopTracker()
    assert not tracker.update(0.05, 3.0, 0.005)   # 5 ms only
    held = 0.0
    done = False
    while held < 1.0:
        held += 0.005
        done = tracker.update(0.05, 3.0, 0.005)
        if done:
            break
    assert done and abs(tracker.held - 0.5) < 1e-6, (held, tracker.held)
    tracker2 = StableStopTracker()
    tracker2.update(0.05, 3.0, 0.40)   # wrong dt (policy period) must not stop early
    assert tracker2.held == 0.40
    tracker2.update(0.30, 3.0, 0.005)  # speed spike resets the hold
    assert tracker2.held == 0.0
    print("comp_part1_course self-test: OK", json.dumps(summary["rules"], ensure_ascii=False))


if __name__ == "__main__":
    run_self_test()
