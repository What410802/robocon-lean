"""Part 2 v5 judge: platform / trench / wall terminal states and evidence rules.

These tests drive ``Part2OfficialJudge`` directly with synthetic support states
(wheel bottoms, body-frame offsets), which is exactly what the trace replay
feeds it.  They pin:

* traverse semantics: reaching/holding on the obstacle is evidence, success
  additionally requires leaving it to normal ground;
* the tightened top band (``top + 0.05`` for platform/plum) -- a wheel hovering
  10 cm above the platform is not "on" it;
* lane/forbidden-footprint, fall and fail-closed evidence handling;
* cadence independence: the same physical trajectory at 20 ms and 2 ms frames
  must give the same verdict.
"""

from __future__ import annotations

import unittest

import numpy as np

from robocon_eval.part2.v5_judge import JUDGE_VERSION, WHEEL_TOP_BAND_M, Part2OfficialJudge

LEGS = ("FL", "FR", "RL", "RR")
OFFSETS = ((0.2, -0.3), (0.2, 0.3), (-0.2, -0.3), (-0.2, 0.3))

PLATFORM = {
    "key": "high_platform_20cm", "category": "high_platform", "level": 20, "unit": "cm",
    "judge": {"kind": "platform", "top_z": 0.2, "x_range": [2.0, 3.25],
              "exit_x_min": 3.35, "exit_ground_tol": 0.08,
              "lane_half_width": 0.7, "hold_s": 0.2},
}

GAP = {
    "key": "trench_30cm", "category": "trench", "level": 30, "unit": "cm",
    "judge": {"kind": "gap", "gap_x": 3.95, "width": 0.3, "plateau_z": 0.3,
              "near_x": 3.8, "far_x": 4.1, "z_cross_min": 0.58,
              "gap_max_drop_m": 0.103, "exit_ramp_end": 6.1, "exit_ground_tol": 0.08,
              "lane_half_width": 0.7, "hold_s": 0.2},
}

WALL = {
    "key": "high_wall_20cm", "category": "high_wall", "level": 20, "unit": "cm",
    "judge": {"kind": "wall", "wall_h": 0.2, "wall_x": 1.86, "goal_x_min": 2.27,
              "climb_z_min": 0.54, "lane_half_width": 0.7, "hold_s": 0.2,
              "wall_span_half": 0.7},
}


def _update(judge: Part2OfficialJudge, t: float, x: float, *, wheel_z=0.2,
            base_z=0.55, y=0.0, tilt=2.0, wheel_y_offsets=None, wheel_z_offsets=None):
    wy = np.array([y + dy for _, dy in OFFSETS], dtype=float)
    if wheel_y_offsets is not None:
        wy = np.array(wheel_y_offsets, dtype=float)
    wz = np.full(4, float(wheel_z), dtype=float)
    if wheel_z_offsets is not None:
        wz = np.array(wheel_z_offsets, dtype=float)
    wx = np.array([x + dx for dx, _ in OFFSETS], dtype=float)
    return judge.update(t, np.array([x, y, base_z]), tilt,
                        wheel_x=wx, wheel_z=wz, wheel_y=wy)


def _feed(judge: Part2OfficialJudge, frames: list[dict]) -> Part2OfficialJudge:
    for frame in frames:
        _update(judge, **frame)
    return judge


def _platform_frames(dt: float = 0.02, top_steps: int = 20, exit_steps: int = 20,
                     top_wz: float = 0.2) -> list[dict]:
    frames = []
    t = 0.0
    for _ in range(top_steps):
        frames.append({"t": t, "x": 2.45, "wheel_z": top_wz, "base_z": 0.55})
        t += dt
    for _ in range(exit_steps):
        frames.append({"t": t, "x": 3.55, "wheel_z": 0.05, "base_z": 0.55})
        t += dt
    return frames


class TestPlatform(unittest.TestCase):
    def test_climb_hold_exit_success(self):
        report = _feed(Part2OfficialJudge(PLATFORM), _platform_frames()).finalize("end")
        self.assertEqual(report["status"], "success", report)
        self.assertTrue(report["detail"]["top_seen"])
        self.assertTrue(report["detail"]["exited"])

    def test_stopping_on_platform_is_not_success(self):
        frames = _platform_frames()[:20]
        report = _feed(Part2OfficialJudge(PLATFORM), frames).finalize("end")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["reason"], "not_exited_platform")

    def test_hovering_above_platform_is_not_on_surface(self):
        # wheel bottoms 10 cm above the top: outside the tightened +0.05 m band
        frames = _platform_frames(top_wz=0.2 + WHEEL_TOP_BAND_M + 0.05)
        report = _feed(Part2OfficialJudge(PLATFORM), frames).finalize("end")
        self.assertEqual(report["status"], "fail", report)
        self.assertFalse(report["detail"]["top_seen"])
        self.assertEqual(report["reason"], "not_on_platform")

    def test_side_corridor_is_forbidden(self):
        frames = []
        t = 0.0
        for _ in range(40):   # > SIDE_GRACE_S (0.6 s)
            frames.append({"t": t, "x": 2.45, "wheel_z": 0.2, "base_z": 0.55,
                           "wheel_y_offsets": [0.8, 0.3, -0.3, -0.3]})
            t += 0.02
        report = _feed(Part2OfficialJudge(PLATFORM), frames).finalize("end")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["reason"], "footprint_forbidden")

    def test_fall_fails(self):
        report = _update(Part2OfficialJudge(PLATFORM), 0.0, 2.45,
                         wheel_z=0.2, tilt=80.0)
        self.assertEqual(report, "fail")
        judge = _feed(Part2OfficialJudge(PLATFORM),
                      [{"t": 0.0, "x": 2.45, "wheel_z": 0.2, "base_z": 0.55}])
        judge.update(0.02, np.array([2.45, 0.0, 0.55]), 80.0,
                     wheel_x=np.array([2.65, 2.65, 2.25, 2.25]),
                     wheel_z=np.full(4, 0.2), wheel_y=np.array([-0.3, 0.3, -0.3, 0.3]))
        report = judge.finalize("end")
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["reason"].startswith("fell"))

    def test_missing_support_evidence_fails_closed(self):
        judge = Part2OfficialJudge(PLATFORM)
        self.assertEqual(judge.update(0.0, np.array([2.45, 0.0, 0.55]), 2.0), "fail")
        self.assertEqual(judge.reason, "missing_support_evidence")

    def test_missing_judge_key_fails_closed(self):
        broken = {"key": "x", "category": "high_platform", "level": 20,
                  "judge": {"kind": "platform", "top_z": 0.2, "x_range": [2.0, 3.25],
                            "hold_s": 0.2}}
        with self.assertRaises(KeyError):
            Part2OfficialJudge(broken)

    def test_frame_cadence_independent(self):
        slow = _feed(Part2OfficialJudge(PLATFORM), _platform_frames(0.02, 20, 20))
        fast = _feed(Part2OfficialJudge(PLATFORM), _platform_frames(0.002, 200, 200))
        self.assertEqual(slow.finalize("end")["status"], "success")
        self.assertEqual(fast.finalize("end")["status"], "success")

    def test_judge_version_is_bumped_for_plum_geometry(self):
        self.assertTrue(JUDGE_VERSION.startswith("v8."))


class TestTrench(unittest.TestCase):
    """Official trench level: ground -> 1 m entry ramp -> near plateau (z=0.3)
    -> 0.3 m slot -> far plateau -> 1 m exit ramp -> normal ground."""

    def _wheel_z(self, x: float) -> float:
        if x < 1.8:
            return 0.0
        if x < 2.8:
            return round(x - 1.8, 4)          # entry ramp up to the plateau
        if x < 5.1:
            return 0.3                        # near/far plateau + slot crossing
        if x < 6.1:
            return round(6.1 - x, 4)          # exit ramp down to the ground
        return 0.0

    def _frames(self) -> list[dict]:
        frames = []
        t = 0.0
        for x in [0.5 + 0.1 * i for i in range(19)] + \
                [3.7, 3.75, 3.85, 3.95, 4.05, 4.15, 4.25, 4.4, 4.8] + \
                [5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.1, 6.2, 6.35]:
            frames.append({"t": t, "x": x, "wheel_z": self._wheel_z(x), "base_z": 0.55})
            t += 0.02
        for _ in range(20):                   # hold on normal ground past the exit ramp
            frames.append({"t": t, "x": 6.35, "wheel_z": 0.0, "base_z": 0.55})
            t += 0.02
        return frames

    def test_cross_plateau_ramp_success(self):
        report = _feed(Part2OfficialJudge(GAP), self._frames()).finalize("end")
        self.assertEqual(report["status"], "success", report)
        self.assertTrue(report["detail"]["exited"])

    def test_wheel_in_slot_fails(self):
        frames = self._frames()
        for frame in frames:
            if 3.76 < frame["x"] < 4.14:
                frame["wheel_z"] = 0.10    # deeper than gap_max_drop below the plateau
                break
        report = _feed(Part2OfficialJudge(GAP), frames).finalize("end")
        self.assertEqual(report["status"], "fail", report)
        self.assertTrue(report["reason"].startswith("gap:wheel_in_slot"), report["reason"])

    def test_stopping_on_far_plateau_is_not_success(self):
        frames = [f for f in self._frames() if f["x"] <= 4.8]
        report = _feed(Part2OfficialJudge(GAP), frames).finalize("end")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["reason"], "not_descended")


class TestWall(unittest.TestCase):
    def _frames(self) -> list[dict]:
        frames = []
        t = 0.0
        for i in range(25):
            x = 1.4 + 0.05 * i
            frames.append({"t": t, "x": x, "wheel_z": 0.0, "base_z": 0.5})
            t += 0.02
        for _ in range(20):
            frames.append({"t": t, "x": 3.0, "wheel_z": 0.0, "base_z": 0.5})
            t += 0.02
        return frames

    def test_support_projection_success(self):
        report = _feed(Part2OfficialJudge(WALL), self._frames()).finalize("end")
        self.assertEqual(report["status"], "success", report)

    def test_side_crossing_fails(self):
        frames = self._frames()
        for frame in frames:
            if 1.6 < frame["x"] < 2.1:
                frame["wheel_y_offsets"] = [1.2, 0.3, -0.3, -0.3]
        report = _feed(Part2OfficialJudge(WALL), frames).finalize("end")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["reason"], "wall_side_bypass")


if __name__ == "__main__":
    unittest.main()
