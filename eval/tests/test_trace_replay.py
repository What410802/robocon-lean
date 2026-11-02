"""Trace format, contact classification and replay (Part 1 + Part 2).

Also pins the judged sampling policy: replay must accept both the legacy 20 ms
frame cadence and the judged-run 2 ms cadence (``eval_state_every:=1``) because
the judges advance hold timers from consecutive frame timestamps.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robocon_eval.trace import classify_contacts, read_trace, replay_part1
from robocon_eval.trace_part2_v5 import replay_part2_v5

FIX = Path(__file__).resolve().parents[1] / "fixtures"
LEGS = ("FL", "FR", "RL", "RR")
OFFSETS = ((0.2, -0.3), (0.2, 0.3), (-0.2, -0.3), (-0.2, 0.3))

PLATFORM_LEVEL = {
    "key": "high_platform_20cm", "category": "high_platform", "level": 20, "unit": "cm",
    "scene": "part2_off_high_platform_20cm.xml",
    "judge": {"kind": "platform", "top_z": 0.2, "x_range": [2.0, 3.25],
              "exit_x_min": 3.35, "exit_ground_tol": 0.08,
              "lane_half_width": 0.7, "hold_s": 0.2},
}


def _support(y):
    return {"FL": [1.55, y, 0.20], "FR": [2.05, y, 0.20],
            "RL": [1.55, y, 0.20], "RR": [2.05, y, 0.20]}


def _write_trace(path: Path, meta: dict, frames: list[dict]) -> None:
    with path.open("w") as fh:
        fh.write(json.dumps(meta) + "\n")
        for frame in frames:
            fh.write(json.dumps({"kind": "frame", **frame}) + "\n")


def _part2_frames(dt: float, top_steps: int, exit_steps: int) -> list[dict]:
    frames: list[dict] = []
    t = 0.0
    for _ in range(top_steps):
        support = {leg: [2.5 + dx, dy, 0.2] for leg, (dx, dy) in zip(LEGS, OFFSETS)}
        frames.append({"t": t, "base": [2.5, 0.0, 0.55], "tilt_deg": 2.0, "support": support})
        t += dt
    for _ in range(exit_steps):
        support = {leg: [3.55 + dx, dy, 0.05] for leg, (dx, dy) in zip(LEGS, OFFSETS)}
        frames.append({"t": t, "base": [3.55, 0.0, 0.55], "tilt_deg": 2.0, "support": support})
        t += dt
    return frames


class TestTraceFormat(unittest.TestCase):
    def test_contact_classification(self):
        tags, ground = classify_contacts([
            ["FL_foot", "t_stairs_y_low1"],
            ["FR_foot", "floor"],
        ])
        self.assertIn("step:s:1", tags["FL"])
        self.assertEqual(ground, {"FR"})
        self.assertEqual(tags["FR"], set())

    def test_unmapped_geom_is_ignored(self):
        tags, ground = classify_contacts([["FL_foot", "some_unknown_geom"]])
        self.assertEqual(tags["FL"], set())
        self.assertEqual(ground, set())

    def test_missing_meta_or_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"kind": "frame", "t": 0.0}) + "\n")
            with self.assertRaises(ValueError):
                read_trace(path)
            path.write_text(json.dumps({"kind": "meta", "schema": 99}) + "\n")
            with self.assertRaises(ValueError):
                read_trace(path)


class TestPart1Replay(unittest.TestCase):
    def test_high_wall_support_frames_score(self):
        frames = [
            {"t": 0.00, "base": [1.8, -5.9, 0.45], "support": _support(-5.9)},
            {"t": 0.02, "base": [1.8, -6.4, 0.45], "support": _support(-6.4)},
            {"t": 0.04, "base": [1.8, -7.2, 0.45], "support": _support(-7.2)},
            {"t": 0.06, "base": [1.8, -7.8, 0.45], "support": _support(-7.8)},
            {"t": 0.08, "base": [1.8, -8.1, 0.45], "support": _support(-8.1)},
        ]
        meta = {"kind": "meta", "schema": 1, "part": 1, "robot": "blackW"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part1.jsonl"
            _write_trace(path, meta, frames)
            report = replay_part1(path)
        self.assertEqual(report["states"]["high_wall"]["status"], "success")
        self.assertEqual(report["states"]["high_wall"]["score"], 100)
        self.assertEqual(report["trace"]["frames"], len(frames))

    def test_golden_part1_replay(self):
        report = replay_part1(FIX / "golden/part1_blackw_flat.jsonl")
        self.assertEqual(report["trace"]["frames"], 134)
        self.assertEqual(report["score"], 0.0)
        self.assertIn("height_bar", report["states"])
        self.assertEqual(report["states"]["height_bar"]["status"], "fail")


class TestPart2Replay(unittest.TestCase):
    def _replay(self, frames: list[dict], level: dict, canonical: dict | None = None) -> dict:
        meta = {"kind": "meta", "schema": 1, "part": 2, "robot": "blackW",
                "level": level, "support_frame": "wheel_lowest"}
        with tempfile.TemporaryDirectory() as tmp:
            levels = Path(tmp) / "part2_official_levels.json"
            levels.write_text(json.dumps(
                {"schema_version": 2, "levels": [canonical or level]}))
            trace = Path(tmp) / "part2.jsonl"
            _write_trace(trace, meta, frames)
            return replay_part2_v5(trace, levels_path=levels)

    def test_golden_part2_replay(self):
        report = replay_part2_v5(FIX / "golden/part2_blackw_high_platform_10cm.jsonl")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["score"], 0)
        self.assertTrue(report["reason"].startswith("fell"), report["reason"])
        self.assertEqual(report["level_key"], "high_platform_10cm")

    def test_judged_run_cadence_2ms(self):
        report = self._replay(_part2_frames(0.002, 250, 250), PLATFORM_LEVEL)
        self.assertEqual(report["status"], "success", report["reason"])
        self.assertEqual(report["score"], 50)   # 20 cm platform = 50

    def test_legacy_cadence_20ms(self):
        report = self._replay(_part2_frames(0.02, 25, 25), PLATFORM_LEVEL)
        self.assertEqual(report["status"], "success", report["reason"])
        self.assertEqual(report["score"], 50)
        self.assertAlmostEqual(report["trace"]["frame_dt_s"], 0.02, places=9)

    def test_forged_level_is_rejected(self):
        forged = json.loads(json.dumps(PLATFORM_LEVEL))
        forged["judge"]["top_z"] = 0.05   # a trace cannot lower the canonical threshold
        frames = _part2_frames(0.02, 25, 25)
        with self.assertRaises(ValueError):
            self._replay(frames, forged, canonical=PLATFORM_LEVEL)

    def test_missing_support_fails_closed(self):
        frames = [{"t": 0.0, "base": [2.5, 0.0, 0.55], "tilt_deg": 1.0}]
        report = self._replay(frames, PLATFORM_LEVEL)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["reason"], "missing_support_evidence")


if __name__ == "__main__":
    unittest.main()
