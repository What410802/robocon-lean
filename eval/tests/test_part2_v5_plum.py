"""Part 2 plum: rules-PDF geometry (6 cubes, two staggered rows, no ramps).

Geometry (``terrain/gen_plum.py``, from the rules-PDF page-4 figure):
entry platform 0.8x0.8x0.2 m at x=[1.8, 2.6]; three cubes per row at
x=2.8/3.4/4.0 (y=+0.2) and x=3.1/3.7/4.3 (y=-0.2); exit platform
0.8x0.8x0.2 m at x=[4.5, 5.3]; the robot steps off the exit platform's far edge
(0.2 m drop) to normal ground.

Success = cross the field without a wheel bottom dropping to the ground, reach
the exit platform (tightened top band), then step down to normal ground.  The
``exit_ramp_end`` manifest field keeps its historical name; with no ramp it is
the exit-platform far edge (5.3).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robocon_eval.part2 import official
from robocon_eval.part2.v5_judge import WHEEL_TOP_BAND_M, Part2OfficialJudge
from robocon_eval.trace_part2_v5 import replay_part2_v5

LEGS = ("FL", "FR", "RL", "RR")
OFFSETS = ((0.2, -0.3), (0.2, 0.3), (-0.2, -0.3), (-0.2, 0.3))

PLUM = {
    "key": "plum_1", "category": "plum", "level": 1, "unit": "bool",
    "scene": "part2_off_plum_1.xml",
    "judge": {"kind": "plum", "top_z": 0.2, "cube_x_range": [2.55, 4.55],
              "goal_x_min": 4.5, "z_margin": 0.35, "z_cross_min": 0.05,
              "lane_half_width": 0.7, "hold_s": 0.2,
              "exit_ramp_end": 5.3, "exit_ground_tol": 0.08},
}


def _frame(t: float, x: float, wheel_z: float, base_z: float = 0.55) -> dict:
    return {"t": t, "base": [x, 0.0, base_z], "tilt_deg": 2.0,
            "support": {leg: [x + dx, dy, wheel_z]
                        for leg, (dx, dy) in zip(LEGS, OFFSETS)}}


def _traverse_frames() -> list[dict]:
    """Ground approach -> entry platform -> cube field -> exit platform -> ground."""
    frames: list[dict] = []
    t = 0.0

    def add(x: float, wheel_z: float, n: int = 1, base_z: float = 0.55) -> None:
        nonlocal t
        for _ in range(n):
            frames.append(_frame(t, x, wheel_z, base_z))
            t += 0.02

    for x in (0.5, 1.0, 1.3):                      # ground approach
        add(x, 0.0)
    for x in (1.85, 2.0, 2.3, 2.55):               # step up onto the entry platform
        add(x, 0.2)
    for x in (2.7, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.4):   # cross the 6 cubes
        add(x, 0.2)
    add(4.8, 0.2, n=20)                            # reach + hold on the exit platform
    add(5.60, 0.0, n=20)                           # step off the far edge, normal ground
    return frames


def _replay(frames: list[dict], level: dict) -> dict:
    meta = {"kind": "meta", "schema": 1, "part": 2, "robot": "blackW",
            "level": level, "support_frame": "wheel_lowest"}
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "part2_official_levels.json"
        manifest.write_text(json.dumps({"schema_version": 2, "levels": [level]}))
        trace = Path(tmp) / "plum.jsonl"
        with trace.open("w") as fh:
            fh.write(json.dumps(meta) + "\n")
            for frame in frames:
                fh.write(json.dumps({"kind": "frame", **frame}) + "\n")
        return replay_part2_v5(trace, levels_path=manifest)


class TestPlumReplay(unittest.TestCase):
    def test_full_traverse_scores_200(self):
        report = _replay(_traverse_frames(), PLUM)
        self.assertEqual(report["status"], "success", report)
        self.assertEqual(report["score"], 200)
        self.assertTrue(report["detail"]["top_seen"])
        self.assertTrue(report["detail"]["exit_platform_seen"])
        self.assertTrue(report["detail"]["exited"])

    def test_stopping_on_exit_platform_is_not_success(self):
        frames = _traverse_frames()
        frames = [f for f in frames if f["base"][0] <= 4.9]
        report = _replay(frames, PLUM)
        self.assertEqual(report["status"], "fail", report)
        self.assertEqual(report["reason"], "not_descended_exit_platform")
        self.assertEqual(report["score"], 0)

    def test_wheel_touching_ground_in_field_fails(self):
        frames = _traverse_frames()
        for frame in frames:
            if 2.55 <= frame["base"][0] <= 4.55:
                for leg in LEGS:
                    frame["support"][leg][2] = 0.0
                break
        report = _replay(frames, PLUM)
        self.assertEqual(report["status"], "fail", report)
        self.assertTrue(report["reason"].startswith("dropped"), report["reason"])

    def test_ground_without_exit_platform_evidence_fails(self):
        # jump from the cube field straight to the ground: never stands on the
        # exit platform, so the terminal state must not be accepted.
        frames = [f for f in _traverse_frames() if f["base"][0] <= 4.0]
        t = max(f["t"] for f in frames) + 0.02
        for i in range(25):
            frames.append(_frame(t + i * 0.02, 5.60, 0.0))
        report = _replay(frames, PLUM)
        self.assertEqual(report["status"], "fail", report)
        self.assertEqual(report["reason"], "not_on_exit_platform")

    def test_hovering_above_exit_platform_is_not_on_surface(self):
        frames = _traverse_frames()
        for frame in frames:
            if 4.5 <= frame["base"][0] < 5.3:
                for leg in LEGS:
                    frame["support"][leg][2] = 0.2 + WHEEL_TOP_BAND_M + 0.05
        report = _replay(frames, PLUM)
        self.assertEqual(report["status"], "fail", report)
        self.assertFalse(report["detail"]["exit_platform_seen"])
        self.assertEqual(report["reason"], "not_on_exit_platform")

    def test_missing_exit_evidence_key_fails_closed(self):
        broken = json.loads(json.dumps(PLUM))
        del broken["judge"]["exit_ramp_end"]
        with self.assertRaises(KeyError):
            Part2OfficialJudge(broken)


@unittest.skipUnless(official.LEVELS_PATH.is_file(),
                     "generated Part 2 manifest not present (terrain/gen_part2_levels.py)")
class TestGeneratedManifest(unittest.TestCase):
    def test_official_20_levels(self):
        levels = official.load_levels()
        self.assertEqual(len(levels), 20)
        keys = [entry["key"] for entry in levels]
        self.assertNotIn("plum_reg_1", keys)
        expected = {"high_platform": 4, "stairs": 4, "slope": 4,
                    "high_wall": 4, "trench": 3, "plum": 1}
        counts: dict[str, int] = {}
        for entry in levels:
            counts[entry["category"]] = counts.get(entry["category"], 0) + 1
        self.assertEqual(counts, expected)

    def test_plum_entry_matches_pdf_geometry(self):
        entry = official.get_level("plum_1")
        judge = entry["judge"]
        self.assertEqual(judge["kind"], "plum")
        self.assertEqual(judge["top_z"], 0.2)
        self.assertEqual(judge["cube_x_range"], [2.55, 4.55])
        self.assertEqual(judge["goal_x_min"], 4.5)
        # no ramps: the exit-platform far edge is the terminal
        self.assertEqual(judge["exit_ramp_end"], 5.3)
        self.assertEqual(entry["obstacle_bbox"]["max"][0], 5.3)
        self.assertEqual(judge["z_cross_min"], 0.05)
        self.assertEqual(judge["exit_ground_tol"], 0.08)

    def test_safety_timeout_only(self):
        for entry in official.load_levels():
            self.assertGreaterEqual(
                float(entry.get("seconds", 0.0)), 120.0,
                f"{entry['key']}: Part 2 is untimed; seconds is a safety cap only")
            self.assertEqual(official.level_scene_path(entry).name, entry["scene"])

    def test_plum_replay_against_generated_manifest(self):
        report = _replay(_traverse_frames(), official.get_level("plum_1"))
        self.assertEqual(report["status"], "success", report)
        self.assertEqual(report["score"], 200)


if __name__ == "__main__":
    unittest.main()
