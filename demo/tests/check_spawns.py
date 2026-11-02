#!/usr/bin/env python3
"""P2-09: validate manifest-driven Part1/Part2 spawns and scene files."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def finite4(v):
    return isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def main() -> int:
    fails = []
    for robot in ("blackW", "black"):
        man = yaml.safe_load((ROOT / "robots" / robot / "robot.yaml").read_text())
        spawn = man["spawn"]["part1"]
        if not finite4(spawn):
            fails.append(f"{robot}: bad part1 spawn {spawn}")
        scene = ROOT / man["scene_roots"]["part1"] / "scene_part1.xml"
        if not scene.is_file():
            fails.append(f"{robot}: missing {scene}")
        levels_path = ROOT / man["spawn"]["part2_source"]
        if not levels_path.is_file():
            fails.append(f"{robot}: missing {levels_path}")
            continue
        levels = yaml.safe_load(levels_path.read_text()) if levels_path.suffix != ".json" else __import__("json").loads(levels_path.read_text())
        entries = levels["levels"] if isinstance(levels, dict) else levels
        for e in entries:
            sp = e.get("spawn") or e.get("start")
            if not finite4(sp):
                fails.append(f"{robot}/{e.get('key')}: bad spawn {sp}")
            if not (ROOT / man["scene_roots"]["part2"] / e["scene"]).is_file():
                fails.append(f"{robot}/{e.get('key')}: missing scene {e['scene']}")
        print(f"[OK] {robot}: part1 scene/spawn, {len(entries)} Part2 levels")
    for f in fails:
        print("[FAIL]", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
