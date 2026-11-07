#!/usr/bin/env python3
"""Fail-loud checks for robots/<robot>/robot.yaml (no ROS, no MuJoCo needed).

Run from the repo root (any Python with pyyaml, e.g. the eval env):
    python demo/tests/check_robot_manifests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # .../demo
ROBOTS = ROOT / "robots"

TERM_WIDTHS = {
    "commands": 3,
    "ang_vel_body": 3,
    "ang_vel": 3,  # black (12 DoF) names the same 3-wide term without the _body suffix
    "gravity_vec": 3,
}


def fail(robot: str, msg: str) -> None:
    print(f"[FAIL] {robot}: {msg}")
    raise SystemExit(1)


def main() -> int:
    manifests = sorted(ROBOTS.glob("*/robot.yaml"))
    if not manifests:
        print("no robot.yaml found")
        return 1
    for path in manifests:
        d = yaml.safe_load(path.read_text())
        name = d.get("name", path.parent.name)
        n = d["num_dofs"]
        for key in ("joint_names", "joint_mapping", "default_dof_pos", "torque_limits"):
            if len(d[key]) != n:
                fail(name, f"{key} length {len(d[key])} != num_dofs {n}")
        if sorted(d["joint_mapping"]) != list(range(n)):
            fail(name, "joint_mapping is not a permutation of 0..num_dofs-1")
        if d["support_kind"] not in ("foot", "wheel"):
            fail(name, f"unknown support_kind {d['support_kind']}")
        wheel_idx = d.get("wheel_indices", [])
        if d["support_kind"] == "wheel" and len(wheel_idx) != 4:
            fail(name, "wheel robot must have 4 wheel_indices")
        if d["support_kind"] == "foot" and wheel_idx:
            fail(name, "foot robot must have no wheel_indices")
        if any(i < 0 or i >= n for i in wheel_idx):
            fail(name, "wheel_indices out of range")
        act = d["action_contract"]
        if len(act["scale"]) != n or act["dim"] != n:
            fail(name, "action_contract dim/scale mismatch")
        # The ordinary contract is always present; depth-conditioned policies
        # declare a second command-free one (see demo/robots/blackW/robot.yaml).
        contracts = {"obs_contract": d.get("obs_contract"),
                     "obs_contract_depth": d.get("obs_contract_depth")}
        for contract_name, obs in contracts.items():
            if obs is None:
                if contract_name == "obs_contract":
                    fail(name, "missing obs_contract")
                continue
            total = 0
            for term in obs["terms"]:
                total += TERM_WIDTHS.get(term, n)
            if total != obs["num_observations"]:
                fail(name, f"{contract_name} terms sum {total} != num_observations {obs['num_observations']}")
            if obs.get("history", 0) < 1:
                fail(name, f"{contract_name} history must be >= 1")
            if contract_name == "obs_contract_depth":
                depth = obs.get("depth") or {}
                for key in ("camera_name", "parent_body", "width", "height", "cutoff", "min_depth"):
                    if key not in depth:
                        fail(name, f"obs_contract_depth.depth has no {key}")
                if depth["min_depth"] >= depth["cutoff"]:
                    fail(name, "obs_contract_depth min_depth must be < cutoff")
        obs = d["obs_contract"]
        desc = ROOT / d["description_file"]
        if not desc.is_file():
            fail(name, f"description_file missing: {desc}")
        for part, rel in d.get("scene_roots", {}).items():
            if not (ROOT / rel).is_dir():
                fail(name, f"scene root missing for {part}: {rel}")
        print(f"[OK] {name}: {n} dof, obs {obs['num_observations']}x{obs['history']}, "
              f"support={d['support_kind']}, desc={d['description_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
