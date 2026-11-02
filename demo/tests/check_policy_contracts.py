#!/usr/bin/env python3
"""Validate every copied demo policy against its robot.yaml contract.

Run from the repo root (any Python with pyyaml, e.g. the eval env):
    python demo/tests/check_policy_contracts.py

Policy weights (policy.pt / best.pt) are local-only: a checkout without them reports
[SKIP] for the weight-dependent checks instead of failing.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # .../demo
POLICY_ROOT = ROOT / "rl_sar" / "policy"
ROBOTS = {p.parent.name: yaml.safe_load(p.read_text()) for p in (ROOT / "robots").glob("*/robot.yaml")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def payload(raw: dict) -> dict:
    if len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        return next(iter(raw.values()))
    return raw


def fail(where: str, msg: str) -> None:
    print(f"[FAIL] {where}: {msg}")
    raise SystemExit(1)


def main() -> int:
    checked = 0
    for robot, man in ROBOTS.items():
        for cfg_path in sorted((POLICY_ROOT / robot).glob("*/config.yaml")):
            where = f"{robot}/{cfg_path.parent.name}"
            cfg = payload(yaml.safe_load(cfg_path.read_text()))
            n = man["num_dofs"]
            if cfg.get("num_of_dofs", n) != n:
                fail(where, f"num_of_dofs {cfg.get('num_of_dofs')} != {n}")
            if cfg.get("joint_names") and cfg["joint_names"] != man["joint_names"]:
                fail(where, "joint_names differ from robot.yaml")
            if cfg.get("joint_mapping") and cfg["joint_mapping"] != man["joint_mapping"]:
                fail(where, "joint_mapping differs from robot.yaml")
            for key in ("torque_limits", "default_dof_pos"):
                if cfg.get(key) and len(cfg[key]) != n:
                    fail(where, f"{key} length != {n}")
            if cfg.get("action_scale") and len(cfg["action_scale"]) != n:
                fail(where, "action_scale length mismatch")
            obs = man["obs_contract"]
            if cfg.get("num_observations") and cfg["num_observations"] != obs["num_observations"]:
                fail(where, "num_observations mismatch")
            if cfg.get("observations") and list(cfg["observations"]) != list(obs["terms"]):
                fail(where, f"observation terms differ: {cfg['observations']} vs {obs['terms']}")
            hist = cfg.get("observations_history")
            if hist is not None and len(hist) != obs["history"]:
                fail(where, f"history length {len(hist)} != {obs['history']}")
            model = cfg.get("model_name")
            if not model:
                fail(where, "missing model_name")
            pt = cfg_path.parent / model
            if not pt.is_file():
                # P2-08 full archive also copies historical config-only round dirs; only dirs with
                # an actual weight are runtime policies.
                print(f"[SKIP] {where}: config-only (no {pt.name})")
                continue
            mpath = cfg_path.parent / "manifest.json"
            if mpath.is_file():
                manifest = yaml.safe_load(mpath.read_text()) if mpath.suffix in (".yaml", ".yml") else __import__("json").loads(mpath.read_text())
                if manifest.get("policy_sha256") != sha(pt):
                    fail(where, "policy sha256 does not match manifest")
            print(f"[OK] {where}: {n} dof, obs {cfg.get('num_observations', obs['num_observations'])}"
                  f"x{len(hist) if hist else obs['history']}, model {model}")
            checked += 1
    if not checked:
        print("no policies found")
        return 1
    print(f"[OK] {checked} policy contracts valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
