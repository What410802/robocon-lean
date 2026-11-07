#!/usr/bin/env python3
"""Validate every copied demo policy against its robot.yaml contract.

Run from the repo root (any Python with pyyaml, e.g. the eval env):
    python demo/tests/check_policy_contracts.py

Policy weights (policy.pt / best.pt) are local-only: a checkout without them reports
[SKIP] for the weight-dependent checks instead of failing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
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


def _camera_attrs(description_file: Path, camera_name: str) -> dict:
    """The named camera element from the robot description, as plain attributes."""
    root = ET.parse(description_file).getroot()
    for cam in root.iter("camera"):
        if cam.get("name") == camera_name:
            return dict(cam.attrib)
    return {}


def _check_depth_contract(where: str, policy_dir: Path, obs: dict, man: dict) -> None:
    """Keep depth.json, the policy config block and the robot description in sync.

    Three places describe the same camera: the policy directory's depth.json
    (drives the eval runner and the demo depth topic), robot.yaml
    obs_contract_depth.depth (declares what the deployment expects) and the
    <camera> element in the robot XML (what sim_node actually renders).  A
    mismatch here is exactly the kind of silent drift that would feed the policy
    an image from a different camera.
    """
    depth_json = policy_dir / "depth.json"
    if not depth_json.is_file():
        fail(where, "depth policy is missing depth.json")
    contract = json.loads(depth_json.read_text())
    expect = obs.get("depth") or {}
    for key in ("camera_name", "parent_body", "width", "height", "cutoff", "min_depth"):
        if contract.get(key) != expect.get(key):
            fail(where, f"depth.json {key}={contract.get(key)!r} != robot.yaml {expect.get(key)!r}")
    # The policy config repeats the camera block so the deployment loader can pick
    # it up from config.yaml alone (rl_sdk.cpp reads it there).
    config = payload(yaml.safe_load((policy_dir / "config.yaml").read_text()))
    for key in ("camera_name", "width", "height", "cutoff", "min_depth"):
        if (config.get("depth") or {}).get(key) != contract.get(key):
            fail(where, f"config.yaml depth.{key} != depth.json {key}")
    # ... and the robot description's <camera> element is what sim_node renders.
    description = ROOT / man["description_file"]
    cam = _camera_attrs(description, contract["camera_name"])
    if not cam:
        fail(where, f"robot description has no camera {contract['camera_name']!r}")
    for key in ("pos", "quat"):
        got = [float(v) for v in cam.get(key, "").split()]
        if len(got) != len(contract[key]) or any(abs(a - b) > 1e-6 for a, b in zip(got, contract[key])):
            fail(where, f"robot camera {key}={cam.get(key)!r} != depth.json {contract[key]!r}")
    if abs(float(cam.get("fovy", "nan")) - float(contract["fovy"])) > 1e-6:
        fail(where, f"robot camera fovy={cam.get('fovy')!r} != depth.json {contract['fovy']!r}")


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
            depth = cfg.get("depth") or {}
            if depth.get("enabled"):
                # Depth policies are command-free (54 x 6) and carry their camera
                # in depth.json; the robot manifest declares the matching
                # observation contract under obs_contract_depth.
                obs = man.get("obs_contract_depth") or {}
                if not obs:
                    fail(where, "depth policy but robot.yaml has no obs_contract_depth")
                _check_depth_contract(where, cfg_path.parent, obs, man)
            else:
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
