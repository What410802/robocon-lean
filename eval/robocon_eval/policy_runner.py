"""Policy -> MuJoCo trace -> judge closed loop.

This is the *independent* eval-side runner.  It must reproduce the deployment
stack's observation/action/control contract closely enough that a model scored
here behaves the same way when the deployment stack runs it:

* joint arrays are in **policy order** (``robot.yaml:joint_names``), while the
  MuJoCo actuators may be in a different order.  ``joint_mapping[i]`` is the
  actuator index of policy joint ``i``; ignoring it silently swaps blackW's
  RL/RR legs and makes every policy fall.
* the policy runs at ``policy_hz`` (50 Hz by default) but PD is applied every
  ``control_dt`` (5 ms) with the 2 ms physics step, and the command limiter
  (gains / dq / slew / implied-torque budget) runs before PD, exactly like
  ``rl_sim`` + ``sim_node``.
* wheel joint positions are zeroed in ``dof_pos`` observations, history is
  newest-first with the first frame backfilled, and actions/observations are
  clipped using the policy config.
* the trace carries the quantities the judges consume plus enough hashes and
  versions to reproduce the trial.

It deliberately does not import ROS or deployment code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

from . import rules
from .trace import classify_contacts, replay_part1
from .trace_part2_v5 import replay_part2_v5
from .part2.official import get_level, level_scene_path

BASE = Path(__file__).resolve().parents[2]      # repo root
POLICY_ROOT = BASE / "demo/rl_sar/policy"       # demo/rl_sar/policy/<robot>/<name>/
ROBOT_ROOT = BASE / "demo/robots"               # demo/robots/<robot>/
SCENE_ROOT = BASE / "demo/scenes"               # generated Part 2 levels live under part2/generated/
LEGS = ("FL", "FR", "RL", "RR")
PHYSICS_DT = 0.002

# Optional depth-camera contract (blackW depth policies).
#
# The eval-side runner mirrors the *training* simulated front camera so that a
# legacy depth policy exported from rsl_rl's ``CNNModel`` is scored on the same
# input it was trained on.  The depth research branch itself was not migrated to
# this repo; the optional path is kept so old checkpoints stay replayable.  It is
# dormant unless a policy directory carries ``depth.json``.
#
# A depth policy is detected by a ``depth.json`` in its policy directory (or by
# passing ``depth_contract=`` to :class:`PolicyRunner` / ``--depth-contract`` to
# ``evaluate-policy``).  Blind policies never touch any of this path, so their
# traces stay byte-identical.
DEPTH_CONTRACT_NAME = "depth.json"
# Geom group used to hide robot geoms from the depth render.  It is disabled in
# the camera's ``MjvOption``; the MJCF never assigns collision geoms to it.
DEPTH_HIDDEN_GEOMGROUP = 5
# Training camera contract (body<-camera quaternion is w,x,y,z).
DEFAULT_DEPTH_CONTRACT: dict[str, Any] = {
    "camera_name": "depth_cam",
    "parent_body": "d435_link",
    "pos": [0.01, 0.0, 0.0],
    "quat": [0.5610, 0.4304, -0.4304, -0.5610],
    "fovy": 70.0,
    "width": 64,
    "height": 48,
    "min_depth": 0.10,
    "cutoff": 3.0,
    "input_layout": "1,1,H,W",
    # Training renders only geom group 0 (terrain + obstacles); the blackW
    # robot's visual meshes are group 1 and its collision geoms group 3.
    "visible_geom_groups": [0],
    "hide_robot_geoms": True,
}


class ContractError(RuntimeError):
    """The policy/robot/scene combination violates the declared contract."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _id(model, obj, name: str) -> int:
    return mujoco.mj_name2id(model, obj, name)


def _unwrap_config(data: dict) -> dict:
    """Policy configs are ``<robot>/<name>: {...}`` (occasionally one anonymous level)."""
    if isinstance(data, dict) and "num_observations" not in data and len(data) == 1:
        only = next(iter(data.values()))
        return only if isinstance(only, dict) else data
    return data


def _scene_root(manifest: dict, part: str, fallback: Path | None) -> Path | None:
    """Resolve a manifest-declared scene root (``robot.yaml:scene_roots.<part>``).

    The value is a path relative to ``demo/`` (e.g. ``robots/blackW`` or
    ``scenes/part2/generated``); absolute paths are used as-is.  Returns the
    fallback when the manifest does not declare one.
    """
    roots = manifest.get("scene_roots") or {}
    raw = roots.get(part)
    if not raw:
        return fallback
    path = Path(str(raw))
    return path if path.is_absolute() else (BASE / "demo" / path)


def _load_level(key: str) -> dict:
    try:
        return get_level(key)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def _normalize_depth_contract(raw: dict, source: Path | None = None) -> dict:
    """Validate/merge a depth contract over :data:`DEFAULT_DEPTH_CONTRACT`."""
    if not isinstance(raw, dict):
        raise ContractError("depth contract must be a JSON object")
    contract = dict(DEFAULT_DEPTH_CONTRACT)
    contract.update(raw)
    camera_name = str(contract.get("camera_name") or "")
    if not camera_name:
        raise ContractError("depth contract camera_name must be non-empty")
    parent = str(contract.get("parent_body") or "")
    if not parent:
        raise ContractError("depth contract parent_body must be non-empty")
    contract["camera_name"] = camera_name
    contract["parent_body"] = parent
    contract["pos"] = [float(v) for v in _vec(contract.get("pos"), None, 3, "depth.pos")]
    quat = _vec(contract.get("quat"), None, 4, "depth.quat")
    if not np.any(np.abs(quat) > 0.0):
        raise ContractError("depth contract quat must be non-zero")
    contract["quat"] = [float(v) for v in quat]
    fovy = float(contract.get("fovy") or 0.0)
    if not 0.0 < fovy < 180.0:
        raise ContractError(f"depth contract fovy out of range: {fovy}")
    contract["fovy"] = fovy
    for key in ("width", "height"):
        value = int(contract.get(key) or 0)
        if value <= 0:
            raise ContractError(f"depth contract {key} must be > 0")
        contract[key] = value
    near = float(contract.get("min_depth") or 0.0)
    cutoff = float(contract.get("cutoff") or 0.0)
    if near < 0.0 or cutoff <= 0.0 or near >= cutoff:
        raise ContractError(
            f"depth contract needs 0 <= min_depth < cutoff, got {near} / {cutoff}")
    contract["min_depth"] = near
    contract["cutoff"] = cutoff
    layout = str(contract.get("input_layout") or "")
    if layout != "1,1,H,W":
        raise ContractError(
            f"depth contract input_layout must be '1,1,H,W', got {layout!r}")
    groups = contract.get("visible_geom_groups")
    if groups is None:
        groups = list(DEFAULT_DEPTH_CONTRACT["visible_geom_groups"])
    groups = [int(g) for g in np.asarray(groups).reshape(-1)]
    if any(g < 0 or g >= 6 for g in groups):
        raise ContractError(f"depth contract visible_geom_groups invalid: {groups}")
    contract["visible_geom_groups"] = groups
    contract["hide_robot_geoms"] = bool(contract.get("hide_robot_geoms", True))
    if source is not None:
        contract["source"] = str(source)
        contract["source_sha256"] = _sha256(source)
    return contract


def _load_depth_contract(policy_dir: Path, explicit: Any) -> dict | None:
    """Resolve the optional depth contract for a policy.

    ``explicit`` may be ``None`` (auto-detect ``<policy_dir>/depth.json``; a
    blind policy returns ``None``), a dict (inline contract merged over the
    defaults) or a path (absolute, or relative to the policy directory) to a
    JSON contract.
    """
    source: Path | None = None
    if explicit is None:
        source = policy_dir / DEPTH_CONTRACT_NAME
        if not source.is_file():
            return None
        raw = json.loads(source.read_text())
    elif isinstance(explicit, dict):
        raw = dict(explicit)
    else:
        source = Path(explicit)
        if not source.is_absolute() and not source.is_file():
            source = policy_dir / source
        if not source.is_file():
            raise ContractError(f"depth contract missing: {source}")
        raw = json.loads(source.read_text())
    return _normalize_depth_contract(raw, source)


def _vec(value: Any, fallback: Any, length: int, name: str) -> np.ndarray:
    raw = value if value is not None else fallback
    if raw is None:
        raise ContractError(f"missing contract array {name!r}")
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size != length:
        raise ContractError(f"{name}: expected {length} values, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ContractError(f"{name}: non-finite values")
    return arr


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    awa, ax, ay, az = np.asarray(a, dtype=np.float64)
    bwa, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        awa * bwa - ax * bx - ay * by - az * bz,
        awa * bx + ax * bwa + ay * bz - az * by,
        awa * by - ax * bz + ay * bwa + az * bx,
        awa * bz + ax * by - ay * bx + az * bwa,
    ], dtype=np.float64)


class AdaptiveCommand:
    """State-feedback forward command: hold speed, center on y=0, face +x.

    The dog receives ``vx`` along its body x, and ``wz`` from current base y and
    yaw.  This mirrors what a human operator does: keep the dog on the course
    centreline and pointing forward instead of blindly holding a body-frame
    command.  The controller is stateless and deterministic, so it is usable for
    automated regression runs.
    """

    def __init__(self, vx: float = 0.5, k_y: float = 0.8, k_yaw: float = 2.0,
                 k_vy: float = 0.6, k_vheading: float = 0.0,
                 vy_gain: float = 0.5, max_vy: float = 0.2,
                 max_wz: float = 0.7, center_y: float = 0.0):
        self.vx = float(vx)
        self.k_y = float(k_y)
        self.k_yaw = float(k_yaw)
        self.k_vy = float(k_vy)
        self.k_vheading = float(k_vheading)
        self.vy_gain = float(vy_gain)
        self.max_vy = float(max_vy)
        self.max_wz = float(max_wz)
        self.center_y = float(center_y)

    def __call__(self, runner, step: int, nominal: np.ndarray) -> np.ndarray:
        qpos = np.asarray(runner.data.qpos, dtype=np.float64)
        qvel = np.asarray(runner.data.qvel, dtype=np.float64)
        y = float(qpos[1])
        yaw = _yaw_from_quat(qpos[3:7])
        # body-frame lateral velocity; use it to damp drift, not just position.
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx_b = cy * float(qvel[0]) + sy * float(qvel[1])
        vy_b = -sy * float(qvel[0]) + cy * float(qvel[1])
        # Current travel direction in the body frame.  A dog can have near-zero
        # base yaw while crabbing (velocity heading ~10-20 deg); feed that error
        # back as well, otherwise yaw-only correction leaves wz near zero.
        heading_err = math.atan2(vy_b, max(abs(vx_b), 0.2))
        err_y = y - self.center_y
        vy = float(np.clip(-self.vy_gain * err_y, -self.max_vy, self.max_vy))
        wz = (-self.k_y * err_y - self.k_yaw * yaw
              - self.k_vy * vy_b - self.k_vheading * heading_err)
        wz = float(np.clip(wz, -self.max_wz, self.max_wz))
        vx = self.vx if self.vx is not None else float(nominal[0])
        return np.asarray((vx, vy, wz), dtype=np.float64)

    def as_dict(self) -> dict:
        return {
            "kind": "adaptive_heading_centerline_vy",
            "vx": self.vx, "k_y": self.k_y, "k_yaw": self.k_yaw,
            "k_vy": self.k_vy, "k_vheading": self.k_vheading,
            "vy_gain": self.vy_gain, "max_vy": self.max_vy,
            "max_wz": self.max_wz, "center_y": self.center_y,
        }


def command_limit(
    q_des: np.ndarray,
    dq_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_ff: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    q_des_prev: np.ndarray | None,
    limits: dict[str, float],
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Port of ``demo/rl_sar/include/command_limit.hpp``.

    Applied once per 5 ms control cycle, in the same order as the C++ code:
    gains, dq_des, slew, implied-torque budget.  Returns copies; the caller
    stores ``q_des`` as the previous target for the next cycle.
    """
    q_des = np.array(q_des, dtype=np.float64, copy=True)
    dq_des = np.array(dq_des, dtype=np.float64, copy=True)
    kp = np.array(kp, dtype=np.float64, copy=True)
    kd = np.array(kd, dtype=np.float64, copy=True)
    tau_ff = np.array(tau_ff, dtype=np.float64, copy=True)
    counts = {"gains": 0, "dq_des": 0, "slew": 0, "torque": 0}
    tau_cap = float(limits["tau"])
    kp_max = float(limits["kp_max"])
    kd_max = float(limits["kd_max"])
    dq_max = float(limits["dq_max"])
    slew = float(limits["slew"])

    for i in range(q_des.size):
        if kp[i] > kp_max:
            kp[i] = kp_max
            counts["gains"] += 1
        if kd[i] > kd_max:
            kd[i] = kd_max
            counts["gains"] += 1
        kp[i] = max(0.0, kp[i])
        kd[i] = max(0.0, kd[i])

        if abs(dq_des[i]) > dq_max:
            dq_des[i] = math.copysign(dq_max, dq_des[i])
            counts["dq_des"] += 1

        if slew > 0.0 and q_des_prev is not None and i < q_des_prev.size:
            step = slew * dt
            lo = q_des_prev[i] - step
            hi = q_des_prev[i] + step
            if q_des[i] < lo or q_des[i] > hi:
                q_des[i] = min(hi, max(lo, q_des[i]))
                counts["slew"] += 1

        touched = False
        if abs(tau_ff[i]) > tau_cap:
            tau_ff[i] = math.copysign(tau_cap, tau_ff[i])
            touched = True
        budget = tau_cap - abs(tau_ff[i])
        if budget <= 0.0:
            if q_des[i] != q[i]:
                q_des[i] = q[i]
                touched = True
            if dq_des[i] != 0.0:
                dq_des[i] = 0.0
                touched = True
        else:
            pd_budget = budget - kd[i] * abs(dq_des[i] - dq[i])
            if pd_budget <= 0.0:
                if dq_des[i] != dq[i]:
                    dq_des[i] = dq[i]
                    touched = True
                if q_des[i] != q[i]:
                    q_des[i] = q[i]
                    touched = True
            elif kp[i] > 0.0:
                allowed = pd_budget / kp[i]
                lo = q[i] - allowed
                hi = q[i] + allowed
                if q_des[i] < lo or q_des[i] > hi:
                    q_des[i] = min(hi, max(lo, q_des[i]))
                    touched = True
        if touched:
            counts["torque"] += 1
    return q_des, dq_des, kp, kd, tau_ff, counts


class PolicyRunner:
    """One policy on one robot in one MuJoCo scene, using the deployment contract."""

    def __init__(
        self,
        robot: str,
        part: int,
        policy: str,
        level: str | None = None,
        scene: Path | None = None,
        *,
        steps: int | None = None,
        duration_s: float | None = None,
        command: tuple[float, float, float] | None = None,
        settle_s: float = 1.0,
        seed: int = 0,
        profile: str = "diagnostic",
        depth_contract: dict | str | Path | None = None,
    ) -> None:
        self.robot = robot
        self.part = int(part)
        self.policy_name = policy
        self.profile = profile
        self.seed = int(seed)
        self.steps_override = steps
        self.settle_s = max(0.0, float(settle_s))
        self.warnings: list[str] = []

        manifest_path = ROBOT_ROOT / robot / "robot.yaml"
        if not manifest_path.is_file():
            raise ContractError(f"robot manifest missing: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = yaml.safe_load(manifest_path.read_text()) or {}
        self.robot_sha256 = _sha256(manifest_path)
        if int(self.manifest.get("num_dofs", -1)) <= 0:
            raise ContractError(f"{manifest_path}: missing num_dofs")

        # ``policy`` is normally a name under policy/<robot>/, but a full
        # filesystem directory (exported depth checkpoint, feasibility dummy)
        # that carries config.yaml is accepted as-is.
        named_dir = POLICY_ROOT / robot / policy
        direct_dir = Path(policy)
        if ((direct_dir / "config.yaml").is_file()
                and (not named_dir.is_dir() or direct_dir.resolve() != named_dir.resolve())):
            policy_dir = direct_dir
        else:
            policy_dir = named_dir
        cfg_path = policy_dir / "config.yaml"
        if not cfg_path.is_file():
            raise ContractError(f"policy config missing: {cfg_path}")
        self.policy_dir = policy_dir
        self.cfg_path = cfg_path
        self.config_sha256 = _sha256(cfg_path)
        raw_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        self.policy_cfg = _unwrap_config(raw_cfg)
        base_path = POLICY_ROOT / robot / "base.yaml"
        base_cfg: dict = {}
        if base_path.is_file():
            raw_base = yaml.safe_load(base_path.read_text()) or {}
            base_cfg = raw_base.get(robot) or next(iter(raw_base.values()), {}) or {}
        # The policy config is the deployment-side source of truth; base.yaml
        # supplies only fields the config may omit (dt/decimation, fixed gains).
        self.cfg: dict = {**base_cfg, **self.policy_cfg}

        model_name = str(self.cfg.get("model_name") or "policy.pt")
        policy_path = policy_dir / model_name
        if not policy_path.is_file():
            fallback = policy_dir / "best.pt"
            if fallback.is_file():
                policy_path = fallback
            else:
                raise ContractError(f"policy weight missing: {policy_dir}")
        self.policy_path = policy_path
        self.policy_sha256 = _sha256(policy_path)

        if self.part == 2:
            if robot != "blackW":
                raise ContractError(
                    "Part2 official scenes and the deployment launch are blackW-only; "
                    "generate black Part2 scene variants before evaluating black/part2"
                )
            if not level:
                raise ContractError("Part2 needs --level")
            self.level = _load_level(level)
            self.level_key = level
            part2_root = _scene_root(self.manifest, "part2", None)
            if scene is not None:
                scene_path = scene
            elif part2_root is not None:
                scene_path = part2_root / str(self.level["scene"])
            else:
                scene_path = level_scene_path(self.level)
            spawn = (self.level.get("spawn_default_pose")
                     or self.level.get("start") or self.level.get("spawn")
                     or (0.0, 0.0, 0.5, 0.0))
            self.command = np.asarray(
                self.level.get("command") or (0.5, 0.0, 0.0), dtype=np.float64)
            # Part 2 is untimed; `seconds` in the manifest is only a safety cap.
            self.duration_s = float(self.level.get("seconds") or rules.PART2_SAFETY_TIMEOUT_S)
        else:
            self.level = None
            self.level_key = None
            # Scene root is manifest-configurable (``robot.yaml:scene_roots.part1``,
            # relative to demo/), defaulting to demo/robots/<robot>/.
            part1_root = _scene_root(self.manifest, "part1", ROBOT_ROOT / robot)
            scene_path = scene or (part1_root / "scene_part1.xml")
            spawn = self.manifest.get("spawn", {}).get("part1") or (0.0, 0.0, 0.5, 0.0)
            self.command = np.asarray((0.3, 0.0, 0.0), dtype=np.float64)
            self.duration_s = 10.0
        if command is not None:
            self.command = np.asarray(command, dtype=np.float64)
        if self.command.size != 3 or not np.isfinite(self.command).all():
            raise ContractError(f"command must be a finite 3-vector, got {self.command!r}")
        if duration_s is not None:
            self.duration_s = float(duration_s)
        if self.steps_override is not None:
            self.duration_s = None
        self.spawn = np.asarray(spawn, dtype=np.float64).reshape(-1)
        if self.spawn.size != 4:
            raise ContractError(f"spawn must be [x,y,z,yaw], got {self.spawn!r}")
        self.scene = Path(scene_path)
        if not self.scene.is_file():
            raise ContractError(f"scene missing: {self.scene}")
        self.scene_sha256 = _sha256(self.scene)

        # -- model / joint contract -----------------------------------------
        # Depth policies get the training camera injected *before* compile.  The
        # blind path keeps the exact original ``from_xml_path`` call so its model
        # and traces stay byte-identical.
        self.depth_contract = _load_depth_contract(policy_dir, depth_contract)
        self.depth_camera_id = -1
        if self.depth_contract is None:
            self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        else:
            self.model = self._compile_depth_scene()
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        n = int(self.manifest["num_dofs"])
        if self.model.nu != n:
            raise ContractError(f"{robot}: model nu={self.model.nu} != robot.yaml num_dofs={n}")
        self.joint_names = list(self.manifest.get("joint_names") or [])
        if len(self.joint_names) != n:
            raise ContractError(f"{self.manifest_path}: joint_names length != num_dofs")
        # The declared ``joint_mapping`` maps policy order to the *ROS message*
        # order used by sim_node, which is not necessarily the MJCF actuator
        # order (black's MJCF is FL/FR/RL/RR, while its motor message is
        # FL/FR/RR/RL).  The eval runner bypasses ROS, so derive the model
        # mapping by joint name; keep the declared mapping only for contract
        # validation and traceability.
        self.contract_mapping = np.asarray(
            self.cfg.get("joint_mapping") or self.manifest.get("joint_mapping"), dtype=int
        ).reshape(-1)
        if (self.contract_mapping.size != n
                or sorted(self.contract_mapping.tolist()) != list(range(n))):
            raise ContractError(
                f"{robot}: joint_mapping is not a permutation of 0..{n - 1}: "
                f"{self.contract_mapping.tolist()}"
            )
        manifest_map = np.asarray(self.manifest.get("joint_mapping"), dtype=int).reshape(-1)
        if manifest_map.size and not np.array_equal(manifest_map, self.contract_mapping):
            raise ContractError(
                f"{robot}: policy config joint_mapping {self.contract_mapping.tolist()} != "
                f"robot.yaml {manifest_map.tolist()}"
            )
        self.joint_ids = np.empty(n, dtype=int)
        self.mapping = np.empty(n, dtype=int)
        for i, name in enumerate(self.joint_names):
            jid = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ContractError(f"{robot}: model has no joint {name!r}")
            hits = [a for a in range(self.model.nu)
                    if int(self.model.actuator_trnid[a, 0]) == jid]
            if len(hits) != 1:
                raise ContractError(
                    f"{robot}: expected exactly one actuator for joint {name!r}, "
                    f"found {len(hits)}"
                )
            self.joint_ids[i] = jid
            self.mapping[i] = hits[0]
        if sorted(self.mapping.tolist()) != list(range(n)):
            raise ContractError(f"{robot}: model joint/actuator mapping is not a permutation")
        self.act_ids = self.mapping.copy()
        self.qpos_adr = np.asarray([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_adr = np.asarray([self.model.jnt_dofadr[j] for j in self.joint_ids])

        # -- policy arrays (policy order) ------------------------------------
        self.q_default = _vec(self.cfg.get("default_dof_pos"),
                              self.manifest.get("default_dof_pos"), n, "default_dof_pos")
        manifest_scale = self.manifest.get("action_contract", {}).get("scale")
        self.action_scale = _vec(self.cfg.get("action_scale"), manifest_scale, n, "action_scale")
        manifest_scale_arr = np.asarray(manifest_scale, dtype=np.float64).reshape(-1) \
            if manifest_scale is not None else None
        if manifest_scale_arr is not None and not np.allclose(manifest_scale_arr, self.action_scale):
            self.warnings.append(
                "policy action_scale differs from robot.yaml action_contract.scale "
                f"(using policy config as deployment does): {self.action_scale.tolist()} vs "
                f"{manifest_scale_arr.tolist()}"
            )
        self.kp = _vec(self.cfg.get("rl_kp"), None, n, "rl_kp")
        self.kd = _vec(self.cfg.get("rl_kd"), None, n, "rl_kd")
        self.fixed_kp = _vec(self.cfg.get("fixed_kp"),
                             self.manifest.get("action_contract", {}).get("fixed_kp"), n,
                             "fixed_kp")
        self.fixed_kd = _vec(self.cfg.get("fixed_kd"),
                             self.manifest.get("action_contract", {}).get("fixed_kd"), n,
                             "fixed_kd")
        config_torque = _vec(self.cfg.get("torque_limits"),
                             self.manifest.get("torque_limits"), n, "torque_limits")
        model_limit_act = np.abs(
            np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float64))
        model_limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        if model_limited.all() and np.isfinite(model_limit_act).all() and np.all(model_limit_act > 0):
            # sim_node uses the MJCF motor ctrlrange as the torque limit, so the
            # eval-side PD must clamp to the same value to be comparable.  The
            # model array is in actuator order; policy arrays are in policy order.
            model_limit = model_limit_act[self.mapping]
            self.torque_limits = model_limit
            if not np.allclose(model_limit, config_torque):
                self.warnings.append(
                    "policy torque_limits differ from MJCF actuator ctrlrange; using MJCF "
                    f"limits {model_limit.tolist()} (config {config_torque.tolist()})"
                )
        else:
            self.torque_limits = config_torque
        self.config_torque_limits = config_torque

        self.wheel_indices = np.asarray(
            self.cfg.get("wheel_indices") or self.manifest.get("wheel_indices") or [], dtype=int
        ).reshape(-1)
        if np.any(self.wheel_indices < 0) or np.any(self.wheel_indices >= n):
            raise ContractError(f"{robot}: wheel_indices out of range: {self.wheel_indices.tolist()}")
        self.zero_wheel_dof_pos = bool(
            self.manifest.get("obs_contract", {}).get("wheel_dof_pos_zeroed",
                                                      self.wheel_indices.size > 0)
        )
        self.wheel_radius = float(self.cfg.get("wheel_radius") or
                                  (0.103 if self.manifest.get("support_kind") == "wheel" else 0.0))
        # Locked-wheel (12-DoF point-foot) policies hold the wheel joints at
        # q=0 with position PD; the policy's 4 wheel outputs are ignored.  The
        # deploy config carries ``lock_wheels: true`` plus non-zero wheel
        # rl_kp/rl_kd (the ordinary wheel mode uses rl_kp=0 / velocity targets).
        self.lock_wheels = bool(self.cfg.get("lock_wheels", False))

        # -- observation contract -------------------------------------------
        self.obs_terms = list(self.cfg.get("observations") or
                              self.manifest.get("obs_contract", {}).get("terms") or [])
        if not self.obs_terms:
            raise ContractError(f"{robot}: no observation terms")
        self.history_indices = self.cfg.get("observations_history")
        if self.history_indices is not None and len(self.history_indices) > 0:
            # PolicyRunner consumes newest-first only; configs must be [0..N-1].
            if list(self.history_indices) != list(range(len(self.history_indices))):
                raise ContractError(
                    f"{robot}/{policy}: observations_history must be [0..N-1], "
                    f"got {self.history_indices}"
                )
        self.use_history = self.history_indices is not None and len(self.history_indices) > 0
        self.history_len = len(self.history_indices) if self.use_history else 0
        scales = self.manifest.get("obs_contract", {}).get("scales", {})
        self.cmd_scale = _vec(self.cfg.get("commands_scale"), scales.get("commands"), 3,
                              "commands_scale")
        self.ang_vel_scale = float(self.cfg.get("ang_vel_scale") or
                                   scales.get("ang_vel_body") or 0.25)
        self.dof_pos_scale = float(self.cfg.get("dof_pos_scale") or scales.get("dof_pos") or 1.0)
        self.dof_vel_scale = float(self.cfg.get("dof_vel_scale") or scales.get("dof_vel") or 0.05)
        self.actions_scale = float(self.cfg.get("actions_scale") or scales.get("actions") or 1.0)
        self.gravity_scale = float(self.cfg.get("gravity_vec_scale") or
                                   scales.get("gravity_vec") or 1.0)
        self.clip_obs = float(self.cfg.get("clip_obs") or
                              self.manifest.get("obs_contract", {}).get("clip") or 100.0)
        self.num_observations = int(self.cfg.get("num_observations") or
                                    self.manifest.get("obs_contract", {}).get(
                                        "num_observations") or 0)
        self._check_obs_dim()
        lower = self.cfg.get("clip_actions_lower")
        upper = self.cfg.get("clip_actions_upper")
        self.action_lower = (np.asarray(lower, dtype=np.float64).reshape(-1)
                             if lower is not None else None)
        self.action_upper = (np.asarray(upper, dtype=np.float64).reshape(-1)
                             if upper is not None else None)
        for arr, name in ((self.action_lower, "clip_actions_lower"),
                          (self.action_upper, "clip_actions_upper")):
            if arr is not None and arr.size != n:
                raise ContractError(f"{name}: expected {n} values, got {arr.size}")

        # -- frequencies ------------------------------------------------------
        control_dt = float(self.cfg.get("dt") or 0.005)
        decimation = int(self.cfg.get("decimation") or 4)
        control_hz = self.cfg.get("control_hz")
        policy_hz = self.cfg.get("policy_hz")
        if control_hz:
            control_dt = 1.0 / float(control_hz)
        if policy_hz:
            policy_dt = 1.0 / float(policy_hz)
        else:
            policy_dt = control_dt * decimation
        if control_dt <= 0.0 or policy_dt <= 0.0:
            raise ContractError(f"{robot}: non-positive control/policy period")
        if abs(policy_dt / PHYSICS_DT - round(policy_dt / PHYSICS_DT)) > 1e-6:
            raise ContractError(
                f"{robot}: policy_dt={policy_dt} is not an integer multiple of "
                f"physics_dt={PHYSICS_DT}"
            )
        self.control_dt = control_dt
        self.policy_dt = policy_dt
        # rl_sim applies the limiter every 5 ms and sim_node recomputes PD every
        # 2 ms.  Applying the limiter once per 20 ms policy period gives the same
        # end-of-window target (the target is constant between inferences), and
        # the 10 PD steps below are exactly the sim_node steps in that window.
        self.substeps = int(round(policy_dt / PHYSICS_DT))

        # -- command limiter --------------------------------------------------
        env = self.manifest.get("command_envelope", {}) or {}
        self.limits = {
            "tau": float(self.cfg.get("command_tau_limit") or env.get("tau_limit") or 33.5),
            "kp_max": float(self.cfg.get("command_kp_max") or env.get("kp_max") or 80.0),
            "kd_max": float(self.cfg.get("command_kd_max") or env.get("kd_max") or 10.0),
            "dq_max": float(self.cfg.get("command_dq_max") or env.get("dq_max") or 20.0),
            "slew": float(self.cfg.get("command_slew") or env.get("slew") or 60.0),
        }

        # -- bodies / sensors -------------------------------------------------
        self.trunk = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        if self.trunk < 0:
            raise ContractError(f"{robot}: model has no 'trunk' body")
        self.imu_site = _id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.support_bodies: dict[str, int] = {}
        for leg in LEGS:
            bid = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
            if bid < 0:
                raise ContractError(f"{robot}: model has no '{leg}_foot' body")
            self.support_bodies[leg] = bid
        self.bar_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "height_bar_beam_body")
        self.wheel_geoms: dict[str, int] = {}
        for leg in LEGS:
            bid = self.support_bodies[leg]
            hits = [g for g in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[g]) == bid
                    and int(self.model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)]
            if len(hits) == 1:
                self.wheel_geoms[leg] = hits[0]
            elif self.manifest.get("support_kind") == "wheel":
                self.warnings.append(f"{leg}: expected one cylinder wheel geom, found {len(hits)}")
        self.support_frame = "wheel_lowest" if len(self.wheel_geoms) == 4 else "body_origin"
        self.spawn_offset = {"dy": 0.0, "dyaw_deg": 0.0}
        self._gyro_adr = self._sensor_adr("imu_gyro")
        self._quat_adr = self._sensor_adr("imu_quat")

        # -- depth camera -----------------------------------------------------
        self.depth_renderer = None
        self.depth_render_count = 0
        self.depth_width = 0
        self.depth_height = 0
        if self.depth_contract is not None:
            self._setup_depth_renderer()

        # -- policy -----------------------------------------------------------
        torch.set_num_threads(1)
        try:
            self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        except Exception as exc:  # noqa: BLE001 - make the failure actionable
            raise ContractError(f"cannot load TorchScript policy {policy_path}: {exc}") from exc
        self.qpos0 = np.array(self.model.qpos0, dtype=np.float64)
        self.reset()

    # -- depth camera --------------------------------------------------------
    def _compile_depth_scene(self) -> mujoco.MjModel:
        """Add the training depth camera to the scene, then compile.

        Runs only for depth policies; the blind path uses
        ``MjModel.from_xml_path`` untouched.  An existing camera with the contract
        name is reused instead of duplicating it.
        """
        contract = self.depth_contract
        assert contract is not None
        spec = mujoco.MjSpec.from_file(str(self.scene))
        body = spec.body(contract["parent_body"])
        if body is None:
            raise ContractError(
                f"{self.robot}: depth camera parent body {contract['parent_body']!r} "
                f"is missing from {self.scene}"
            )
        camera_name = contract["camera_name"]
        if any(c.name == camera_name for c in spec.cameras):
            self.warnings.append(
                f"scene already has camera {camera_name!r}; using the scene camera "
                "instead of injecting the training contract camera"
            )
        else:
            body.add_camera(
                name=camera_name,
                pos=tuple(contract["pos"]),
                quat=tuple(contract["quat"]),
                fovy=float(contract["fovy"]),
            )
        try:
            model = spec.compile()
        except Exception as exc:  # noqa: BLE001 - make the failure actionable
            raise ContractError(
                f"{self.robot}: cannot compile scene with depth camera: {exc}") from exc
        camera_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
        if camera_id < 0:
            raise ContractError(
                f"{self.robot}: depth camera {camera_name!r} missing after compile")
        self.depth_camera_id = camera_id
        return model

    def _robot_body_ids(self) -> set[int]:
        """Body ids of the robot: the trunk and all of its descendants."""
        ids: set[int] = set()
        for bid in range(self.model.nbody):
            parent = bid
            while parent != 0:
                if parent == self.trunk:
                    ids.add(bid)
                    break
                parent = int(self.model.body_parentid[parent])
        return ids

    def _setup_depth_renderer(self) -> None:
        """Create one offscreen depth renderer with the training camera contract.

        Robot geoms are moved to a disabled geom group and only the contract's
        visible groups are rendered, matching the training camera's
        ``enabled_geom_groups=(0,)`` (terrain/obstacles; no self-occlusion).
        """
        contract = self.depth_contract
        assert contract is not None
        if contract["hide_robot_geoms"]:
            robot_bodies = self._robot_body_ids()
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) in robot_bodies:
                    self.model.geom_group[gid] = DEPTH_HIDDEN_GEOMGROUP
        self.depth_option = mujoco.MjvOption()
        self.depth_option.geomgroup[:] = 0
        for group in contract["visible_geom_groups"]:
            self.depth_option.geomgroup[int(group)] = 1
        self.depth_option.geomgroup[DEPTH_HIDDEN_GEOMGROUP] = 0
        self.depth_width = int(contract["width"])
        self.depth_height = int(contract["height"])
        try:
            renderer = mujoco.Renderer(
                self.model, height=self.depth_height, width=self.depth_width)
            renderer.enable_depth_rendering()
        except Exception as exc:  # noqa: BLE001 - headless backend problems
            raise ContractError(
                "cannot create an offscreen depth renderer (set MUJOCO_GL=egl and "
                f"check the OpenGL/EGL driver): {exc}") from exc
        self.depth_renderer = renderer

    def render_depth(self) -> np.ndarray:
        """Render one normalized depth frame ``(1, 1, H, W)`` float32 in [0, 1].

        Mirrors ``camera_depth()`` in training: no-hit/sky pixels become the
        cutoff, the near clamp is ``min_depth`` and the image is divided by the
        cutoff.  The actor receives this as ``policy(obs_1d, [depth])``.
        """
        contract = self.depth_contract
        if contract is None or self.depth_renderer is None:
            raise ContractError(f"{self.robot}: depth rendering is not enabled")
        self.depth_renderer.update_scene(
            self.data, camera=self.depth_camera_id, scene_option=self.depth_option)
        raw = np.asarray(self.depth_renderer.render(), dtype=np.float32)
        cutoff = float(contract["cutoff"])
        near = float(contract["min_depth"])
        depth = np.where(np.isfinite(raw) & (raw > 0.0), raw, cutoff)
        depth = np.clip(depth, near, cutoff) / cutoff
        self.depth_render_count += 1
        return depth[None, None, :, :]

    def depth_meta(self) -> dict | None:
        """Contract/camera description stored in the trace meta for a depth policy."""
        contract = self.depth_contract
        if contract is None:
            return None
        return {
            "enabled": True,
            "camera_name": contract["camera_name"],
            "camera_id": int(self.depth_camera_id),
            "parent_body": contract["parent_body"],
            "pos": list(contract["pos"]),
            "quat": list(contract["quat"]),
            "quat_convention": "wxyz, body<-camera",
            "fovy": contract["fovy"],
            "width": self.depth_width,
            "height": self.depth_height,
            "min_depth": contract["min_depth"],
            "cutoff": contract["cutoff"],
            "normalization": "clamp(depth, min_depth, cutoff) / cutoff",
            "input_layout": contract["input_layout"],
            "visible_geom_groups": list(contract["visible_geom_groups"]),
            "hide_robot_geoms": bool(contract["hide_robot_geoms"]),
            "policy_call": "policy(obs_1d, [depth])",
            "contract": contract,
        }

    def close(self) -> None:
        """Release the offscreen renderer (no-op for blind policies).

        Closing explicitly avoids EGL-teardown noise from garbage collection at
        interpreter shutdown; ``run()`` calls it after the judge replay.
        """
        renderer = getattr(self, "depth_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self.depth_renderer = None

    # -- contract helpers ----------------------------------------------------
    def _sensor_adr(self, name: str) -> int | None:
        sid = _id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return None
        return int(self.model.sensor_adr[sid])

    def _term_dim(self, term: str) -> int:
        if term == "commands":
            return 3
        if term in ("ang_vel", "ang_vel_body", "ang_vel_world", "gravity_vec"):
            return 3
        if term in ("dof_pos", "dof_vel", "actions"):
            return int(self.manifest["num_dofs"])
        if term == "lin_vel":
            return 3
        raise ContractError(f"{self.robot}: unsupported observation term {term!r}")

    def _check_obs_dim(self) -> None:
        dim = sum(self._term_dim(term) for term in self.obs_terms)
        if self.num_observations and dim != self.num_observations:
            raise ContractError(
                f"{self.robot}: observation terms sum to {dim}, config says "
                f"num_observations={self.num_observations}"
            )
        self.obs_dim = dim

    def _gyro(self) -> np.ndarray:
        if self._gyro_adr is not None:
            return np.asarray(self.data.sensordata[self._gyro_adr:self._gyro_adr + 3],
                              dtype=np.float64)
        # Free-joint qvel[3:6] is the trunk angular velocity in the body frame
        # for these models; this fallback is only used if a scene lacks the
        # imu_gyro sensor (all current Part1/Part2 scenes have it).
        if self.data.qvel.size >= 6:
            return np.asarray(self.data.qvel[3:6], dtype=np.float64)
        return np.zeros(3)

    def _orientation_mat(self) -> np.ndarray:
        if self.imu_site >= 0:
            return np.array(self.data.site_xmat[self.imu_site], dtype=np.float64).reshape(3, 3)
        return np.array(self.data.xmat[self.trunk], dtype=np.float64).reshape(3, 3)

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        d = self.data
        d.qpos[:] = self.qpos0
        d.qvel[:] = 0.0
        d.qpos[0:3] = self.spawn[:3]
        yaw = float(self.spawn[3])
        d.qpos[3:7] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        d.qpos[self.qpos_adr] = self.q_default
        mujoco.mj_forward(self.model, d)
        self.history: list[np.ndarray] = []
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float64)
        self.prev_q_des: np.ndarray | None = None
        self.t0 = float(d.time)
        self.sim_time = 0.0
        self.settle_drift = None

    def mark_start(self) -> None:
        self.t0 = float(self.data.time)

    def _measure(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array(self.data.qpos[self.qpos_adr], dtype=np.float64),
                np.array(self.data.qvel[self.qvel_adr], dtype=np.float64))

    def settle(self, seconds: float | None = None) -> dict:
        """Hold the default stance with the fixed gains before the policy starts.

        The deployment stack enters RL after the get-up FSM has put the robot in
        its default stance.  A short fixed-gain settle reproduces that entry
        state without importing the FSM, and it removes the ~5 cm drop that a
        raw ``qpos0 + default joints`` reset would otherwise inject into the
        first policy observation.
        """
        seconds = self.settle_s if seconds is None else max(0.0, float(seconds))
        start = np.array(self.data.qpos[0:3], dtype=np.float64)
        cycles = int(round(seconds / self.control_dt))
        for cycle in range(cycles):
            q_meas, dq_meas = self._measure()
            q_des, dq_des, kp, kd, tau_ff, _ = command_limit(
                self.q_default, np.zeros_like(self.q_default),
                self.fixed_kp, self.fixed_kd, np.zeros_like(self.q_default),
                q_meas, dq_meas, self.prev_q_des, self.limits, self.control_dt,
            )
            self.prev_q_des = q_des.copy()
            # 5 ms control / 2 ms physics: alternate 2 and 3 physics steps.
            n_steps = 2 if cycle % 2 == 0 else 3
            for _ in range(n_steps):
                self._pd_step(q_des, dq_des, kp, kd, tau_ff)
        self.settle_drift = [round(float(v), 6) for v in
                             (np.array(self.data.qpos[0:3]) - start)]
        self.prev_q_des = None
        self.history = []
        self.last_action[:] = 0.0
        self.mark_start()
        return {"seconds": seconds, "drift": self.settle_drift}

    def _pd_step(
        self,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        q, dq = self._measure()
        tau = (self._kp_vec(kp) * (q_des - q) + kd * (dq_des - dq)
               + (tau_ff if tau_ff is not None else 0.0))
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.act_ids] = tau
        mujoco.mj_step(self.model, self.data)
        return tau

    @staticmethod
    def _kp_vec(kp: np.ndarray) -> np.ndarray:
        return np.asarray(kp, dtype=np.float64)

    # -- observation / action ------------------------------------------------
    def _obs(self, command: np.ndarray) -> np.ndarray:
        q, dq = self._measure()
        R = self._orientation_mat()
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = self._gyro()
        pieces: list[np.ndarray] = []
        for term in self.obs_terms:
            if term == "commands":
                pieces.append(np.asarray(command, dtype=np.float64) * self.cmd_scale)
            elif term in ("ang_vel", "ang_vel_body"):
                pieces.append(gyro * self.ang_vel_scale)
            elif term == "ang_vel_world":
                pieces.append(gyro * self.ang_vel_scale)
            elif term == "gravity_vec":
                pieces.append(gravity * self.gravity_scale)
            elif term == "lin_vel":
                # rl_sim initializes obs.lin_vel to zero and never fills it for the
                # feedback-driven path; match that rather than inventing a signal.
                pieces.append(np.zeros(3))
            elif term == "dof_pos":
                rel = q - self.q_default
                if self.zero_wheel_dof_pos and self.wheel_indices.size:
                    rel[self.wheel_indices] = 0.0
                pieces.append(rel * self.dof_pos_scale)
            elif term == "dof_vel":
                pieces.append(dq * self.dof_vel_scale)
            elif term == "actions":
                pieces.append(self.last_action * self.actions_scale)
            else:
                raise ContractError(f"{self.robot}: unsupported observation term {term!r}")
        obs = np.concatenate(pieces).astype(np.float64)
        return np.clip(obs, -self.clip_obs, self.clip_obs)

    def _infer(self, command: np.ndarray) -> np.ndarray:
        obs = self._obs(command)
        if self.use_history:
            if not self.history:
                self.history = [obs.copy() for _ in range(self.history_len)]
            else:
                self.history.append(obs.copy())
                self.history = self.history[-self.history_len:]
            policy_input = np.concatenate(list(reversed(self.history))).astype(np.float32)
        else:
            policy_input = obs.astype(np.float32)
        with torch.no_grad():
            if self.depth_contract is not None:
                # rsl_rl CNNModel export contract: proprio 1-D tensor plus a
                # one-element list holding the (1, 1, H, W) depth image.
                depth = torch.from_numpy(self.render_depth())
                out = self.policy(torch.from_numpy(policy_input).unsqueeze(0), [depth])
            else:
                out = self.policy(torch.from_numpy(policy_input).unsqueeze(0))
        action = out.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if action.size != self.manifest["num_dofs"]:
            raise ContractError(
                f"{self.robot}: policy output {action.size} != num_dofs "
                f"{self.manifest['num_dofs']}"
            )
        if not np.isfinite(action).all():
            self.warnings.append("non-finite policy action replaced with zeros")
            action = np.zeros_like(action)
        if self.action_lower is not None and self.action_upper is not None:
            action = np.clip(action, self.action_lower, self.action_upper)
        self.last_action = action.copy()
        return action

    def _targets(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = action * self.action_scale
        is_wheel = np.zeros(scaled.size, dtype=bool)
        is_wheel[self.wheel_indices] = True
        if self.lock_wheels:
            # Legs position; wheel hinges held at the default angle with the
            # lock PD gains from the policy config.
            q_des = self.q_default + np.where(is_wheel, 0.0, scaled)
            q_des[self.wheel_indices] = 0.0
            dq_des = np.zeros_like(q_des)
            return q_des, dq_des
        q_des = self.q_default + np.where(is_wheel, 0.0, scaled)
        dq_des = np.where(is_wheel, scaled, 0.0)
        return q_des, dq_des

    def step(self, command: np.ndarray | tuple[float, float, float] | None = None) -> dict:
        cmd = self.command if command is None else np.asarray(command, dtype=np.float64)
        return self.apply_action(self._infer(cmd), cmd)

    def apply_action(self, action: np.ndarray, command: np.ndarray | None = None) -> dict:
        """Apply a raw 16-D policy action (diagnostics / fixed-action probes)."""
        action = np.asarray(action, dtype=np.float64)
        cmd = self.command if command is None else np.asarray(command, dtype=np.float64)
        q_des, dq_des = self._targets(action)
        # rl_sim runs the limiter every 5 ms (4 times per 20 ms policy period)
        # and re-reads /motor_state each time.  Physics is 2 ms, so distribute
        # the 10 physics steps over the four cycles as evenly as possible
        # (2/3/2/3 => limiter updates at 4/10/14/20 ms of sim time).
        cycles = max(1, int(round(self.policy_dt / self.control_dt)))
        counts = [self.substeps // cycles] * cycles
        for i in range(self.substeps % cycles):
            counts[(2 * i + 1) % cycles] += 1
        for cycle in range(cycles):
            q_meas, dq_meas = self._measure()
            limited_q, limited_dq, kp, kd, tau_ff, _ = command_limit(
                q_des, dq_des, self.kp, self.kd, np.zeros_like(q_des),
                q_meas, dq_meas, self.prev_q_des, self.limits, self.control_dt,
            )
            self.prev_q_des = limited_q.copy()
            for _ in range(counts[cycle]):
                # sim_node recomputes PD every physics step from the current q/dq.
                self._pd_step(limited_q, limited_dq, kp, kd, tau_ff)
        self.sim_time = float(self.data.time - self.t0)
        return self.frame(cmd)

    def offset_spawn(self, dy: float = 0.0, dyaw_deg: float = 0.0) -> dict:
        """Shift base y and yaw after settle, before the policy starts.

        Deterministic initial-state perturbation for judge/controller coverage;
        the offset itself is not a policy action.
        """
        qpos = self.data.qpos
        qpos[1] += float(dy)
        if abs(dyaw_deg) > 1e-12:
            half = math.radians(float(dyaw_deg)) * 0.5
            qz = np.array((math.cos(half), 0.0, 0.0, math.sin(half)))
            q = _quat_mul(qz, qpos[3:7])
            qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(self.model, self.data)
        self.spawn_offset = {"dy": float(dy), "dyaw_deg": float(dyaw_deg)}
        return dict(self.spawn_offset)

    def warmup(self, command: np.ndarray, seconds: float) -> None:
        n = int(round(max(0.0, seconds) / self.policy_dt))
        for _ in range(n):
            self.step(command)
        self.mark_start()
        self.history = []
        self.last_action[:] = 0.0
        self.prev_q_des = None

    # -- trace ---------------------------------------------------------------
    def _support_point(self, leg: str) -> np.ndarray:
        """Spec P2-JUDGE-01: wheel support is the cylinder's lowest world point.

        For a cylinder with centre c, axis a (unit), radius r and half-length h,
        the point minimizing z is c - h*sign(a_z)*a + r*normalize(down - a*(a·down)).
        This is exact for the collision cylinder, not body-origin minus radius.
        """
        bid = self.support_bodies[leg]
        g = self.wheel_geoms.get(leg)
        if g is None:
            return np.asarray(self.data.xpos[bid], dtype=np.float64)
        center = np.asarray(self.data.geom_xpos[g], dtype=np.float64)
        axis = np.asarray(self.data.geom_xmat[g], dtype=np.float64).reshape(3, 3)[:, 2]
        radius = float(self.model.geom_size[g][0])
        half = float(self.model.geom_size[g][1])
        down = np.array([0.0, 0.0, -1.0])
        az = float(axis[2])
        axial = -math.copysign(half, az) * axis if abs(az) > 1e-9 else np.zeros(3)
        radial = down - axis * float(np.dot(axis, down))
        norm = float(np.linalg.norm(radial))
        if norm > 1e-9:
            radial = radial / norm * radius
        else:
            radial = np.zeros(3)
        return center + axial + radial

    def frame(self, command: np.ndarray | None = None) -> dict:
        d = self.data
        quat = np.array(d.qpos[3:7], dtype=np.float64)
        base = np.array(d.qpos[0:3], dtype=np.float64)
        R = _quat_to_mat(quat)
        tilt = float(math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2]))))))
        contacts: list[list[str]] = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom[0]))
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom[1]))
            if g1 and g2:
                contacts.append([g1, g2])
        tags, ground = classify_contacts(contacts)
        support = {
            leg: [float(v) for v in self._support_point(leg)]
            for leg in self.support_bodies
        }
        bar_z = float(d.xpos[self.bar_body][2]) if self.bar_body >= 0 else None
        return {
            "kind": "frame",
            "t": round(float(d.time - self.t0), 6),
            "base": [round(float(v), 9) for v in base],
            "quat": [round(float(v), 9) for v in quat],
            "yaw": round(_yaw_from_quat(quat), 9),
            "tilt_deg": round(tilt, 4),
            "foot_tags": {leg: sorted(tags[leg]) for leg in LEGS},
            "ground_feet": sorted(ground),
            "bar_z": bar_z,
            "support": {k: [round(float(v), 9) for v in xyz] for k, xyz in support.items()},
            "contacts": contacts,
            # Keep full precision: adaptive commands are closed-loop; rounding
            # to 6 decimals makes a 9 s replay diverge (chaotic contact).
            "cmd": [float(v) for v in (self.command if command is None else command)],
        }

    # -- result --------------------------------------------------------------
    def meta(self, steps: int | None = None) -> dict:
        meta: dict[str, Any] = {
            "kind": "meta",
            "schema": 1,
            "part": self.part,
            "robot": self.robot,
            "policy": self.policy_name,
            "scene": str(self.scene.relative_to(BASE)) if self.scene.is_relative_to(BASE)
            else str(self.scene),
            "scene_sha256": self.scene_sha256,
            "policy_sha256": self.policy_sha256,
            "config_sha256": self.config_sha256,
            "robot_sha256": self.robot_sha256,
            "command": [float(v) for v in self.command],
            "joint_mapping": self.mapping.tolist(),
            "contract_joint_mapping": self.contract_mapping.tolist(),
            "support_frame": self.support_frame,
            "spawn_offset": self.spawn_offset,
            "physics_dt": PHYSICS_DT,
            "control_dt": self.control_dt,
            "policy_dt": self.policy_dt,
            "duration_s": self.duration_s,
            "steps": steps,
            "seed": self.seed,
            "profile": self.profile,
            "warnings": self.warnings,
            "versions": {
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
        }
        if self.part == 2:
            meta["level"] = self.level
            meta["level_entry"] = self.level
        if self.depth_contract is not None:
            meta["depth"] = self.depth_meta()
        return meta

    def result_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and
                    np.isfinite(self.data.qvel).all())


def _judge_for(robot: str, part: int, out: Path) -> dict:
    """Replay the freshly written trace through the independent judge."""
    if part == 1:
        return replay_part1(out)
    if robot != "blackW":
        raise ContractError(
            "Part 2 is blackW-only in this repo (no black Part 2 scene variants)")
    return replay_part2_v5(out)


def run(
    robot: str,
    part: int,
    policy: str,
    level: str | None,
    out: Path,
    steps: int | None = None,
    scene: Path | None = None,
    *,
    duration_s: float | None = None,
    command: tuple[float, float, float] | None = None,
    settle_s: float = 1.0,
    warmup_s: float = 0.0,
    seed: int = 0,
    stop_on_fall: bool = True,
    profile: str = "diagnostic",
    command_fn=None,
    spawn_offset: tuple[float, float] | None = None,
    depth_contract: dict | str | Path | None = None,
) -> dict:
    """Run one policy trace and return the judge result.

    ``steps`` is kept for the existing tests/CLI.  When it is omitted the trial
    length is the official level ``seconds`` (Part 2) or ``duration_s``.
    ``depth_contract`` opts a policy into the depth-camera path; when it is
    omitted a ``depth.json`` next to the policy config is auto-detected.
    """
    if steps is not None and int(steps) < 1:
        raise ContractError(f"steps must be >= 1, got {steps}")
    started = time.time()
    runner = PolicyRunner(
        robot, part, policy, level, scene,
        steps=steps, duration_s=duration_s, command=command,
        settle_s=settle_s, seed=seed, profile=profile,
        depth_contract=depth_contract,
    )
    settle_info = runner.settle(settle_s)
    if spawn_offset is not None:
        settle_info["offset"] = runner.offset_spawn(*spawn_offset)
    if warmup_s > 0:
        runner.warmup(runner.command, warmup_s)
    if steps is not None:
        n_steps = max(0, int(steps))
        duration = n_steps * runner.policy_dt
    else:
        duration = float(runner.duration_s or 10.0)
        n_steps = max(1, int(math.ceil(duration / runner.policy_dt)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(json.dumps(runner.meta(n_steps)) + "\n")
        frames = 0
        stopped = "steps"
        for i in range(n_steps):
            cmd = command_fn(runner, i, runner.command) if command_fn else runner.command
            frame = runner.step(cmd)
            fh.write(json.dumps(frame) + "\n")
            frames += 1
            if not runner.result_finite():
                stopped = "non_finite"
                break
            if stop_on_fall and runner.part == 2:
                if float(frame["tilt_deg"]) > 80.0 or float(frame["base"][2]) < 0.12:
                    stopped = "fall"
                    break
        if i + 1 >= n_steps and stopped == "steps":
            stopped = "duration"
    finite = runner.result_finite()
    judge = _judge_for(robot, part, out)
    status = judge.get("status")
    if status is None and part == 1:
        status = "scored"
    result = {
        "robot": robot,
        "part": part,
        "policy": policy,
        "level": level,
        "scene": str(runner.scene),
        "steps": n_steps,
        "duration_s": duration,
        "sim_time_s": round(float(runner.data.time - runner.t0), 6),
        "frames": frames,
        "stopped": stopped,
        "settle": settle_info,
        "warmup_s": warmup_s,
        "trace": str(out),
        "finite": finite,
        "status": status,
        "score": judge.get("score"),
        "reason": judge.get("reason"),
        "policy_sha256": runner.policy_sha256,
        "scene_sha256": runner.scene_sha256,
        "config_sha256": runner.config_sha256,
        "robot_sha256": runner.robot_sha256,
        "warnings": runner.warnings,
        "command_mode": getattr(command_fn, "as_dict", lambda: None)(),
        "spawn_offset": runner.spawn_offset,
        "wall_time_s": round(time.time() - started, 3),
        "judge": judge,
    }
    if part == 2:
        result["category"] = runner.level.get("category")
        result["level_entry"] = runner.level
    if runner.depth_contract is not None:
        result["depth"] = runner.depth_meta()
    runner.close()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one policy through the eval judge")
    ap.add_argument("--robot", choices=["black", "blackW"], required=True)
    ap.add_argument("--part", type=int, choices=[1, 2], required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--level", default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="override duration with a fixed number of policy steps")
    ap.add_argument("--duration", type=float, default=None,
                    help="trial duration in simulated seconds (default: level seconds)")
    ap.add_argument("--command", type=float, nargs=3, default=None,
                    metavar=("VX", "VY", "WZ"))
    ap.add_argument("--adaptive", action="store_true",
                    help="state-feedback vx/wz: keep speed, y=0 and heading +x")
    ap.add_argument("--adaptive-vx", type=float, default=0.5)
    ap.add_argument("--k-y", type=float, default=0.6)
    ap.add_argument("--k-yaw", type=float, default=1.2)
    ap.add_argument("--max-wz", type=float, default=0.7)
    ap.add_argument("--spawn-y", type=float, default=0.0,
                    help="initial base y offset after settle (m)")
    ap.add_argument("--spawn-yaw", type=float, default=0.0,
                    help="initial base yaw offset after settle (deg)")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="fixed-gain settle before the policy starts")
    ap.add_argument("--warmup", type=float, default=0.0)
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--depth-contract", default=None,
                    help="depth contract JSON (default: auto-detect <policy>/depth.json)")
    ap.add_argument("--out", type=Path, default=BASE / "artifacts/traces/policy_trace.jsonl")
    ap.add_argument("--result", type=Path, default=None)
    a = ap.parse_args(argv)
    command_fn = AdaptiveCommand(vx=a.adaptive_vx, k_y=a.k_y, k_yaw=a.k_yaw,
                                 max_wz=a.max_wz) if a.adaptive else None
    result = run(
        a.robot, a.part, a.policy, a.level, a.out, a.steps, a.scene,
        duration_s=a.duration, command=tuple(a.command) if a.command else None,
        settle_s=a.settle, warmup_s=a.warmup, profile="diagnostic",
        command_fn=command_fn,
        spawn_offset=(a.spawn_y, a.spawn_yaw) if (a.spawn_y or a.spawn_yaw) else None,
        depth_contract=a.depth_contract,
    )
    if a.result:
        a.result.parent.mkdir(parents=True, exist_ok=True)
        a.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "robot": result["robot"],
        "part": result["part"],
        "policy": result["policy"],
        "level": result["level"],
        "status": result["status"],
        "score": result["score"],
        "reason": result["reason"],
        "finite": result["finite"],
        "sim_time_s": result["sim_time_s"],
        "frames": result["frames"],
        "stopped": result["stopped"],
        "warnings": result["warnings"],
        "trace": result["trace"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
