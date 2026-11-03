#!/usr/bin/env python3
"""Export a true 12-DoF locked-wheel actor and its 16-DoF deployment adapter.

The 12-DoF policy sees 45 values per frame:

    command 3 | base_ang_vel 3 | projected_gravity 3
    | leg joint_pos 12 | leg joint_vel 12 | last_action 12

and emits 12 leg actions (hip/thigh/calf in FL, FR, RL, RR order).  Wheel
joints are not part of the policy IO; the deployment lock controller owns them.

The existing C++/PolicyRunner interface is still the 16-DoF wheel-width vector.
``--adapter-out`` writes a TorchScript adapter that strips the four wheel
columns from the 57-wide observation, runs the 12-DoF policy, and pads four zero
wheel actions back to 16.  The lock code ignores those zeros and holds the
wheel hinges; the policy itself remains true 12-DoF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from alldog_mjlab.tasks.velocity.common.config import BLACK_CONFIG as C

# Deployment frame layouts.  frame-major newest->oldest for the C++/PolicyRunner.
LAYOUT16 = {
    "frame_dim": 57,
    "terms": ["command", "base_ang_vel", "projected_gravity",
              "joint_pos", "joint_vel", "actions"],
    "term_dims": [3, 3, 3, 16, 16, 16],
    "offsets": [0, 3, 6, 9, 25, 41],
    "wheel": [3, 7, 11, 15],
}
LAYOUT12 = {
    "frame_dim": 45,
    "terms": LAYOUT16["terms"],
    "term_dims": [3, 3, 3, 12, 12, 12],
    "offsets": [0, 3, 6, 9, 21, 33],
    "wheel": [],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_linear(actor):
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            return module
    raise ValueError("actor has no Linear layer")


def native_permutation(history: int, frame_dim: int, term_dims, offsets) -> np.ndarray:
    """native term-major oldest->newest -> frame-major newest->oldest."""
    perm = np.zeros(frame_dim * history, dtype=np.int64)
    native_block = 0
    for term_dim, term_offset in zip(term_dims, offsets):
        for frame in range(history):
            native_frame = history - 1 - frame
            for d in range(term_dim):
                target = frame * frame_dim + term_offset + d
                native = native_block + native_frame * term_dim + d
                perm[target] = native
        native_block += term_dim * history
    return perm


def prune16_to_12_columns() -> list[int]:
    """Columns of the 342-wide 16-DoF deployment observation to keep."""
    idx = []
    for frame in range(6):
        base = frame * LAYOUT16["frame_dim"]
        idx += [base + i for i in range(9)]  # command/ang/gravity
        for term, dim in (("joint_pos", 16), ("joint_vel", 16), ("actions", 16)):
            off = LAYOUT16["offsets"][LAYOUT16["terms"].index(term)]
            idx += [base + off + i for i in range(dim)
                    if i not in LAYOUT16["wheel"]]
    return idx


def make_adapter(policy12: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
    keep = torch.tensor(prune16_to_12_columns(), dtype=torch.long)

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = policy12
            self.keep = keep

        def forward(self, obs):
            y = self.policy(obs[:, self.keep])
            z = torch.zeros((obs.shape[0], 1), dtype=y.dtype, device=y.device)
            return torch.cat([y[:, 0:3], z, y[:, 3:6], z,
                              y[:, 6:9], z, y[:, 9:12], z], dim=1)

    return torch.jit.trace(Adapter().eval(), torch.zeros(1, 342))


def write_manifest(out_dir: Path, policy: Path, checkpoint: Path, actor_dim: int,
                   action_dim: int, layout: dict, label: str,
                   adapter: Path | None = None) -> None:
    exports = {
        "schema_version": 1,
        "robot": "blackW",
        "task": "blackw-vt-locked-s2",
        "frequency": {
            "physics_dt": C.control.physics_dt,
            "control_period": C.control.control_period,
            "policy_period": C.control.policy_period,
            "sim_decimation": C.control.sim_decimation,
            "deployment_decimation": C.control.deployment_decimation,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy": str(policy),
        "sha256": sha256(policy),
        "layout": "frame-major newest->oldest (HIM deployment)",
        "history_length": actor_dim // layout["frame_dim"],
        "actor_dim": actor_dim,
        "cnn_latent_dim": 0,
        "camera_shape": None,
        "action_contract": "locked12",
        "observation_layout": {
            "terms": layout["terms"],
            "term_dims": layout["term_dims"],
            "frame_dim": layout["frame_dim"],
            "history": actor_dim // layout["frame_dim"],
            "deploy_layout": [[d, o] for d, o in
                              zip(layout["term_dims"], layout["offsets"])],
        },
        "action_shape": [1, action_dim],
    }
    if adapter is not None:
        exports["adapter"] = {
            "policy": str(adapter),
            "sha256": sha256(adapter),
            "input_dim": 342,
            "output_dim": 16,
            "input_layout": "16-DoF frame-major newest->oldest",
        }
    manifest = {
        "schema_version": 1,
        "robot": "blackW",
        "policy": label,
        "weight_file": policy.name,
        "policy_sha256": sha256(policy),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "contract": {
            "num_dofs": action_dim,
            "num_observations": actor_dim,
            "history": actor_dim // layout["frame_dim"],
            "support_kind": "wheel",
            "action_contract": "locked12",
        },
        "export": exports,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="blackw-vt-locked-s2")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for the true-12-DoF policy")
    ap.add_argument("--adapter-out", type=Path, default=None,
                    help="optional directory for the 16-in/16-out adapter")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(args.task)), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    actor = runner.alg.get_policy()
    actor_dim = int(env.unwrapped.observation_manager.group_obs_dim["actor"][0])
    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    if actor_dim != 270 or action_dim != 12:
        raise SystemExit(f"expected 270 actor / 12 action, got {actor_dim}/{action_dim}")
    history = actor_dim // LAYOUT12["frame_dim"]
    perm = native_permutation(history, LAYOUT12["frame_dim"],
                              LAYOUT12["term_dims"], LAYOUT12["offsets"])
    linear = first_linear(actor)
    if int(linear.weight.shape[1]) != actor_dim:
        raise SystemExit(f"first Linear in={linear.weight.shape[1]} != {actor_dim}")
    linear.weight.data = linear.weight.data[:, torch.from_numpy(perm).to(linear.weight.device)].contiguous()
    args.out.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(str(args.out))
    policy_path = args.out / "policy.pt"
    policy12 = torch.jit.load(str(policy_path), map_location="cpu").eval()
    with torch.no_grad():
        a12 = policy12(torch.zeros(1, actor_dim))
    if tuple(a12.shape) != (1, 12):
        raise SystemExit(f"12-DoF policy output {tuple(a12.shape)}")
    write_manifest(args.out, policy_path, Path(args.checkpoint), actor_dim,
                   action_dim, LAYOUT12, args.out.name)
    print(f"exported 12-DoF policy -> {policy_path} action={tuple(a12.shape)}")
    if args.adapter_out is not None:
        args.adapter_out.mkdir(parents=True, exist_ok=True)
        adapter = make_adapter(policy12)
        adapter_path = args.adapter_out / "policy.pt"
        adapter.save(str(adapter_path))
        write_manifest(args.adapter_out, adapter_path, Path(args.checkpoint),
                       342, 16, LAYOUT16, args.adapter_out.name,
                       adapter=adapter_path)
        print(f"exported 16-DoF adapter -> {adapter_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
